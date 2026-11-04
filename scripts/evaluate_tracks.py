"""Held-out track and gene-level accuracy for Elaphros, Elaphros-Conv and the comparators.

    python scripts/evaluate_tracks.py --data DATA --model elaphros --ckpt runs/elaphros/snap_step060000.pt \
        --match-targets borzoi_targets_human.txt --gtf gencode.v48.basic.annotation.gtf --out results/elaphros.json
    python scripts/evaluate_tracks.py --data DATA --model borzoi --targets borzoi_targets_human.txt \
        --gtf gencode.v48.basic.annotation.gtf --out results/borzoi.json
    python scripts/evaluate_tracks.py --data DATA --model enformer --targets enformer_targets_human.txt \
        --span 32768 --exclude-bed enformer_train_intervals.bed --out results/enformer.json

Every window is the model's native input centred on one fold 3 (valid) or fold 4 (test) crop. Predictions on the
central --span bp are decoded to measured units per 128 bp bin (Borzoi's 32 bp bins are summed). Comparators
are scored on our tracks that share an experiment accession, assay and strand with one of their output
channels (several channels are averaged); --match-targets restricts an Elaphros run to the same tracks.

Track metric: per-track Pearson pooled over all measured bins of all windows of a fold, on raw values and on
log1p values after a per-track scale fitted on the other fold (prediction total = truth total) and frozen;
mean within assay, macro mean across assays. Gene metric (--gtf): RNA-seq summed over exon bins and CAGE over
the TSS bin and its neighbours, log2(x + 1) Pearson across genes per track with the same cross-fold scale.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elaphros import baselines as BL  # noqa: E402
from elaphros.data import BLOCKS, Cache, Tracks  # noqa: E402
from elaphros.losses import unscale_targets  # noqa: E402
from elaphros.model import load_checkpoint  # noqa: E402

FAMILY = {"CHIP_TF": "TF", "CHIP_HISTONE": "HISTONE"}


class Predictor:
    def __init__(self, a, tracks, dev):
        self.name, self.dev, self.tracks = a.model, dev, tracks
        rows = tracks.rows
        if a.model.startswith("elaphros"):
            self.m, self.step = load_checkpoint(a.ckpt, dev)
            self.L, self.bin, self.off = 524288, 128, 0
            keep = [bool(c) for c in BL.exact_map(rows, BL.targets(a.match_targets))] if a.match_targets else [True] * len(rows)
            self.cols = [i for i, k in enumerate(keep) if k]
        else:
            self.m, self.step = BL.load(a.model, dev, flash=a.flash), None
            self.L, self.bin, self.off = BL.GEOM[a.model]
            self.chans = BL.targets(a.targets)
            cm = BL.exact_map(rows, self.chans)
            self.cols = [i for i, c in enumerate(cm) if c]
            self.cmap = {i: cm[i] for i in self.cols}
            self.sel = sorted({c for i in self.cols for c in cm[i]})
            self.pos = {c: j for j, c in enumerate(self.sel)}

    @torch.no_grad()
    def __call__(self, x, lo, nb):
        """x (1, L, 4); returns (n_cols, nb) decoded 128 bp values for bins [lo, lo + nb) of the window."""
        if self.name.startswith("elaphros"):
            out = self.m(x, heads_1bp=False)
            p = []
            for b in BLOCKS:
                k = self.tracks.n(b)
                z = out[f"{b.lower()}@128"][:, :k, lo:lo + nb].float().cpu()
                p.append(unscale_targets(z, self.tracks.nzm[b], 128, b == "RNA_SEQ")[0])
            return torch.cat(p)[self.cols].double().numpy()
        y = BL.predict(self.name, self.m, x)[0].float().cpu()
        s, f = (lo * 128 - self.off) // self.bin, nb * 128 // self.bin
        assert 0 <= s and s + f <= y.shape[-1], f"--span {nb * 128} exceeds {self.name}'s output; use --span <= {y.shape[-1] * self.bin}"
        y = BL.decode(self.name, y[self.sel, s:s + f].double(), [self.chans[c] for c in self.sel])
        if self.bin == 32:
            y = y.view(len(self.sel), nb, 4).sum(-1)
        return torch.stack([y[[self.pos[c] for c in self.cmap[i]]].mean(0) for i in self.cols]).numpy()


def windows(cache, L, exclude):
    for fn, c, cs, *_ in sorted(cache.crops):
        mid = cs + 16384
        s = mid - L // 2
        if s < 0 or s + L > cache.genome.idx[c][0]:
            continue
        if any(c == ec and es < mid + 16384 and ee > mid - 16384 for ec, es, ee in exclude):
            continue
        yield fn, c, mid, s


def genes_in(gtf, chrom_spans):
    """Protein-coding genes (GENCODE gene_type, 'basic' transcripts when tagged) whose exons and TSSs lie inside one
    scored span: [(span index, strand, exon bins, TSS bins)]."""
    ex, tss, st, ch = defaultdict(list), defaultdict(set), {}, {}
    for l in open(gtf):
        if l[0] == "#":
            continue
        f = l.split("\t", 9)
        if f[2] not in ("exon", "transcript") or 'gene_type "protein_coding"' not in f[8] \
                or ('tag "basic"' not in f[8] and "tag " in f[8]) or "readthrough" in f[8]:
            continue
        g = re.search(r'gene_id "([^".]+)', f[8]).group(1)
        a, b = int(f[3]) - 1, int(f[4]); st[g], ch[g] = f[6], f[0]
        if f[2] == "exon":
            ex[g].append((a, b))
        else:
            tss[g].add(a if f[6] == "+" else b - 1)
    out, seen = [], set()
    for w, (c, s0, s1) in enumerate(chrom_spans):
        for g in ex:
            if g in seen or ch[g] != c or not tss[g]:
                continue
            lo, hi = min(a for a, _ in ex[g]), max(b for _, b in ex[g])
            if lo >= s0 and hi <= s1 and min(tss[g]) - 128 >= s0 and max(tss[g]) + 128 < s1:
                e = sorted({i for a, b in ex[g] for i in range((a - s0) // 128, (b - 1 - s0) // 128 + 1)})
                t = sorted({j for x in tss[g] for j in ((x - s0) // 128 - 1, (x - s0) // 128, (x - s0) // 128 + 1)})
                out.append((w, st[g], e, t)); seen.add(g)
    return out


def pearson_rows(p, t):
    p = p - p.mean(1, keepdims=True); t = t - t.mean(1, keepdims=True)
    r = (p * t).sum(1) / np.maximum(np.sqrt((p * p).sum(1) * (t * t).sum(1)), 1e-30)
    r[(t * t).sum(1) == 0] = np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="elaphros", choices=["elaphros", "elaphros-conv", "borzoi", "flashzoi", "enformer"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--targets", default="", help="comparator target table")
    ap.add_argument("--match-targets", default="", help="score Elaphros only on tracks this target table matches")
    ap.add_argument("--flash", default="port", help="Flashzoi attention: 'port' (plain PyTorch) or 'native'")
    ap.add_argument("--span", type=int, default=196608, help="scored bp at the window centre")
    ap.add_argument("--exclude-bed", default="", help="skip windows whose central 32 kb overlaps these intervals")
    ap.add_argument("--gtf", default="", help="GENCODE annotation for the gene-level metric")
    ap.add_argument("--limit", type=int, default=0, help="windows per fold (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = Tracks(a.data)
    P = Predictor(a, tracks, dev)
    exclude = [(l.split()[0], int(l.split()[1]), int(l.split()[2])) for l in open(a.exclude_bed)] if a.exclude_bed else []
    nb = a.span // 128
    folds = {}
    for sp in ("valid", "test"):
        cache = Cache(a.data, sp)
        T, Y, M, spans = [], [], [], []
        for n, (fn, c, mid, s) in enumerate(windows(cache, P.L, exclude)):
            if a.limit and n >= a.limit:
                break
            x = torch.from_numpy(cache.genome.onehot(c, s, s + P.L))[None].to(dev)
            lo = (mid - a.span // 2 - s) // 128
            y, fill = cache.window(c, mid - a.span // 2, a.span)
            T.append(np.clip(y[P.cols].astype(np.float64), 0, None)); Y.append(np.clip(P(x, lo, nb), 0, None))
            M.append(fill); spans.append((c, mid - a.span // 2, mid + a.span // 2))
        folds[sp] = (np.stack(T), np.stack(Y), np.stack(M), spans)        # (windows, cols, bins)
    scale = {}
    for sp, (T, Y, M, _) in folds.items():
        m = M[:, None, :]
        scale[sp] = np.nan_to_num((T * m).sum((0, 2)) / np.maximum((Y * m).sum((0, 2)), 1e-9), nan=1.0, posinf=1.0)
    blocks = [r["block"] for r in tracks.rows]
    res = {"model": a.model, "step": P.step, "n_tracks": len(P.cols), "span_bp": a.span,
           "tracks": [tracks.rows[i]["track_id"] for i in P.cols]}
    for sp, other in (("valid", "test"), ("test", "valid")):
        T, Y, M, spans = folds[sp]
        t = T.transpose(1, 0, 2)[:, M]; y = Y.transpose(1, 0, 2)[:, M]
        r = {"raw": pearson_rows(y, t), "log1p": pearson_rows(np.log1p(y * scale[other][:, None]), np.log1p(t))}
        out = {"windows": len(spans), "per_track": {k: [None if np.isnan(v) else round(float(v), 4) for v in vals]
                                                     for k, vals in r.items()}}
        for k, vals in r.items():
            fam = defaultdict(list)
            for j, i in enumerate(P.cols):
                fam[FAMILY.get(blocks[i], blocks[i])].append(vals[j])
            out[k] = {f: round(float(np.nanmean(v)), 4) for f, v in fam.items()}
            out[k]["macro"] = round(float(np.mean(list(out[k].values()))), 4)
        if a.gtf:
            gl = genes_in(a.gtf, spans)
            gene = defaultdict(list)
            for j, i in enumerate(P.cols):
                b, strand = blocks[i], tracks.rows[i]["strand"]
                if b not in ("RNA_SEQ", "CAGE"):
                    continue
                g = [(w, e if b == "RNA_SEQ" else tt) for w, gs, e, tt in gl if strand == "." or strand == gs]
                if len(g) < 3:
                    continue
                tv = np.array([T[w, j, bins].sum() for w, bins in g])
                yv = np.array([Y[w, j, bins].sum() for w, bins in g]) * scale[other][j]
                if tv.std() > 0 and yv.std() > 0:
                    gene[b].append(np.corrcoef(np.log2(yv + 1), np.log2(tv + 1))[0, 1])
            out["gene"] = {b: round(float(np.mean(v)), 4) for b, v in gene.items()}
            out["n_genes"] = len(gl)
        res[sp] = out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print(json.dumps({sp: {k: v for k, v in res[sp].items() if k != "per_track"} for sp in ("valid", "test")}, indent=1))


if __name__ == "__main__":
    main()
