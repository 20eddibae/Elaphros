"""Training data: a genome-wide 128 bp track cache, 32,768 bp crops of 1 bp tracks, and the batch sampler.

Data root layout (written by preprocessing/assemble.py):

    genome/hg38.fa, hg38.fa.fai     reference genome
    folds.bed.gz                     Borzoi's 8-fold split (chrom, start, end, foldN)
    tracks.tsv                       one row per track column (block, track_id, experiment, assay, target, term_id,
                                     term_name, classification, strand, source, file, nonzero_mean)
    layout.json                      {"columns": {BLOCK: [first, last+1]}, "n_cols", "crop", "bin"}
    g128/<chrom>.u16, <chrom>.fill   all columns at 128 bp, bf16 bits, (bins, n_cols); fill = 1 where measured
    crops/<split>/<file>.npz.zst     1 bp tracks over one 32,768 bp crop: one (32768, n) bf16 array per 1 bp block
    crops/<split>.json               [[file, chrom, start, interval_start, interval_length], ...]

Splits follow Borzoi: train = folds 0, 1, 2, 5, 6, 7; valid = fold 3; test = fold 4.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

B1 = ["ATAC", "DNASE", "PROCAP", "CAGE", "RNA_SEQ"]       # predicted at 1 bp and 128 bp
B128 = ["CHIP_TF", "CHIP_HISTONE"]                         # predicted at 128 bp
BLOCKS = B1 + B128
HELDOUT_FOLDS = ("fold3", "fold4")
STAGES = [(65536, 8), (131072, 4), (524288, 1)]           # (window bp, batch): 512 kb of sequence per step
VAL_L, VAL_CORE = 524288, 131072                           # validation window and its scored centre


def stage_at(step, total, fracs=(0.4, 0.4, 0.2)):
    acc = 0.0
    for (L, B), f in zip(STAGES, fracs):
        acc += f
        if step < acc * total:
            return L, B
    return STAGES[-1]


def bf16_to_f32(u):
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def _read(path, tries=6):
    for i in range(tries):
        try:
            return Path(path).read_bytes()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def read_crop(path):
    import zstandard
    return dict(np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(_read(path)))))


class Genome:
    """Indexed FASTA reader (needs the .fai)."""

    def __init__(self, fa):
        self.fa = Path(fa)
        self.idx = {}
        for line in open(str(self.fa) + ".fai"):
            c, ln, off, lb, lbytes = line.split("\t")[:5]
            self.idx[c] = (int(ln), int(off), int(lb), int(lbytes))
        self.fh = open(self.fa, "rb")

    def fetch(self, c, s, e):
        ln, off, lb, lbytes = self.idx[c]
        s2, e2 = max(0, s), min(ln, e)
        out = bytearray(b"N" * (e - s))
        if e2 > s2:
            a = off + (s2 // lb) * lbytes + s2 % lb
            b = off + (e2 // lb) * lbytes + e2 % lb
            self.fh.seek(a)
            out[s2 - s:e2 - s] = self.fh.read(b - a).replace(b"\n", b"").upper()
        return bytes(out)

    def onehot(self, c, s, e):
        lut = np.full(256, -1, np.int8)
        for i, b in enumerate(b"ACGT"):
            lut[b] = i
        idx = lut[np.frombuffer(self.fetch(c, s, e), np.uint8)]
        x = np.zeros((e - s, 4), np.float32)
        ok = idx >= 0
        x[np.nonzero(ok)[0], idx[ok]] = 1
        return x



def read_folds(root):
    rows = [l.rstrip("\n").split("\t") for l in gzip.open(Path(root) / "folds.bed.gz", "rt")]
    return [(c, int(s), int(e), f) for c, s, e, f in rows]


class Tracks:
    """Track table: column ranges, per-track normalisers and the strand-swap permutation per block.
    `use[b]` lists the columns (positions within the block) that enter the loss; track j of block b is
    head slot j."""

    def __init__(self, root):
        root = Path(root)
        lay = json.load(open(root / "layout.json"))
        rows = list(csv.DictReader(open(root / "tracks.tsv"), delimiter="\t"))
        self.rows = rows
        self.cols, self.nzm, self.swap, self.use = {}, {}, {}, {}
        for b in BLOCKS:
            c0, c1 = lay["columns"][b]
            self.cols[b] = (c0, c1)
            rs = rows[c0:c1]
            assert all(r["block"] == b for r in rs), f"tracks.tsv rows are out of column order for {b}"
            self.nzm[b] = torch.tensor([float(r["nonzero_mean"]) for r in rs], dtype=torch.float32)
            other = {"+": "-", "-": "+"}
            pos = {(r["experiment"], r["strand"]): j for j, r in enumerate(rs)}
            self.swap[b] = torch.tensor([pos.get((r["experiment"], other.get(r["strand"], r["strand"])), j)
                                         for j, r in enumerate(rs)], dtype=torch.long)
            self.use[b] = torch.arange(c1 - c0)

    def n(self, b):
        return self.cols[b][1] - self.cols[b][0]


class Cache:
    """Reads g128 and crops for one split, and samples training windows clear of held-out intervals."""

    def __init__(self, root, split="train", rng=None):
        self.root = Path(root)
        lay = json.load(open(self.root / "layout.json"))
        self.n, self.crop = lay["n_cols"], lay["crop"]
        self.genome = Genome(self.root / "genome/hg38.fa")
        self.G, self.F = {}, {}
        for f in (self.root / "g128").glob("*.u16"):
            nb = f.stat().st_size // (2 * self.n)
            self.G[f.stem] = np.memmap(f, np.uint16, "r", shape=(nb, self.n))
            self.F[f.stem] = np.memmap(f.with_suffix(".fill"), np.uint8, "r", shape=(nb,))
        self.split = split
        self.crops = [e for e in json.load(open(self.root / "crops" / f"{split}.json")) if e[1] in self.G]
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # training windows may not overlap the 1 Mb around any valid/test interval, and held-out bins never
        # enter the loss
        ho = [r for r in read_folds(self.root) if r[3] in HELDOUT_FOLDS]
        self.block, self.ho_bins = {}, {}
        for c, s, e, _ in ho:
            m = (s + e) // 2
            self.block.setdefault(c, []).append((m - 524288, m + 524288))
            self.ho_bins.setdefault(c, []).append((s // 128, (e + 127) // 128))
        self.block = {c: np.array(v) for c, v in self.block.items()}
        self.ho_bins = {c: np.array(v) for c, v in self.ho_bins.items()}

    def _clear(self, c, s, e):
        b = self.block.get(c)
        return b is None or not np.any((b[:, 0] < e) & (b[:, 1] > s))

    def window(self, c, s, L):
        """y128 (n_cols, L/128) float32 and fill (L/128,) bool for [s, s+L)."""
        b0, b1 = s // 128, (s + L) // 128
        y = bf16_to_f32(self.G[c][b0:b1]).T
        return y, np.asarray(self.F[c][b0:b1]).astype(bool)

    def sample(self, L, off_tok):
        """A training window of L bp containing a random 1 bp crop at token offset off_tok."""
        for _ in range(200):
            fn, c, cs = self.crops[self.rng.integers(len(self.crops))][:3]
            s = cs - off_tok * 128
            if s < 0 or (s + L) // 128 > self.G[c].shape[0] or not self._clear(c, s, s + L):
                continue
            y, fill = self.window(c, s, L)
            hb = self.ho_bins.get(c)
            if hb is not None:
                for b0, b1 in hb[(hb[:, 0] < (s + L) // 128) & (hb[:, 1] > s // 128)]:
                    fill[max(b0 - s // 128, 0):max(b1 - s // 128, 0)] = False
            if fill.mean() < 0.5:
                continue
            cr = read_crop(self.root / "crops" / self.split / fn)
            return self.genome.onehot(c, s, s + L), y, fill, cr, (cs - s) // 128
        raise RuntimeError("no admissible training window found")


def make_batch(cache, B, L, crop_len):
    off_tok = int(cache.rng.integers(0, (L - crop_len) // 128 + 1))      # one crop offset per batch
    xs, ys, fs, y1, cs = [], [], [], {b: [] for b in B1}, []
    for _ in range(B):
        x, y, f, cr, off = cache.sample(L, off_tok)
        xs.append(torch.from_numpy(x)); ys.append(torch.from_numpy(np.ascontiguousarray(y))); fs.append(torch.from_numpy(f))
        for b in B1:
            y1[b].append(torch.from_numpy(cr[b][:crop_len].view(np.int16)).view(torch.bfloat16))
        cs.append(off)
    return {"x": torch.stack(xs), "y128": torch.stack(ys), "fill": torch.stack(fs),
            "y1": {b: torch.stack(v) for b, v in y1.items()}, "crop_tok": cs}


def rc_batch(bt, tracks, ctok):
    """Reverse-complement a batch: DNA, every target (positions reversed, stranded pairs swapped), the
    DNase 1 bp cut-site shift, and the crop offset."""
    bt["x"] = bt["x"].flip(1).flip(2)
    bt["y128"] = bt["y128"].flip(-1)
    bt["fill"] = bt["fill"].flip(-1)
    for b in BLOCKS:
        c0, c1 = tracks.cols[b]
        bt["y128"][:, c0:c1] = bt["y128"][:, c0:c1][:, tracks.swap[b]]
    for b in B1:
        y = bt["y1"][b].flip(1)[..., tracks.swap[b]]
        if b == "DNASE":
            y = torch.roll(y, 1, 1)
            y[:, 0] = 0
        bt["y1"][b] = y
    ntok = bt["fill"].shape[-1]
    bt["crop_tok"] = [ntok - c - ctok for c in bt["crop_tok"]]
    return bt


class BatchStream(torch.utils.data.IterableDataset):
    def __init__(self, root, L, B, crop, seed):
        self.root, self.L, self.B, self.crop, self.seed = root, L, B, crop, seed

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, w.id if w else 0, self.L])
        tracks, cache = Tracks(self.root), Cache(self.root, "train", rng)
        while True:
            bt = make_batch(cache, self.B, self.L, self.crop)
            if rng.random() < 0.5:
                bt = rc_batch(bt, tracks, self.crop // 128)
            yield bt


def loader(root, L, B, crop, seed, workers, prefetch=2):
    return iter(torch.utils.data.DataLoader(
        BatchStream(root, L, B, crop, seed), batch_size=None, num_workers=workers,
        prefetch_factor=prefetch if workers else None, persistent_workers=bool(workers),
        pin_memory=torch.cuda.is_available()))


def eval_batches(root, split, n, crop_len, L=VAL_L):
    """Fixed evaluation windows of L bp, each centred on one valid/test interval and fully measured over its
    central VAL_CORE. Returns (batches, windows)."""
    cache = Cache(root, split)
    out, wins = [], []
    for fn, c, cs, start, ln in sorted(cache.crops):
        s = ((start + ln // 2 - L // 2) // 128) * 128
        if s < 0 or (s + L) // 128 > cache.G[c].shape[0]:
            continue
        y, fill = cache.window(c, s, L)
        if not fill[(L - VAL_CORE) // 256:(L + VAL_CORE) // 256].all():
            continue
        cr = read_crop(cache.root / "crops" / split / fn)
        out.append({"x": torch.from_numpy(cache.genome.onehot(c, s, s + L))[None],
                    "y128": torch.from_numpy(np.ascontiguousarray(y))[None], "fill": torch.from_numpy(fill)[None],
                    "y1": {b: torch.from_numpy(cr[b][:crop_len].view(np.int16)).view(torch.bfloat16)[None] for b in B1},
                    "crop_tok": [(cs - s) // 128]})
        wins.append((c, s, s + L))
        if len(out) == n:
            break
    return out, wins
