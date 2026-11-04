"""Assemble the per-track intermediates into the training data root read by elaphros/data.py.

    python preprocessing/assemble.py --work /path/to/work --genome hg38.fa --folds sequences_human.bed.gz \
        --out /path/to/data --workers 8

Tracks that fail QC (missing autosomes, empty, crop/128 bp mismatch, many negative values) or whose strand
partner failed are dropped and listed in <out>/assembly_report.json. Each track's normaliser is the mean of
its non-zero values at native resolution over training regions only (1 bp assays: the training crops; ChIP:
128 bp bins outside folds 3 and 4).

Memory: each crop worker holds one 512-crop chunk of every 1 bp track (about 60 GB at 1,854 tracks) and each
g128 worker one chromosome of every track (about 18 GB for chr1), so size --workers to the host.
"""
import argparse
import csv
import gzip
import io
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import B1, B128, BIN, CHROMS, CHUNK, CROP, SLOTS, bf16, bins_of, chrom_sizes, load_crops, load_zst, unbf16

ORDER = B1 + B128
FIELDS = ["block", "track_id", "experiment", "assay", "target", "term_id", "term_name", "classification", "strand",
          "source", "file", "nonzero_mean"]


def columns(work):
    rows = list(csv.DictReader(open(Path(work) / "manifest.tsv"), delimiter="\t"))
    keep, drop = [], []
    for r in rows:
        d = Path(work) / "tracks" / r["track_id"]
        if not (d / "DONE").exists():
            drop.append((r["track_id"], "not processed")); continue
        st = json.load(open(d / "stats.json"))
        bad = []
        if r["block"] in B1 and not st.get("crop_vs_g128_ok", True):
            bad.append("crop_vs_g128")
        if [c for c in st.get("chroms_missing", []) if c != "chrX"]:
            bad.append(f"missing {st['chroms_missing']}")
        if st.get("nz_count", 0) == 0:
            bad.append("empty")
        if st.get("neg_clipped", 0) > 0.001 * max(st.get("nz_count", 1), 1):
            bad.append("negative values")
        if bad:
            drop.append((r["track_id"], ";".join(bad))); continue
        keep.append(r)
    by = {}
    for r in keep:
        by.setdefault((r["block"], r["experiment"]), []).append(r)
    final = []
    for rs in by.values():
        st = {r["strand"] for r in rs}
        if st & {"+", "-"} and st != {"+", "-"}:
            drop += [(r["track_id"], "strand pair incomplete") for r in rs]
        else:
            final += rs
    cols = {}
    for b in ORDER:
        cols[b] = sorted((r for r in final if r["block"] == b),
                         key=lambda r: (r["source"], r["term_id"], r["target"], r["assay"], r["experiment"], r["strand"]))
        assert len(cols[b]) <= SLOTS[b], (b, len(cols[b]))
    return cols, drop


def heldout_mask(genome, folds):
    sz = chrom_sizes(genome); nb = bins_of(sz)
    off = dict(zip(CHROMS, np.cumsum([0] + [nb[c] for c in CHROMS[:-1]])))
    m = np.zeros(int(sum(nb.values())), bool)
    for line in gzip.open(folds, "rt"):
        c, s, e, f = line.split()
        if c in off and f in ("fold3", "fold4"):
            m[off[c] + int(s) // BIN:off[c] + (int(e) + BIN - 1) // BIN] = True
    return m


_HO = {}


def nonzero_mean(args):
    tid, block, n_train, work, genome, folds = args
    d = Path(work) / "tracks" / tid
    if block in B1:
        s = n = 0
        for k in range((n_train + CHUNK - 1) // CHUNK):
            x = load_zst(d / f"crops_{k}.zst")[:max(0, n_train - k * CHUNK)]
            v = unbf16(x[x != 0]); s += float(v.sum(dtype=np.float64)); n += v.size
    else:
        if "m" not in _HO:                              # computed once per worker process
            _HO["m"] = heldout_mask(genome, folds)
        g = np.fromfile(d / "g128.u16", np.uint16)[~_HO["m"]]
        v = unbf16(g[g != 0]); s, n = float(v.sum(dtype=np.float64)), v.size
    return tid, s / n if n else float("nan")


def g128_chrom(args):
    """One chromosome's (bins, n_cols) array. ChIP intermediates hold 128 bp means; the cache stores every
    128 bp value as a bin sum, so ChIP is multiplied by 128 (exact in bf16)."""
    c, ids, chip, out, work, genome = args
    sz = chrom_sizes(genome); nb = bins_of(sz)
    off = int(sum(nb[x] for x in CHROMS[:CHROMS.index(c)])); tot = int(sum(nb.values()))
    a = np.empty((nb[c], len(ids)), np.uint16)
    for j, t in enumerate(ids):
        a[:, j] = np.memmap(Path(work) / "tracks" / t / "g128.u16", np.uint16, "r", shape=(tot,))[off:off + nb[c]]
    if chip.any():
        a[:, chip] = bf16(unbf16(a[:, chip]) * np.float32(128))
    a.tofile(out / f"{c}.u16.tmp"); os.replace(out / f"{c}.u16.tmp", out / f"{c}.u16")
    np.ones(nb[c], np.uint8).tofile(out / f"{c}.fill")
    return c


def crops_chunk(args):
    import zstandard
    k, cols, crops, out, work = args
    sub = crops[k * CHUNK:(k + 1) * CHUNK]
    arr = {}
    for b in B1:
        ids = [r["track_id"] for r in cols[b]]
        x = np.empty((len(sub), CROP, len(ids)), np.uint16)
        for j, t in enumerate(ids):
            x[:, :, j] = load_zst(Path(work) / "tracks" / t / f"crops_{k}.zst")
        arr[b] = x
    zc = zstandard.ZstdCompressor(level=3)
    for i, c in enumerate(sub):
        buf = io.BytesIO()
        np.savez(buf, chrom=np.asarray(c["chrom"]), start=np.asarray(int(c["start"])), **{b: arr[b][i] for b in B1})
        dst = out / "crops" / c["split"] / c["fn"]
        tmp = Path(str(dst) + ".tmp"); tmp.write_bytes(zc.compress(buf.getvalue())); os.replace(tmp, dst)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    for d in ("genome", "g128", "crops/train", "crops/valid", "crops/test"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for src, dst in ((a.genome, "genome/hg38.fa"), (a.genome + ".fai", "genome/hg38.fa.fai")):
        if not (out / dst).exists():
            os.symlink(Path(src).resolve(), out / dst)
    shutil.copy(a.folds, out / "folds.bed.gz")
    cols, drop = columns(a.work)
    crops = load_crops(a.work)
    n_train = sum(c["split"] == "train" for c in crops)
    assert all(c["split"] == "train" for c in crops[:n_train]), "crops.tsv must list training crops first"

    jobs = [(r["track_id"], b, n_train, a.work, a.genome, a.folds) for b in ORDER for r in cols[b]]
    with ProcessPoolExecutor(a.workers) as pp:
        nzm = dict(pp.map(nonzero_mean, jobs, chunksize=4))
    with open(out / "tracks.tsv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for b in ORDER:
            for r in cols[b]:
                assert nzm[r["track_id"]] > 0, r["track_id"]
                w.writerow({**r, "nonzero_mean": f"{nzm[r['track_id']]:.8g}"})
    lay, o = {}, 0
    for b in ORDER:
        lay[b] = [o, o + len(cols[b])]; o += len(cols[b])
    json.dump({"columns": lay, "n_cols": o, "crop": CROP, "bin": BIN}, open(out / "layout.json", "w"), indent=1)
    for split in ("train", "valid", "test"):
        json.dump([[c["fn"], c["chrom"], int(c["start"]), int(c["rec_start"]), int(c["rec_len"])]
                   for c in crops if c["split"] == split], open(out / "crops" / f"{split}.json", "w"))

    ids = [r["track_id"] for b in ORDER for r in cols[b]]
    chip = np.array([b in B128 for b in ORDER for _ in cols[b]])
    with ProcessPoolExecutor(a.workers) as pp:
        for c in pp.map(g128_chrom, [(c, ids, chip, out / "g128", a.work, a.genome) for c in CHROMS]):
            print("g128", c, flush=True)
        for k in pp.map(crops_chunk, [(k, cols, crops, out, a.work) for k in range((len(crops) + CHUNK - 1) // CHUNK)]):
            print("crops chunk", k, flush=True)
    json.dump({"columns": {b: len(cols[b]) for b in ORDER}, "n_cols": o, "dropped": drop},
              open(out / "assembly_report.json", "w"), indent=1)
    print({b: len(cols[b]) for b in ORDER}, "dropped", len(drop))


if __name__ == "__main__":
    main()
