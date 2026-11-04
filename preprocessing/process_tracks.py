"""Download each manifest track from ENCODE and convert it to the per-track intermediate format (common.py).

    python preprocessing/process_tracks.py --work /path/to/work --genome hg38.fa --shard 0/8 --workers 16

FANTOM5 CTSS (output_type 'ctss'): 1 bp tag counts of the row's strand.
bigWig tracks: replicate files are summed; minus-strand signal that ENCODE stores as negative values is
sign-flipped; NaN is set to 0. BAM tracks (ATAC, DNase): 5' read ends of every mate that passes
`samtools view -F 780`, shifted per BAM_TRANSFORM, summed over replicates. Each raw file is md5-checked,
converted and deleted, so local disk holds at most `workers` raw files. Shards split the manifest by crc32 of
the track id, so independent jobs can run in parallel. Finished tracks are skipped.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import B1, BIN, CHROMS, CHUNK, CROP, bf16, bins_of, chrom_sizes, load_crops, save_zst, unbf16

FLAG_MASK = 0x30C        # unmapped, mate unmapped, secondary, QC-fail


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(row, tmp):
    paths = []
    for u, m, acc in zip(row["url"].split("|"), row["md5"].split("|"), row["file"].split("|")):
        p = Path(tmp) / (f"{acc}.bam" if u.endswith(".bam") else f"{acc}.bed.gz" if u.endswith(".gz") else f"{acc}.bigWig")
        for _ in range(4):
            try:
                with urllib.request.urlopen(u, timeout=600) as r, open(p, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 24)
            except OSError:
                continue
            if md5(p) == m:
                break
        else:
            raise RuntimeError(f"download or md5 failed: {acc}")
        paths.append(p)
    return paths


def bigwig_signal(paths, strand, sizes):
    """Yields (chrom, 1 bp signal) with replicate files summed; also returns QC counters."""
    import pyBigWig
    bws = [pyBigWig.open(str(p)) for p in paths]
    flips = [strand == "-" and h["maxVal"] <= 0 < -h["minVal"] for h in (b.header() for b in bws)]
    qc = {"neg_flipped": any(flips), "nan": 0, "neg_clipped": 0, "chroms_missing": []}
    for c in CHROMS:
        v = np.zeros(sizes[c], np.float32)
        seen = False
        for bw, fl in zip(bws, flips):
            if c not in bw.chroms():
                continue
            seen = True
            x = bw.values(c, 0, min(sizes[c], bw.chroms()[c]), numpy=True).astype(np.float32, copy=False)
            nan = np.isnan(x)
            qc["nan"] += int(nan.sum()); x[nan] = 0
            v[:len(x)] += -x if fl else x
        if not seen:
            qc["chroms_missing"].append(c)
        qc["neg_clipped"] += int((v < 0).sum())
        yield c, np.maximum(v, 0), qc
    for b in bws:
        b.close()


def ctss_signal(paths, strand, sizes):
    """FANTOM5 CTSS bed (chrom, start, end, name, count, strand): counts of one strand, duplicates summed."""
    import gzip
    pos = {c: ([], []) for c in CHROMS}
    qc = {"chroms_missing": [], "neg_clipped": 0}
    for p in paths:
        for line in gzip.open(p, "rt"):
            f = line.split("\t")
            if f[0] in pos and f[5].strip() == strand:
                pos[f[0]][0].append(int(f[1])); pos[f[0]][1].append(float(f[4]))
    for c in CHROMS:
        x, w = np.asarray(pos[c][0], np.int64), np.asarray(pos[c][1], np.float64)
        ok = (x >= 0) & (x < sizes[c])
        qc["neg_clipped"] += int((w < 0).sum())
        v = np.bincount(x[ok], np.clip(w[ok], 0, None), minlength=sizes[c]).astype(np.float32)
        if not v.any():
            qc["chroms_missing"].append(c)
        yield c, v, qc


def bam_signal(paths, transform, sizes):
    import pysam
    from array import array
    sp, sm = (int(x) for x in transform.split(":")[1].split("/"))
    pos = {c: array("i") for c in CHROMS}
    qc = {"reads": 0, "kept": 0, "chroms_missing": []}
    for p in paths:
        with pysam.AlignmentFile(str(p), "rb", threads=2) as f:
            for r in f.fetch(until_eof=True):
                qc["reads"] += 1
                if r.flag & FLAG_MASK or r.reference_name not in pos:
                    continue
                pos[r.reference_name].append(r.reference_end - 1 + sm if r.is_reverse else r.reference_start + sp)
                qc["kept"] += 1
    for c in CHROMS:
        x = np.frombuffer(pos.pop(c), dtype=np.int32).astype(np.int64)
        x = x[(x >= 0) & (x < sizes[c])]
        v = np.bincount(x, minlength=sizes[c]).astype(np.float32)
        if not v.any():
            qc["chroms_missing"].append(c)
        yield c, v, qc


def convert(row, paths, work, genome):
    sizes = chrom_sizes(genome); nb = bins_of(sizes)
    crops = load_crops(work)
    is1 = row["block"] in B1
    bych = {}
    for i, c in enumerate(crops):
        bych.setdefault(c["chrom"], []).append((i, int(c["start"])))
    cr = np.zeros((len(crops), CROP), np.uint16) if is1 else None
    if row["output_type"] == "ctss":
        sig = ctss_signal(paths, row["strand"], sizes)
    elif row["transform"]:
        sig = bam_signal(paths, row["transform"], sizes)
    else:
        sig = bigwig_signal(paths, row["strand"], sizes)
    g, st = [], {"track_id": row["track_id"], "nz_count": 0}
    ar = np.arange(CROP)
    for c, v, qc in sig:
        L, n = sizes[c], nb[c]
        full = L // BIN * BIN
        s128 = np.empty(n, np.float64)
        s128[:L // BIN] = v[:full].reshape(-1, BIN).sum(1, dtype=np.float64)
        if n > L // BIN:
            s128[-1] = v[full:].sum(dtype=np.float64)
        if is1:                                         # 128 bp sums, and the 1 bp crops
            g.append(bf16(s128.astype(np.float32)))
            st["nz_count"] += int(np.count_nonzero(v))
            if c in bych:
                idx = np.array([i for i, _ in bych[c]]); s0 = np.array([s for _, s in bych[c]])
                cr[idx] = bf16(v[s0[:, None] + ar[None, :]])
        else:                                           # ChIP: 128 bp means
            m = (s128 / BIN).astype(np.float32)
            if n > L // BIN:
                m[-1] = float(s128[-1] / (L - full))
            g.append(bf16(m))
            st["nz_count"] += int(np.count_nonzero(m))
    st.update(qc)
    out = Path(work) / "tracks" / row["track_id"]; out.mkdir(parents=True, exist_ok=True)
    gg = np.concatenate(g)
    gg.tofile(out / "g128.u16.tmp"); os.replace(out / "g128.u16.tmp", out / "g128.u16")
    if is1:
        # QC: a crop's 1 bp sum must match the 128 bp bins it covers
        off = dict(zip(CHROMS, np.cumsum([0] + [nb[c] for c in CHROMS[:-1]])))
        chk = []
        for i in np.random.default_rng(zlib.crc32(row["track_id"].encode())).choice(len(crops), 3, replace=False):
            c, s = crops[i]["chrom"], int(crops[i]["start"])
            b0, b1 = -(-s // BIN), (s + CROP) // BIN
            chk.append((float(unbf16(cr[i][b0 * BIN - s:b1 * BIN - s]).sum()), float(unbf16(gg[off[c] + b0:off[c] + b1]).sum())))
        st["crop_vs_g128_ok"] = all(abs(x - y) <= 0.02 * max(abs(y), 1.0) for x, y in chk)
        for k in range(0, len(crops), CHUNK):
            save_zst(out / f"crops_{k // CHUNK}.zst", cr[k:k + CHUNK], level=1)
    json.dump(st, open(out / "stats.json", "w"))
    (out / "DONE").touch()
    return st


def run(args):
    row, work, genome, tmp = args
    d = Path(tmp) / row["track_id"]; d.mkdir(parents=True, exist_ok=True)
    try:
        return convert(row, download(row, d), work, genome)
    except Exception as e:
        return {"track_id": row["track_id"], "error": repr(e)}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    work = Path(a.work)
    rows = [r for r in csv.DictReader(open(work / "manifest.tsv"), delimiter="\t")
            if zlib.crc32(r["track_id"].encode()) % sn == si and not (work / "tracks" / r["track_id"] / "DONE").exists()]
    print(f"shard {si}/{sn}: {len(rows)} tracks", flush=True)
    with ProcessPoolExecutor(a.workers) as pp, open(work / f"process_log_{si}of{sn}.jsonl", "a") as log:
        for st in pp.map(run, [(r, str(work), a.genome, a.tmp) for r in rows]):
            log.write(json.dumps(st) + "\n"); log.flush()
            print(st["track_id"], "FAILED " + st["error"] if "error" in st else "ok", flush=True)


if __name__ == "__main__":
    main()
