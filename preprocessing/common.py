"""Shared constants and helpers for the preprocessing pipeline.

Per-track intermediates live under <work>/tracks/<track_id>/:
    g128.u16       all chromosomes concatenated, ceil(len/128) bins each, bf16 bits. 1 bp assays hold 128 bp
                   sums of the 1 bp signal; ChIP holds 128 bp means of fold change over control
    crops_<k>.zst  1 bp assays only: crops k*CHUNK ... of crops.tsv as (n, CROP) bf16 bits
    stats.json     provenance and QC
    DONE           written last
"""
import io
import os
from pathlib import Path

import numpy as np

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
CROP, BIN, CHUNK = 32768, 128, 512
B1 = ["ATAC", "DNASE", "PROCAP", "CAGE", "RNA_SEQ"]
B128 = ["CHIP_TF", "CHIP_HISTONE"]
SLOTS = {"ATAC": 256, "DNASE": 384, "PROCAP": 128, "CAGE": 640, "RNA_SEQ": 768,
         "CHIP_TF": 1664, "CHIP_HISTONE": 1152}       # model head sizes


def chrom_sizes(genome):
    sz = {}
    for line in open(str(genome) + ".fai"):
        c, n = line.split("\t")[:2]
        if c in CHROMS:
            sz[c] = int(n)
    return {c: sz[c] for c in CHROMS}


def bins_of(sizes):
    return {c: (n + BIN - 1) // BIN for c, n in sizes.items()}


def bf16(x):
    """float32 -> bf16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + ((u >> 16) & 1) + np.uint32(0x7FFF)) >> 16).astype(np.uint16)


def unbf16(u):
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def load_crops(work):
    import csv
    return list(csv.DictReader(open(Path(work) / "crops.tsv"), delimiter="\t"))


def save_zst(path, arr, level=3):
    import zstandard
    b = io.BytesIO(); np.save(b, arr)
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(zstandard.ZstdCompressor(level=level, threads=2).compress(b.getvalue()))
    os.replace(tmp, path)


def load_zst(path):
    import zstandard
    return np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(Path(path).read_bytes())))
