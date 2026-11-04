"""Plan the 32,768 bp crops that carry 1 bp targets, from Borzoi's fold split, and write <work>/crops.tsv.

    python preprocessing/make_crops.py --work /path/to/work --genome hg38.fa --folds sequences_human.bed.gz

One crop centred in every training interval (folds 0, 1, 2, 5, 6, 7) and in 400 randomly chosen intervals
each of fold 3 (valid) and fold 4 (test). Starts are snapped down to the 128 bp grid so 1 bp targets line up
with the model's 128 bp tokens.
"""
import argparse
import csv
import gzip
import random
from pathlib import Path

from common import CHROMS, CROP, chrom_sizes

N_EVAL = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--folds", required=True)
    a = ap.parse_args()
    sz = chrom_sizes(a.genome)
    rows = [l.rstrip("\n").split("\t") for l in gzip.open(a.folds, "rt")]
    rows = [(c, int(s), int(e), f) for c, s, e, f in rows if c in CHROMS]
    start = lambda s, e: ((s + e) // 2 - CROP // 2) // 128 * 128  # noqa: E731
    out = []
    for i, (c, s, e, f) in enumerate(sorted(r for r in rows if r[3] not in ("fold3", "fold4"))):
        cs = start(s, e)
        if cs >= 0 and cs + CROP <= sz[c]:
            out.append(("train", f"t_{i:06d}.npz.zst", c, cs, s, e - s))
    rng = random.Random(0)
    for split, fold in (("valid", "fold3"), ("test", "fold4")):
        pool = sorted(r for r in rows if r[3] == fold)
        for i, (c, s, e, f) in enumerate(sorted(rng.sample(pool, min(N_EVAL, len(pool))))):
            if 0 <= start(s, e) and start(s, e) + CROP <= sz[c]:
                out.append((split, f"{split[0]}_{i:05d}.npz.zst", c, start(s, e), s, e - s))
    Path(a.work).mkdir(parents=True, exist_ok=True)
    with open(Path(a.work) / "crops.tsv", "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "split", "fn", "chrom", "start", "rec_start", "rec_len"])
        for i, r in enumerate(out):
            w.writerow([i, *r])
    print({sp: sum(r[0] == sp for r in out) for sp in ("train", "valid", "test")})


if __name__ == "__main__":
    main()
