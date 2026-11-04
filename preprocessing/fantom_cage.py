"""Add FANTOM5 CAGE libraries to <work>/manifest.tsv, one '+' and one '-' column per library.

    python preprocessing/fantom_cage.py --work /path/to/work --libraries fantom_libraries.tsv

`fantom_libraries.tsv` lists the chosen libraries with columns
    library (CNhs id), url (hg38 CTSS .bed.gz), md5, term_id, term_name, classification
CTSS files: https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/basic/ (directories
human.{primary_cell,tissue,cell_line}.{hCAGE,LQhCAGE}; md5 values in each 00MD5SUM.txt). FANTOM5 data are
CC BY 4.0 (RIKEN; Lizio et al. 2019).

Selection used for the paper: exclude samples whose name marks a treatment, stimulation, infection, mock,
expansion, universal or reference RNA, time course (dayN, Nh, hr), induction or knockdown; keep one library per
(category, sample term), preferring hCAGE over LQhCAGE, then the larger file, then the library id; take every
primary-cell and tissue term, then cell lines by depth, up to the CAGE head size. That gave 296 libraries
(159 primary cell, 137 tissue) and 592 columns. Sample terms are mapped to CL / UBERON / CLO / EFO through the
FANTOM5 sample ontology.

process_tracks.py converts these rows (output_type 'ctss') into 1 bp tag counts on the given strand.
"""
import argparse
import csv
from pathlib import Path

from common import SLOTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--libraries", required=True)
    a = ap.parse_args()
    man = Path(a.work) / "manifest.tsv"
    rows = list(csv.DictReader(open(man), delimiter="\t"))
    have = {r["track_id"] for r in rows}
    for lib in csv.DictReader(open(a.libraries), delimiter="\t"):
        for strand, sfx in (("+", "p"), ("-", "m")):
            tid = f"CAGE__{lib['library']}_{sfx}"
            if tid in have:
                continue
            rows.append({"track_id": tid, "block": "CAGE", "strand": strand, "experiment": lib["library"],
                         "file": lib["library"], "output_type": "ctss", "assay": "CAGE", "target": "",
                         "term_id": lib["term_id"], "term_name": lib["term_name"],
                         "classification": lib.get("classification", ""), "n_rep": 1, "url": lib["url"],
                         "md5": lib["md5"], "bytes": 0, "transform": "", "source": "fantom"})
    n_cage = sum(r["block"] == "CAGE" for r in rows)
    if n_cage > SLOTS["CAGE"]:
        raise SystemExit(f"{n_cage} CAGE columns exceed the {SLOTS['CAGE']} CAGE head slots; shorten the library list")
    with open(man, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(len(rows), "manifest rows")


if __name__ == "__main__":
    main()
