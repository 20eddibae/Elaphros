"""Select ENCODE GRCh38 tracks from the ENCODE portal and write <work>/manifest.tsv (one row per output column).

    python preprocessing/select_tracks.py --work /path/to/work

Rules, per assay family, one track per key among released, unrestricted ENCODE-consortium experiments with
no genetic modification, no treatment and no ERROR-level audit, using files of a released analysis:
  DNASE, ATAC    key = biosample; filtered alignments (BAM), converted to 1 bp cut-site counts
  CHIP_TF/HISTONE key = (target, biosample); pooled 'fold change over control'
  RNA_SEQ        key = (assay, biosample); plus/minus strand signal of unique reads, or unstranded signal
  CAGE           key = biosample; plus/minus strand signal of unique reads
  PROCAP         the six ENCODE PRO-cap experiments used by ProCapNet; plus/minus strand signal
Ties go to more biological replicates, then the more recent release. Each family is capped at its model head
size (ChIP keeps the most widely profiled targets first).
"""
import argparse
import collections
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

from common import SLOTS

API = "https://www.encodeproject.org/search/"
PROCAP = ["ENCSR046BCI", "ENCSR100LIJ", "ENCSR261KBX", "ENCSR935RNW", "ENCSR098LLB", "ENCSR799DGV"]
# ATAC / DNase: 5' read ends shifted +4/-4 (ATAC, Tn5 insertion) or 0/+1 (DNase cut site), as in ChromBPNet
BAM_TRANSFORM = {"ATAC": "insertions:4/-4", "DNASE": "insertions:0/1"}
SPEC = [
    ("DNASE", ["DNase-seq"], ["alignments"]),
    ("ATAC", ["ATAC-seq"], ["alignments"]),
    ("CHIP_TF", ["TF ChIP-seq"], ["fold change over control"]),
    ("CHIP_HISTONE", ["Histone ChIP-seq"], ["fold change over control"]),
    ("RNA_SEQ", ["total RNA-seq", "polyA plus RNA-seq"],
     ["plus strand signal of unique reads", "minus strand signal of unique reads", "signal of unique reads"]),
    ("CAGE", ["CAGE"], ["plus strand signal of unique reads", "minus strand signal of unique reads"]),
    ("PROCAP", ["PRO-cap"], ["plus strand signal of unique reads", "minus strand signal of unique reads"]),
]


def get(params):
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params, doseq=True),
                                 headers={"Accept": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["@graph"]


def experiments(assays):
    g = get({"type": "Experiment", "status": "released", "assay_title": assays, "limit": "all", "format": "json",
             "field": ["accession", "assay_title", "date_released", "target.label", "award.project",
                       "biosample_ontology.term_id", "biosample_ontology.term_name",
                       "biosample_ontology.classification", "replicates.library.biosample.genetic_modifications",
                       "replicates.library.biosample.treatments", "audit.ERROR.category"]})
    out = {}
    for e in g:
        bios = [r.get("library", {}).get("biosample", {}) for r in e.get("replicates", [])]
        e["_ok"] = (e.get("award", {}).get("project") == "ENCODE"
                    and not any(b.get("genetic_modifications") or b.get("treatments") for b in bios)
                    and not e.get("audit", {}).get("ERROR"))
        out[e["accession"]] = e
    return out


def files(assays, outputs, fmt):
    return get({"type": "File", "file_format": fmt, "assembly": "GRCh38", "status": "released",
                "assay_title": assays, "output_type": outputs, "limit": "all", "format": "json",
                "field": ["accession", "dataset", "output_type", "file_size", "md5sum", "cloud_metadata.url",
                          "biological_replicates", "restricted", "preferred_default", "analyses.status"]})


def url(f):
    return f["cloud_metadata"]["url"].replace("s3://encode-public/", "https://encode-public.s3.amazonaws.com/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    rows = []
    for block, assays, outputs in SPEC:
        ex = experiments(assays)
        fs = [f for f in files(assays, outputs, "bam" if block in BAM_TRANSFORM else "bigWig")
              if not f.get("restricted") and any(x.get("status") == "released" for x in f.get("analyses", []))]
        byexp = collections.defaultdict(list)
        for f in fs:
            byexp[f["dataset"].split("/")[2]].append(f)
        cands = []
        for acc, flist in byexp.items():
            e = ex.get(acc)
            if e is None or not e["_ok"] or (block == "PROCAP" and acc not in PROCAP):
                continue
            nrep = max(len(f.get("biological_replicates", [])) for f in flist)
            pick = {}
            for f in sorted((f for f in flist if len(f.get("biological_replicates", [])) == nrep),
                            key=lambda f: (not f.get("preferred_default"), f["accession"])):
                pick.setdefault(f["output_type"], f)
            if block in BAM_TRANSFORM:            # every replicate's filtered BAM, counts summed
                trk = [(".", sorted(flist, key=lambda f: f["accession"]))]
                nrep = len({b for f in flist for b in f.get("biological_replicates", [])})
            elif block in ("RNA_SEQ", "CAGE", "PROCAP"):
                p, m = "plus strand signal of unique reads", "minus strand signal of unique reads"
                if p in pick and m in pick:
                    trk = [("+", [pick[p]]), ("-", [pick[m]])]
                elif "signal of unique reads" in pick:
                    trk = [(".", [pick["signal of unique reads"]])]
                else:
                    continue
            elif outputs[0] in pick:
                trk = [(".", [pick[outputs[0]]])]
            else:
                continue
            bt = e["biosample_ontology"]
            tgt = (e.get("target") or {}).get("label", "")
            key = {"CHIP_TF": (tgt, bt["term_id"]), "CHIP_HISTONE": (tgt, bt["term_id"]),
                   "RNA_SEQ": (e["assay_title"], bt["term_id"])}.get(block, (bt["term_id"],))
            cands.append((key, nrep, e.get("date_released", ""), acc, e, trk))
        best = {}
        for c in sorted(cands, key=lambda c: (c[0], c[3])):
            if c[0] not in best or (c[1], c[2]) > (best[c[0]][1], best[c[0]][2]):
                best[c[0]] = c
        chosen = sorted(best.values(), key=lambda c: c[0])
        if block.startswith("CHIP"):
            freq = collections.Counter(c[0][0] for c in chosen)
            chosen.sort(key=lambda c: (-freq[c[0][0]], c[0]))
        n_cols = 0
        for key, nrep, date, acc, e, trk in chosen:
            if n_cols + len(trk) > SLOTS[block]:
                break
            n_cols += len(trk)
            bt = e["biosample_ontology"]
            for strand, fl in trk:
                tid = f"{block}__{acc}" if block in BAM_TRANSFORM else f"{block}__{fl[0]['accession']}"
                rows.append({"track_id": tid, "block": block, "strand": strand, "experiment": acc,
                             "file": "|".join(f["accession"] for f in fl), "output_type": fl[0]["output_type"],
                             "assay": e["assay_title"], "target": (e.get("target") or {}).get("label", ""),
                             "term_id": bt["term_id"], "term_name": bt["term_name"],
                             "classification": bt.get("classification", ""), "n_rep": nrep,
                             "url": "|".join(url(f) for f in fl), "md5": "|".join(f["md5sum"] for f in fl),
                             "bytes": sum(f["file_size"] for f in fl), "transform": BAM_TRANSFORM.get(block, ""),
                             "source": "encode"})
        print(block, {"candidates": len(cands), "keys": len(chosen), "columns": n_cols, "slots": SLOTS[block]}, flush=True)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    with open(Path(a.work) / "manifest.tsv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(len(rows), "columns,", round(sum(r["bytes"] for r in rows) / 1e12, 2), "TB to download")


if __name__ == "__main__":
    main()
