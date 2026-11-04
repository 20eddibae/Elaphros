"""Score a cell-conditioned checkpoint on held-out biosamples.

    python scripts/evaluate_cellcond.py --data /path/to/data --ckpt runs/film/best_cc.pt --split heldout.json \
        --biosamples test --windows 48 --out results/film_test.json

The statistic is the partial Pearson r(log1p prediction, log1p truth | log1p training-cell mean track) on
the central 131 kb of 524 kb windows from the test fold (fold 4) or, with --biosamples val, the validation
fold (fold 3), averaged over (window, biosample, track) rows. Bars on the same rows: the nearest training
biosample's measured track (bar_a), the biosample's own accessibility (bar_c), and the model's sequence-only
head for the nearest training biosample (bar_d). The 'wrong' control conditions each biosample on a random
training biosample's accessibility instead of its own. A checkpoint without the cell branch (the Base model
from train.py --heldout-split) is scored on its sequence-only rows: Base = its bar_d.

With --baseline borzoi|enformer --targets <target table>, the comparator is scored on the same rows in three
ways (EXACT: its own channel for the cell, a ceiling since it trained on that cell; NEAREST: its channel for the
nearest training cell; MEAN: the mean of its same-assay channels), and the paired gap model minus comparator is
reported per way and modality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elaphros.cellcond import CellCondConfig, CellCondElaphros  # noqa: E402
from elaphros import baselines as BL  # noqa: E402
from elaphros.cellcond_data import Pools, baseline_rows, paired_gaps, score_rows, summarise  # noqa: E402
from elaphros.data import Tracks, eval_batches  # noqa: E402
from elaphros.model import Elaphros, ElaphrosConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--biosamples", default="test", choices=["val", "test"])
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--baseline", default="", choices=["", "borzoi", "enformer"])
    ap.add_argument("--targets", default="", help="comparator target table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ElaphrosConfig.from_dict(st["config"])
    model = CellCondElaphros(cfg, CellCondConfig(**st["cc_config"])) if "cc_config" in st else Elaphros(cfg)
    model.load_state_dict({k[len("module."):]: v for k, v in st["ema"].items() if k.startswith("module.")})
    model = model.to(dev).eval()
    tracks = Tracks(a.data)
    pools = Pools(tracks, json.load(open(a.split)), root=a.data)
    if hasattr(model, "ccb"):
        assert pools.n_keys == model.cc_cfg.n_keys, "the split does not match the checkpoint's assay keys"
    bios = pools.test if a.biosamples == "test" else pools.val
    batches, wins = eval_batches(a.data, "test" if a.biosamples == "test" else "valid", a.windows, 32768)
    rows = score_rows(model, batches, pools, tracks, bios, dev)
    wrong = score_rows(model, batches, pools, tracks, bios, dev,
                       cond_src=lambda b, r: pools.train[int(r.integers(len(pools.train)))], rng_seed=2) \
        if hasattr(model, "ccb") else []
    res = {"step": st.get("step"), "biosamples": bios, "windows": wins, **summarise(rows),
           **{k: v for k, v in summarise(wrong, "wrong/").items() if "/model/" in k}}
    if a.baseline:
        base = baseline_rows(a.baseline, BL.load(a.baseline, dev), BL.targets(a.targets), batches, pools, tracks, bios, dev)
        res[f"model_minus_{a.baseline}"] = paired_gaps(rows, base) if hasattr(model, "ccb") else {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("ALL/", "wrong/ALL", "n_rows", "model_minus"))}, indent=1))


if __name__ == "__main__":
    main()
