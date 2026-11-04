"""Train Elaphros with the accessibility-conditioned cell branch, with held-out biosamples removed from
every loss.

    python scripts/train_cellcond.py --data /path/to/data --out runs/film --split heldout.json
    python scripts/train_cellcond.py ... --lora-ranks 1,2,5,10 --lora-scale 1.0      # FiLM + conditional LoRA
    python scripts/train_cellcond.py ... --lora-ranks 1,2,5,10 --film-freeze          # LoRA only (global)
    python scripts/train_cellcond.py ... --pool                                       # FiLM on pooled accessibility

The base model and the branch train together from scratch with the pretraining recipe (40,000 steps for
the released runs). Each step also draws K = 8 training biosamples per window and supervises up to M = 24
of each one's tracks through the branch, conditioned on one of its accessibility tracks. The trunk gets the
branch's gradient scaled so that its norm is rho = 0.5 of the base gradient norm at the shared activation.
Checkpoints are selected on held-out VAL biosamples only; TEST biosamples are scored by evaluate_cellcond.py.
If --split does not exist it is created with cellcond_data.make_split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elaphros.cellcond import CellCondConfig, CellCondElaphros  # noqa: E402
from elaphros.cellcond_data import Pools, cc_loss, make_cc_batch, make_split, score_rows, summarise  # noqa: E402
from elaphros.data import B1, Tracks, eval_batches, loader, stage_at  # noqa: E402
from elaphros.losses import track_loss  # noqa: E402
from elaphros.model import elaphros_config, elaphros_conv_config, n_params  # noqa: E402
from elaphros.training import autocast, is_muon, make_optimizers, sample_tracks, track_losses, validate  # noqa: E402


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, help="held-out biosamples JSON {'val': [...], 'test': [...]}")
    ap.add_argument("--model", default="elaphros", choices=["elaphros", "elaphros-conv"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--muon-lr", type=float, default=8e-4)
    ap.add_argument("--muon-wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--stage-fracs", default="0.4,0.4,0.2")
    ap.add_argument("--crop", type=int, default=32768)
    ap.add_argument("--seg-bp", type=int, default=8192)
    ap.add_argument("--shape-weight", type=float, default=5.0)
    ap.add_argument("--track-frac", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-windows", type=int, default=48)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--snap-every", type=int, default=10000)
    ap.add_argument("--ckpt-every-s", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of ElaphrosConfig fields")
    g = ap.add_argument_group("cell branch")
    g.add_argument("--cc-k", type=int, default=8, help="training biosamples per window per step")
    g.add_argument("--cc-m", type=int, default=24, help="target tracks per biosample per step")
    g.add_argument("--cc-rho", type=float, default=0.5, help="target trunk gradient-norm ratio, branch / base")
    g.add_argument("--cc-cap", type=float, default=4.0, help="maximum trunk gradient scale")
    g.add_argument("--cc-warmup", type=int, default=2000)
    g.add_argument("--pool", action="store_true", help="mean-pool the accessibility features over position")
    g.add_argument("--lora-ranks", default="", help="conditional LoRA ranks for (proj, block 0, block 1, out)")
    g.add_argument("--lora-scale", type=float, default=1.0, help="alpha / r")
    g.add_argument("--lora-lr", type=float, default=0.0, help="AdamW lr of the LoRA generators (0 = --lr)")
    g.add_argument("--film-freeze", action="store_true", help="keep FiLM at its zero init (LoRA-only arm)")
    return ap.parse_args()


def main():
    a = args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tracks = Tracks(a.data)
    split = json.load(open(a.split)) if Path(a.split).exists() else make_split(tracks, path=a.split)
    pools = Pools(tracks, split, root=a.data)
    pools.drop_heldout(tracks)
    cfg = (elaphros_config if a.model == "elaphros" else elaphros_conv_config)(**json.loads(a.cfg))
    ccc = CellCondConfig(n_keys=pools.n_keys, pool=a.pool, lora_ranks=a.lora_ranks, lora_scale=a.lora_scale)
    model = CellCondElaphros(cfg, ccc).to(dev)
    if a.film_freeze:
        if not a.lora_ranks:
            raise SystemExit("--film-freeze needs --lora-ranks")
        for p in model.ccb.film.parameters():
            p.requires_grad_(False)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))

    def lora_gen(n):
        return n.startswith("ccb.") and any(t in n for t in (".up.", ".down.", ".emb."))

    def muon_rule(n, p):
        # AdamW for the branch readout tables (q, b, s), the branch blocks (matched by the 'ccb.b' prefix, as in
        # the released runs) and the LoRA generators; Muon for the rest of the branch
        return is_muon(n, p) and not n.startswith(("ccb.q", "ccb.b", "ccb.s")) and not lora_gen(n)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gen = [p for n, p in named if lora_gen(n)]
    opts, scheds = make_optimizers(named, a, dev, muon_rule, (gen, a.lora_lr or a.lr) if gen else None)
    base_params = [p for n, p in named if not n.startswith("ccb.")]
    cc_params = [p for n, p in named if n.startswith("ccb.")]
    step, s_cc = 0, 0.0
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        for o, so in zip(opts, st["opt"]):
            o.load_state_dict(so)
        for sc, ss in zip(scheds, st["sched"]):
            sc.load_state_dict(ss)
        step, s_cc = st["step"], st["s_cc"]
        rng = np.random.default_rng(a.seed + step)
    loss_fn = torch.compile(track_loss) if (dev == "cuda" and not a.no_compile) else track_loss

    nz = {b: v.to(dev) for b, v in tracks.nzm.items()}
    ctok = a.crop // 128
    vb, wins = eval_batches(a.data, "valid", a.val_windows, a.crop)
    (out / "val_windows.json").write_text(json.dumps(wins))
    best = json.load(open(out / "best.json"))["mean"] if (out / "best.json").exists() else -1.0
    best_cc = json.load(open(out / "best_cc.json"))["sel"] if (out / "best_cc.json").exists() else -9.0
    meta = {"config": cfg.to_dict(), "cc_config": model.cc_dict()}
    print(json.dumps({"params_M": n_params(model) / 1e6, "branch_params_M": sum(p.numel() for p in cc_params) / 1e6,
                      "n_keys": pools.n_keys, "train_biosamples": len(pools.train), "heldout_val": len(pools.val),
                      "heldout_test": len(pools.test), "args": vars(a)}), flush=True)

    def cc_validate(m):
        rows = score_rows(m, vb, pools, tracks, pools.val, dev)
        s = summarise(rows)
        v = [r["model/partial"] for r in rows if r["block"] in ("RNA_SEQ", "CHIP_HISTONE", "CHIP_TF")
             and r["model/partial"] == r["model/partial"]]
        s["sel"] = round(float(np.mean(v)), 5) if v else -9.0     # selection statistic, VAL biosamples only
        return s, rows

    log = open(out / "log.jsonl", "a")
    t_ck, cur, it = time.time(), None, None
    fracs = tuple(float(f) for f in a.stage_fracs.split(","))
    while step < a.steps:
        t0 = time.time()
        L, B = stage_at(step, a.steps, fracs)
        if (L, B) != cur:
            it = loader(a.data, L, B, a.crop, a.seed + step, a.workers)
            cur = (L, B)
        bt = next(it)
        c0 = bt["crop_tok"][0]
        sub = sample_tracks(tracks, a.track_frac)
        x = bt["x"].to(dev, non_blocking=True)
        org = torch.zeros(B, dtype=torch.long, device=dev)
        fill = bt["fill"].to(dev, non_blocking=True)
        ccb = make_cc_batch(bt["y128"], pools, rng, a.cc_k, a.cc_m)
        model.s_cc.fill_(s_cc * min(1.0, (step + 1) / max(1, a.cc_warmup)))
        model.rec.clear()
        cc = {"acc": ccb["acc"].to(dev), "fill": fill, "keys": ccb["keys"].to(dev)}
        with autocast(dev):
            pred = model(x, org, track_idx={b.lower(): sub[b].to(dev) for b in B1}, crop=(c0, c0 + ctok), cc=cc)
        losses = track_losses(pred, bt, tracks, nz, sub, dev, loss_fn, a.seg_bp, a.shape_weight)
        losses["cc"] = cc_loss(pred["cc"], ccb["tgt"].to(dev), cc["keys"], fill, a.seg_bp, a.shape_weight, loss_fn)
        loss = sum(losses.values())
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        n_base, n_cc = float(model.rec.get("trunk_base", 0.0)), float(model.rec.get("trunk_cc", 0.0))
        if step == 0 and not (n_cc > 0 and math.isfinite(n_cc)):
            raise RuntimeError(f"cell branch is not connected to the trunk (gradient norm {n_cc})")
        if n_cc > 0 and n_base > 0:                    # n_cc is measured before scaling
            s_new = min(a.cc_cap, a.cc_rho * n_base / n_cc)
            s_cc = s_new if step == 0 else 0.9 * s_cc + 0.1 * s_new
        gn = torch.nn.utils.clip_grad_norm_(base_params, a.clip)      # clipped separately: the branch cannot
        torch.nn.utils.clip_grad_norm_(cc_params, a.clip)             # shrink the base model's update
        for o in opts:
            o.step()
        for sc in scheds:
            sc.step()
        ema.update_parameters(model)
        step += 1
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "L": L, "B": B, "loss": float(loss), "gnorm": float(gn), "s_cc": float(model.s_cc),
                   "t": round(time.time() - t0, 3), **{k: round(float(v), 4) for k, v in losses.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if not math.isfinite(rec["loss"]):
                raise RuntimeError("non-finite loss")
        if vb and (step % a.val_every == 0 or step == a.steps):
            v = validate(ema.module, vb, tracks, nz, dev, ctok); v["step"] = step
            cv, rows = cc_validate(ema.module); cv["step"] = step
            (out / "cc_rows").mkdir(exist_ok=True)
            (out / "cc_rows" / f"step{step:06d}.json").write_text(json.dumps(rows))
            log.write(json.dumps({"val": v, "cc_val": cv}) + "\n"); log.flush()
            print(json.dumps({"val": v, "cc_val": cv}), flush=True)
            if v["mean"] > best:
                best = v["mean"]
                torch.save({"ema": ema.state_dict(), "step": step, **meta}, out / "best.tmp")
                os.replace(out / "best.tmp", out / "best.pt"); json.dump(v, open(out / "best.json", "w"))
            if cv["sel"] > best_cc:
                best_cc = cv["sel"]
                torch.save({"ema": ema.state_dict(), "step": step, **meta}, out / "best_cc.tmp")
                os.replace(out / "best_cc.tmp", out / "best_cc.pt"); json.dump(cv, open(out / "best_cc.json", "w"))
        if step % a.snap_every == 0:
            torch.save({"ema": ema.state_dict(), "step": step, **meta}, out / f"snap_step{step:06d}.pt")
        if time.time() - t_ck > a.ckpt_every_s or step == a.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": [o.state_dict() for o in opts],
                        "sched": [s.state_dict() for s in scheds], "step": step, "s_cc": s_cc, **meta}, out / "last.tmp")
            os.replace(out / "last.tmp", out / "last.pt"); t_ck = time.time()


if __name__ == "__main__":
    main()
