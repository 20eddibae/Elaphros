"""Pretrain Elaphros or Elaphros-Conv on the track cache.

    python scripts/train.py --data /path/to/data --out runs/elaphros --model elaphros

Recipe used for the released models: 60,000 steps; a context curriculum of 64 kb x 8, 131 kb x 4 and
524 kb x 1 windows (40/40/20% of steps); Muon on block projections and AdamW elsewhere, both at lr 8e-4
with 2,000 warmup steps and cosine decay; EMA 0.999; reverse-complement augmentation with p = 0.5; half of
each 1 bp block's tracks sampled per step; bf16 autocast.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elaphros.data import B1, Tracks, eval_batches, loader, stage_at  # noqa: E402
from elaphros.losses import track_loss  # noqa: E402
from elaphros.model import Elaphros, elaphros_config, elaphros_conv_config, n_params  # noqa: E402
from elaphros.training import autocast, make_optimizers, sample_tracks, track_losses, validate  # noqa: E402


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="elaphros", choices=["elaphros", "elaphros-conv"])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=8e-4, help="AdamW learning rate")
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay")
    ap.add_argument("--muon-lr", type=float, default=8e-4)
    ap.add_argument("--muon-wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--stage-fracs", default="0.4,0.4,0.2", help="fraction of steps at 64 kb, 131 kb and 524 kb")
    ap.add_argument("--crop", type=int, default=32768, help="1 bp supervision span per window")
    ap.add_argument("--seg-bp", type=int, default=8192, help="segment length of the multinomial shape term")
    ap.add_argument("--shape-weight", type=float, default=5.0)
    ap.add_argument("--track-frac", type=float, default=0.5, help="share of 1 bp tracks supervised per step")
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
    ap.add_argument("--heldout-split", default="", help="cell-conditioning split JSON: drop those biosamples' tracks "
                    "from the loss (the unconditioned Base model of the cell-type experiment)")
    return ap.parse_args()


def main():
    a = args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = (elaphros_config if a.model == "elaphros" else elaphros_conv_config)(**json.loads(a.cfg))
    model = Elaphros(cfg).to(dev)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
    opts, scheds = make_optimizers(list(model.named_parameters()), a, dev)
    step = 0
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        for o, so in zip(opts, st["opt"]):
            o.load_state_dict(so)
        for sc, ss in zip(scheds, st["sched"]):
            sc.load_state_dict(ss)
        step = st["step"]
    use_compile = dev == "cuda" and not a.no_compile
    fwd = torch.compile(model) if use_compile else model
    loss_fn = torch.compile(track_loss) if use_compile else track_loss

    tracks = Tracks(a.data)
    if a.heldout_split:
        from elaphros.cellcond_data import Pools
        Pools(tracks, json.load(open(a.heldout_split))).drop_heldout(tracks)
    nz = {b: v.to(dev) for b, v in tracks.nzm.items()}
    ctok = a.crop // 128
    vb, wins = eval_batches(a.data, "valid", a.val_windows, a.crop)
    (out / "val_windows.json").write_text(json.dumps(wins))
    best = json.load(open(out / "best.json"))["mean"] if (out / "best.json").exists() else -1.0
    print(json.dumps({"params_M": n_params(model) / 1e6, "config": cfg.to_dict(), "args": vars(a),
                      "val_windows": len(vb)}), flush=True)
    log = open(out / "log.jsonl", "a")
    t_ck, cur, it = time.time(), None, None
    fracs = tuple(float(f) for f in a.stage_fracs.split(","))
    while step < a.steps:
        t0 = time.time()
        L, B = stage_at(step, a.steps, fracs)
        if (L, B) != cur:                                          # new curriculum stage
            it = loader(a.data, L, B, a.crop, a.seed + step, a.workers)
            cur = (L, B)
            if use_compile:
                torch._dynamo.reset(); fwd = torch.compile(model); loss_fn = torch.compile(track_loss)
        bt = next(it)
        c0 = bt["crop_tok"][0]
        sub = sample_tracks(tracks, a.track_frac)
        x = bt["x"].to(dev, non_blocking=True)
        org = torch.zeros(B, dtype=torch.long, device=dev)
        with autocast(dev):
            pred = fwd(x, org, track_idx={b.lower(): sub[b].to(dev) for b in B1}, crop=(c0, c0 + ctok))
        losses = track_losses(pred, bt, tracks, nz, sub, dev, loss_fn, a.seg_bp, a.shape_weight)
        loss = sum(losses.values())
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        for o in opts:
            o.step()
        for sc in scheds:
            sc.step()
        ema.update_parameters(model)
        step += 1
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "L": L, "B": B, "loss": float(loss), "gnorm": float(gn), "t": round(time.time() - t0, 3),
                   **{k: round(float(v), 4) for k, v in losses.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if not math.isfinite(rec["loss"]):
                raise RuntimeError("non-finite loss")
        if vb and (step % a.val_every == 0 or step == a.steps):
            v = validate(ema.module, vb, tracks, nz, dev, ctok); v["step"] = step
            log.write(json.dumps({"val": v}) + "\n"); log.flush(); print(json.dumps({"val": v}), flush=True)
            if v["mean"] > best:                                   # selected on the validation fold only
                best = v["mean"]
                torch.save({"ema": ema.state_dict(), "step": step, "config": cfg.to_dict()}, out / "best.tmp")
                os.replace(out / "best.tmp", out / "best.pt"); json.dump(v, open(out / "best.json", "w"))
        if step % a.snap_every == 0:
            torch.save({"ema": ema.state_dict(), "step": step, "config": cfg.to_dict()}, out / f"snap_step{step:06d}.pt")
        if time.time() - t_ck > a.ckpt_every_s or step == a.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": [o.state_dict() for o in opts],
                        "sched": [s.state_dict() for s in scheds], "step": step, "config": cfg.to_dict()}, out / "last.tmp")
            os.replace(out / "last.tmp", out / "last.pt"); t_ck = time.time()


if __name__ == "__main__":
    main()
