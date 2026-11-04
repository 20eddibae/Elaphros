"""Pieces shared by the pretraining and cell-conditioning trainers."""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch

from .data import B1, BLOCKS, VAL_CORE, VAL_L
from .losses import log1p_pearson, scale_targets
from .muon import Muon


def is_muon(name, p):
    """Muon for every parameter with two or more dimensions inside the encoder, mixer and decoder (projections,
    and also the (1, C, 1) gains of per-position norms); AdamW for the stem, depthwise convs, biases, 1-D
    parameters (including the SSM time scales), the organism embedding and the track heads."""
    return p.ndim >= 2 and ".dw." not in name and not name.startswith(("stem", "h1.", "h128.", "splice", "org"))


def lr_lambda(step, warmup, total):
    """Linear warmup from 1% then cosine decay to zero."""
    w = 0.01 + 0.99 * min(1.0, (step + 1) / warmup)
    return w * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, step - warmup) / max(1, total - warmup))))


def make_optimizers(named, a, dev, muon_rule=is_muon, extra_adamw=None):
    """named: list of (name, param). extra_adamw: optional (params, lr) group added to the AdamW optimiser."""
    mu = [p for n, p in named if muon_rule(n, p)]
    extra = set(map(id, extra_adamw[0])) if extra_adamw else set()
    ad = [p for n, p in named if not muon_rule(n, p) and id(p) not in extra]
    groups = [{"params": ad}] + ([{"params": extra_adamw[0], "lr": extra_adamw[1]}] if extra_adamw else [])
    opts = [torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd, fused=(dev == "cuda"))]
    if mu:
        opts.append(Muon(mu, lr=a.muon_lr, weight_decay=a.muon_wd))
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, partial(lr_lambda, warmup=a.warmup, total=a.steps)) for o in opts]
    return opts, scheds


def sample_tracks(tracks, frac):
    """A random subset of each 1 bp block's tracks for this step (the 1 bp heads compute only these)."""
    return {b: tracks.use[b][torch.randperm(len(tracks.use[b]))[:max(1, int(frac * len(tracks.use[b])))]].sort().values
            for b in B1}


def track_losses(pred, bt, tracks, nz, sub, dev, loss_fn, seg_bp, shape_weight):
    """Per-head loss on every 128 bp track in use and on the sampled 1 bp tracks."""
    losses = {}
    fill = bt["fill"].to(dev, non_blocking=True)[:, None, :]
    for b in BLOCKS:
        c0, c1 = tracks.cols[b]
        u = tracks.use[b].to(dev)
        tgt = scale_targets(bt["y128"][:, c0:c1].to(dev)[:, u], nz[b][u], 128, b == "RNA_SEQ")
        losses[f"{b}@128"] = loss_fn(pred[f"{b.lower()}@128"].float()[:, u], tgt, fill, 128, seg_bp, shape_weight)
    for b in B1:
        j = sub[b].to(dev)
        y = bt["y1"][b].to(dev, non_blocking=True)[..., j].transpose(1, 2).float()
        tgt = scale_targets(y, nz[b][j], 1, b == "RNA_SEQ")
        losses[f"{b}@1"] = loss_fn(pred[f"{b.lower()}@1"].float(), tgt, None, 1, seg_bp, shape_weight)
    return losses


@torch.no_grad()
def validate(model, batches, tracks, nz, dev, ctok):
    """Mean per-track log1p Pearson per head on the central 131 kb of fixed 524 kb windows; 'mean' averages
    the twelve head x resolution cells."""
    model.eval()
    acc = {}
    k0 = (VAL_L // 128 - VAL_CORE // 128) // 2
    n = VAL_CORE // 128
    for bt in batches:
        c0 = bt["crop_tok"][0]
        org = torch.zeros(1, dtype=torch.long, device=dev)
        fill = bt["fill"].to(dev)[:, k0:k0 + n]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = model(bt["x"].to(dev), org, crop=(c0, c0 + ctok))
        for b in BLOCKS:
            col0, col1 = tracks.cols[b]
            u = tracks.use[b].to(dev)
            t = scale_targets(bt["y128"][:, col0:col1, k0:k0 + n].to(dev)[:, u], nz[b][u], 128, b == "RNA_SEQ")
            acc.setdefault(f"{b}@128", []).append(log1p_pearson(pred[f"{b.lower()}@128"].float()[:, u, k0:k0 + n], t, fill))
        for b in B1:
            u = tracks.use[b].to(dev)
            y = bt["y1"][b].to(dev)[..., u].transpose(1, 2).float()
            t = scale_targets(y, nz[b][u], 1, b == "RNA_SEQ")
            acc.setdefault(f"{b}@1", []).append(log1p_pearson(pred[f"{b.lower()}@1"].float()[:, u], t))
    model.train()
    out = {k: round(float(torch.cat(v).nanmean()), 4) for k, v in acc.items()}
    out["mean"] = float(np.mean(list(out.values())))
    return out


def autocast(dev):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
