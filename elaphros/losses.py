"""Target transform, training loss and the validation metric."""
from __future__ import annotations

import torch


def scale_targets(y, nzm, res, rna):
    """Measured units -> model space. y (B, T, S); nzm (T,) per-track non-zero mean; res = bp per bin.
    Divide by nzm * res, raise RNA-seq to the power 0.75, soft-clip above 10."""
    y = y / (nzm[None, :, None] * res)
    if rna:
        y = y.clamp_min(0).pow(0.75)
    return torch.where(y > 10, 2 * torch.sqrt(10 * y.clamp_min(10)) - 10, y)


def unscale_targets(y, nzm, res, rna):
    """Inverse of scale_targets: model space -> measured units, per bin."""
    x = torch.where(y > 10, ((y + 10) / 2) ** 2 / 10, y.clamp_min(0))
    if rna:
        x = x.pow(1 / 0.75)
    return x * nzm[None, :, None] * res


def track_loss(pred, tgt, mask, res, seg_bp=8192, shape_weight=5.0):
    """Poisson NLL of segment totals + shape_weight x multinomial NLL of the positions within each segment,
    both divided by the segment length and averaged over filled bins. pred/tgt (B, T, S) in model space,
    mask (B, 1, S) or None."""
    S = pred.shape[-1]
    seg = max(1, min(S, seg_bp // res))
    if mask is None:
        mask = torch.ones(pred.shape[0], 1, S, device=pred.device)
    pred = pred.float() * mask
    tgt = tgt.float().clamp_min(0) * mask
    p = pred.reshape(*pred.shape[:2], S // seg, seg)
    t = tgt.reshape(*tgt.shape[:2], S // seg, seg)
    ps, ts = p.sum(-1), t.sum(-1)
    pois = (ps - ts * torch.log(ps + 1e-7)) - (ts - ts * torch.log(ts + 1e-7))
    mult = -(t * torch.log(p / (ps[..., None] + 1e-7) + 1e-7)).sum(-1)
    m = mask.reshape(mask.shape[0], 1, S // seg, seg).float().mean(-1)
    return ((pois + shape_weight * mult) / seg).sum() / (m.sum() * pred.shape[1] + 1e-7)


def log1p_pearson(p, t, fill=None):
    """Per-row Pearson of log1p values over the last axis. p, t (B, T, S); fill (B, S) bool or None.
    Returns a flat tensor over rows whose target is not constant."""
    p, t = torch.log1p(p.clamp_min(0)), torch.log1p(t.clamp_min(0))
    if fill is not None:
        m = fill[:, None, :].float()
        n = m.sum(-1).clamp_min(1)
        cp = (p - ((p * m).sum(-1) / n)[..., None]) * m
        ct = (t - ((t * m).sum(-1) / n)[..., None]) * m
    else:
        cp, ct = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
    r = (cp * ct).sum(-1) / ((cp.pow(2).sum(-1) * ct.pow(2).sum(-1)).sqrt() + 1e-8)
    return r[ct.pow(2).sum(-1) > 0].flatten()
