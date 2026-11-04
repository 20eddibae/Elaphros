"""Cell-state conditioning: predict tracks of a cell type never seen in training from its measured
accessibility (one ATAC-seq or DNase-seq track).

    DNA -> Elaphros trunk (cell-blind) --------------------------+-> the usual track heads
                                                                  |  (128 bp features)
    target-cell accessibility -> AccEncoder -> a (per 128 bp bin) |
                                                 |                v
                                    FiLM(a) -> conv block -> FiLM(a) -> conv block -> FiLM(a) -> out -> e
    shared readout: y[cell, key] = softplus(<e, q_key> + b_key) * softplus(s_key)

`key` is an assay key (modality, assay, target, strand), so a new cell type is read with the same
parameters. FiLM is position-specific (gamma, beta at every bin) and zero-initialised. Optional conditional
LoRA (Hu et al. 2021; as in Bolero, Liu et al. 2026) adds a per-window low-rank update to each 1x1 map of the branch,
generated from the window-mean accessibility features. The trunk receives the branch's gradient through a
scale set from measured gradient norms, so the branch cannot dominate the trunk.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CheriBlock, ConvTowerBlock, Elaphros, ElaphrosConfig


@dataclass
class CellCondConfig:
    n_keys: int
    acc_dim: int = 128
    branch_dim: int = 384
    n_blocks: int = 2
    emb_dim: int = 256
    pool: bool = False            # True = mean-pool the accessibility features over position (global ablation)
    lora_ranks: str = ""          # "" = FiLM only; else n_blocks + 2 ranks for (proj, each block, out), e.g. "1,2,5,10"
    lora_scale: float = 1.0       # alpha / r at every site
    lora_hidden: int = 32         # width of the LoRA generator


class GradScale(torch.autograd.Function):
    """Identity forward; backward scales the gradient by s and records its norm under rec[key]."""

    @staticmethod
    def forward(ctx, x, s, rec, key):
        ctx.s, ctx.rec, ctx.key = s, rec, key
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if ctx.rec is not None:
            ctx.rec[ctx.key] = g.detach().float().norm()
        return g * (ctx.s.to(g.dtype) if torch.is_tensor(ctx.s) else ctx.s), None, None, None


@torch._dynamo.disable
def tap(x, s, rec, key):
    return GradScale.apply(x, s, rec, key) if torch.is_grad_enabled() and x.requires_grad else x


class FiLMZero(nn.Module):
    """h * (1 + gamma(a)) + beta(a), per position, zero-initialised (identity at init)."""

    def __init__(self, ca, d):
        super().__init__()
        self.loc = nn.Conv1d(ca, 2 * d, 1)
        nn.init.zeros_(self.loc.weight)
        nn.init.zeros_(self.loc.bias)

    def forward(self, h, a):
        ga, be = self.loc(a).chunk(2, 1)
        return h * (1 + ga) + be


class AccEncoder(nn.Module):
    """(log1p accessibility, mask, fill) channels -> per-bin features: conv k=9, then Cheri blocks at
    dilations 1, 4 and 16."""

    def __init__(self, c):
        super().__init__()
        self.inp = nn.Conv1d(3, c, 9, padding=4)
        self.blocks = nn.ModuleList([CheriBlock(c, 9, d, 2, "position") for d in (1, 4, 16)])

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h


def cond_features(a):
    """(N, C, T) -> (N, C + 1): RMS-normalised window mean plus log1p(rms), so sequencing depth enters as one
    feature instead of scaling every update."""
    m = a.mean(-1)
    rms = m.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return torch.cat([m / rms, torch.log1p(rms)], dim=-1)


class CondLoRAConv1x1(nn.Module):
    """base(x) + (alpha/r) B(e) A(e) x for a 1x1 conv, with A and B generated per sample from e."""

    def __init__(self, base: nn.Conv1d, cond_dim: int, r: int, alpha: float, hidden: int = 32):
        super().__init__()
        self.base, self.r, self.alpha = base, int(r), float(alpha)
        self.c_in, self.c_out = base.in_channels, base.out_channels
        self.emb = nn.Linear(cond_dim, hidden)
        self.down = nn.Linear(hidden, r * self.c_in)
        self.up = nn.Linear(hidden, r * self.c_out)
        nn.init.normal_(self.emb.weight, std=1.0 / math.sqrt(cond_dim))
        nn.init.zeros_(self.emb.bias)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(hidden * self.c_in))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)              # update is exactly zero at init
        nn.init.zeros_(self.up.bias)

    def forward(self, x, e=None):
        y = self.base(x)
        if e is None:
            return y
        h = torch.tanh(self.emb(e.to(x.dtype)))
        A = self.down(h).view(-1, self.r, self.c_in)
        B = self.up(h).view(-1, self.c_out, self.r)
        return y + (self.alpha / self.r) * torch.einsum("nor,nrt->not", B, torch.einsum("nrc,nct->nrt", A, x))


class CondConvTowerBlock(nn.Module):
    """ConvTowerBlock whose two 1x1 maps carry conditional LoRA."""

    def __init__(self, tower_blk, cond_dim, r, alpha, hidden):
        super().__init__()
        self.blk, self.drop = tower_blk.blk, tower_blk.drop
        self.blk.fc1 = CondLoRAConv1x1(self.blk.fc1, cond_dim, r, alpha, hidden)
        self.blk.fc2 = CondLoRAConv1x1(self.blk.fc2, cond_dim, r, alpha, hidden)

    def forward(self, x, e=None):
        b = self.blk
        h = F.gelu(b.fc1(b.norm(b.dw(x)), e), approximate="tanh")
        return x + self.drop(x + 0.15 * b.fc2(h, e) - x)


class CellBranch(nn.Module):
    def __init__(self, d_in, pool_in, cc: CellCondConfig, cfg: ElaphrosConfig):
        super().__init__()
        self.cc, self.pool_in = cc, pool_in
        self.enc = AccEncoder(cc.acc_dim)
        self.proj = nn.Conv1d(d_in, cc.branch_dim, 1)
        self.film = nn.ModuleList([FiLMZero(cc.acc_dim, cc.branch_dim) for _ in range(cc.n_blocks + 1)])
        self.blocks = nn.ModuleList([ConvTowerBlock(cc.branch_dim, cfg, 2 ** i) for i in range(cc.n_blocks)])
        self.out = nn.Conv1d(cc.branch_dim, cc.emb_dim, 1)
        self.q = nn.Parameter(torch.randn(cc.n_keys + 1, cc.emb_dim) / cc.emb_dim ** 0.5)
        self.b = nn.Parameter(torch.zeros(cc.n_keys + 1))
        self.s = nn.Parameter(torch.ones(cc.n_keys + 1))
        self.lora = False
        if cc.lora_ranks:
            ranks = [int(x) for x in re.split(r"[,:;\s]+", cc.lora_ranks.strip()) if x]
            if len(ranks) != cc.n_blocks + 2:
                raise ValueError(f"lora_ranks needs {cc.n_blocks + 2} values (proj, each block, out), got {ranks}")
            cd, h = cc.acc_dim + 1, cc.lora_hidden
            self.proj = CondLoRAConv1x1(self.proj, cd, ranks[0], cc.lora_scale * ranks[0], h)
            self.blocks = nn.ModuleList([CondConvTowerBlock(blk, cd, r, cc.lora_scale * r, h)
                                         for blk, r in zip(self.blocks, ranks[1:-1])])
            self.out = CondLoRAConv1x1(self.out, cd, ranks[-1], cc.lora_scale * ranks[-1], h)
            self.lora = True

    def body(self, t, acc, fill):
        """t (B, D, T*pool_in) trunk features; acc (B, K, T) model-space accessibility of K cells;
        fill (B, T). Returns the readout embedding (B*K, emb_dim, T)."""
        B, K, T = acc.shape
        x = torch.stack([torch.log1p(acc.clamp_min(0)), torch.zeros_like(acc),
                         fill[:, None, :].expand(B, K, T).to(acc.dtype)], 2).reshape(B * K, 3, T)
        tp = F.avg_pool1d(t, self.pool_in) if self.pool_in > 1 else t
        a = self.enc(x.to(tp.dtype))
        if self.cc.pool:
            a = a.mean(-1, keepdim=True).expand_as(a)
        if self.lora:
            e = cond_features(a)
            h = self.proj(tp.repeat_interleave(K, 0), e)
            for f, blk in zip(self.film, self.blocks):
                h = blk(f(h, a), e)
            return F.gelu(self.out(self.film[-1](h, a), e))
        h = self.proj(tp).repeat_interleave(K, 0)
        for f, blk in zip(self.film, self.blocks):
            h = blk(f(h, a))
        return F.gelu(self.out(self.film[-1](h, a)))

    def head(self, e, keys, B, K, T):
        kk = keys.reshape(B * K, -1)
        kc = kk.clamp_min(0)
        z = torch.einsum("net,nme->nmt", e, self.q[kc].to(e.dtype)) + self.b[kc][..., None]
        pred = F.softplus(z.float()) * F.softplus(self.s[kc].float())[..., None]
        return (pred * (kk >= 0)[..., None]).reshape(B, K, -1, T)


class CellCondElaphros(Elaphros):
    """Elaphros (either variant) plus the cell branch. forward(..., cc={'acc', 'fill', 'keys'}) adds
    out['cc'] = (B, K, M, T) predictions for M assay keys in each of K cells."""

    def __init__(self, cfg: ElaphrosConfig, cc: CellCondConfig):
        super().__init__(cfg)
        self.cc_cfg = cc
        d_in, pool_in = (list(cfg.widths)[5], 4) if self.ssm else (cfg.trunk_width, 1)
        self.ccb = CellBranch(d_in, pool_in, cc, cfg)
        self.rec = {}
        self.register_buffer("s_cc", torch.ones(()), persistent=False)

    def forward(self, x, org=None, heads_1bp=True, track_idx=None, crop=None, halo=8, cc=None):
        h, skips, trunk = self.trunk(x, org)
        on = cc is not None
        h1 = tap(h, 1.0, self.rec, "trunk_base") if on else h       # the tower output feeds both paths
        if self.ssm:
            trunk = self.to_trunk(F.avg_pool1d(h1, 4))
        else:
            trunk = h1
        out = self.heads(h1, skips, trunk, org, heads_1bp, track_idx, crop, halo)
        if on:
            t = tap(h, self.s_cc, self.rec, "trunk_cc")
            acc = cc["acc"]
            B, K, T = acc.shape
            out["cc"] = self.ccb.head(self.ccb.body(t, acc, cc["fill"]), cc["keys"], B, K, T)
        return out

    def cc_dict(self):
        return asdict(self.cc_cfg)
