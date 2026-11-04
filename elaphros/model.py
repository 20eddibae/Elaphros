"""Elaphros: a ~50M-parameter sequence-to-function U-Net with 524,288 bp of context.

Both variants share the stem, the Cheri convolution encoder, the decoder and the track heads.

  Elaphros       encoder levels at 1, 2, 4, 8, 16 and 32 bp -> 14 bidirectional Hydra SSM blocks at 32 bp
                 -> (a) average-pool x4 to 128 bp for the 128 bp heads, (b) a five-level decoder back to 1 bp.
  Elaphros-Conv  encoder levels at 1 ... 64 bp -> max-pool to 128 bp -> 12 dilated Cheri blocks at 128 bp
                 -> the 128 bp heads, and a seven-level decoder back to 1 bp.

The 1 bp heads read the decoder output plus a broadcast copy of the 128 bp embedding. Every head is a
low-rank linear readout with a softplus output.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hydra import HydraBlock

# Output slots per head. Each assay family fills its first n slots; the rest are unused padding.
HEADS_1BP_128BP = {"atac": 256, "dnase": 384, "procap": 128, "cage": 640, "rna_seq": 768}
HEADS_128BP = {"chip_tf": 1664, "chip_histone": 1152}
N_SPLICE_CLS, N_SPLICE_USAGE = 5, 734


@dataclass
class ElaphrosConfig:
    widths: Sequence[int] = (160, 192, 256, 320, 384, 512, 640)   # channels at 1, 2, 4, ..., 64 bp
    trunk_width: int = 768
    enc_depth: int = 2                     # Cheri blocks per encoder level
    dec_depth: int = 1                     # Cheri blocks per decoder level
    cheri_kernel: int = 9
    cheri_expand: int = 2
    stem_kernel: int = 15
    conv_tower: int = 12                   # Elaphros-Conv: dilated Cheri blocks at 128 bp
    tower_mlp_ratio: float = 2.0
    dropout: float = 0.3
    emb1_dim: int = 192
    emb128_dim: int = 768
    head_rank: int = 128
    chip_head_rank: int = 256
    n_organisms: int = 2                   # organism slot 0 = human; slot 1 is unused by the released models
    splice: bool = True                    # splice heads exist so trained checkpoints load; never trained or computed
    ssm_blocks: int = 0                    # Elaphros: Hydra blocks at 32 bp (0 = the Elaphros-Conv geometry)
    ssm_expand: float = 2.0
    ssm_headdim: int = 64
    ssm_dstate: int = 1
    ssm_dconv: int = 7
    ssm_mlp: float = 2.0
    ssm_dropout: float = 0.1
    ssm_dt_min: float = 1e-3
    ssm_dt_max: float = 0.1
    ssm_backend: str = "auto"              # 'auto' | 'triton' | 'torch'

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


def elaphros_config(**kw) -> ElaphrosConfig:
    return ElaphrosConfig(**{"ssm_blocks": 14, **kw})


def elaphros_conv_config(**kw) -> ElaphrosConfig:
    return ElaphrosConfig(**{**kw, "ssm_blocks": 0})


# --------------------------------------------------------------------------- blocks
class ChanRMSNorm(nn.Module):
    """Per-position RMS norm over channels (B, C, L)."""

    def __init__(self, c: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-6) * self.g


class PlaneLayerNorm(nn.Module):
    """Per-example layer norm over the whole (C, L) plane, no affine (Cherimoya's encoder norm)."""

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(dim=(1, 2), keepdim=True)
        var = xf.var(dim=(1, 2), keepdim=True, unbiased=False)
        return ((xf - mu) * torch.rsqrt(var + 1e-3)).to(x.dtype)


class CheriBlock(nn.Module):
    """Cherimoya block (Schreiber 2024): depthwise dilated conv -> norm -> 1x1 expand, GELU, 1x1 project,
    added back with a fixed 0.15 residual scale."""

    def __init__(self, c: int, k: int = 9, dilation: int = 1, expand: int = 2, norm: str = "plane"):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=dilation * (k // 2), dilation=dilation, groups=c, bias=False)
        self.norm = PlaneLayerNorm() if norm == "plane" else ChanRMSNorm(c)
        self.fc1 = nn.Conv1d(c, expand * c, 1, bias=False)
        self.fc2 = nn.Conv1d(expand * c, c, 1, bias=False)
        for w in (self.dw.weight, self.fc1.weight, self.fc2.weight):
            nn.init.trunc_normal_(w, std=0.02)

    def forward(self, x):
        return x + 0.15 * self.fc2(F.gelu(self.fc1(self.norm(self.dw(x))), approximate="tanh"))


class EncLevel(nn.Module):
    def __init__(self, cin, cout, depth, cfg: ElaphrosConfig, norm: str = "plane"):
        super().__init__()
        self.proj = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1, bias=False)
        self.blocks = nn.ModuleList([CheriBlock(cout, cfg.cheri_kernel, (1, 2, 4, 8)[i % 4], cfg.cheri_expand, norm)
                                     for i in range(depth)])

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return x


class DecLevel(nn.Module):
    """Upsample by 2 (repeat), add the projected encoder skip, refine."""

    def __init__(self, cin, cskip, depth, cfg: ElaphrosConfig):
        super().__init__()
        self.proj = nn.Identity() if cin == cskip else nn.Conv1d(cin, cskip, 1, bias=False)
        self.skip = nn.Conv1d(cskip, cskip, 1)
        self.scale = nn.Parameter(torch.ones(1))
        self.enc = EncLevel(cskip, cskip, depth, cfg, norm="position")

    def forward(self, x, skip):
        x = self.proj(x).repeat_interleave(2, dim=2) * self.scale
        return self.enc(x + self.skip(skip))


class ConvTowerBlock(nn.Module):
    """Cheri block at 128 bp with a per-position norm and dropout on the residual branch."""

    def __init__(self, d, cfg: ElaphrosConfig, dilation):
        super().__init__()
        self.blk = CheriBlock(d, cfg.cheri_kernel, dilation, int(cfg.tower_mlp_ratio), "position")
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return x + self.drop(self.blk(x) - x)


class ConvTower(nn.Module):
    def __init__(self, d, cfg: ElaphrosConfig):
        super().__init__()
        self.layers = nn.ModuleList([ConvTowerBlock(d, cfg, 2 ** (i % 10)) for i in range(cfg.conv_tower)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TrackHead(nn.Module):
    """Low-rank per-organism readout: a shared E -> r projection and an r x T track embedding."""

    def __init__(self, e, t, cfg: ElaphrosConfig, rank=None, act="softplus"):
        super().__init__()
        self.act = act
        r = rank or cfg.head_rank
        self.down = nn.Conv1d(e, r, 1, bias=False)
        self.w = nn.Parameter(torch.randn(cfg.n_organisms, t, r) / math.sqrt(r))
        self.b = nn.Parameter(torch.zeros(cfg.n_organisms, t, 1))
        self.s = nn.Parameter(torch.ones(cfg.n_organisms, t, 1)) if act == "softplus" else None

    def forward(self, x, org, idx=None):
        """idx: optional track subset, so only those rows of the track embedding are computed."""
        x = self.down(x)
        w, b = self.w[org], self.b[org]
        s = self.s[org] if self.s is not None else None
        if idx is not None:
            w, b = w[:, idx], b[:, idx]
            s = s[:, idx] if s is not None else None
        z = torch.einsum("bcs,btc->bts", x, w) + b
        return z if self.act == "logit" else F.softplus(z) * F.softplus(s)


# --------------------------------------------------------------------------- model
class Elaphros(nn.Module):
    def __init__(self, cfg: ElaphrosConfig = None):
        super().__init__()
        self.cfg = c = cfg or elaphros_config()
        W = list(c.widths)
        self.ssm = c.ssm_blocks > 0
        n_enc = 6 if self.ssm else len(W)                     # Elaphros stops the encoder at 32 bp
        self.stem = nn.Conv1d(4, W[0], c.stem_kernel, padding=c.stem_kernel // 2)
        self.enc = nn.ModuleList()
        cin = W[0]
        for i in range(n_enc):                                # level i runs at 2**i bp
            self.enc.append(EncLevel(cin, W[i], c.enc_depth, c, norm="plane"))
            cin = W[i]
        if self.ssm:
            w5 = W[5]
            self.org = nn.Embedding(c.n_organisms, w5)
            self.tower = nn.Sequential(*[
                HydraBlock(w5, c.ssm_expand, c.ssm_headdim, c.ssm_dstate, c.ssm_dconv, c.ssm_mlp, c.ssm_dropout,
                           c.ssm_dt_min, c.ssm_dt_max, c.ssm_backend) for _ in range(c.ssm_blocks)])
            self.to_trunk = nn.Conv1d(w5, c.trunk_width, 1)
            dec_in = [w5] + W[1:5][::-1]                      # 32 -> 16 -> ... -> 1 bp
            dec_out = W[0:5][::-1]
        else:
            self.to_trunk = nn.Conv1d(W[-1], c.trunk_width, 1)
            self.org = nn.Embedding(c.n_organisms, c.trunk_width)
            self.tower = ConvTower(c.trunk_width, c)
            dec_in = [c.trunk_width] + W[1:][::-1]            # 128 -> 64 -> ... -> 1 bp
            dec_out = W[::-1]
        self.dec = nn.ModuleList([DecLevel(i, o, c.dec_depth, c) for i, o in zip(dec_in, dec_out)])
        self.emb128 = nn.Conv1d(c.trunk_width, c.emb128_dim, 1)
        self.emb1 = nn.Conv1d(W[0], c.emb1_dim, 1)
        self.emb1_skip = nn.Conv1d(c.emb128_dim, c.emb1_dim, 1, bias=False)
        self.h1 = nn.ModuleDict({k: TrackHead(c.emb1_dim, t, c) for k, t in HEADS_1BP_128BP.items()})
        self.h128 = nn.ModuleDict({k: TrackHead(c.emb128_dim, t, c) for k, t in HEADS_1BP_128BP.items()})
        self.h128.update({k: TrackHead(c.emb128_dim, t, c, rank=c.chip_head_rank) for k, t in HEADS_128BP.items()})
        if c.splice:
            self.splice_cls = nn.Conv1d(c.emb1_dim, N_SPLICE_CLS, 1)
            self.splice_use = TrackHead(c.emb1_dim, N_SPLICE_USAGE, c, act="logit")

    # The decoder input runs at 32 bp (Elaphros) or 128 bp (Elaphros-Conv).
    @property
    def dec_bp(self) -> int:
        return 32 if self.ssm else 128

    def trunk(self, x, org=None):
        """x (B, L, 4) one-hot, L a multiple of 128. Returns (tower output, encoder skips, 128 bp trunk)."""
        if org is None:
            org = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h = self.stem(x.transpose(1, 2))
        skips = []
        for i, lvl in enumerate(self.enc):
            if i > 0:
                h = F.max_pool1d(h, 2)
            h = lvl(h)
            skips.append(h)
        if self.ssm:
            h = self.tower(h + self.org(org)[:, :, None])                 # (B, W5, L/32)
            return h, skips[:-1], self.to_trunk(F.avg_pool1d(h, 4))       # the 32 bp map is the decoder input
        t = self.tower(self.to_trunk(F.max_pool1d(h, 2)) + self.org(org)[:, :, None])
        return t, skips, t

    def heads(self, h, skips, trunk, org=None, heads_1bp=True, track_idx=None, crop=None, halo=8):
        if org is None:
            org = torch.zeros(trunk.shape[0], dtype=torch.long, device=trunk.device)
        track_idx = track_idx or {}
        e128 = F.gelu(self.emb128(trunk))
        out = {f"{k}@128": head(e128, org) for k, head in self.h128.items()}
        if not heads_1bp:
            return out
        e1 = self.embed_1bp(h, skips, e128, crop, halo)
        out.update({f"{k}@1": head(e1, org, track_idx.get(k)) for k, head in self.h1.items()})
        return out

    def embed_1bp(self, h, skips, e128, crop=None, halo=8):
        """Decode to the 1 bp embedding. crop = (start, end) in 128 bp tokens decodes only that span (plus a
        halo of `halo` tokens each side for the decoder's receptive field)."""
        r = 128 // self.dec_bp
        if crop is not None:
            a, b = max(0, crop[0] - halo), min(e128.shape[-1], crop[1] + halo)
            h = h[..., a * r:b * r]
            skips = [sk[..., a * (128 >> i):b * (128 >> i)] for i, sk in enumerate(skips)]
            e128 = e128[..., a:b]
        for lvl, sk in zip(self.dec, reversed(skips)):
            h = lvl(h, sk)
        e1 = F.gelu(self.emb1(h) + self.emb1_skip(e128).repeat_interleave(128, dim=2))
        if crop is not None:
            e1 = e1[..., (crop[0] - a) * 128:(crop[1] - a) * 128]
        return e1

    def forward(self, x, org=None, heads_1bp: bool = True, track_idx: Optional[Dict[str, torch.Tensor]] = None,
                crop: Optional[Tuple[int, int]] = None, halo: int = 8):
        """x (B, L, 4) one-hot. Returns {'<assay>@128': (B, slots, L/128), '<assay>@1': (B, slots, L or crop)}.
        track_idx limits the 1 bp heads to a subset of slots; crop limits 1 bp outputs to a token span."""
        h, skips, trunk = self.trunk(x, org)
        return self.heads(h, skips, trunk, org, heads_1bp, track_idx, crop, halo)

    def features(self, x, org=None):
        """Embeddings for fine-tuning: the encoder levels feeding the decoder (enc[i] at 2**i bp), the 1 bp
        embedding and the 128 bp trunk."""
        h, skips, trunk = self.trunk(x, org)
        e128 = F.gelu(self.emb128(trunk))
        return {"enc": skips, "e1": self.embed_1bp(h, skips, e128), "trunk": trunk, "e128": e128}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def load_checkpoint(path, device="cpu", weights="ema"):
    """Load a training checkpoint. Returns (model, step). `weights` = 'ema' (default, every checkpoint) or
    'model' (the raw weights, last.pt only)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    model = Elaphros(ElaphrosConfig.from_dict(st["config"]))
    if weights == "ema" and "ema" in st:
        sd = {k[len("module."):]: v for k, v in st["ema"].items() if k.startswith("module.")}
    else:
        sd = st["model"]
    model.load_state_dict(sd)
    return model.to(device).eval(), int(st.get("step", -1))
