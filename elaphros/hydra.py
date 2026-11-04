"""Bidirectional Hydra SSM block (Hwang et al. 2024), the sequence mixer of Elaphros.

Hydra is Mamba-2's state-space duality scan run in both directions with a shared input projection:
each direction's output is shifted by one position, the two are summed, and a data-dependent diagonal
term is added. Adapted from calico/baskerville `layers.Hydra` (Apache-2.0), which builds on mamba_ssm
`mamba2_simple` (Apache-2.0).

The scan uses mamba_ssm's Triton kernel when it is installed and the input is on CUDA, and otherwise a
pure-PyTorch chunked scan (the Mamba-2 paper's `ssd_minimal`), so the model also runs on CPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRITON = None


def _triton_scan():
    global _TRITON
    if _TRITON is None:
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            _TRITON = mamba_chunk_scan_combined
        except Exception:
            _TRITON = False
    return _TRITON or None


def _segsum(x):
    """x (..., T) -> (..., T, T) with out[i, j] = sum_{k=j+1..i} x_k for i >= j and -inf above the diagonal."""
    T = x.size(-1)
    xr = x[..., None].expand(*x.shape, T)
    xr = xr.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1), 0)
    s = torch.cumsum(xr, dim=-2)
    return s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), 0), -torch.inf)


def ssd_scan(x, dt, A, B, C, chunk: int = 256):
    """Causal SSD scan in fp32. x (b,l,h,p), dt (b,l,h) after softplus, A (h,) negative, B/C (b,l,g,n)."""
    b, l, h, p = x.shape
    g = B.shape[2]
    pad = (-l) % chunk
    x, dt, B, C = (t.float() for t in (x, dt, B, C))
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad)); dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad)); C = F.pad(C, (0, 0, 0, 0, 0, pad))
    L = l + pad; nc = L // chunk
    B = B.repeat_interleave(h // g, dim=2); C = C.repeat_interleave(h // g, dim=2)
    X = (x * dt[..., None]).view(b, nc, chunk, h, p)
    Ad = (A.float()[None, None] * dt).view(b, nc, chunk, h).permute(0, 3, 1, 2)
    B = B.view(b, nc, chunk, h, -1); C = C.view(b, nc, chunk, h, -1)
    Acs = torch.cumsum(Ad, dim=-1)
    Y = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, torch.exp(_segsum(Ad)), X)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, torch.exp(Acs[..., -1:] - Acs), X)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    states = torch.einsum("bhzc,bchpn->bzhpn", torch.exp(_segsum(F.pad(Acs[..., -1], (1, 0)))), states)[:, :-1]
    Y = Y + torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(Acs))
    return Y.reshape(b, L, h, p)[:, :l]


class GatedRMSNorm(nn.Module):
    """RMSNorm(y) * w * silu(z), as mamba_ssm's RMSNormGated with norm_before_gate=True."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, y, z):
        yf = y.float()
        yf = yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (yf * F.silu(z.float())).to(y.dtype)


class Hydra(nn.Module):
    """(B, L, D) -> (B, L, D) bidirectional selective SSM."""

    def __init__(self, d_model, d_state=1, d_conv=7, expand=2.0, headdim=64, ngroups=1,
                 dt_min=1e-3, dt_max=0.1, dt_init_floor=1e-4, chunk_size=256, backend="auto"):
        super().__init__()
        self.d_inner = max(1, int(round(expand * d_model / headdim))) * headdim
        self.headdim, self.d_state, self.ngroups, self.chunk = headdim, d_state, ngroups, chunk_size
        self.nheads = self.d_inner // headdim
        self.backend = backend
        gn = ngroups * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 4 * gn + 2 * self.nheads, bias=False)
        cdim = self.d_inner + 4 * gn
        self.dw = nn.Conv1d(cdim, cdim, d_conv, groups=cdim, padding=d_conv // 2, bias=True)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False)
        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan(self, x, dt, A, B, C):
        use_triton = self.backend == "triton" or (self.backend == "auto" and x.is_cuda)
        fn = _triton_scan() if use_triton else None
        if self.backend == "triton" and fn is None:
            raise RuntimeError("backend='triton' but mamba_ssm's SSD kernel could not be imported")
        if fn is not None:
            return fn(x, dt, A, B, C, chunk_size=self.chunk, D=None, z=None)
        return ssd_scan(x, dt, A, B, C, self.chunk).to(x.dtype)

    def forward(self, u):
        b, l, _ = u.shape
        gn = self.ngroups * self.d_state
        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 4 * gn, 2 * self.nheads], dim=-1)
        # forward and reverse directions share one scan at batch 2B
        dt = torch.cat((dt[..., :self.nheads], torch.flip(dt[..., self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        xBC = F.silu(self.dw(xBC.transpose(1, 2)).transpose(1, 2))
        x, BC = torch.split(xBC, [self.d_inner, 4 * gn], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat((BC[..., :2 * gn], torch.flip(BC[..., 2 * gn:], (1,))), dim=0)
        B, C = torch.split(BC, [gn, gn], dim=-1)
        A = -torch.exp(self.A_log.float())
        y = self._scan(x.reshape(2 * b, l, self.nheads, self.headdim), dt, A,
                       B.reshape(2 * b, l, self.ngroups, self.d_state), C.reshape(2 * b, l, self.ngroups, self.d_state))
        y = y.reshape(2 * b, l, self.d_inner)
        y = F.pad(y[:, :-1], (0, 0, 1, 0))                          # shift by one: strictly off-diagonal
        y = y[:b] + torch.flip(y[b:], (1,))
        y = y + x_og * F.linear(x_og, self.fc_D.weight, self.D).repeat_interleave(self.headdim, dim=-1)
        return self.out_proj(self.norm(y, z))


class ChanRMS(nn.Module):
    """Per-position RMS norm over channels, (B, C, L)."""

    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(1, keepdim=True) + 1e-6) * self.w.float()[None, :, None]).to(x.dtype)


class HydraBlock(nn.Module):
    """(B, C, L) residual block: x + drop(Hydra(norm(x))), then x + drop(MLP(norm(x)))."""

    def __init__(self, d, expand=2.0, headdim=64, d_state=1, d_conv=7, mlp=2.0, dropout=0.1,
                 dt_min=1e-3, dt_max=0.1, backend="auto"):
        super().__init__()
        self.norm = ChanRMS(d)
        self.mix = Hydra(d, d_state, d_conv, expand, headdim, dt_min=dt_min, dt_max=dt_max, backend=backend)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.mlp = None
        if mlp > 0:
            h = int(round(mlp * d))
            self.norm2 = ChanRMS(d)
            self.mlp = nn.Sequential(nn.Conv1d(d, h, 1, bias=False), nn.GELU(approximate="tanh"),
                                     nn.Conv1d(h, d, 1, bias=False))

    def forward(self, x):
        x = x + self.drop(self.mix(self.norm(x).transpose(1, 2)).transpose(1, 2))
        if self.mlp is not None:
            x = x + self.drop(self.mlp(self.norm2(x)))
        return x
