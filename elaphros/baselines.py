"""Comparator models for evaluation: Borzoi, Flashzoi and Enformer from their public PyTorch ports.

    Borzoi    johahi/borzoi-replicate-0             borzoi-pytorch      524,288 bp in, 32 bp bins, 6,144 bins
    Flashzoi  johahi/flashzoi-replicate-0           borzoi-pytorch      same geometry
    Enformer  EleutherAI/enformer-official-rough    enformer-pytorch    196,608 bp in, 128 bp bins, 896 bins

Target tables: Borzoi `examples/targets_human.txt` (calico/borzoi) and Enformer `targets_human.txt`
(calico/basenji, manuscripts/cross2020). Flashzoi's attention kernel needs flash-attn; `flash="port"`
swaps in an equivalent plain-PyTorch attention so it runs anywhere.
"""
from __future__ import annotations

import csv
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

HF = {"borzoi": "johahi/borzoi-replicate-0", "flashzoi": "johahi/flashzoi-replicate-0",
      "enformer": "EleutherAI/enformer-official-rough"}
GEOM = {"borzoi": (524288, 32, 163840), "flashzoi": (524288, 32, 163840), "enformer": (196608, 128, 40960)}
# (input bp, output bin bp, bp offset of output bin 0 from the input start)


class FlashAttentionCPU(nn.Module):
    """Plain-PyTorch replacement for Flashzoi's flash-attn layer (same parameters and computation)."""

    def __init__(self, dim=1536, heads=8, dropout=0.15, pos_dropout=0.15, rotary_emb_base=20000.0, rotary_emb_scale_base=None):
        super().__init__()
        self.mha = nn.Module()
        self.h, self.hkv, self.d, self.rot = heads, heads // 2, dim // heads, 128
        self.mha.Wqkv = nn.Linear(dim, self.d * (self.h + 2 * self.hkv), bias=True)
        self.mha.out_proj = nn.Linear(dim, dim, bias=True)
        self.scale = (dim / heads) ** -0.5
        self.base = rotary_emb_base

    def _rope(self, x, cos, sin):
        r = self.rot
        x1, x2, rest = x[..., :r // 2], x[..., r // 2:r], x[..., r:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.mha.Wqkv(x)
        q = qkv[..., :self.h * self.d].view(B, S, self.h, self.d)
        kv = qkv[..., self.h * self.d:].view(B, S, 2, self.hkv, self.d)
        k, v = kv[:, :, 0], kv[:, :, 1]
        inv = 1.0 / (self.base ** (torch.arange(0, self.rot, 2, dtype=torch.float32, device=x.device) / self.rot))
        fr = torch.outer(torch.arange(S, dtype=torch.float32, device=x.device), inv)
        cos, sin = fr.cos()[None, :, None, :].to(q.dtype), fr.sin()[None, :, None, :].to(q.dtype)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        k = k.repeat_interleave(self.h // self.hkv, dim=2); v = v.repeat_interleave(self.h // self.hkv, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=self.scale)
        return self.mha.out_proj(o.transpose(1, 2).reshape(B, S, self.h * self.d))


def load(name, device="cpu", heads=True, pretrained=True, flash="port"):
    """Returns the comparator model in eval mode. heads=False replaces the track heads with Identity."""
    if name in ("borzoi", "flashzoi"):
        import borzoi_pytorch.pytorch_borzoi_model as bm
        from borzoi_pytorch.config_borzoi import BorzoiConfig
        if name == "flashzoi" and flash == "port":
            bm.FlashAttention = FlashAttentionCPU
        cfg = BorzoiConfig.from_pretrained(HF[name])
        cfg.enable_mouse_head = False
        m = bm.Borzoi.from_pretrained(HF[name], config=cfg) if pretrained else bm.Borzoi(cfg)
        if name == "borzoi":                       # rebuild relative-position tables left on the meta device
            from borzoi_pytorch.pytorch_borzoi_transformer import get_positional_embed
            for mod in m.modules():
                p = getattr(mod, "positions", None)
                if torch.is_tensor(p) and p.is_meta:
                    mod.register_buffer("positions", get_positional_embed(4096, mod.num_rel_pos_features, "cpu"),
                                        persistent=False)
        if not heads:
            m.human_head = nn.Identity()
    else:
        from enformer_pytorch import Enformer, EnformerConfig, from_pretrained
        m = from_pretrained(HF[name], use_tf_gamma=False) if pretrained else Enformer(EnformerConfig.from_pretrained(HF[name]))
        if not heads:
            m._heads = nn.ModuleDict({})
    return m.to(device).eval()


def predict(name, m, x):
    """x (B, L, 4) one-hot at the native length -> (B, tracks, bins) in model space."""
    if name == "enformer":
        return m(x, head="human").transpose(1, 2)
    return m(x.transpose(1, 2))


# ----------------------------------------------------------------------------- trunk features (fine-tuning)
SOURCES = {"borzoi": {"cdna": (2, 512), "unet0": (32, 1280), "tok": (128, 1536), "out32": (32, 1536), "final": (32, 1920)},
           "enformer": {"stem": (2, 768), "final": (128, 3072)}}     # source -> (bp per position, channels)


def trunk(name, m, x):
    """Crop-free embeddings at any input length that is a multiple of 128. Returns {source: (B, C, positions)}."""
    if name == "enformer":
        return {"stem": m.stem(x.transpose(1, 2)),
                "final": m(x, return_only_embeddings=True, target_length=-1).transpose(1, 2)}
    h = m.conv_dna(x.transpose(1, 2))
    u0 = m.res_tower(h)
    u1 = m.unet1(u0)
    tok = m.transformer(m._max_pool(u1).permute(0, 2, 1)).permute(0, 2, 1)
    u = m.separable1(m.upsampling_unet1(tok) + m.horizontal_conv1(u1))
    out32 = m.separable0(m.upsampling_unet0(u) + m.horizontal_conv0(u0))
    return {"cdna": h, "unet0": u0, "tok": tok, "out32": out32, "final": m.final_joined_convs(out32)}


def reset_positions(m):
    """Borzoi caches a relative-position table sized to the first input length; clear it to run other lengths."""
    for mod in m.modules():
        if hasattr(mod, "positions") and hasattr(mod, "num_rel_pos_features"):
            mod.positions = None


# ----------------------------------------------------------------------------- target tables
def targets(path):
    """One dict per output channel: type (DNASE, ATAC, CHIP, CAGE, RNA), target, biosample text, strand, the
    accession in the source file name (ENCSR... or FANTOM5 CNhs...), and Borzoi's decoding fields."""
    rows = list(csv.reader(open(path), delimiter="\t"))
    h = rows[0]
    out = []
    for i, r in enumerate(rows[1:]):
        g = dict(zip(h, r))
        parts = g["description"].split(":")
        typ = parts[0]
        tgt, bs = (parts[1], ":".join(parts[2:])) if typ == "CHIP" else ("", ":".join(parts[1:]))
        acc = re.search(r"(ENCSR\w+|CNhs\d+)", g.get("file", ""))
        out.append({"ch": i, "id": g["identifier"], "type": typ, "target": tgt, "bs": bs,
                    "strand": g["identifier"][-1] if g["identifier"][-1] in "+-" else ".",
                    "accession": acc.group(1) if acc else None, "gtex": g["identifier"].upper().startswith("GTEX"),
                    "clip_soft": float(g.get("clip_soft", "nan") or "nan"), "scale": float(g.get("scale", "nan") or "nan"),
                    "sqrt": g.get("sum_stat", "") == "sum_sqrt"})
    return out


def decode(name, y, chans):
    """Model space -> additive units per bin, y (..., n_chans, bins). Borzoi inverts its per-track scale, soft
    clip and square-root transform; Enformer's outputs are used as they are."""
    if name == "enformer":
        return y
    sc = torch.tensor([c["scale"] for c in chans], dtype=y.dtype)[:, None]
    cs = torch.tensor([c["clip_soft"] for c in chans], dtype=y.dtype)[:, None]
    sq = torch.tensor([c["sqrt"] for c in chans])[:, None]
    v = y / sc
    v = torch.where(v > cs, (v - cs) ** 2 + cs, v)
    return torch.where(sq, v.clamp_min(0) ** (4.0 / 3.0), v)


FAMILY = {"DNASE": "DNASE", "ATAC": "ATAC", "CHIP_TF": "CHIP", "CHIP_HISTONE": "CHIP", "RNA_SEQ": "RNA", "CAGE": "CAGE"}


def exact_map(rows, chans):
    """Our track rows -> comparator channels with the same experiment accession, assay family and strand.
    Several channels on one key are averaged by the caller. Unstranded comparator channels can stand in for
    a strand pair summed (handled by the caller)."""
    by = {}
    for c in chans:
        if c["accession"] and not c["gtex"]:
            by.setdefault((c["type"], c["accession"], c["strand"]), []).append(c["ch"])
    return [by.get((FAMILY.get(r["block"]), r["experiment"], r["strand"]), []) for r in rows]


# biosample-name matching for the cell-conditioning comparison
_STOP = {"male", "female", "adult", "child", "embryo", "newborn", "fetal", "originated", "donor", "pool1", "pool2",
         "pool3", "unknown", "donor0", "donor1", "donor2", "donor3", "sex"}
ALIAS = {"prostate gland": ["prostate"]}


def _words(s):
    ws = [re.sub(r"[^a-z0-9]", "", w) for w in re.split(r"[\s,;/]+", s.lower())]
    return [w[:-1] if (w.endswith("s") and len(w) > 4) else w for w in ws if w]


def bs_match(ours, desc_bs, typ, is_line=False):
    """'exact' | 'approx' | None: does a comparator channel's biosample text name our biosample?"""
    d = _words(desc_bs.split(":")[-1] if (typ == "CAGE" and "cell line:" in desc_bs)
               else (desc_bs.split(",")[0] if typ == "CAGE" else desc_bs))
    for nm in [ours] + ALIAS.get(ours.lower(), []):
        o = _words(nm)
        if d[:len(o)] == o and (len(d) == len(o) or d[len(o)] in _STOP or d[len(o)][:1].isdigit()):
            return "exact" if nm == ours else "approx"
    o = _words(ours)
    if is_line and o and len(o[0]) >= 4 and d and d[0] == o[0] and (len(d) == 1 or len(o) == 1):
        return "approx"
    return None


def key_compatible(ch, block, target, strand):
    typ = {"CAGE": "CAGE", "RNA_SEQ": "RNA", "CHIP_HISTONE": "CHIP", "CHIP_TF": "CHIP"}.get(block)
    if typ is None or ch["type"] != typ or ch["gtex"]:
        return False
    if typ == "CHIP" and ch["target"].lower() != target.lower():
        return False
    return strand == "." or ch["strand"] in (strand, ".")
