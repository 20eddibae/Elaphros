"""Cell-type pools, conditioning batches, the conditioned loss and held-out-cell scoring.

A biosample is an ontology term (tracks.tsv `term_id`). Held-out biosamples are listed in a split file
{"val": [...], "test": [...]}; every track of a held-out biosample is removed from the training loss.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

from .data import BLOCKS, VAL_CORE, VAL_L, Cache, Tracks
from .losses import log1p_pearson, track_loss

ACC = ("ATAC", "DNASE")
TARGET_MODS = ("CAGE", "RNA_SEQ", "CHIP_HISTONE", "CHIP_TF", "PROCAP")
NEVER_HOLD = ("EFO:0002067", "EFO:0001187", "EFO:0002784")   # K562, HepG2, GM12878


def key_of(t):
    return (t["block"], t["assay"], t["target"], t["strand"])


def track_records(tracks: Tracks):
    out = []
    for b in BLOCKS:
        c0, c1 = tracks.cols[b]
        for j, r in enumerate(tracks.rows[c0:c1]):
            out.append({**r, "col": c0 + j, "j": j, "nzm": float(r["nonzero_mean"])})
    return out


def make_split(tracks: Tracks, frac=0.15, path=None):
    """Deterministic held-out split: among biosamples with accessibility and at least one target track,
    excluding the three most densely profiled cell lines, hold out the `frac` with the smallest crc32 of the
    ontology term id; split them val / test by crc32 parity."""
    by = {}
    for t in track_records(tracks):
        by.setdefault(t["term_id"], set()).add(t["block"])
    elig = sorted(b for b, m in by.items() if m & set(ACC) and m & set(TARGET_MODS) and b not in NEVER_HOLD)
    crc = lambda s: zlib.crc32(s.encode())
    ho = sorted(sorted(elig, key=crc)[:max(2, round(frac * len(elig)))])
    split = {"val": [b for b in ho if crc(b) % 2 == 0], "test": [b for b in ho if crc(b) % 2 == 1]}
    if path:
        Path(path).write_text(json.dumps(split, indent=1))
    return split


class Pools:
    """Training and held-out biosamples, their accessibility and target tracks, and assay keys (defined on
    training biosamples only)."""

    def __init__(self, tracks: Tracks, split, root=None, n_sim_bins=100_000):
        T = track_records(tracks)
        self.val, self.test = list(split["val"]), list(split["test"])
        self.ho = set(self.val) | set(self.test)
        self.ho_cols = {t["col"] for t in T if t["term_id"] in self.ho}
        keys = sorted({key_of(t) for t in T if t["term_id"] not in self.ho and t["block"] in TARGET_MODS})
        self.key_id = {k: i for i, k in enumerate(keys)}
        self.n_keys = len(keys)
        self.acc, self.tgt = {}, {}
        for t in T:
            if t["block"] in ACC:
                self.acc.setdefault(t["term_id"], []).append(t)
            elif t["block"] in TARGET_MODS and key_of(t) in self.key_id:
                self.tgt.setdefault(t["term_id"], []).append(t)
        self.train = sorted(b for b in self.acc if b in self.tgt and b not in self.ho)
        self.key_train = {}
        for b in self.train:
            for t in self.tgt[b]:
                self.key_train.setdefault(key_of(t), []).append(t)
        self.R = self._similarity(root, n_sim_bins) if root else None

    def _similarity(self, root, n_bins):
        """Correlation of log1p accessibility between biosamples on random training-region 128 bp bins."""
        cache = Cache(root, "train", np.random.default_rng(20261008))
        names = sorted(b for b in self.acc if b in self.tgt)
        cols = [self.acc[b][0]["col"] for b in names]
        rows = []
        chroms = sorted(cache.G)
        while sum(len(r) for r in rows) < n_bins:
            c = chroms[cache.rng.integers(len(chroms))]
            s = int(cache.rng.integers(0, cache.G[c].shape[0] - 1024)) * 128
            if not cache._clear(c, s, s + 1024 * 128):
                continue
            y, fill = cache.window(c, s, 1024 * 128)
            rows.append(y[cols][:, fill].T)
        X = np.log1p(np.clip(np.concatenate(rows), 0, None))
        R = np.corrcoef(X.T)
        return {"names": names, "ix": {n: i for i, n in enumerate(names)}, "R": R}

    def nearest_train_with_key(self, b, key):
        cands = self.key_train.get(key, [])
        if not cands or self.R is None:
            return None
        ix, R = self.R["ix"], self.R["R"]
        return max(cands, key=lambda t: (R[ix[b], ix[t["term_id"]]], -t["col"]))

    def drop_heldout(self, tracks: Tracks):
        """Remove every held-out biosample's columns from the base model's loss."""
        for b in BLOCKS:
            c0, _ = tracks.cols[b]
            tracks.use[b] = torch.tensor([int(j) for j in tracks.use[b] if c0 + int(j) not in self.ho_cols], dtype=torch.long)
            assert not any(c0 + int(j) in self.ho_cols for j in tracks.use[b])


def scaled(y, nz, rna, res=128):
    """Measured -> model space for rows y (N, T) with normalisers nz (N,) and RNA flags rna (N,)."""
    y = y / torch.as_tensor(nz * res, dtype=torch.float64).to(y.dtype)[:, None]
    y = torch.where(torch.as_tensor(rna)[:, None], y.clamp_min(0).pow(0.75), y)
    return torch.where(y > 10, 2 * torch.sqrt(10 * y.clamp_min(10)) - 10, y)


def _nz(t):
    return (t["nzm"] if np.isfinite(t["nzm"]) and t["nzm"] > 0 else 1.0), t["block"] == "RNA_SEQ"


def make_cc_batch(y128, pools: Pools, rng, K, M):
    """y128 (B, n_cols, T) measured. For each window draw K training biosamples; condition on one of each
    biosample's accessibility tracks and target up to M of its other tracks."""
    B, _, T = y128.shape
    acc_col = np.zeros((B, K), np.int64); acc_nz = np.ones((B, K)); acc_rna = np.zeros((B, K), bool)
    t_col = np.zeros((B, K, M), np.int64); t_nz = np.ones((B, K, M)); t_rna = np.zeros((B, K, M), bool)
    keys = np.full((B, K, M), -1, np.int64)
    for b in range(B):
        for k, i in enumerate(rng.choice(len(pools.train), size=K, replace=len(pools.train) < K)):
            c = pools.train[i]
            at = pools.acc[c][int(rng.integers(len(pools.acc[c])))]
            acc_col[b, k] = at["col"]; acc_nz[b, k], acc_rna[b, k] = _nz(at)
            tt = pools.tgt[c]
            for m, p in enumerate(rng.choice(len(tt), size=min(M, len(tt)), replace=False)):
                t = tt[int(p)]
                keys[b, k, m] = pools.key_id[key_of(t)]
                t_col[b, k, m] = t["col"]; t_nz[b, k, m], t_rna[b, k, m] = _nz(t)
    y = y128.float()
    bi = np.arange(B)
    acc = scaled(y[np.repeat(bi, K), acc_col.reshape(-1)], acc_nz.reshape(-1), acc_rna.reshape(-1)).reshape(B, K, T)
    valid = keys >= 0
    tgt = torch.zeros(B, K, M, T)
    if valid.any():
        bb = np.broadcast_to(bi[:, None, None], keys.shape)[valid]
        tgt[torch.from_numpy(valid)] = scaled(y[bb, t_col[valid]], t_nz[valid], t_rna[valid])
    return {"acc": acc, "keys": torch.from_numpy(keys), "tgt": tgt}


def cc_loss(pred, tgt, keys, fill, seg_bp=8192, shape_weight=5.0, loss_fn=track_loss):
    """track_loss on the stacked (window, cell, key) predictions, normalised to a mean over valid slots."""
    B, K, M, T = pred.shape
    valid = (keys >= 0).float()
    p = (pred * valid[..., None]).reshape(B, K * M, T)
    t = (tgt * valid[..., None]).reshape(B, K * M, T)
    mk = fill[:, None, :].float()
    raw = loss_fn(p, t, mk, 128, seg_bp, shape_weight)
    seg = max(1, min(T, seg_bp // 128))
    m = mk.reshape(B, 1, T // seg, seg).mean(-1).sum(-1).squeeze(1)
    return raw * (m.sum() * K * M) / ((m * valid.sum((1, 2))).sum() + 1e-7)


def partial_r(x, y, z):
    """Partial correlation r(x, y | z): 0 for any predictor that is a function of z alone."""
    if y.std() == 0 or z.std() == 0:
        return float("nan")
    if x.std() == 0:
        return 0.0
    c = torch.corrcoef(torch.stack([x, y, z]).double())
    rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
    return float((rxy - rxz * ryz) / torch.sqrt((1 - rxz ** 2).clamp_min(1e-12) * (1 - ryz ** 2).clamp_min(1e-12)))


def _one(y, t):
    nz, rna = _nz(t)
    return scaled(y[None], np.array([nz]), np.array([rna]))[0]


@torch.no_grad()
def score_rows(model, batches, pools: Pools, tracks: Tracks, biosamples, dev, cond_src=None, rng_seed=0):
    """Score held-out biosamples on the central 131 kb of fixed windows. One row per (window, biosample,
    target track) with plain log1p Pearson and the partial Pearson given the training-cell mean track for:
      model  the conditioned prediction (conditioned on cond_src(b) if given, e.g. a random training cell)
      bar_a  the nearest training biosample's measured track with the same assay key
      bar_c  the biosample's own accessibility
      bar_d  the model's sequence-only head for that nearest training biosample's track
    """
    model.eval()
    k0 = (VAL_L // 128 - VAL_CORE // 128) // 2
    sl = slice(k0, k0 + VAL_CORE // 128)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for wi, bt in enumerate(batches):
        y = bt["y128"][0]
        bl = [b for b in biosamples if b in pools.acc and b in pools.tgt]
        if not bl:
            continue
        K, M = len(bl), max(len(pools.tgt[b]) for b in bl)
        acc = torch.zeros(1, K, y.shape[-1]); keys = torch.full((1, K, M), -1, dtype=torch.long)
        for k, b in enumerate(bl):
            src = cond_src(b, rng) if cond_src is not None else b
            acc[0, k] = _one(y[pools.acc[src][0]["col"]].float(), pools.acc[src][0])
            for m, t in enumerate(pools.tgt[b]):
                keys[0, k, m] = pools.key_id[key_of(t)]
        cc = {"acc": acc.to(dev), "fill": bt["fill"].to(dev), "keys": keys.to(dev)}
        org = torch.zeros(1, dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            # a model without the cell branch (the Base model) is scored on its sequence-only rows only
            out = model(bt["x"].to(dev), org, heads_1bp=False, **({"cc": cc} if hasattr(model, "ccb") else {}))
        pred = out["cc"][0].float()[..., sl].cpu() if "cc" in out else None
        f = bt["fill"][0, sl]
        for k, b in enumerate(bl):
            for m, t in enumerate(pools.tgt[b]):
                truth = _one(y[t["col"], sl].float(), t)
                rec = {"win": wi, "biosample": b, "block": t["block"], "col": t["col"]}
                lt = torch.log1p(truth.clamp_min(0))
                tr_k = pools.key_train.get(key_of(t), [])
                if lt.std() == 0 or not tr_k:
                    continue
                mean_tr = torch.stack([_one(y[u["col"], sl].float(), u) for u in tr_k]).mean(0)
                cand = {"bar_c": _one(y[pools.acc[b][0]["col"], sl].float(), pools.acc[b][0])}
                if pred is not None:
                    cand["model"] = pred[k, m]
                nn_t = pools.nearest_train_with_key(b, key_of(t))
                if nn_t is not None:
                    cand["bar_a"] = _one(y[nn_t["col"], sl].float(), nn_t)
                    cand["bar_d"] = out[f"{nn_t['block'].lower()}@128"][0, nn_t["j"], sl].float().cpu()
                lb = torch.log1p(mean_tr.clamp_min(0))
                for nm, v in cand.items():
                    rec[f"{nm}/plain"] = float(log1p_pearson(v[None, None], truth[None, None], f[None]).mean())
                    rec[f"{nm}/partial"] = partial_r(torch.log1p(v.clamp_min(0)), lt, lb)
                rows.append(rec)
    model.train()
    return rows


def summarise(rows, prefix=""):
    out = {f"{prefix}n_rows": len(rows)}
    stats = sorted({k for r in rows for k in r if "/" in k})
    for mod in ("ALL",) + TARGET_MODS:
        sub = rows if mod == "ALL" else [r for r in rows if r["block"] == mod]
        for s in stats:
            v = [r[s] for r in sub if s in r and r[s] == r[s]]
            if v:
                out[f"{prefix}{mod}/{s}"] = round(float(np.mean(v)), 4)
    return out


# ----------------------------------------------------------------------------- comparators (Borzoi, Enformer)
@torch.no_grad()
def baseline_core(name, m, chans, keep, bt, dev):
    """Decoded comparator predictions (len(keep), core bins) at 128 bp over the scored core of one window."""
    from . import baselines as BL
    k0, nb = (VAL_L // 128 - VAL_CORE // 128) // 2, VAL_CORE // 128
    x = bt["x"].to(dev)
    if name == "enformer":            # 114,688 bp of output per pass: a left- and a right-aligned pass cover the core
        Lin, flank = 196608, (196608 - 114688) // 2
        parts = [BL.predict(name, m, x[:, st:st + Lin]).float()[0][keep].cpu()
                 for st in (k0 * 128 - flank, k0 * 128 + VAL_CORE - 114688 - flank)]
        return torch.cat([parts[0], parts[1][:, 896 - (nb - 896):]], 1)
    y = BL.decode(name, BL.predict(name, m, x).float()[0][keep].cpu(), [chans[c] for c in keep])
    y = y.view(len(keep), -1, 4).sum(-1)                         # 32 bp -> 128 bp; output starts at bin 1280
    return y[:, k0 - 1280:k0 - 1280 + nb]


def baseline_rows(name, m, chans, batches, pools, tracks, biosamples, dev):
    """Comparator rows on the same (window, held-out biosample, track) rows as score_rows, for three ways of
    asking a model with fixed output tracks about a new cell type:
      EXACT    its own channel for that biosample, when it has one (it trained on that cell: a ceiling)
      NEAREST  its channel for the nearest training biosample (by accessibility) that it has a channel for
      MEAN     the mean of its channels with the same assay key, excluding every held-out biosample
    Each prediction is put on the measured scale of the training-cell mean track for its key, then scored with
    the same partial Pearson as the model."""
    from . import baselines as BL
    name_of = {t["term_id"]: (t["term_name"], t["classification"] == "cell line") for t in track_records(tracks)}
    match = {u: [(c, q) for c in chans if (q := BL.bs_match(name_of[u][0], c["bs"], c["type"], name_of[u][1]))]
             for u in set(pools.train) | pools.ho if u in name_of}
    ho_ch = {c["ch"] for u in pools.ho for c, _ in match.get(u, [])}
    ix, R = pools.R["ix"], pools.R["R"]

    def exact(u, t):
        return [c["ch"] for c, q in match.get(u, []) if q == "exact" and BL.key_compatible(c, t["block"], t["target"], t["strand"])]

    biosamples = [b for b in biosamples if b in pools.acc and b in pools.tgt]
    plan = {}
    for b in biosamples:
        rank = sorted(pools.train, key=lambda u: -R[ix[b], ix[u]])
        for t in pools.tgt.get(b, []):
            near = next((ch for u in rank if (ch := exact(u, t))), [])
            mean = [c["ch"] for c in chans if BL.key_compatible(c, t["block"], t["target"], t["strand"]) and c["ch"] not in ho_ch]
            plan[(b, t["col"])] = {"EXACT": exact(b, t), "NEAREST": near, "MEAN": mean}
    keep = sorted({c for p in plan.values() for v in p.values() for c in v})
    pos = {c: i for i, c in enumerate(keep)}
    k0, nb = (VAL_L // 128 - VAL_CORE // 128) // 2, VAL_CORE // 128
    sl = slice(k0, k0 + nb)
    core = [baseline_core(name, m, chans, keep, bt, dev) for bt in batches]
    anchor = {}                                                  # training-cell mean level per key, model space
    rows = []
    for b in biosamples:
        for t in pools.tgt.get(b, []):
            k = key_of(t)
            if not pools.key_train.get(k):
                continue
            if k not in anchor:
                anchor[k] = float(np.mean([np.mean([float(bt["y128"][0, u["col"], sl].float().mean()) / (_nz(u)[0] * 128)
                                                    for u in pools.key_train[k]]) for bt in batches]))
            for proto, chs in plan[(b, t["col"])].items():
                if not chs:
                    continue
                raw = [c[[pos[x] for x in chs]].mean(0) for c in core]
                s = anchor[k] / max(float(torch.stack(raw).mean()), 1e-12)
                for wi, bt in enumerate(batches):
                    y = bt["y128"][0]
                    truth = _one(y[t["col"], sl].float(), t)
                    lt = torch.log1p(truth.clamp_min(0))
                    if lt.std() == 0:
                        continue
                    lb = torch.log1p(torch.stack([_one(y[u["col"], sl].float(), u) for u in pools.key_train[k]]).mean(0).clamp_min(0))
                    pred = scaled((raw[wi] * s * 128)[None].double(), np.array([1.0]), np.array([t["block"] == "RNA_SEQ"]))[0]
                    rows.append({"win": wi, "biosample": b, "block": t["block"], "col": t["col"], "protocol": proto,
                                 "partial": partial_r(torch.log1p(pred.float().clamp_min(0)), lt, lb)})
    return rows


def paired_gaps(model_rows, base_rows):
    """Mean of (model - comparator) partial Pearson over rows both scored, per protocol and modality."""
    mr = {(r["win"], r["biosample"], r["col"]): r["model/partial"] for r in model_rows}
    out = {}
    for proto in ("EXACT", "NEAREST", "MEAN"):
        d = [(r["block"], mr[(r["win"], r["biosample"], r["col"])] - r["partial"]) for r in base_rows
             if r["protocol"] == proto and (r["win"], r["biosample"], r["col"]) in mr and r["partial"] == r["partial"]]
        if d:
            out[proto] = {"ALL": round(float(np.mean([v for _, v in d])), 4), "n_rows": len(d),
                          **{m: round(float(np.mean([v for b, v in d if b == m])), 4) for m in TARGET_MODS if any(b == m for b, _ in d)}}
    return out
