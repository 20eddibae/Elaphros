"""Fine-tune on the 5'UTR MPRA of Sample et al. 2019 (GEO GSE114002, GSM3130435_egfp_unmod_1.csv) and
compare with cheap baselines at matched label budgets.

    python scripts/finetune_utr.py --csv GSM3130435_egfp_unmod_1.csv --ckpt runs/elaphros/snap_step060000.pt \
        --arms bar_kmer,cnn_small,probe,full_ft,scratch --budgets 1024,16384 --seeds 0,1 --out results/utr

Target: mean ribosome load in polysome-bin units. Fixed disjoint test (20,000 rows), selection (10,000) and
training pool; each budget N is a prefix of the shuffled pool. Every arm's hyperparameters (ridge alpha and
k, or learning rate) are chosen on the selection rows, and the selected model is scored on test.

Arms
  bar_kmer   ridge on k-mer counts, k in {3, 4, 5, 6}
  cnn_small  a 0.47M one-hot CNN trained from scratch (3 x conv(120, k=8) + dense 40)
  probe      ridge on frozen pooled Elaphros features
  full_ft    every Elaphros parameter + a linear readout
  scratch    the same architecture from random initialisation
  borzoi_probe, enformer_probe   ridge on frozen pooled comparator features
  borzoi_ft, enformer_ft         full fine-tuning of the comparator trunk + a linear readout
                                 (Enformer's BatchNorm statistics frozen; Borzoi's updating unless --freeze-bn)

Each 50 bp insert is centred in a 512 bp input with a fixed random flank shared by every row. The readout
mean- and max-pools encoder levels 0, 2 and 4, the 1 bp embedding and the 128 bp trunk over the insert.
Comparators run their crop-free trunks on the same 512 bp input and pool their 2 bp stem plus their tower
outputs (Borzoi: 32 and 128 bp; Enformer: 128 bp).
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elaphros import baselines as BL  # noqa: E402
from elaphros.model import Elaphros, ElaphrosConfig, load_checkpoint  # noqa: E402

L_WIN, INSERT = 512, 50
LRS = {"full_ft": (1e-5, 3e-5, 1e-4, 3e-4), "scratch": (1e-4, 3e-4, 1e-3, 3e-3), "cnn_small": (3e-4, 1e-3, 3e-3, 1e-2),
       "borzoi_ft": (1e-5, 3e-5, 1e-4, 3e-4), "enformer_ft": (1e-5, 3e-5, 1e-4, 3e-4)}
SOURCES = ("enc0", "enc2", "enc4", "e1", "trunk")


# ----------------------------------------------------------------------------- data and metrics
def load_utr(csv):
    import pandas as pd
    df = pd.read_csv(csv, index_col=0)
    P = df[[f"r{i}" for i in range(14)]].to_numpy(np.float64)
    y = P @ np.arange(14, dtype=np.float64)
    ok = np.isfinite(y)
    return [s for s, k in zip(df.utr.tolist(), ok) if k], y[ok]


def onehot(seqs, L=INSERT):
    lut = np.zeros((256, 4), np.float32)
    for j, c in enumerate("ACGT"):
        lut[ord(c), j] = lut[ord(c.lower()), j] = 1.0
    a = np.frombuffer("".join(s[:L].ljust(L, "N") for s in seqs).encode(), np.uint8).reshape(len(seqs), L)
    return lut[a]


def kmer_counts(seqs, k):
    B, n = len(seqs), len(seqs[0])
    code = np.full(256, -1, np.int64)
    for j, c in enumerate("ACGT"):
        code[ord(c)] = code[ord(c.lower())] = j
    c = code[np.frombuffer("".join(seqs).encode(), np.uint8).reshape(B, n)]
    bad = c < 0
    c = np.where(bad, 0, c)
    idx = np.zeros((B, n - k + 1), np.int64); ok = np.ones((B, n - k + 1), bool)
    for j in range(k):
        idx = idx * 4 + c[:, j:j + n - k + 1]
        ok &= ~bad[:, j:j + n - k + 1]
    X = np.zeros((B, 4 ** k), np.float32)
    np.add.at(X, (np.repeat(np.arange(B), n - k + 1), idx.ravel()), ok.ravel().astype(np.float32))
    return X


def ridge_sweep(Xt, yt, Xs, ys, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6)):
    Xt, Xs = np.asarray(Xt, np.float64), np.asarray(Xs, np.float64)
    mx, my = Xt.mean(0), yt.mean()
    Xc, yc = Xt - mx, yt - my
    G, g = Xc.T @ Xc, Xc.T @ yc
    best = None
    for al in alphas:
        try:
            w = np.linalg.solve(G + al * np.eye(len(G)), g)
        except np.linalg.LinAlgError:
            continue
        r2 = r2_score(ys, (Xs - mx) @ w + my)
        if best is None or r2 > best[0]:
            best = (r2, lambda X, w=w: (np.asarray(X, np.float64) - mx) @ w + my, al)
    return best


def r2_score(y, p):
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def pearson(y, p):
    y, p = y - y.mean(), p - p.mean()
    return float((y * p).sum() / np.sqrt((y ** 2).sum() * (p ** 2).sum()))


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------------------- models
class Windower:
    """Places the insert at the centre of a fixed 512 bp flank drawn once at hg38 base composition."""

    def __init__(self, seed=20261006):
        rng = np.random.default_rng(seed)
        p = np.array([0.295, 0.205, 0.205, 0.295])
        self.pad = (L_WIN - INSERT) // 2
        self.left = np.eye(4, dtype=np.float32)[rng.choice(4, self.pad, p=p)]
        self.right = np.eye(4, dtype=np.float32)[rng.choice(4, L_WIN - INSERT - self.pad, p=p)]

    def __call__(self, oh):
        B = oh.shape[0]
        return np.concatenate([np.broadcast_to(self.left, (B,) + self.left.shape), oh,
                               np.broadcast_to(self.right, (B,) + self.right.shape)], 1).copy()


def pooled(feats, a, b):
    out = []
    for s in SOURCES:
        z, res = (feats["enc"][int(s[3:])], 2 ** int(s[3:])) if s.startswith("enc") else \
                 (feats["e1"], 1) if s == "e1" else (feats["trunk"], 128)
        z = z[..., a // res:max(a // res + 1, -(-b // res))].float()
        out += [z.mean(-1), z.amax(-1)]
    return torch.cat(out, 1)


class ElaphrosScalar(nn.Module):
    def __init__(self, trunk: Elaphros, span):
        super().__init__()
        c = trunk.cfg
        W = list(c.widths)
        d = 2 * (W[0] + W[2] + W[4] + c.emb1_dim + c.trunk_width)
        self.trunk, self.span = trunk, span
        self.norm, self.lin = nn.LayerNorm(d), nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(self.norm(pooled(self.trunk.features(x), *self.span))).squeeze(-1)


def pooled_baseline(name, feats, a, b):
    out = []
    for s, (res, _) in BL.SOURCES[name].items():
        z = feats[s][..., a // res:max(a // res + 1, -(-b // res))].float()
        out += [z.mean(-1), z.amax(-1)]
    return torch.cat(out, 1)


def load_baseline(name, dev):
    m = BL.load(name, dev, heads=False)
    BL.reset_positions(m)
    return m


class BaselineScalar(nn.Module):
    """Comparator trunk + readout (Borzoi: LayerNorm + linear, as for Elaphros; Enformer: linear)."""

    def __init__(self, name, trunk, span):
        super().__init__()
        self.name, self.trunk, self.span = name, trunk, span
        d = 2 * sum(c for _, c in BL.SOURCES[name].values())
        self.norm = nn.LayerNorm(d) if name == "borzoi" else nn.Identity()
        self.lin = nn.Linear(d, 1)
        if name == "enformer":
            nn.init.normal_(self.lin.weight, std=0.01); nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(self.norm(pooled_baseline(self.name, BL.trunk(self.name, self.trunk, x), *self.span))).squeeze(-1)


def freeze_bn_stats(m):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.eval()


class SmallCNN(nn.Module):
    """3 x conv(120, k=8) + dense 40 on the insert alone, after Sample et al. 2019."""

    def __init__(self, width=120, k=8, hidden=40, dropout=0.2):
        super().__init__()
        ch = [4, width, width, width]
        self.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], k, padding=k // 2) for i in range(3)])
        self.drop = nn.Dropout(dropout)
        self.fc1, self.fc2 = nn.Linear(width * INSERT, hidden), nn.Linear(hidden, 1)

    def forward(self, x):
        h = x.transpose(1, 2)
        for c in self.convs:
            h = F.relu(c(h))[..., :INSERT]
        return self.fc2(self.drop(F.relu(self.fc1(self.drop(h.flatten(1)))))).squeeze(-1)


# ----------------------------------------------------------------------------- arms
def train_sgd(arm, seqs, y, tr, sel, te, lr, a, win, dev, seed):
    torch.manual_seed(seed_of(arm, lr, seed))
    ym, ys = float(y[tr].mean()), float(y[tr].std() + 1e-9)
    if arm == "cnn_small":
        model = SmallCNN().to(dev)
        inp = lambda rows: torch.from_numpy(onehot([seqs[j] for j in rows])).to(dev)  # noqa: E731
    elif arm in ("borzoi_ft", "enformer_ft"):
        name = arm.split("_")[0]
        model = BaselineScalar(name, load_baseline(name, dev), (win.pad, win.pad + INSERT)).to(dev)
        inp = lambda rows: torch.from_numpy(win(onehot([seqs[j] for j in rows]))).to(dev)  # noqa: E731
    else:
        trunk = load_checkpoint(a.ckpt, dev)[0] if arm == "full_ft" else \
            Elaphros(ElaphrosConfig.from_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"])).to(dev)
        model = ElaphrosScalar(trunk, (win.pad, win.pad + INSERT)).to(dev)
        inp = lambda rows: torch.from_numpy(win(onehot([seqs[j] for j in rows]))).to(dev)  # noqa: E731
    if arm != "cnn_small":                       # no dropout in pretrained or from-scratch trunks
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    frozen_bn = arm == "enformer_ft" or (arm == "borzoi_ft" and a.freeze_bn)
    model.train()
    if frozen_bn:
        freeze_bn_stats(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    B = min(a.batch, len(tr))
    steps = int(np.clip(int(np.ceil(len(tr) / B)) * a.epochs, a.min_steps, a.max_steps))
    warm = max(5, int(0.05 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + np.cos(np.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    rng = np.random.default_rng(seed_of("order", arm, lr, seed))
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = []
        for i in range(0, len(rows), 192):
            with torch.autocast(**amp):
                out.append(model(inp(rows[i:i + 192])).float().cpu().numpy())
        model.train()
        if frozen_bn:
            freeze_bn_stats(model)
        return np.concatenate(out) * ys + ym

    best = (-1e9, None, 0)
    every = max(1, steps // 10)
    for s in range(steps):
        rows = tr[rng.integers(0, len(tr), B)] if len(tr) > B else tr
        with torch.autocast(**amp):
            loss = F.mse_loss(model(inp(rows)).float(), torch.as_tensor((y[rows] - ym) / ys, dtype=torch.float32, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        if (s + 1) % every == 0 or s == steps - 1:
            r2 = r2_score(y[sel], predict(sel))
            if r2 > best[0]:
                best = (r2, {k: v.detach().clone() for k, v in model.state_dict().items()}, s + 1)
    model.load_state_dict(best[1])
    return predict(te), best[0], {"lr": lr, "best_step": best[2], "steps": steps,
                                  "trainable": sum(p.numel() for p in params)}


@torch.no_grad()
def frozen_features(rows, seqs, model, win, dev, name="elaphros"):
    model.eval()
    out = []
    for i in range(0, len(rows), 192):
        x = torch.from_numpy(win(onehot([seqs[j] for j in rows[i:i + 192]]))).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            f = pooled(model.features(x), win.pad, win.pad + INSERT) if name == "elaphros" else \
                pooled_baseline(name, BL.trunk(name, model, x), win.pad, win.pad + INSERT)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="bar_kmer,cnn_small,probe,full_ft,scratch")
    ap.add_argument("--budgets", default="1024,16384")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--n-sel", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--freeze-bn", action="store_true", help="borzoi_ft: hold BatchNorm running statistics fixed")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seqs, y = load_utr(a.csv)
    perm = np.random.default_rng(0).permutation(len(y))
    te, sel, pool = perm[:a.n_test], perm[a.n_test:a.n_test + a.n_sel], perm[a.n_test + a.n_sel:]
    win = Windower()
    budgets = [int(b) for b in a.budgets.split(",")]
    arms = a.arms.split(",")
    feats = {}
    allrows = np.concatenate([te, sel, pool[:max(budgets)]])
    pos = {int(r): i for i, r in enumerate(allrows)}
    for arm in arms:
        if arm.endswith("probe"):
            name = "elaphros" if arm == "probe" else arm.split("_")[0]
            base = load_checkpoint(a.ckpt, dev)[0] if name == "elaphros" else load_baseline(name, dev)
            F_ = frozen_features(allrows, seqs, base, win, dev, name)
            feats[arm] = lambda rows, F_=F_: F_[[pos[int(r)] for r in rows]]
            del base
    for N in budgets:
        tr = pool[:N]
        for arm in arms:
            for seed in [int(s) for s in a.seeds.split(",")]:
                cell = out / f"{arm}__N{N}__s{seed}.json"
                if cell.exists():
                    continue
                if arm == "bar_kmer" or arm.endswith("probe"):
                    if seed != int(a.seeds.split(",")[0]):
                        continue                                 # deterministic: one seed is enough
                    cands = [(f"kmer{k}", lambda r, k=k: kmer_counts([seqs[j] for j in r], k)) for k in (3, 4, 5, 6)] \
                        if arm == "bar_kmer" else [(arm, feats[arm])]
                    best = None
                    for name, fx in cands:
                        r2, f, al = ridge_sweep(fx(tr), y[tr], fx(sel), y[sel])
                        if best is None or r2 > best[0]:
                            best = (r2, f, al, name, fx)
                    pred, hp = best[1](best[4](te)), {"alpha": best[2], "features": best[3]}
                    sel_r2 = best[0]
                else:
                    runs = [train_sgd(arm, seqs, y, tr, sel, te, lr, a, win, dev, seed) for lr in LRS[arm]]
                    pred, sel_r2, hp = max(runs, key=lambda r: r[1])
                rec = {"arm": arm, "N": N, "seed": seed, "test_r2": r2_score(y[te], pred),
                       "test_pearson": pearson(y[te], pred), "sel_r2": sel_r2, "hp": hp}
                cell.write_text(json.dumps(rec, indent=1))
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
