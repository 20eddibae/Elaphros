"""Inference speed on one GPU.

    python scripts/benchmark_speed.py variant --model elaphros --precision bf16 --compile
    python scripts/benchmark_speed.py length  --model elaphros-conv

variant  latency of one scored variant: a REF and an ALT forward at the model's native input length, the
         128 bp heads, and the summed difference of one track over the 384 bp around the variant.
         Batches of 1, 2 and 4 variants; 10 warm-up and 3 x 50 timed calls. --model borzoi, flashzoi or
         enformer times the published comparators (needs borzoi-pytorch / enformer-pytorch).
length   forward time at 131,072, 524,288 and 1,048,576 bp, and the exponent of a power-law fit of time
         against length. Elaphros models run with their 128 bp heads; Borzoi runs its crop-free trunk and
         Enformer its embeddings (target_length=-1), so head size does not dominate.

Timing does not depend on the weights, so --ckpt is optional (randomly initialised models time the same).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elaphros.model import Elaphros, elaphros_config, elaphros_conv_config, load_checkpoint, n_params  # noqa: E402

NATIVE = {"elaphros": 524288, "elaphros-conv": 524288, "borzoi": 524288, "flashzoi": 524288, "enformer": 196608}


def build(name, ckpt):
    """Returns (forward(x (B, L, 4)) -> (B, tracks, bins), bp per output bin, bp offset of bin 0, params, module)."""
    if name.startswith("elaphros"):
        m = load_checkpoint(ckpt, "cuda")[0] if ckpt else \
            Elaphros(elaphros_config() if name == "elaphros" else elaphros_conv_config()).cuda().eval()
        return (lambda x: m(x, heads_1bp=False)["dnase@128"]), 128, 0, n_params(m), m
    from elaphros import baselines as BL
    m = BL.load(name, "cuda", flash="native")             # Flashzoi speed needs the flash-attn kernel
    _, bin_bp, off = BL.GEOM[name]
    return (lambda x: BL.predict(name, m, x)), bin_bp, off, n_params(m), m


def timed(fn, n_warm=10, n=50, groups=3):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(groups):
        for _ in range(n):
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def ctx(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(precision == "bf16"))


@torch.inference_mode()
def variant(a):
    fwd, bin_bp, off, params, m = build(a.model, a.ckpt)
    if a.compile:
        m.compile(mode="max-autotune")
    L = NATIVE[a.model]
    rng = np.random.default_rng(0)
    res = {"model": a.model, "params": params, "precision": a.precision, "compiled": a.compile, "input_bp": L, "cells": {}}
    for bs in (1, 2, 4):
        ref = torch.nn.functional.one_hot(torch.from_numpy(rng.integers(0, 4, (bs, L))), 4).float()
        alt = ref.clone()
        alt[:, L // 2] = alt[:, L // 2].roll(1, -1)                         # one substitution at the centre
        x = torch.stack([ref, alt], 1).reshape(2 * bs, L, 4).cuda()
        lo = (L // 2 // 128 * 128 - 128 - off) // bin_bp

        def score():
            with ctx(a.precision):
                y = fwd(x)[:, 0, lo:lo + 384 // bin_bp].float().sum(-1)
            return (y[1::2] - y[0::2]).cpu()

        try:
            torch.cuda.reset_peak_memory_stats()
            med, p95 = timed(score)
            res["cells"][bs] = {"ms_per_variant": 1e3 * med / bs, "p95_ms": 1e3 * p95 / bs,
                                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        except torch.cuda.OutOfMemoryError:
            res["cells"][bs] = "OOM"
            torch.cuda.empty_cache()
        print(json.dumps({bs: res["cells"][bs]}), flush=True)
    return res


@torch.inference_mode()
def length(a):
    fwd, _, _, params, m = build(a.model, a.ckpt)
    if a.model in ("borzoi", "flashzoi"):
        from elaphros import baselines as BL
        state = {"n": None}

        def fwd(x):                                       # rebuild the relative-position table per length
            if state["n"] != x.shape[1]:
                BL.reset_positions(m); state["n"] = x.shape[1]
            return BL.trunk(a.model, m, x)["final"]
    elif a.model == "enformer":
        def fwd(x):
            return m(x, return_only_embeddings=True, target_length=-1)
    if a.compile:
        fwd = torch.compile(fwd)
    res = {"model": a.model, "params": params, "precision": a.precision, "compiled": a.compile, "lengths": {}}
    for L in (131072, 524288, 1048576):
        x = torch.nn.functional.one_hot(torch.randint(0, 4, (1, L)), 4).float().cuda()

        def f():
            with ctx(a.precision):
                fwd(x)
        res["lengths"][L] = 1e3 * timed(f, 3, 10, 1)[0]
        print(json.dumps({L: res["lengths"][L]}), flush=True)
    ls = np.log(list(res["lengths"])); ts = np.log(list(res["lengths"].values()))
    res["exponent"] = float(np.polyfit(ls, ts, 1)[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["variant", "length"])
    ap.add_argument("--model", default="elaphros", choices=list(NATIVE))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "the benchmark needs a GPU"
    res = variant(a) if a.mode == "variant" else length(a)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
