#!/usr/bin/env python3
"""IndexerPrologueK performance / accuracy harness.

Rebuilds the unfused production baseline from the five official CANN ops and
compares it with the fused custom entry, all through llm_ops:

    projected      = llm_ops.matmul(latent, wk, out)                          # MatMulV2
    key, rstd      = llm_ops.rms_norm(projected, gamma, out, rstd, eps)       # RmsNorm
    llm_ops.inplace_partial_rotary_mul(key_4d, cos, sin, 64, 128)             # InplacePartialRotaryMul
    quant, qscale  = llm_ops.dynamic_quant(key, out, scale)                   # DynamicQuant (INT8)
    llm_ops.scatter_nd_update(cache_3d, indices, quant)                      # ScatterNdUpdate

    llm_ops.indexer_prologue_k(latent, wk, gamma, cos, sin, indices,
                               k_cache, k_scale_cache, eps)                   # fused

Prerequisites:
    ./build.sh                                  # builds libllm_ops.so + wheel
    pip install output/llm_ops*.whl             # or PYTHONPATH=utils/python
    # fused entry needs the ops-transformer custom package:
    #   cd ops-transformer && bash build.sh --pkg --soc=ascend910b --ops=indexer_prologue_k
    #   ./build_out/cann-ops-transformer-custom_linux-*.run
    source $CANN/set_env.sh
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH

Usage:
    python3 perf_indexer_prologue_k.py --check --mode both --T 96 --nb 22594 --ns 64 --device 7

Kernel time (msprof op_summary, one collection per mode):
    msprof --output=prof_base  python3 perf_indexer_prologue_k.py --mode baseline ...
    msprof --output=prof_fused python3 perf_indexer_prologue_k.py --mode fused ...
"""

import argparse
import time

import torch
import torch_npu  # noqa: F401

import llm_ops

HEAD_DIM = 128
HIDDEN = 512
ROPE_DIM = 64


def make_inputs(tokens, nb, ns, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16).to(device)
    wk = (torch.randn(HEAD_DIM, HIDDEN, generator=gen) / HIDDEN**0.5).to(torch.bfloat16).to(device)
    gamma = (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1.0).to(torch.bfloat16).to(device)

    angles = torch.rand(tokens, ROPE_DIM, generator=gen) * 6.2831853
    cos = torch.cos(angles).contiguous().view(tokens, 1, 1, ROPE_DIM).to(device)
    sin = torch.sin(angles).contiguous().view(tokens, 1, 1, ROPE_DIM).to(device)

    # production-like contiguous slots: row r -> (block 3 + (r+5)//ns, (r+5)%ns)
    flat = torch.arange(tokens, dtype=torch.int32) + 5
    indices = torch.stack([3 + flat // ns, flat % ns], dim=1).to(device)

    k_cache = torch.randint(-128, 127, (nb, ns, 1, HEAD_DIM), dtype=torch.int8, device=device)
    k_scale_cache = (torch.rand(nb, ns, 1, 1, device=device) * 0.05).to(torch.float16)
    return latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache


def make_buffers(tokens, device):
    # Preallocated so the timed loop has no torch allocator traffic.
    return {
        "projected": torch.empty(tokens, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "key": torch.empty(tokens, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "key_4d": None,
        "rstd": torch.empty(tokens, 1, dtype=torch.float32, device=device),
        "quant": torch.empty(tokens, HEAD_DIM, dtype=torch.int8, device=device),
        "qscale": torch.empty(tokens, dtype=torch.float32, device=device),
    }


def run_baseline(bufs, latent, wk, gamma, cos, sin, indices, cache_3d, eps):
    projected = llm_ops.matmul(latent, wk, bufs["projected"])
    key, _ = llm_ops.rms_norm(projected, gamma, bufs["key"], bufs["rstd"], eps)
    if bufs["key_4d"] is None:
        bufs["key_4d"] = key.view(key.shape[0], 1, 1, HEAD_DIM)
    llm_ops.inplace_partial_rotary_mul(bufs["key_4d"], cos, sin, ROPE_DIM, HEAD_DIM)
    quant, qscale = llm_ops.dynamic_quant(key, bufs["quant"], bufs["qscale"])
    llm_ops.scatter_nd_update(cache_3d, indices, quant)
    return quant, qscale


def run_fused(latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache, eps):
    llm_ops.indexer_prologue_k(latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache, eps)


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def check(bufs, latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache, eps):
    nb, ns = k_cache.shape[0], k_cache.shape[1]
    init_cache = k_cache.clone()
    init_scale = k_scale_cache.clone()

    quant, qscale = run_baseline(bufs, latent, wk, gamma, cos, sin, indices,
                                 k_cache.view(nb, ns, HEAD_DIM), eps)
    torch.npu.synchronize()

    fused_cache = init_cache.clone()
    fused_scale = init_scale.clone()
    run_fused(latent, wk, gamma, cos, sin, indices, fused_cache, fused_scale, eps)
    torch.npu.synchronize()

    flat = indices[:, 0].long() * ns + indices[:, 1].long()
    touched = torch.zeros(nb * ns, dtype=torch.bool, device=k_cache.device)
    touched[flat] = True

    got = fused_cache.view(nb * ns, HEAD_DIM)[flat].to(torch.int16)
    diff = (got - quant.to(torch.int16)).abs()
    print(f"k_cache: max|diff|={int(diff.max())} count(>1)={int((diff > 1).sum())} elems={diff.numel()}")

    changed = (fused_cache.view(nb * ns, HEAD_DIM)[~touched] !=
               init_cache.view(nb * ns, HEAD_DIM)[~touched]).sum()
    print(f"untouched slots: changed bytes={int(changed)}")

    got_scale = fused_scale.view(nb * ns)[flat].to(torch.float32)
    want_scale = qscale.to(torch.float16).to(torch.float32)
    rel = ((got_scale - want_scale).abs() / want_scale.abs().clamp_min(1e-12)).max()
    bad = ((got_scale - want_scale).abs() > 9e-2 + 1e-3 * want_scale.abs()).sum()
    print(f"k_scale: max_rel={float(rel):.3e} rows_out_of_tol={int(bad)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "fused", "both"], default="both")
    parser.add_argument("--T", type=int, default=96)
    parser.add_argument("--nb", type=int, default=22594)
    parser.add_argument("--ns", type=int, default=64)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--check", action="store_true", help="compare fused vs baseline outputs")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache = make_inputs(
        args.T, args.nb, args.ns, 20261116, f"npu:{args.device}")
    cache_3d = k_cache.view(args.nb, args.ns, HEAD_DIM)
    bufs = make_buffers(args.T, k_cache.device)

    if args.check:
        check(bufs, latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache, args.eps)

    results = {}
    if args.mode in ("baseline", "both"):
        results["baseline"] = bench(
            lambda: run_baseline(bufs, latent, wk, gamma, cos, sin, indices, cache_3d, args.eps),
            args.iters)
    if args.mode in ("fused", "both"):
        results["fused"] = bench(
            lambda: run_fused(latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache, args.eps),
            args.iters)

    for name, us in results.items():
        print(f"{name}: {us:.3f} us/call (T={args.T} NB={args.nb} NS={args.ns} iters={args.iters})")
    if len(results) == 2:
        print(f"ratio baseline/fused = {results['baseline'] / results['fused']:.3f}")


if __name__ == "__main__":
    main()
