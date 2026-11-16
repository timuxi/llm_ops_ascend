#!/usr/bin/env python3
"""IndexerPrologueQW 精度 / 性能测试入口（baseline vs fused）。

baseline（7 个独立算子，逐步调用）:
    activation DynamicQuant -> weight DynamicQuant -> Permute -> NpuFormatCast
    -> QuantMatmulWeightNz -> W MatMul -> Post(torch 参考实现，融合核无公开 aclnn)

fused:
    dense:       llm_ops.indexer_prologue_qw              (7 kernels，含 weight 预处理)
    prequantized: llm_ops.indexer_prologue_qw_prequantized (4 kernels，weight 由调用方持有)

默认只跑 kernel（配合 msprof 做算子级对比，不输出 wall/精度）：

    msprof --output=prof_base  python3 perf_indexer_prologue_qw.py --mode kernel-baseline
    msprof --output=prof_dense python3 perf_indexer_prologue_qw.py --mode kernel-dense
    msprof --output=prof_pre   python3 perf_indexer_prologue_qw.py --mode kernel-prequant

精度判定（仅 --mode precision）：
  - baseline / dense 各自与 torch_npu W8A8 golden 比：q max|diff| <= 1，q_scale/w 逐位
  - baseline vs dense vs prequant 三者逐位一致（同一数值路径）

kernel 时间统一用裸 msprof 采 op_summary；`--mode wall` 只保留给需要 wall 的场合。

依赖：
  - ops-transformer 已编包并安装 indexer_prologue_qw（custom_transformer vendor）
  - llm_ops 可导入（llm_ops_ascend ./build.sh + pip install，或 PYTHONPATH=<repo>/utils/python）
  - PYTHONPATH 要保留 CANN 的 site-packages（torch_npu 的 npu_dynamic_quant 需要 tbe）
"""

import argparse
import time

import torch
import torch_npu  # noqa: F401

try:
    import llm_ops
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "llm_ops 不可导入：先在 llm_ops_ascend 仓执行 ./build.sh 并安装 "
        "output/llm_ops_ascend-*.whl（或 PYTHONPATH=<repo>/utils/python），"
        "并确认 LD_LIBRARY_PATH 含 ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib"
    ) from exc

N_HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64
DIM_X = 5120
DIM_QR = 1280
WQB_N = N_HEADS * HEAD_DIM
DEFAULT_SCALE = HEAD_DIM**-0.5


def make_inputs(tokens, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qr = torch.randn(tokens, DIM_QR, generator=g).to(torch.bfloat16).to(device)
    wqb = (torch.randn(WQB_N, DIM_QR, generator=g) / DIM_QR**0.5).to(torch.bfloat16).to(device)
    x = torch.randn(tokens, DIM_X, generator=g).to(torch.bfloat16).to(device)
    ww = (torch.randn(N_HEADS, DIM_X, generator=g) / DIM_X**0.5).to(torch.bfloat16).to(device)
    ang = torch.rand(tokens, ROPE_DIM, generator=g) * 6.2831853
    sin = torch.sin(ang).contiguous().to(device)
    cos = torch.cos(ang).contiguous().to(device)
    return x, qr, wqb, ww, sin, cos


def make_outputs(tokens, device):
    return (
        torch.empty(tokens, N_HEADS, HEAD_DIM, dtype=torch.int8, device=device),
        torch.empty(tokens, N_HEADS, dtype=torch.float16, device=device),
        torch.empty(tokens, N_HEADS, dtype=torch.float16, device=device),
    )


def make_baseline_buffers(tokens, device):
    return {
        "act_q": torch.empty(tokens, DIM_QR, dtype=torch.int8, device=device),
        "act_s": torch.empty(tokens, dtype=torch.float32, device=device),
        "w_q": torch.empty(WQB_N, DIM_QR, dtype=torch.int8, device=device),
        "w_s": torch.empty(WQB_N, dtype=torch.float32, device=device),
        "w_t": torch.empty(DIM_QR, WQB_N, dtype=torch.int8, device=device),
        # FRACTAL_NZ 内部格式：需要 torch.npu.config.allow_internal_format = True
        "w_nz": torch_npu.empty_with_format((DIM_QR, WQB_N), dtype=torch.int8, device=device, acl_format=29),
        "qb": torch.empty(tokens, WQB_N, dtype=torch.bfloat16, device=device),
        "wb": torch.empty(tokens, N_HEADS, dtype=torch.bfloat16, device=device),
    }


def post_reference(qb, wb, sin, cos, softmax_scale):
    """Post 阶段的 torch 参考：tail interleave RoPE + per-(token,head) INT8 quant + W scale。"""
    tokens = qb.shape[0]
    q = qb.view(tokens, N_HEADS, HEAD_DIM)
    tail = q[..., -ROPE_DIM:].float()
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rot = torch.empty_like(tail)
    rot[..., 0::2] = even * cos[:, None, 0::2] - odd * sin[:, None, 0::2]
    rot[..., 1::2] = odd * cos[:, None, 1::2] + even * sin[:, None, 1::2]
    q = torch.cat([q[..., :-ROPE_DIM], rot.to(torch.bfloat16)], dim=-1)
    amax = q.float().abs().amax(dim=-1, keepdim=True)
    scale16 = (amax / 127.0).clamp_min(2**-24).to(torch.float16)
    q8 = torch.round(q.float() / scale16.float()).clamp_(-127, 127).to(torch.int8)
    w = (wb.float() * (softmax_scale * N_HEADS**-0.5)).to(torch.float16)
    return q8, scale16.squeeze(-1), w


def run_baseline(x, qr, wqb, ww, sin, cos, scale, bufs):
    """7 个独立算子：6 个官方 aclnn + Post 的 torch 参考实现。"""
    act_q, act_s = llm_ops.dynamic_quant(qr, bufs["act_q"], bufs["act_s"])
    w_q, w_s = llm_ops.dynamic_quant(wqb, bufs["w_q"], bufs["w_s"])
    w_t = llm_ops.permute(w_q, [1, 0], bufs["w_t"])
    w_nz = llm_ops.npu_format_cast(w_t, bufs["w_nz"])
    qb = llm_ops.quant_matmul_weight_nz(act_q, w_nz, act_s, w_s, bufs["qb"])
    wb = llm_ops.matmul(x, ww, bufs["wb"])
    return post_reference(qb, wb, sin, cos, scale)


def golden(x, qr, wqb, ww, sin, cos, softmax_scale):
    """Production W8A8 reference：全 torch_npu 算子。"""
    wq8, wscale = torch_npu.npu_dynamic_quant(wqb, dst_type=torch.int8)
    wnz = torch_npu.npu_format_cast(wq8.t().contiguous(), 29)
    aq, ascale = torch_npu.npu_dynamic_quant(qr, dst_type=torch.int8)
    proj = torch_npu.npu_quant_matmul(aq, wnz, wscale, pertoken_scale=ascale, output_dtype=torch.bfloat16)
    wb = torch.matmul(x, ww.t())
    return post_reference(proj, wb, sin, cos, softmax_scale)


def diff_report(name, got, want, int8_data):
    if int8_data:
        d = (got.to(torch.int16) - want.to(torch.int16)).abs()
        print(f"{name}: max|diff|={int(d.max())} count(>1)={int((d > 1).sum())} elems={d.numel()}")
        return
    eq = torch.equal(got, want)
    tail = "" if eq else f" max|diff|={float((got.float() - want.float()).abs().max())}"
    print(f"{name}: bitwise_equal={eq}{tail}")


def run_dense(x, qr, wqb, ww, sin, cos, scale, q, qs, w):
    llm_ops.indexer_prologue_qw(x, qr, wqb, ww, sin, cos, scale, q, qs, w)


def run_prequant(x, qr, wnz, wscale, ww, sin, cos, scale, q, qs, w):
    llm_ops.indexer_prologue_qw_prequantized(x, qr, wnz, wscale, ww, sin, cos, scale, q, qs, w)


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["kernel", "kernel-baseline", "kernel-dense", "kernel-prequant",
                             "precision", "wall"],
                    default="kernel")
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--softmax-scale", type=float, default=DEFAULT_SCALE)
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True  # INT8/NZ weight
    dev = f"npu:{args.device}"
    x, qr, wqb, ww, sin, cos = make_inputs(args.T, 20261116, dev)
    q, qs, w = make_outputs(args.T, dev)
    q2, qs2, w2 = make_outputs(args.T, dev)
    bufs = make_baseline_buffers(args.T, dev)

    if args.mode in ("precision", "both"):
        gq, gqs, gw = golden(x, qr, wqb, ww, sin, cos, args.softmax_scale)
        torch.npu.synchronize()

        bq, bqs, bw = run_baseline(x, qr, wqb, ww, sin, cos, args.softmax_scale, bufs)
        run_dense(x, qr, wqb, ww, sin, cos, args.softmax_scale, q, qs, w)
        wnz, wscale = llm_ops.prepare_indexer_prologue_qw_weight(wqb)
        run_prequant(x, qr, wnz, wscale, ww, sin, cos, args.softmax_scale, q2, qs2, w2)
        torch.npu.synchronize()

        print(f"== precision: T={args.T} softmax_scale={args.softmax_scale}")
        print("-- baseline(7 ops) vs torch golden")
        diff_report("q      ", bq, gq, True)
        diff_report("q_scale", bqs, gqs.to(torch.float16), False)
        diff_report("w      ", bw, gw, False)
        print("-- fused dense vs torch golden")
        diff_report("q      ", q, gq, True)
        diff_report("q_scale", qs, gqs.to(torch.float16), False)
        diff_report("w      ", w, gw, False)
        print("-- baseline vs dense")
        diff_report("q      ", bq, q, True)
        diff_report("q_scale", bqs, qs, False)
        diff_report("w      ", bw, w, False)
        print("-- dense vs prequantized")
        diff_report("q      ", q2, q, True)
        diff_report("q_scale", qs2, qs, False)
        diff_report("w      ", w2, w, False)

    if args.mode.startswith("kernel") or args.mode == "wall":
        wnz, wscale = llm_ops.prepare_indexer_prologue_qw_weight(wqb)
        torch.npu.synchronize()

        calls = []
        if args.mode in ("kernel", "kernel-baseline", "wall"):
            calls.append(("baseline", lambda: run_baseline(x, qr, wqb, ww, sin, cos,
                                                           args.softmax_scale, bufs)))
        if args.mode in ("kernel", "kernel-dense", "wall"):
            calls.append(("dense", lambda: run_dense(x, qr, wqb, ww, sin, cos,
                                                     args.softmax_scale, q, qs, w)))
        if args.mode in ("kernel", "kernel-prequant", "wall"):
            calls.append(("prequant", lambda: run_prequant(x, qr, wnz, wscale, ww, sin, cos,
                                                           args.softmax_scale, q2, qs2, w2)))

        for name, fn in calls:
            if args.mode == "wall":
                us = bench(fn, args.iters)
                print(f"{name}: {us:.3f} us/call")
            else:
                for _ in range(3):
                    fn()
                for _ in range(args.iters):
                    fn()
                torch.npu.synchronize()
                print(f"{name}: {args.iters} iterations done (msprof only)")


if __name__ == "__main__":
    main()
