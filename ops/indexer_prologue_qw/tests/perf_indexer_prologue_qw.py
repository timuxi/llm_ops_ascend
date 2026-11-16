#!/usr/bin/env python3
"""IndexerPrologueQW 精度 / 性能测试入口（dense vs prequantized）。

dense:       llm_ops.indexer_prologue_qw              (7 个 kernel，每次调用都做 weight 预处理)
prequantized: llm_ops.indexer_prologue_qw_prequantized (4 个 kernel，weight 由调用方持有)

精度判定（--mode precision / both）：
  - q           : dense/prequant 的 INT8 结果与 torch_npu golden 比，要求 max|diff| <= 1
  - q_scale     : 与 fp16(golden fp32 scale) 逐位比
  - w           : 与 golden fp16 逐位比
  - dense vs prequant：三个输出必须逐位相同（同一数值路径，只差 weight 预处理时机）

性能（--mode perf / both）：每次调用后 torch.npu.synchronize()，输出 us/call。
  kernel 时间用裸 msprof 采 op_summary，见 indexer_prologue_qw.md。

依赖：
  - ops-transformer 已编包并安装 indexer_prologue_qw（custom_transformer vendor）
  - llm_ops 可导入（llm_ops_ascend ./build.sh + pip install，或 PYTHONPATH=<repo>/utils/python）
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
DEFAULT_SCALE = HEAD_DIM**-0.5


def make_inputs(tokens, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qr = torch.randn(tokens, DIM_QR, generator=g).to(torch.bfloat16).to(device)
    wqb = (torch.randn(N_HEADS * HEAD_DIM, DIM_QR, generator=g) / DIM_QR**0.5).to(torch.bfloat16).to(device)
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


def golden(x, qr, wqb, ww, sin, cos, softmax_scale):
    """Production W8A8 reference: real DynamicQuant + INT8/NZ QuantBatchMatmul."""
    tokens = qr.shape[0]
    wq8, wscale = torch_npu.npu_dynamic_quant(wqb, dst_type=torch.int8)
    wnz = torch_npu.npu_format_cast(wq8.t().contiguous(), 29)
    aq, ascale = torch_npu.npu_dynamic_quant(qr, dst_type=torch.int8)
    proj = torch_npu.npu_quant_matmul(aq, wnz, wscale, pertoken_scale=ascale, output_dtype=torch.bfloat16)
    q = proj.view(tokens, N_HEADS, HEAD_DIM)
    tail = q[..., -ROPE_DIM:].float()
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rot = torch.empty_like(tail)
    rot[..., 0::2] = even * cos[:, None, 0::2] - odd * sin[:, None, 0::2]
    rot[..., 1::2] = odd * cos[:, None, 1::2] + even * sin[:, None, 1::2]
    q = torch.cat([q[..., :-ROPE_DIM], rot.to(torch.bfloat16)], dim=-1)  # [T,32,128]
    amax = q.float().abs().amax(dim=-1, keepdim=True)
    scale16 = (amax / 127.0).clamp_min(2**-24).to(torch.float16)
    q8 = torch.round(q.float() / scale16.float()).clamp_(-127, 127).to(torch.int8)
    w_scale = softmax_scale * N_HEADS**-0.5
    w = (torch.matmul(x, ww.t()).float() * w_scale).to(torch.float16)
    return q8, scale16.squeeze(-1), w, wnz, wscale


def diff_report(name, got, want, int8_data):
    if int8_data:
        d = (got.to(torch.int16) - want.to(torch.int16)).abs()
        print(f"{name}: max|diff|={int(d.max())} count(>1)={int((d > 1).sum())} elems={d.numel()}")
        return int(d.max())
    eq = torch.equal(got, want)
    if not eq:
        d = (got.float() - want.float()).abs()
        print(f"{name}: bitwise_equal=False max|diff|={float(d.max())}")
    else:
        print(f"{name}: bitwise_equal=True")
    return 0 if eq else 1


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
    ap.add_argument("--mode", choices=["precision", "perf", "perf-dense", "perf-prequant", "both"],
                    default="precision")
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--softmax-scale", type=float, default=DEFAULT_SCALE)
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True  # needed for the INT8/NZ weight
    dev = f"npu:{args.device}"
    x, qr, wqb, ww, sin, cos = make_inputs(args.T, 20261116, dev)
    q, qs, w = make_outputs(args.T, dev)
    q2, qs2, w2 = make_outputs(args.T, dev)

    if args.mode in ("precision", "both"):
        gq, gqs, gw, gwnz, gwscale = golden(x, qr, wqb, ww, sin, cos, args.softmax_scale)
        torch.npu.synchronize()

        run_dense(x, qr, wqb, ww, sin, cos, args.softmax_scale, q, qs, w)
        torch.npu.synchronize()
        print(f"== precision: T={args.T} softmax_scale={args.softmax_scale}")
        print("-- dense vs torch golden")
        diff_report("q      ", q, gq, True)
        diff_report("q_scale", qs, gqs.to(torch.float16), False)
        diff_report("w      ", w, gw, False)

        wnz, wscale = llm_ops.prepare_indexer_prologue_qw_weight(wqb)
        torch.npu.synchronize()
        print("-- weight prep vs torch golden")
        print("wqb_nz bitwise_equal:", torch.equal(wnz, gwnz))
        print("wqb_scale bitwise_equal:", torch.equal(wscale, gwscale))

        run_prequant(x, qr, wnz, wscale, ww, sin, cos, args.softmax_scale, q2, qs2, w2)
        torch.npu.synchronize()
        print("-- dense vs prequantized (must be bitwise equal)")
        diff_report("q      ", q2, q, True)
        diff_report("q_scale", qs2, qs, False)
        diff_report("w      ", w2, w, False)

    if args.mode in ("perf", "perf-dense", "perf-prequant", "both"):
        wnz, wscale = llm_ops.prepare_indexer_prologue_qw_weight(wqb)
        torch.npu.synchronize()

        dense = pre = None
        if args.mode in ("perf", "perf-dense", "both"):
            dense = bench(lambda: run_dense(x, qr, wqb, ww, sin, cos, args.softmax_scale, q, qs, w),
                          args.iters)
            print(f"dense:       {dense:.3f} us/call (iters={args.iters})")
        if args.mode in ("perf", "perf-prequant", "both"):
            pre = bench(lambda: run_prequant(x, qr, wnz, wscale, ww, sin, cos, args.softmax_scale,
                                             q2, qs2, w2), args.iters)
            print(f"prequant:    {pre:.3f} us/call (iters={args.iters})")
        if dense is not None and pre is not None:
            print(f"wall ratio dense/prequant = {dense / pre:.3f}")


if __name__ == "__main__":
    main()
