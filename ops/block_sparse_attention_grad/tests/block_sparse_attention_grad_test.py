#!/usr/bin/env python3
"""Smoke test for llm_ops.block_sparse_attention_grad (BNSD)."""

import math
import sys

import llm_ops
import torch


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available, skip")
        return 0

    device = torch.device("npu:0")
    batch, num_heads, num_kv_heads = 1, 1, 1
    q_seqlen, kv_seqlen, head_dim = 128, 128, 128
    bx, by = 64, 64
    ceil_q = (q_seqlen + bx - 1) // bx
    ceil_kv = (kv_seqlen + by - 1) // by

    q_shape = (batch, num_heads, q_seqlen, head_dim)
    kv_shape = (batch, num_kv_heads, kv_seqlen, head_dim)

    query = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    key = torch.full(kv_shape, 0.1, dtype=torch.float16, device=device)
    value = torch.full(kv_shape, 0.1, dtype=torch.float16, device=device)
    dout = torch.full(q_shape, 0.01, dtype=torch.float16, device=device)
    attention_out = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    softmax_lse = torch.full((batch, num_heads, q_seqlen), 5.0, dtype=torch.float32, device=device)
    mask = torch.ones((batch, num_heads, ceil_q, ceil_kv), dtype=torch.uint8, device=device)

    dq, dk, dv = llm_ops.block_sparse_attention_grad(
        dout,
        query,
        key,
        value,
        attention_out,
        softmax_lse,
        mask,
        [bx, by],
        num_kv_heads,
        actual_seq_lengths=[q_seqlen],
        actual_seq_lengths_kv=[kv_seqlen],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        scale_value=1.0 / math.sqrt(head_dim),
    )
    print("dq", tuple(dq.shape), "dk", tuple(dk.shape), "dv", tuple(dv.shape))
    assert tuple(dq.shape) == q_shape
    assert tuple(dk.shape) == kv_shape
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
