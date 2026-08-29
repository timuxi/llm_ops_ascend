#!/usr/bin/env python3
"""Smoke test for llm_ops.flash_attention_score_grad (BNSD and TND)."""

import math
import sys

import llm_ops
import torch


def _run_bnsd(device: torch.device) -> None:
    batch, num_heads = 1, 1
    seq_len, head_dim = 256, 128
    q_shape = (batch, num_heads, seq_len, head_dim)

    query = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    key = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    value = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    dy = torch.full(q_shape, 0.01, dtype=torch.float16, device=device)
    attention_in = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    softmax_max = torch.full((batch, num_heads, seq_len, 8), 3.0, dtype=torch.float32, device=device)
    softmax_sum = torch.full((batch, num_heads, seq_len, 8), 3.0, dtype=torch.float32, device=device)
    atten_mask = torch.zeros((seq_len, seq_len), dtype=torch.uint8, device=device)

    dq, dk, dv = llm_ops.flash_attention_score_grad(
        query,
        key,
        value,
        dy,
        softmax_max,
        softmax_sum,
        attention_in,
        num_heads,
        atten_mask=atten_mask,
        scale_value=1.0 / math.sqrt(head_dim),
        input_layout="BNSD",
    )
    print("BNSD dq", tuple(dq.shape), "dk", tuple(dk.shape), "dv", tuple(dv.shape))
    assert tuple(dq.shape) == q_shape
    assert tuple(dk.shape) == q_shape
    assert tuple(dv.shape) == q_shape


def _run_tnd(device: torch.device) -> None:
    num_heads = 1
    head_dim = 128
    actual_seq_qlen = [128, 256]
    actual_seq_kvlen = [128, 256]
    t_q, t_kv = actual_seq_qlen[-1], actual_seq_kvlen[-1]
    q_shape = (t_q, num_heads, head_dim)
    kv_shape = (t_kv, num_heads, head_dim)

    query = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    key = torch.full(kv_shape, 0.1, dtype=torch.float16, device=device)
    value = torch.full(kv_shape, 0.1, dtype=torch.float16, device=device)
    dy = torch.full(q_shape, 0.01, dtype=torch.float16, device=device)
    attention_in = torch.full(q_shape, 0.1, dtype=torch.float16, device=device)
    softmax_max = torch.full((t_q, num_heads, 8), 3.0, dtype=torch.float32, device=device)
    softmax_sum = torch.full((t_q, num_heads, 8), 3.0, dtype=torch.float32, device=device)

    dq, dk, dv = llm_ops.flash_attention_score_grad(
        query,
        key,
        value,
        dy,
        softmax_max,
        softmax_sum,
        attention_in,
        num_heads,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        scale_value=1.0 / math.sqrt(head_dim),
        input_layout="TND",
    )
    print("TND dq", tuple(dq.shape), "dk", tuple(dk.shape), "dv", tuple(dv.shape))
    assert tuple(dq.shape) == q_shape
    assert tuple(dk.shape) == kv_shape
    assert tuple(dv.shape) == kv_shape


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available, skip")
        return 0

    device = torch.device("npu:0")
    _run_bnsd(device)
    _run_tnd(device)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
