#!/usr/bin/env python3
"""Smoke test for llm_ops.all_gather_matmul_v2 (2-rank HCCL)."""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from torch.distributed.distributed_c10d import _get_default_group

import llm_ops


def _hccl_name(rank: int) -> str:
    pg = _get_default_group()
    backend = pg._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def _run_rank(rank: int, world_size: int, master_port: str) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )
    hcom = _hccl_name(rank)
    m, k, n = 32, 256, 128
    x1 = torch.randn(m, k, dtype=torch.float16, device=f"npu:{rank}")
    x2 = torch.randn(k, n, dtype=torch.float16, device=f"npu:{rank}")
    output, gather_out = llm_ops.all_gather_matmul_v2(x1, x2, hcom, world_size, comm_mode="aiv")
    assert tuple(output.shape) == (m * world_size, n), output.shape
    assert tuple(gather_out.shape) == (m * world_size, k), gather_out.shape
    if rank == 0:
        print("output", tuple(output.shape), "gather_out", tuple(gather_out.shape))
        print("OK")
    dist.destroy_process_group()


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available, skip")
        return 0
    if torch.npu.device_count() < 2:
        print("Need at least 2 NPUs for all_gather_matmul_v2, skip")
        return 0

    world_size = 2
    master_port = os.environ.get("MASTER_PORT", "29531")
    mp.spawn(_run_rank, args=(world_size, master_port), nprocs=world_size, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
