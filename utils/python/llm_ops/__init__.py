"""llm_ops: Python APIs for Ascend LLM operators (ACLNN via EXEC_NPU_CMD)."""

from __future__ import annotations

import pathlib

import torch
import torch_npu  # noqa: F401  # registers PrivateUse1 / npu backend


def _load_native_library() -> None:
    pkg_dir = pathlib.Path(__file__).resolve().parent
    so_path = pkg_dir / "lib" / "libllm_ops.so"
    if not so_path.is_file():
        raise ImportError(
            f"Native library not found: {so_path}. "
            "Build with ./build.sh and install the generated wheel."
        )
    torch.ops.load_library(str(so_path))


_load_native_library()


def all_gather_matmul_v2(*args, **kwargs):
    """All-gather + matmul fusion (MC2). Returns (output, gather_out)."""
    return torch.ops.llm_ops.all_gather_matmul_v2(*args, **kwargs)


def block_sparse_attention_grad(*args, **kwargs):
    """Block sparse attention backward. Returns (dq, dk, dv)."""
    return torch.ops.llm_ops.block_sparse_attention_grad(*args, **kwargs)


def flash_attention_score_grad(*args, **kwargs):
    """Flash attention score backward (BNSD or TND). Returns (dq, dk, dv)."""
    return torch.ops.llm_ops.flash_attention_score_grad(*args, **kwargs)


__all__ = [
    "all_gather_matmul_v2",
    "block_sparse_attention_grad",
    "flash_attention_score_grad",
]
