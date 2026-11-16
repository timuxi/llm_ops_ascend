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


def flash_attention_score(*args, **kwargs):
    """Flash attention score forward (BNSD or TND). Returns (attention_out, softmax_max, softmax_sum)."""
    return torch.ops.llm_ops.flash_attention_score(*args, **kwargs)


def flash_attention_score_grad(*args, **kwargs):
    """Flash attention score backward (BNSD or TND). Returns (dq, dk, dv)."""
    return torch.ops.llm_ops.flash_attention_score_grad(*args, **kwargs)


def indexer_prologue_k(*args, **kwargs):
    """Fused Indexer K prologue: projection + RMSNorm + tail RoPE + INT8 quant + paged cache scatter.

    k_cache / k_scale_cache are updated in place and returned.
    """
    return torch.ops.llm_ops.indexer_prologue_k(*args, **kwargs)


def matmul(*args, **kwargs):
    """MatMulV2 via aclnnMatmul; out is caller-preallocated and returned."""
    return torch.ops.llm_ops.matmul(*args, **kwargs)


def rms_norm(*args, **kwargs):
    """RMSNorm via aclnnRmsNorm; out/rstd are caller-preallocated. Returns (out, rstd)."""
    return torch.ops.llm_ops.rms_norm(*args, **kwargs)


def inplace_partial_rotary_mul(*args, **kwargs):
    """Interleaved partial RoPE on [start, end) of the last dim, in place."""
    return torch.ops.llm_ops.inplace_partial_rotary_mul(*args, **kwargs)


def dynamic_quant(*args, **kwargs):
    """Per-token INT8 dynamic quant via aclnnDynamicQuantV2. Returns (int8, fp32 scale)."""
    return torch.ops.llm_ops.dynamic_quant(*args, **kwargs)


def scatter_nd_update(*args, **kwargs):
    """ScatterNdUpdate: var[indices[i]] = updates[i], in place."""
    return torch.ops.llm_ops.scatter_nd_update(*args, **kwargs)


def indexer_prologue_qw(*args, **kwargs):
    """Dense Indexer Q/W prologue: weight prep + Q projection + RoPE + INT8 query quant + W scale.

    q / q_scale / w are caller-preallocated and returned.
    """
    return torch.ops.llm_ops.indexer_prologue_qw(*args, **kwargs)


def indexer_prologue_qw_prequantized(*args, **kwargs):
    """Prequantized Indexer Q/W prologue using caller-owned INT8/NZ weight + FP32 scale."""
    return torch.ops.llm_ops.indexer_prologue_qw_prequantized(*args, **kwargs)


def prepare_indexer_prologue_qw_weight(*args, **kwargs):
    """wqb -> (INT8 FRACTAL_NZ [1280,4096], FP32 [4096]) via DynamicQuant + transpose + FormatCast.

    FRACTAL_NZ is an internal format; torch_npu only creates it while
    ``torch.npu.config.allow_internal_format`` is enabled.
    """
    torch.npu.config.allow_internal_format = True
    return torch.ops.llm_ops.prepare_indexer_prologue_qw_weight(*args, **kwargs)


def permute(*args, **kwargs):
    """aclnnPermute; out is caller-preallocated and returned."""
    return torch.ops.llm_ops.permute(*args, **kwargs)


def npu_format_cast(*args, **kwargs):
    """aclnnNpuFormatCast; dst must be a FRACTAL_NZ tensor (allow_internal_format)."""
    return torch.ops.llm_ops.npu_format_cast(*args, **kwargs)


def quant_matmul_weight_nz(*args, **kwargs):
    """aclnnQuantMatmulWeightNz: INT8 activation x FRACTAL_NZ weight -> bf16."""
    return torch.ops.llm_ops.quant_matmul_weight_nz(*args, **kwargs)


__all__ = [
    "all_gather_matmul_v2",
    "block_sparse_attention_grad",
    "flash_attention_score",
    "flash_attention_score_grad",
    "indexer_prologue_k",
    "matmul",
    "rms_norm",
    "inplace_partial_rotary_mul",
    "dynamic_quant",
    "scatter_nd_update",
    "indexer_prologue_qw",
    "indexer_prologue_qw_prequantized",
    "prepare_indexer_prologue_qw_weight",
    "permute",
    "npu_format_cast",
    "quant_matmul_weight_nz",
]
