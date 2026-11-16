/**
 * Python/Torch wrappers for IndexerPrologueQW.
 *
 * indexer_prologue_qw              : dense entry (7 kernels, weight prep per call)
 * indexer_prologue_qw_prequantized : caller-owned INT8/NZ weight + FP32 scale (4 kernels)
 * prepare_indexer_prologue_qw_weight: wqb -> (INT8 FRACTAL_NZ [1280,4096], FP32 [4096])
 *
 * The two fused entries need aclnn symbols from libcust_opapi.so, which this repo
 * does not link directly; they are resolved through dlsym like the shared helper.
 */

#ifndef LLM_OPS_INDEXER_PROLOGUE_QW_H
#define LLM_OPS_INDEXER_PROLOGUE_QW_H

#include <ATen/Tensor.h>

#include <tuple>

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> indexer_prologue_qw(
    const at::Tensor &x, const at::Tensor &qr, const at::Tensor &wqb, const at::Tensor &ww,
    const at::Tensor &rope_sin, const at::Tensor &rope_cos, double softmax_scale,
    const at::Tensor &q, const at::Tensor &q_scale, const at::Tensor &w);

std::tuple<at::Tensor, at::Tensor, at::Tensor> indexer_prologue_qw_prequantized(
    const at::Tensor &x, const at::Tensor &qr, const at::Tensor &wqb_nz, const at::Tensor &wqb_scale,
    const at::Tensor &ww, const at::Tensor &rope_sin, const at::Tensor &rope_cos, double softmax_scale,
    const at::Tensor &q, const at::Tensor &q_scale, const at::Tensor &w);

// Production weight preparation: DynamicQuant -> transpose/contiguous -> NpuFormatCast(29).
std::tuple<at::Tensor, at::Tensor> prepare_indexer_prologue_qw_weight(const at::Tensor &wqb);

// Standalone baseline operators (the unfused 7-op chain).
at::Tensor permute(const at::Tensor &self, at::IntArrayRef dims, const at::Tensor &out);
at::Tensor npu_format_cast(const at::Tensor &src, const at::Tensor &dst_nz);
at::Tensor quant_matmul_weight_nz(const at::Tensor &x1, const at::Tensor &x2_nz, const at::Tensor &x1_scale,
                                  const at::Tensor &x2_scale, const at::Tensor &out);

}  // namespace llm_ops

#endif  // LLM_OPS_INDEXER_PROLOGUE_QW_H
