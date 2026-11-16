/**
 * Python/Torch wrappers for the IndexerPrologueK production chain.
 *
 * indexer_prologue_k: fused custom operator from ops-transformer
 *   (MatMul + official RMSNorm + fused Post: tail RoPE / INT8 quant / cache scatter).
 *
 * matmul / rms_norm / inplace_partial_rotary_mul / dynamic_quant / scatter_nd_update:
 *   official CANN aclnn operators, used to rebuild the unfused production baseline
 *   for accuracy and performance comparison against the fused entry.
 *
 * The baseline wrappers take preallocated output tensors so a timed loop does not
 * include torch allocator traffic; the fused entry updates its caches in place.
 */

#ifndef LLM_OPS_INDEXER_PROLOGUE_K_H
#define LLM_OPS_INDEXER_PROLOGUE_K_H

#include <ATen/Tensor.h>

#include <tuple>

namespace llm_ops {

// fused IndexerPrologueK: k_cache / k_scale_cache are updated in place
at::Tensor indexer_prologue_k(const at::Tensor &latent, const at::Tensor &wk, const at::Tensor &norm_weight,
                              const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &indices,
                              const at::Tensor &k_cache, const at::Tensor &k_scale_cache, double norm_eps);

// baseline step 1: out[T,D] = x[T,H] @ weight[D,H]^T  (aclnnMatmul / MatMulV2)
at::Tensor matmul(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &out);

// baseline step 2: RMSNorm over the last dim (aclnnRmsNorm)
std::tuple<at::Tensor, at::Tensor> rms_norm(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &out,
                                            const at::Tensor &rstd, double eps);

// baseline step 3: interleaved partial RoPE on [start, end) of the last dim,
// applied in place (aclnnInplacePartialRotaryMul, rotary_mode = 1)
at::Tensor inplace_partial_rotary_mul(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin,
                                      int64_t start, int64_t end);

// baseline step 4: per-token INT8 dynamic quant + fp32 scale (aclnnDynamicQuantV2)
std::tuple<at::Tensor, at::Tensor> dynamic_quant(const at::Tensor &x, const at::Tensor &out,
                                                 const at::Tensor &scale);

// baseline step 5: var[indices[i]] = updates[i] (aclnnScatterNdUpdate)
at::Tensor scatter_nd_update(const at::Tensor &var, const at::Tensor &indices, const at::Tensor &updates);

}  // namespace llm_ops

#endif  // LLM_OPS_INDEXER_PROLOGUE_K_H
