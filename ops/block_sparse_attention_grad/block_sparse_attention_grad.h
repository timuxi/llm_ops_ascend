#ifndef LLM_OPS_BLOCK_SPARSE_ATTENTION_GRAD_H
#define LLM_OPS_BLOCK_SPARSE_ATTENTION_GRAD_H

#include <tuple>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> block_sparse_attention_grad(
    const at::Tensor &dout, const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &attention_out, const at::Tensor &softmax_lse, const at::Tensor &block_sparse_mask,
    at::IntArrayRef block_shape, int64_t num_key_value_heads, const c10::optional<at::Tensor> &atten_mask,
    c10::optional<at::IntArrayRef> actual_seq_lengths, c10::optional<at::IntArrayRef> actual_seq_lengths_kv,
    c10::string_view q_input_layout, c10::string_view kv_input_layout, int64_t mask_type, double scale_value,
    int64_t pre_tokens, int64_t next_tokens);

}  // namespace llm_ops

#endif  // LLM_OPS_BLOCK_SPARSE_ATTENTION_GRAD_H
