#ifndef LLM_OPS_FLASH_ATTENTION_SCORE_GRAD_H
#define LLM_OPS_FLASH_ATTENTION_SCORE_GRAD_H

#include <tuple>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_score_grad(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dy,
    const at::Tensor &softmax_max, const at::Tensor &softmax_sum, const at::Tensor &attention_in, int64_t head_num,
    const c10::optional<at::Tensor> &pse_shift, const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_in, c10::optional<at::IntArrayRef> prefix,
    c10::optional<at::IntArrayRef> actual_seq_qlen, c10::optional<at::IntArrayRef> actual_seq_kvlen, double scale_value,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t inner_precise,
    int64_t sparse_mode);

}  // namespace llm_ops

#endif  // LLM_OPS_FLASH_ATTENTION_SCORE_GRAD_H
