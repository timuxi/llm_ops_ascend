/**
 * Python/Torch wrapper for flash attention score backward.
 * BNSD/BSND/BSH/SBH -> aclnnFlashAttentionScoreGrad
 * TND (varlen)      -> aclnnFlashAttentionUnpaddingScoreGrad
 * Operator C++ is provided by ops-transformer (custom OPP) or CANN libopapi.
 */

#include <string>
#include <tuple>

#include "flash_attention_score_grad.h"
#include "torch_aclnn_helper.h"

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_score_grad(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dy,
    const at::Tensor &softmax_max, const at::Tensor &softmax_sum, const at::Tensor &attention_in, int64_t head_num,
    const c10::optional<at::Tensor> &pse_shift, const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_in, c10::optional<at::IntArrayRef> prefix,
    c10::optional<at::IntArrayRef> actual_seq_qlen, c10::optional<at::IntArrayRef> actual_seq_kvlen, double scale_value,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t inner_precise,
    int64_t sparse_mode)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on NPU device");
    TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1, "key must be on NPU device");
    TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1, "value must be on NPU device");
    TORCH_CHECK(dy.device().type() == c10::DeviceType::PrivateUse1, "dy must be on NPU device");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be float16 or bfloat16, got ", query.scalar_type());
    TORCH_CHECK(softmax_max.scalar_type() == at::kFloat, "softmax_max must be float32, got ",
                softmax_max.scalar_type());
    TORCH_CHECK(softmax_sum.scalar_type() == at::kFloat, "softmax_sum must be float32, got ",
                softmax_sum.scalar_type());
    TORCH_CHECK(head_num > 0, "head_num must be greater than 0, got ", head_num);

    const bool is_tnd = (input_layout == "TND");
    if (is_tnd) {
        TORCH_CHECK(query.dim() == 3, "TND query must be 3-D [T, N, D], got dim ", query.dim());
        TORCH_CHECK(actual_seq_qlen.has_value() && !actual_seq_qlen->empty(),
                    "actual_seq_qlen is required when input_layout is TND");
        TORCH_CHECK(actual_seq_kvlen.has_value() && !actual_seq_kvlen->empty(),
                    "actual_seq_kvlen is required when input_layout is TND");
    }

    at::Tensor dq =
        at_npu::native::empty_with_format(query.sizes(), query.options(), at_npu::native::get_npu_format(query));
    at::Tensor dk =
        at_npu::native::empty_with_format(key.sizes(), key.options(), at_npu::native::get_npu_format(key));
    at::Tensor dv =
        at_npu::native::empty_with_format(value.sizes(), value.options(), at_npu::native::get_npu_format(value));

    c10::optional<at::Tensor> dpse;
    if (pse_shift.has_value() && pse_shift.value().defined() && pse_shift.value().numel() > 0) {
        const at::Tensor &pse = pse_shift.value();
        dpse = at_npu::native::empty_with_format(pse.sizes(), pse.options(), at_npu::native::get_npu_format(pse));
    }

    std::string layout_storage(input_layout);
    char *layout = const_cast<char *>(layout_storage.c_str());

    if (is_tnd) {
        EXEC_NPU_CMD(aclnnFlashAttentionUnpaddingScoreGrad, query, key, value, dy, pse_shift, drop_mask, padding_mask,
                     atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, prefix, actual_seq_qlen,
                     actual_seq_kvlen, scale_value, keep_prob, pre_tokens, next_tokens, head_num, layout,
                     inner_precise, sparse_mode, dq, dk, dv, dpse);
    } else {
        EXEC_NPU_CMD(aclnnFlashAttentionScoreGrad, query, key, value, dy, pse_shift, drop_mask, padding_mask,
                     atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, prefix, scale_value, keep_prob,
                     pre_tokens, next_tokens, head_num, layout, inner_precise, sparse_mode, dq, dk, dv, dpse);
    }

    return std::make_tuple(dq, dk, dv);
}

}  // namespace llm_ops
