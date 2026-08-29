/**
 * Python/Torch wrapper for flash attention score forward.
 * BNSD/BSND/BSH/SBH -> aclnnFlashAttentionScore
 * TND (varlen)      -> aclnnFlashAttentionVarLenScore
 * Operator C++ is provided by ops-transformer (custom OPP) or CANN libopapi.
 */

#include <string>
#include <tuple>
#include <vector>

#include "flash_attention_score.h"
#include "torch_aclnn_helper.h"

namespace llm_ops {
namespace {

std::vector<int64_t> SoftmaxMaxShape(const at::Tensor &query, int64_t head_num, c10::string_view layout)
{
    if (layout == "TND") {
        // [T, N, 8]
        return {query.size(0), head_num, 8};
    }
    int64_t b = 0;
    int64_t s = 0;
    if (layout == "BNSD") {
        b = query.size(0);
        s = query.size(2);
    } else if (layout == "BSND") {
        b = query.size(0);
        s = query.size(1);
    } else if (layout == "BSH") {
        b = query.size(0);
        s = query.size(1);
    } else if (layout == "SBH") {
        s = query.size(0);
        b = query.size(1);
    } else {
        TORCH_CHECK(false, "unsupported input_layout '", layout, "', expect BNSD/BSND/BSH/SBH/TND");
    }
    // [B, N, S, 8]
    return {b, head_num, s, 8};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, int64_t head_num,
    const c10::optional<at::Tensor> &pse_shift, const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    c10::optional<at::IntArrayRef> prefix, c10::optional<at::IntArrayRef> actual_seq_qlen,
    c10::optional<at::IntArrayRef> actual_seq_kvlen, double scale_value, double keep_prob, int64_t pre_tokens,
    int64_t next_tokens, c10::string_view input_layout, int64_t inner_precise, int64_t sparse_mode)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on NPU device");
    TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1, "key must be on NPU device");
    TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1, "value must be on NPU device");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 ||
                    query.scalar_type() == at::kFloat,
                "query must be float16, bfloat16 or float32, got ", query.scalar_type());
    TORCH_CHECK(head_num > 0, "head_num must be greater than 0, got ", head_num);

    const bool is_tnd = (input_layout == "TND");
    if (is_tnd) {
        TORCH_CHECK(query.dim() == 3, "TND query must be 3-D [T, N, D], got dim ", query.dim());
        TORCH_CHECK(actual_seq_qlen.has_value() && !actual_seq_qlen->empty(),
                    "actual_seq_qlen is required when input_layout is TND");
        TORCH_CHECK(actual_seq_kvlen.has_value() && !actual_seq_kvlen->empty(),
                    "actual_seq_kvlen is required when input_layout is TND");
    }

    at::Tensor attention_out =
        at_npu::native::empty_with_format(query.sizes(), query.options(), at_npu::native::get_npu_format(query));

    const auto softmax_sizes = SoftmaxMaxShape(query, head_num, input_layout);
    auto softmax_opts = query.options().dtype(at::kFloat);
    at::Tensor softmax_max =
        at_npu::native::empty_with_format(softmax_sizes, softmax_opts, at_npu::native::get_npu_format(query));
    at::Tensor softmax_sum =
        at_npu::native::empty_with_format(softmax_sizes, softmax_opts, at_npu::native::get_npu_format(query));

    // softmaxOut is reserved / unused; pass null optional.
    c10::optional<at::Tensor> softmax_out;

    std::string layout_storage(input_layout);
    char *layout = const_cast<char *>(layout_storage.c_str());

    if (is_tnd) {
        EXEC_NPU_CMD(aclnnFlashAttentionVarLenScore, query, key, value, pse_shift, drop_mask, padding_mask,
                     atten_mask, prefix, actual_seq_qlen, actual_seq_kvlen, scale_value, keep_prob, pre_tokens,
                     next_tokens, head_num, layout, inner_precise, sparse_mode, softmax_max, softmax_sum, softmax_out,
                     attention_out);
    } else {
        EXEC_NPU_CMD(aclnnFlashAttentionScore, query, key, value, pse_shift, drop_mask, padding_mask, atten_mask,
                     prefix, scale_value, keep_prob, pre_tokens, next_tokens, head_num, layout, inner_precise,
                     sparse_mode, softmax_max, softmax_sum, softmax_out, attention_out);
    }

    return std::make_tuple(attention_out, softmax_max, softmax_sum);
}

}  // namespace llm_ops
