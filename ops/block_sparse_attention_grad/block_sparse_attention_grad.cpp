/**
 * Python/Torch wrapper for aclnnBlockSparseAttentionGrad.
 * Operator C++ is provided by ops-transformer (custom OPP) or CANN libopapi.
 */

#include <string>
#include <tuple>

#include "block_sparse_attention_grad.h"
#include "torch_aclnn_helper.h"

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> block_sparse_attention_grad(
    const at::Tensor &dout, const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &attention_out, const at::Tensor &softmax_lse, const at::Tensor &block_sparse_mask,
    at::IntArrayRef block_shape, int64_t num_key_value_heads, const c10::optional<at::Tensor> &atten_mask,
    c10::optional<at::IntArrayRef> actual_seq_lengths, c10::optional<at::IntArrayRef> actual_seq_lengths_kv,
    c10::string_view q_input_layout, c10::string_view kv_input_layout, int64_t mask_type, double scale_value,
    int64_t pre_tokens, int64_t next_tokens)
{
    TORCH_CHECK(dout.device().type() == c10::DeviceType::PrivateUse1, "Input tensor must be on NPU device");
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on NPU device");
    TORCH_CHECK(block_shape.size() >= 2, "block_shape must contain at least two elements [x, y]");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be float16 or bfloat16, got ", query.scalar_type());
    TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat, "softmax_lse must be float32, got ",
                softmax_lse.scalar_type());

    at::Tensor dq =
        at_npu::native::empty_with_format(query.sizes(), query.options(), at_npu::native::get_npu_format(query));
    at::Tensor dk =
        at_npu::native::empty_with_format(key.sizes(), key.options(), at_npu::native::get_npu_format(key));
    at::Tensor dv =
        at_npu::native::empty_with_format(value.sizes(), value.options(), at_npu::native::get_npu_format(value));

    std::string q_layout_storage(q_input_layout);
    std::string kv_layout_storage(kv_input_layout);
    char *q_layout = const_cast<char *>(q_layout_storage.c_str());
    char *kv_layout = const_cast<char *>(kv_layout_storage.c_str());

    EXEC_NPU_CMD(aclnnBlockSparseAttentionGrad, dout, query, key, value, attention_out, softmax_lse, block_sparse_mask,
                 atten_mask, block_shape, actual_seq_lengths, actual_seq_lengths_kv, q_layout, kv_layout,
                 num_key_value_heads, mask_type, scale_value, pre_tokens, next_tokens, dq, dk, dv);

    return std::make_tuple(dq, dk, dv);
}

}  // namespace llm_ops
