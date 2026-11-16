/**
 * Python/Torch wrappers for the five official CANN operators used by the
 * IndexerPrologueK production baseline.
 *
 * They exist so the unfused reference chain can be rebuilt in Python:
 *
 *   projected = matmul(latent, wk, out)
 *   key, rstd = rms_norm(projected, gamma, out, rstd, eps)
 *   inplace_partial_rotary_mul(key_4d, cos, sin, 64, 128)
 *   quant, qscale = dynamic_quant(key, out, scale)
 *   scatter_nd_update(cache_3d, indices, quant)
 *
 * Outputs are caller-preallocated (and returned for convenience) so a timed loop
 * does not pay torch allocator traffic. Operator C++ comes from libopapi.so.
 */

#include "indexer_prologue_k.h"

#include "torch_aclnn_helper.h"

namespace llm_ops {

at::Tensor matmul(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &out)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU device");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU device");
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1, "out must be on NPU device");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && out.dim() == 2, "matmul expects 2-D tensors");
    TORCH_CHECK(x.size(1) == weight.size(1), "matmul: x.size(1) must equal weight.size(1)");
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == weight.size(0),
                "matmul: out must be [x.size(0), weight.size(0)]");

    // weight is [N, K]; aclnnMatmul takes [K, N], so feed a strided view (no copy).
    const at::Tensor weight_t = weight.transpose(0, 1);
    const int8_t cube_math_type = 0;  // default cube math type
    EXEC_NPU_CMD(aclnnMatmul, x, weight_t, out, cube_math_type);
    return out;
}

std::tuple<at::Tensor, at::Tensor> rms_norm(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &out,
                                            const at::Tensor &rstd, double eps)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU device");
    TORCH_CHECK(gamma.device().type() == c10::DeviceType::PrivateUse1, "gamma must be on NPU device");
    TORCH_CHECK(out.sizes() == x.sizes(), "rms_norm: out must have the same shape as x");
    TORCH_CHECK(rstd.scalar_type() == at::kFloat && rstd.numel() == x.size(0),
                "rms_norm: rstd must be float32 with one element per row");

    EXEC_NPU_CMD(aclnnRmsNorm, x, gamma, eps, out, rstd);
    return std::make_tuple(out, rstd);
}

at::Tensor inplace_partial_rotary_mul(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin,
                                      int64_t start, int64_t end)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU device");
    TORCH_CHECK(cos.scalar_type() == x.scalar_type() || cos.scalar_type() == at::kFloat,
                "inplace_partial_rotary_mul: cos/sin must be fp32 or match x");
    TORCH_CHECK(start >= 0 && end > start && end <= x.size(-1), "inplace_partial_rotary_mul: bad [start, end)");

    int64_t slice[2] = {start, end};
    const at::IntArrayRef partial(slice, 2);
    const int64_t rotary_mode = 1;  // 1 = interleaved (adjacent even/odd pairs)
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, cos, sin, rotary_mode, partial);
    return x;
}

std::tuple<at::Tensor, at::Tensor> dynamic_quant(const at::Tensor &x, const at::Tensor &out,
                                                 const at::Tensor &scale)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU device");
    TORCH_CHECK(out.scalar_type() == at::kChar && out.sizes() == x.sizes(),
                "dynamic_quant: out must be int8 with the same shape as x");
    TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == x.size(0),
                "dynamic_quant: scale must be float32 with one element per row");

    const c10::optional<at::Tensor> none;
    const int64_t dst_type = 2;  // ACL_INT8
    EXEC_NPU_CMD(aclnnDynamicQuantV2, x, none, none, dst_type, out, scale, none);
    return std::make_tuple(out, scale);
}

at::Tensor scatter_nd_update(const at::Tensor &var, const at::Tensor &indices, const at::Tensor &updates)
{
    TORCH_CHECK(var.device().type() == c10::DeviceType::PrivateUse1, "var must be on NPU device");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "scatter_nd_update: indices must be int32 or int64");
    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == 2,
                "scatter_nd_update: indices must be [T, 2] for the paged cache layout");
    TORCH_CHECK(updates.size(0) == indices.size(0), "scatter_nd_update: updates/indices row mismatch");
    TORCH_CHECK(updates.scalar_type() == var.scalar_type(), "scatter_nd_update: dtype mismatch");

    EXEC_NPU_CMD(aclnnScatterNdUpdate, var, indices, updates);
    return var;
}

}  // namespace llm_ops
