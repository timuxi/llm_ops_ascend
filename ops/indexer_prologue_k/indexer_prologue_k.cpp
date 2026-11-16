/**
 * Python/Torch wrapper for aclnnIndexerPrologueK.
 * Operator C++ is provided by ops-transformer (custom OPP) or CANN libopapi.
 */

#include "indexer_prologue_k.h"

#include "torch_aclnn_helper.h"

namespace llm_ops {

at::Tensor indexer_prologue_k(const at::Tensor &latent, const at::Tensor &wk, const at::Tensor &norm_weight,
                              const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &indices,
                              const at::Tensor &k_cache, const at::Tensor &k_scale_cache, double norm_eps)
{
    TORCH_CHECK(latent.device().type() == c10::DeviceType::PrivateUse1, "latent must be on NPU device");
    TORCH_CHECK(wk.device().type() == c10::DeviceType::PrivateUse1, "wk must be on NPU device");
    TORCH_CHECK(norm_weight.device().type() == c10::DeviceType::PrivateUse1,
                "norm_weight must be on NPU device");
    TORCH_CHECK(rope_cos.device().type() == c10::DeviceType::PrivateUse1, "rope_cos must be on NPU device");
    TORCH_CHECK(rope_sin.device().type() == c10::DeviceType::PrivateUse1, "rope_sin must be on NPU device");
    TORCH_CHECK(indices.device().type() == c10::DeviceType::PrivateUse1, "indices must be on NPU device");
    TORCH_CHECK(k_cache.device().type() == c10::DeviceType::PrivateUse1, "k_cache must be on NPU device");
    TORCH_CHECK(k_scale_cache.device().type() == c10::DeviceType::PrivateUse1,
                "k_scale_cache must be on NPU device");
    TORCH_CHECK(k_cache.scalar_type() == at::kChar, "k_cache must be int8, got ", k_cache.scalar_type());
    TORCH_CHECK(k_scale_cache.scalar_type() == at::kHalf, "k_scale_cache must be float16, got ",
                k_scale_cache.scalar_type());
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32, got ", indices.scalar_type());
    TORCH_CHECK(latent.dim() == 2 && wk.dim() == 2, "latent and wk must be 2-D");
    TORCH_CHECK(k_cache.dim() == 4 && k_scale_cache.dim() == 4,
                "k_cache / k_scale_cache must be 4-D [blocks, slots, 1, width]");
    TORCH_CHECK(latent.size(1) == wk.size(1), "latent.size(1) must equal wk.size(1)");
    TORCH_CHECK(wk.size(0) == k_cache.size(3), "wk.size(0) must equal the cache row width");

    // The ACLNN ABI is in-place: the out tensors must alias the cache inputs.
    EXEC_NPU_CMD(aclnnIndexerPrologueK, latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache,
                 k_scale_cache, norm_eps, k_cache, k_scale_cache);
    return k_cache;
}

}  // namespace llm_ops
