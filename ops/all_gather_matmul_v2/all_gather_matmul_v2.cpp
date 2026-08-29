/**
 * Python/Torch wrapper for aclnnAllGatherMatmulV2 (all-gather + matmul fusion).
 * Operator C++ is provided by ops-transformer (custom OPP) or CANN libopapi.
 */

#include <string>
#include <tuple>

#include "all_gather_matmul_v2.h"
#include "torch_aclnn_helper.h"

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor> all_gather_matmul_v2(
    const at::Tensor &x1, const at::Tensor &x2, c10::string_view group, int64_t world_size,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &x1_scale,
    const c10::optional<at::Tensor> &x2_scale, const c10::optional<at::Tensor> &quant_scale, int64_t block_size,
    int64_t gather_index, int64_t comm_turn, int64_t stream_mode, int64_t group_size, c10::string_view comm_mode,
    bool gather_output)
{
    TORCH_CHECK(x1.device().type() == c10::DeviceType::PrivateUse1, "x1 must be on NPU device");
    TORCH_CHECK(x2.device().type() == c10::DeviceType::PrivateUse1, "x2 must be on NPU device");
    TORCH_CHECK(x1.dim() == 2, "x1 must be 2-D [M, K], got dim ", x1.dim());
    TORCH_CHECK(x2.dim() == 2, "x2 must be 2-D [K, N], got dim ", x2.dim());
    TORCH_CHECK(x1.scalar_type() == at::kHalf || x1.scalar_type() == at::kBFloat16 || x1.scalar_type() == at::kChar,
                "x1 must be float16, bfloat16 or int8, got ", x1.scalar_type());
    TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
                "world_size must be 2, 4 or 8, got ", world_size);
    TORCH_CHECK(gather_index == 0, "gather_index currently only supports 0, got ", gather_index);
    TORCH_CHECK(x1.size(1) >= 256 && x1.size(1) < 65535, "K axis must be in [256, 65535), got ", x1.size(1));

    const int64_t m = x1.size(0);
    const int64_t k = x1.size(1);
    const int64_t n = x2.size(1);

    at::ScalarType out_dtype = x1.scalar_type();
    if (x2_scale.has_value() && x2_scale->defined()) {
        if (x2_scale->scalar_type() == at::kLong) {
            out_dtype = at::kHalf;
        } else {
            out_dtype = at::kBFloat16;
        }
    }
    auto out_opts = x1.options().dtype(out_dtype);
    at::Tensor output = at_npu::native::empty_with_format({m * world_size, n}, out_opts,
                                                          at_npu::native::get_npu_format(x1));

    at::Tensor gather_out;
    if (gather_output) {
        gather_out = at_npu::native::empty_with_format({m * world_size, k}, x1.options(),
                                                       at_npu::native::get_npu_format(x1));
    } else {
        gather_out = at::empty({0}, x1.options());
    }
    at::Tensor amax_out = at::empty({0}, x1.options().dtype(at::kFloat));

    std::string group_storage(group);
    char *group_ptr = const_cast<char *>(group_storage.c_str());
    std::string comm_mode_storage(comm_mode);
    char *comm_mode_ptr = const_cast<char *>(comm_mode_storage.c_str());

    EXEC_NPU_CMD(aclnnAllGatherMatmulV2, x1, x2, bias, x1_scale, x2_scale, quant_scale, block_size, group_ptr,
                 gather_index, comm_turn, stream_mode, group_size, comm_mode_ptr, output, gather_out, amax_out);

    return std::make_tuple(output, gather_out);
}

}  // namespace llm_ops
