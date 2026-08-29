#ifndef LLM_OPS_ALL_GATHER_MATMUL_V2_H
#define LLM_OPS_ALL_GATHER_MATMUL_V2_H

#include <tuple>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

namespace llm_ops {

std::tuple<at::Tensor, at::Tensor> all_gather_matmul_v2(
    const at::Tensor &x1, const at::Tensor &x2, c10::string_view group, int64_t world_size,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &x1_scale,
    const c10::optional<at::Tensor> &x2_scale, const c10::optional<at::Tensor> &quant_scale, int64_t block_size,
    int64_t gather_index, int64_t comm_turn, int64_t stream_mode, int64_t group_size, c10::string_view comm_mode,
    bool gather_output);

}  // namespace llm_ops

#endif  // LLM_OPS_ALL_GATHER_MATMUL_V2_H
