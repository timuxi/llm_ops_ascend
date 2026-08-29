#include <torch/extension.h>
#include <torch/library.h>

#include "all_gather_matmul_v2/all_gather_matmul_v2.h"
#include "block_sparse_attention_grad/block_sparse_attention_grad.h"
#include "flash_attention_score_grad/flash_attention_score_grad.h"

namespace {

TORCH_LIBRARY(llm_ops, m)
{
    m.def(
        "all_gather_matmul_v2(Tensor x1, Tensor x2, str group, int world_size, *, "
        "Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, Tensor? quant_scale=None, "
        "int block_size=0, int gather_index=0, int comm_turn=0, int stream_mode=1, "
        "int group_size=0, str comm_mode=\"aiv\", bool gather_output=True) -> (Tensor, Tensor)");
    m.def(
        "block_sparse_attention_grad(Tensor dout, Tensor query, Tensor key, Tensor value, "
        "Tensor attention_out, Tensor softmax_lse, Tensor block_sparse_mask, int[] block_shape, "
        "int num_key_value_heads, *, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, "
        "int[]? actual_seq_lengths_kv=None, str q_input_layout=\"BNSD\", str kv_input_layout=\"BNSD\", "
        "int mask_type=0, float scale_value=1.0, int pre_tokens=2147483647, "
        "int next_tokens=2147483647) -> (Tensor, Tensor, Tensor)");
    m.def(
        "flash_attention_score_grad(Tensor query, Tensor key, Tensor value, Tensor dy, "
        "Tensor softmax_max, Tensor softmax_sum, Tensor attention_in, int head_num, *, "
        "Tensor? pse_shift=None, Tensor? drop_mask=None, Tensor? padding_mask=None, "
        "Tensor? atten_mask=None, Tensor? softmax_in=None, int[]? prefix=None, "
        "int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, "
        "float scale_value=1.0, float keep_prob=1.0, int pre_tokens=2147483647, "
        "int next_tokens=2147483647, str input_layout=\"BNSD\", int inner_precise=0, "
        "int sparse_mode=0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llm_ops, PrivateUse1, m)
{
    m.impl("all_gather_matmul_v2", TORCH_FN(llm_ops::all_gather_matmul_v2));
    m.impl("block_sparse_attention_grad", TORCH_FN(llm_ops::block_sparse_attention_grad));
    m.impl("flash_attention_score_grad", TORCH_FN(llm_ops::flash_attention_score_grad));
}

}  // namespace
