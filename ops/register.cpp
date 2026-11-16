#include <torch/extension.h>
#include <torch/library.h>

#include "all_gather_matmul_v2/all_gather_matmul_v2.h"
#include "block_sparse_attention_grad/block_sparse_attention_grad.h"
#include "flash_attention_score/flash_attention_score.h"
#include "flash_attention_score_grad/flash_attention_score_grad.h"
#include "indexer_prologue_k/indexer_prologue_k.h"

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
        "flash_attention_score(Tensor query, Tensor key, Tensor value, int head_num, *, "
        "Tensor? pse_shift=None, Tensor? drop_mask=None, Tensor? padding_mask=None, "
        "Tensor? atten_mask=None, int[]? prefix=None, "
        "int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, "
        "float scale_value=1.0, float keep_prob=1.0, int pre_tokens=2147483647, "
        "int next_tokens=2147483647, str input_layout=\"BNSD\", int inner_precise=0, "
        "int sparse_mode=0) -> (Tensor, Tensor, Tensor)");
    m.def(
        "flash_attention_score_grad(Tensor query, Tensor key, Tensor value, Tensor dy, "
        "Tensor softmax_max, Tensor softmax_sum, Tensor attention_in, int head_num, *, "
        "Tensor? pse_shift=None, Tensor? drop_mask=None, Tensor? padding_mask=None, "
        "Tensor? atten_mask=None, Tensor? softmax_in=None, int[]? prefix=None, "
        "int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, "
        "float scale_value=1.0, float keep_prob=1.0, int pre_tokens=2147483647, "
        "int next_tokens=2147483647, str input_layout=\"BNSD\", int inner_precise=0, "
        "int sparse_mode=0) -> (Tensor, Tensor, Tensor)");
    m.def(
        "indexer_prologue_k(Tensor latent, Tensor wk, Tensor norm_weight, Tensor rope_cos, Tensor rope_sin, "
        "Tensor indices, Tensor(a!) k_cache, Tensor(b!) k_scale_cache, float norm_eps=0.000001) -> Tensor");
    m.def("matmul(Tensor x, Tensor weight, Tensor(a!) out) -> Tensor");
    m.def("rms_norm(Tensor x, Tensor gamma, Tensor(a!) out, Tensor(b!) rstd, float eps=0.000001) -> (Tensor, Tensor)");
    m.def("inplace_partial_rotary_mul(Tensor(a!) x, Tensor cos, Tensor sin, int start, int end) -> Tensor");
    m.def("dynamic_quant(Tensor x, Tensor(a!) out, Tensor(b!) scale) -> (Tensor, Tensor)");
    m.def("scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> Tensor");
}

TORCH_LIBRARY_IMPL(llm_ops, PrivateUse1, m)
{
    m.impl("all_gather_matmul_v2", TORCH_FN(llm_ops::all_gather_matmul_v2));
    m.impl("block_sparse_attention_grad", TORCH_FN(llm_ops::block_sparse_attention_grad));
    m.impl("flash_attention_score", TORCH_FN(llm_ops::flash_attention_score));
    m.impl("flash_attention_score_grad", TORCH_FN(llm_ops::flash_attention_score_grad));
    m.impl("indexer_prologue_k", TORCH_FN(llm_ops::indexer_prologue_k));
    m.impl("matmul", TORCH_FN(llm_ops::matmul));
    m.impl("rms_norm", TORCH_FN(llm_ops::rms_norm));
    m.impl("inplace_partial_rotary_mul", TORCH_FN(llm_ops::inplace_partial_rotary_mul));
    m.impl("dynamic_quant", TORCH_FN(llm_ops::dynamic_quant));
    m.impl("scatter_nd_update", TORCH_FN(llm_ops::scatter_nd_update));
}

}  // namespace
