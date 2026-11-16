/**
 * Python/Torch wrappers for IndexerPrologueQW (dense + prequantized) and the
 * production weight-preparation helper.
 *
 * The fused ACLNN symbols live in libcust_opapi.so, which this repo does not link;
 * they are resolved with dlsym through the shared helper (same lookup order:
 * libcust_opapi.so -> libopapi.so). The prequantized weight is physically
 * FRACTAL_NZ, so its aclTensor descriptor is built from the tensor's NPU format
 * instead of the helper's dim-count heuristic.
 */

#include "indexer_prologue_qw.h"

#include "torch_aclnn_helper.h"

#include "indexer_prologue_k/indexer_prologue_k.h"  // reuse llm_ops::dynamic_quant


#include <vector>

namespace llm_ops {
namespace {

aclTensor *MakeAcl(const at::Tensor &x, bool use_npu_format)
{
    static const auto create = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create != nullptr, "aclCreateTensor not found in libopapi");
    std::vector<int64_t> dims(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
    std::vector<int64_t> storage{static_cast<int64_t>(x.storage().nbytes() / x.itemsize())};
    aclFormat fmt = ACL_FORMAT_ND;
    if (use_npu_format) {
        fmt = static_cast<aclFormat>(at_npu::native::get_npu_format(x));
    } else {
        switch (dims.size()) {
            case 3: fmt = ACL_FORMAT_NCL; break;
            case 4: fmt = ACL_FORMAT_NCHW; break;
            case 5: fmt = ACL_FORMAT_NCDHW; break;
            default: fmt = ACL_FORMAT_ND; break;
        }
    }
    auto *t = create(dims.data(), dims.size(), ConvertType(x.scalar_type()), strides.data(), x.storage_offset(),
                     fmt, storage.data(), storage.size(), const_cast<void *>(x.storage().data()));
    TORCH_CHECK(t != nullptr, "aclCreateTensor failed");
    return t;
}

struct AclGuard {
    aclTensor *t = nullptr;
    explicit AclGuard(const at::Tensor &x, bool use_npu_format = false) : t(MakeAcl(x, use_npu_format)) {}
    ~AclGuard()
    {
        if (t != nullptr) {
            (void)GET_OP_API_FUNC(aclDestroyTensor)(t);
        }
    }
    AclGuard(const AclGuard &) = delete;
    AclGuard &operator=(const AclGuard &) = delete;
};

void CheckStatus(int st, const char *name)
{
    TORCH_CHECK(st == 0, name, " failed, status=", st, ", detail=", aclGetRecentErrMsg());
}

using RunFn = int (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using DenseGetWs = int (*)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                           const aclTensor *, const aclTensor *, double, aclTensor *, aclTensor *, aclTensor *,
                           uint64_t *, aclOpExecutor **);
using FormatCastGetWs = int (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using QmmNzGetWs = int (*)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                           const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                           const aclTensor *, bool, bool, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using PrequantGetWs = int (*)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                              const aclTensor *, const aclTensor *, const aclTensor *, double, aclTensor *,
                              aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);

void RunCommon(void *runAddr, uint64_t ws, aclOpExecutor *ex, const char *name)
{
    at::Tensor holder;
    void *wsAddr = nullptr;
    if (ws != 0) {
        holder = at::empty({static_cast<int64_t>(ws)},
                           at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte));
        wsAddr = const_cast<void *>(holder.storage().data());
    }
    auto run = reinterpret_cast<RunFn>(runAddr);
    CheckStatus(run(wsAddr, ws, ex, c10_npu::getCurrentNPUStream().stream(false)), name);
}

void CheckCommon(const at::Tensor &x, const at::Tensor &qr, const at::Tensor &ww, const at::Tensor &sin,
                 const at::Tensor &cos, const at::Tensor &q, const at::Tensor &qs, const at::Tensor &w)
{
    for (const auto *t : {&x, &qr, &ww, &sin, &cos, &q, &qs, &w}) {
        TORCH_CHECK(t->device().type() == c10::DeviceType::PrivateUse1, "all tensors must be on NPU");
    }
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && qr.scalar_type() == at::kBFloat16 &&
                    ww.scalar_type() == at::kBFloat16,
                "x/qr/ww must be bf16");
    TORCH_CHECK(sin.scalar_type() == at::kFloat && cos.scalar_type() == at::kFloat,
                "rope_sin/rope_cos must be fp32");
    TORCH_CHECK(q.scalar_type() == at::kChar && qs.scalar_type() == at::kHalf,
                "q must be int8 and q_scale fp16");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "w must be fp16");
    TORCH_CHECK(x.dim() == 2 && qr.dim() == 2, "x/qr must be 2-D");
    TORCH_CHECK(q.dim() == 3, "q must be [T, heads, head_dim]");
}

void PermuteNd(const at::Tensor &self, at::IntArrayRef dims, const at::Tensor &out)
{
    const at::IntArrayRef dims_l = dims;  // keep the macro's lvalue requirement
    EXEC_NPU_CMD(aclnnPermute, self, dims_l, out);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> indexer_prologue_qw(
    const at::Tensor &x, const at::Tensor &qr, const at::Tensor &wqb, const at::Tensor &ww,
    const at::Tensor &rope_sin, const at::Tensor &rope_cos, double softmax_scale, const at::Tensor &q,
    const at::Tensor &q_scale, const at::Tensor &w)
{
    CheckCommon(x, qr, ww, rope_sin, rope_cos, q, q_scale, w);
    TORCH_CHECK(wqb.scalar_type() == at::kBFloat16, "wqb must be bf16");
    static void *getwsAddr = GetOpApiFuncAddr("aclnnIndexerPrologueQWGetWorkspaceSize");
    static void *runAddr = GetOpApiFuncAddr("aclnnIndexerPrologueQW");
    TORCH_CHECK(getwsAddr != nullptr && runAddr != nullptr,
                "aclnnIndexerPrologueQW not found; install the ops-transformer custom package and set "
                "LD_LIBRARY_PATH to .../custom_transformer/op_api/lib");

    AclGuard tx(x), tqr(qr), twqb(wqb), tww(ww), tsin(rope_sin), tcos(rope_cos), tq(q), tqs(q_scale), tw(w);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto getws = reinterpret_cast<DenseGetWs>(getwsAddr);
    CheckStatus(getws(tx.t, tqr.t, twqb.t, tww.t, tsin.t, tcos.t, softmax_scale, tq.t, tqs.t, tw.t, &ws, &ex),
                "aclnnIndexerPrologueQW");
    RunCommon(runAddr, ws, ex, "aclnnIndexerPrologueQW");
    return std::make_tuple(q, q_scale, w);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> indexer_prologue_qw_prequantized(
    const at::Tensor &x, const at::Tensor &qr, const at::Tensor &wqb_nz, const at::Tensor &wqb_scale,
    const at::Tensor &ww, const at::Tensor &rope_sin, const at::Tensor &rope_cos, double softmax_scale,
    const at::Tensor &q, const at::Tensor &q_scale, const at::Tensor &w)
{
    CheckCommon(x, qr, ww, rope_sin, rope_cos, q, q_scale, w);
    TORCH_CHECK(wqb_nz.scalar_type() == at::kChar && wqb_scale.scalar_type() == at::kFloat,
                "wqb_nz must be int8 and wqb_scale fp32");
    static void *getwsAddr = GetOpApiFuncAddr("aclnnIndexerPrologueQWPrequantizedGetWorkspaceSize");
    static void *runAddr = GetOpApiFuncAddr("aclnnIndexerPrologueQWPrequantized");
    TORCH_CHECK(getwsAddr != nullptr && runAddr != nullptr,
                "aclnnIndexerPrologueQWPrequantized not found; install the ops-transformer custom package");

    AclGuard tx(x), tqr(qr), tnz(wqb_nz, true), tsc(wqb_scale), tww(ww), tsin(rope_sin), tcos(rope_cos),
        tq(q), tqs(q_scale), tw(w);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto getws = reinterpret_cast<PrequantGetWs>(getwsAddr);
    CheckStatus(getws(tx.t, tqr.t, tnz.t, tsc.t, tww.t, tsin.t, tcos.t, softmax_scale, tq.t, tqs.t, tw.t, &ws,
                      &ex),
                "aclnnIndexerPrologueQWPrequantized");
    RunCommon(runAddr, ws, ex, "aclnnIndexerPrologueQWPrequantized");
    return std::make_tuple(q, q_scale, w);
}

std::tuple<at::Tensor, at::Tensor> prepare_indexer_prologue_qw_weight(const at::Tensor &wqb)
{
    TORCH_CHECK(wqb.device().type() == c10::DeviceType::PrivateUse1, "wqb must be on NPU");
    TORCH_CHECK(wqb.dim() == 2 && wqb.scalar_type() == at::kBFloat16, "wqb must be 2-D bf16");
    const int64_t n = wqb.size(0);
    const int64_t k = wqb.size(1);

    auto weight_q = at::empty({n, k}, wqb.options().dtype(at::kChar));
    auto weight_scale = at::empty({n}, wqb.options().dtype(at::kFloat));
    llm_ops::dynamic_quant(wqb, weight_q, weight_scale);

    auto weight_t = at::empty({k, n}, wqb.options().dtype(at::kChar));
    int64_t perm_values[2] = {1, 0};
    at::IntArrayRef perm(perm_values, 2);
    PermuteNd(weight_q, perm, weight_t);

    // NOTE: the Python wrapper sets torch.npu.config.allow_internal_format = True;
    // without it npu_format_cast silently falls back to the base (ND) format.
    auto weight_nz = at_npu::native::npu_format_cast(weight_t, ACL_FORMAT_FRACTAL_NZ);
    TORCH_CHECK(at_npu::native::get_npu_format(weight_nz) == ACL_FORMAT_FRACTAL_NZ,
                "weight_nz is not FRACTAL_NZ");
    return std::make_tuple(weight_nz, weight_scale);
}

at::Tensor permute(const at::Tensor &self, at::IntArrayRef dims, const at::Tensor &out)
{
    PermuteNd(self, dims, out);
    return out;
}

at::Tensor npu_format_cast(const at::Tensor &src, const at::Tensor &dst_nz)
{
    static void *getwsAddr = GetOpApiFuncAddr("aclnnNpuFormatCastGetWorkspaceSize");
    static void *runAddr = GetOpApiFuncAddr("aclnnNpuFormatCast");
    TORCH_CHECK(getwsAddr != nullptr && runAddr != nullptr, "aclnnNpuFormatCast not found");
    TORCH_CHECK(at_npu::native::get_npu_format(dst_nz) == ACL_FORMAT_FRACTAL_NZ,
                "dst must be FRACTAL_NZ (torch.npu.config.allow_internal_format = True)");
    AclGuard s(src);
    AclGuard d(dst_nz, true);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto getws = reinterpret_cast<FormatCastGetWs>(getwsAddr);
    CheckStatus(getws(s.t, d.t, &ws, &ex), "aclnnNpuFormatCast");
    RunCommon(runAddr, ws, ex, "aclnnNpuFormatCast");
    return dst_nz;
}

at::Tensor quant_matmul_weight_nz(const at::Tensor &x1, const at::Tensor &x2_nz, const at::Tensor &x1_scale,
                                  const at::Tensor &x2_scale, const at::Tensor &out)
{
    static void *getwsAddr = GetOpApiFuncAddr("aclnnQuantMatmulWeightNzGetWorkspaceSize");
    static void *runAddr = GetOpApiFuncAddr("aclnnQuantMatmulWeightNz");
    TORCH_CHECK(getwsAddr != nullptr && runAddr != nullptr, "aclnnQuantMatmulWeightNz not found");
    TORCH_CHECK(at_npu::native::get_npu_format(x2_nz) == ACL_FORMAT_FRACTAL_NZ,
                "x2 must be FRACTAL_NZ (torch.npu.config.allow_internal_format = True)");
    TORCH_CHECK(x1.scalar_type() == at::kChar && x2_nz.scalar_type() == at::kChar,
                "x1/x2 must be int8");
    TORCH_CHECK(x1_scale.scalar_type() == at::kFloat && x2_scale.scalar_type() == at::kFloat,
                "scales must be fp32");

    AclGuard t1(x1);
    AclGuard t2(x2_nz, true);
    AclGuard ts1(x1_scale);
    AclGuard ts2(x2_scale);
    AclGuard to(out);
    aclTensor *none = nullptr;
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto getws = reinterpret_cast<QmmNzGetWs>(getwsAddr);
    CheckStatus(getws(t1.t, t2.t, ts1.t, ts2.t, none, none, none, none, none, false, false, 0, to.t, &ws, &ex),
                "aclnnQuantMatmulWeightNz");
    RunCommon(runAddr, ws, ex, "aclnnQuantMatmulWeightNz");
    return out;
}

}  // namespace llm_ops
