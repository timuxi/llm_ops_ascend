# llm_ops_ascend

本仓为昇腾 LLM 算子提供 **Python / PyTorch** 接口（`import llm_ops` / `torch.ops.llm_ops.*`）。

算子的 kernel、tiling、ACLNN **不在本仓编译**。C++ 实现来自：

1. 官方 CANN 的 `libopapi.so`（CANN 已内置该 `aclnnXxx` 时），或
2. **ops-transformer** 编出的自定义算子包（`libcust_opapi.so` + kernel）

本仓只做一层 `EXEC_NPU_CMD` 封装。不要把 ops-transformer 的 `op_host/` / `op_kernel/` 拷进本仓，也不要 git submodule。

```
ops-transformer  ──编译/安装──►  libcust_opapi.so + kernel
CANN libopapi.so ──────────────►  已有同名 ACLNN 时可直接用
llm_ops_ascend   ──封装────────►  llm_ops.<op>(...)
```

运行时查找顺序：先 `libcust_opapi.so`，找不到再回落官方 `libopapi.so`。没装自定义包时看起来也能跑，实际用的是官方库。

---

## 目录

```
llm_ops_ascend/
├── ops/
│   ├── register.cpp                              # TORCH_LIBRARY 注册
│   ├── CMakeLists.txt                            # 自动编 ops/*/*.cpp
│   ├── all_gather_matmul_v2/
│   │   ├── all_gather_matmul_v2.h / .cpp
│   │   └── test.py
│   ├── block_sparse_attention_grad/
│   │   ├── block_sparse_attention_grad.h / .cpp
│   │   └── tests/block_sparse_attention_grad_test.py
│   ├── flash_attention_score/
│   │   ├── flash_attention_score.h / .cpp
│   │   └── tests/flash_attention_score_test.py
│   ├── flash_attention_score_grad/
│   │   ├── flash_attention_score_grad.h / .cpp
│   │   └── tests/flash_attention_score_grad_test.py
│   ├── indexer_prologue_k/
│   │   ├── indexer_prologue_k.h / .cpp            # fused 入口（需 ops-transformer 自定义包）
│   │   ├── indexer_prologue_baseline.cpp          # 5 个官方算子的基线封装
│   │   └── tests/perf_indexer_prologue_k.py
│   └── indexer_prologue_qw/
│       ├── indexer_prologue_qw.h / .cpp           # dense + prequantized + weight 预处理
│       └── tests/perf_indexer_prologue_qw.py
├── utils/
│   ├── torch_aclnn_helper.h                      # EXEC_NPU_CMD（不要随意改）
│   ├── cmake/config_envs.cmake                   # 找 torch / torch_npu / CANN
│   └── python/                                   # wheel 打包
│       ├── setup.py                              # 包名 llm_ops_ascend，import 仍是 llm_ops
│       └── llm_ops/__init__.py
├── build.sh
├── CMakeLists.txt
├── config.ini                                    # 版本号
├── requirements.txt
└── README.md
```

`ops/<op_name>/` 只放 Python/Torch 绑定，不要放 kernel 源码。编译产物在 `build/`、`output/`，已 gitignore。

---

## 环境

- Ascend NPU + 与芯片匹配的 CANN Toolkit
- Python 3.8+，PyTorch + 配套 `torch_npu`

先 `source` CANN 的 `set_env.sh`（或本机等价环境脚本），再编译或运行 Python。`./build.sh` 会按系统 CANN 安装信息自动 `source set_env.sh`。

---

## 使用方法

### 1. （按需）编译并安装 ops-transformer

官方 CANN **没有** 对应 `aclnnXxx`，或要跑 ops-transformer 自己的实现时，才需要这一步。

```bash
cd /path/to/ops-transformer
bash build.sh --pkg --soc=<soc> --vendor_name=custom --ops=<op_name>
./build_out/cann-ops-transformer-custom_linux-*.run
```

- `--soc`：芯片，如 `ascend910b`、`ascend910_93`、`ascend950`。每次只能编一个 SOC。
- `--ops`：逗号分隔多个算子；不指定则编该仓默认算子集。
- `--vendor_name`：默认 `custom`，安装目录一般为  
  `${ASCEND_HOME_PATH}/opp/vendors/custom_transformer`

**必须在启动 Python 之前** 让动态链接器能找到 `libcust_opapi.so`：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

若安装时用了 `--install-path`：

```bash
source ${install_path}/vendors/<vendor>/bin/set_env.bash
```

进程已经起来后再改 `LD_LIBRARY_PATH` **对 `dlopen` 无效**。`ASCEND_CUSTOM_OPP_PATH` 由 `set_env.bash` 设置，helper 也会按  
`${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so` 再搜一遍。

### 2. 编译本仓 Python 封装

```bash
cd /path/to/llm_ops_ascend
./build.sh <SOC_VERSION>          # 例：Ascend910B3
pip3 install --force-reinstall --no-deps output/llm_ops_ascend-*.whl
```

`./build.sh` **只编** `libllm_ops.so` 和 wheel，不编 kernel。参数是 SOC 名，只影响 torch 封装侧；kernel 的 SOC 由 ops-transformer 的 `--soc` 决定。各算子 smoke test 在对应 `ops/<op_name>/` 目录下。

### 3. 调用

输入必须在 NPU 上。

```python
import llm_ops   # 加载 libllm_ops.so，并 import torch_npu

result = llm_ops.<op_name>(...)
# 或 torch.ops.llm_ops.<op_name>(...)
```

参数与约束以上游 ACLNN 头文件和 `ops/<op_name>/` 下的 smoke test 为准。

### IndexerPrologueK 与基线对照

`ops/indexer_prologue_k/` 除 fused 入口外，还把生产基线拆成的 5 个官方算子封成了 torch 接口。
基线接口的输出由调用方预分配（并原样返回），避免把 allocator 开销算进计时：

| Python | ACLNN | 作用 |
|---|---|---|
| `llm_ops.matmul(x, weight, out)` | `aclnnMatmul` | MatMulV2，weight 用 stride 视图转置，不拷贝 |
| `llm_ops.rms_norm(x, gamma, out, rstd, eps)` | `aclnnRmsNorm` | RMSNorm |
| `llm_ops.inplace_partial_rotary_mul(x, cos, sin, start, end)` | `aclnnInplacePartialRotaryMul` | interleave 模式 partial RoPE，原地 |
| `llm_ops.dynamic_quant(x, out, scale)` | `aclnnDynamicQuantV2` | per-token INT8 + fp32 scale |
| `llm_ops.scatter_nd_update(var, indices, updates)` | `aclnnScatterNdUpdate` | paged cache 写回，原地 |
| `llm_ops.indexer_prologue_k(latent, wk, gamma, cos, sin, indices, k_cache, k_scale_cache, eps)` | `aclnnIndexerPrologueK` | fused（需 ops-transformer 自定义包） |

跑法（`--check` 会逐字节比对 fused 与基线的 cache/scale）：

```bash
python3 ops/indexer_prologue_k/tests/perf_indexer_prologue_k.py \
    --check --mode both --T 96 --nb 22594 --ns 64 --device 7
```

kernel 时间用裸 `msprof` 采 `op_summary`（两条命令各采一次）：

```bash
msprof --output=prof_base  python3 ops/indexer_prologue_k/tests/perf_indexer_prologue_k.py --mode baseline --T 96 --nb 22594 --ns 64 --device 7
msprof --output=prof_fused python3 ops/indexer_prologue_k/tests/perf_indexer_prologue_k.py --mode fused    --T 96 --nb 22594 --ns 64 --device 7
```

`--skip=15`（基线 3 warmup × 5 算子）和 `--skip=9`（fused 3 warmup × 3 算子）可用于丢掉 warmup 后再统计。

### IndexerPrologueQW（dense / prequantized）

Q/W 侧两个入口和 weight 预处理：

| Python | ACLNN | kernel 数 | 说明 |
|---|---|---|---|
| `llm_ops.indexer_prologue_qw(x, qr, wqb, ww, rope_sin, rope_cos, softmax_scale, q, q_scale, w)` | `aclnnIndexerPrologueQW` | 7 | 每次调用都做 weight DQ + permute + NZ |
| `llm_ops.indexer_prologue_qw_prequantized(x, qr, wqb_nz, wqb_scale, ww, rope_sin, rope_cos, softmax_scale, q, q_scale, w)` | `aclnnIndexerPrologueQWPrequantized` | 4 | weight 由调用方持有（INT8/FRACTAL_NZ + FP32 scale） |
| `llm_ops.prepare_indexer_prologue_qw_weight(wqb)` | — | — | `DynamicQuant → transpose/contiguous → NpuFormatCast(29)`，返回 `(wqb_nz, wqb_scale)` |
| `llm_ops.permute(self, dims, out)` | `aclnnPermute` | — | 基线：weight transpose |
| `llm_ops.npu_format_cast(src, dst)` | `aclnnNpuFormatCast` | — | 基线：ND → NZ，dst 需是 FRACTAL_NZ |
| `llm_ops.quant_matmul_weight_nz(x1, x2, x1_scale, x2_scale, out)` | `aclnnQuantMatmulWeightNz` | — | 基线：INT8 activation × NZ weight |

`tests/perf_indexer_prologue_qw.py` 现在同时跑三条链做对照：
**baseline（7 个独立算子，Post 用 torch 参考）** / **dense** / **prequantized**，
精度要求三者逐位一致。

输出（调用方预分配，三个接口都返回）：`q` INT8 `[T,32,128]`、`q_scale` FP16 `[T,32]`、`w` FP16 `[T,32]`。

注意：`wqb_nz` 是 **FRACTAL_NZ 内部格式**，torch_npu 只在 `torch.npu.config.allow_internal_format = True` 时才会真正创建它（Python 包装里已经替 `prepare_*` 设置）；自己造 NZ 权重时要先打开这个开关。

跑法（`--mode precision` 会对比 torch_npu W8A8 golden 和 dense/prequant 一致性）：

```bash
python3 ops/indexer_prologue_qw/tests/perf_indexer_prologue_qw.py \
    --mode both --T 96 --device 7 --iters 20
```

kernel 时间（裸 msprof，两条命令分别采）：

```bash
msprof --output=prof_qw_dense python3 ops/indexer_prologue_qw/tests/perf_indexer_prologue_qw.py --mode perf-dense    --T 96 --device 7 --iters 20
msprof --output=prof_qw_pre   python3 ops/indexer_prologue_qw/tests/perf_indexer_prologue_qw.py --mode perf-prequant --T 96 --device 7 --iters 20
```

---

## 新增算子

对照 ops-transformer 的 `op_host/op_api/aclnn_<op>.h` 和 `examples/test_aclnn_*.cpp`。

1. 新建 `ops/<op_name>/`：`<op_name>.h`、`<op_name>.cpp`，以及 smoke test  
   （`test.py` 或 `tests/<op_name>_test.py`）。
2. 在 `ops/register.cpp` 里加 schema 和 `m.impl`。
3. 在 `utils/python/llm_ops/__init__.py` 里导出同名函数。
4. `./build.sh`，安装 wheel，跑该算子目录下的 test。

命名与参数：

| ACLNN / C++ | Python schema |
|-------------|---------------|
| 算子名 | `snake_case`；ACLNN 保持 `aclnn` + PascalCase |
| `aclTensor*` | `Tensor` |
| 可选 Tensor | `Tensor?` |
| `aclIntArray*` | `int[]` / `int[]?` |
| `char *layout` | `str`（先拷到 `std::string` 再 `c_str()`） |
| 多输出 | `std::tuple` / `(Tensor, ...)` |

`EXEC_NPU_CMD` 的实参必须是**左值**，顺序与 `aclnnXxxGetWorkspaceSize` 的业务参数一致（不含 workspace / executor / stream）。用 `empty_with_format` 建输出。

`ops/CMakeLists.txt` 会自动编 `ops/*/*.cpp`，一般不用改 cmake。

不要改 `utils/torch_aclnn_helper.h`，除非修公共 helper 的 bug。

---

## 注意点

1. **本仓不编 kernel。** `ops/` 不是 ops-transformer 的源码树。
2. **先设库路径再开 Python。** 否则 `dlopen("libcust_opapi.so")` 失败，会静默用官方 `libopapi.so`。可用 `/proc/<pid>/maps` 看实际映射的是哪份 so。
3. **layout 字符串** 必须先落到局部 `std::string`，再把 `c_str()` 传给 `EXEC_NPU_CMD`。
4. **dtype / 可选 mask / layout** 以上游 `ValidateParams` 和 ACLNN 头为准，不要猜。
5. **SOC 两处要一致：** ops-transformer 的 `--soc` 与本仓 `./build.sh` 的 SOC 对应同一芯片。
6. 自定义包安装后可用  
   `grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini`  
   确认 vendor 已写入；没有的话检查 `opp/vendors` 权限后重装。
7. 官方 CANN 已有同名算子时，不装 ops-transformer 也能跑通测试，那只说明封装对了，**不代表**跑的是 ops-transformer 的实现。

---

## 当前算子

| Python API | ACLNN | 本仓 | C++ |
|------------|--------|------|-----|
| `block_sparse_attention_grad` | `aclnnBlockSparseAttentionGrad` | `ops/block_sparse_attention_grad/` | ops-transformer `attention/block_sparse_attention_grad/` |
| `flash_attention_score` | `aclnnFlashAttentionScore` / `aclnnFlashAttentionVarLenScore`（TND） | `ops/flash_attention_score/` | ops-transformer `attention/flash_attention_score/` |
| `flash_attention_score_grad` | `aclnnFlashAttentionScoreGrad` / `aclnnFlashAttentionUnpaddingScoreGrad`（TND） | `ops/flash_attention_score_grad/` | ops-transformer `attention/flash_attention_score_grad/` |
| `all_gather_matmul_v2` | `aclnnAllGatherMatmulV2` | `ops/all_gather_matmul_v2/` | ops-transformer `mc2/all_gather_matmul_v2/` |
