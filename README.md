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
│   └── flash_attention_score_grad/
│       ├── flash_attention_score_grad.h / .cpp
│       └── tests/flash_attention_score_grad_test.py
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

- Ascend NPU + 匹配的 CANN Toolkit（本机示例：CANN 9.1.0，Ascend910B3）
- Python 3.8+，PyTorch + 配套 `torch_npu`
- 本机可先 `source /home/xcy/model/proxy.env`（CANN + conda）

`./build.sh` 会从 `/etc/Ascend/ascend_cann_install.info` 读 Toolkit 路径并 `source set_env.sh`。

---

## 使用方法

### 1. （按需）编译并安装 ops-transformer

官方 CANN **没有** 对应 `aclnnXxx`，或你要跑 ops-transformer 自己的实现时，才需要这一步。

```bash
source /home/xcy/model/proxy.env   # 或本机 CANN set_env
cd /path/to/ops-transformer
bash build.sh --pkg --soc=ascend910b --vendor_name=custom --ops=block_sparse_attention_grad
./build_out/cann-ops-transformer-custom_linux-*.run
```

- `--soc`：芯片，如 `ascend910b`（910B）、`ascend910_93`、`ascend950`。每次只能编一个 SOC。
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

确认符号：

```bash
nm -D ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so | grep aclnnBlockSparseAttentionGrad
```

### 2. 编译本仓 Python 封装

```bash
source /home/xcy/model/proxy.env
cd /path/to/llm_ops_ascend
./build.sh Ascend910B3          # 默认即 Ascend910B3
pip3 install --force-reinstall --no-deps output/llm_ops_ascend-*.whl
python3 ops/block_sparse_attention_grad/tests/block_sparse_attention_grad_test.py
python3 ops/flash_attention_score_grad/tests/flash_attention_score_grad_test.py
python3 ops/all_gather_matmul_v2/test.py
```

`./build.sh` **只编** `libllm_ops.so` 和 wheel，不编 kernel。参数是 SOC 名（如 `Ascend910B3`），只影响 torch 封装侧；kernel 的 SOC 由 ops-transformer 的 `--soc` 决定。

### 3. 调用

输入必须在 NPU 上。

```python
import math
import torch
import llm_ops   # 会加载 libllm_ops.so，并 import torch_npu

dq, dk, dv = llm_ops.block_sparse_attention_grad(
    dout, query, key, value, attention_out, softmax_lse, mask,
    [64, 64],          # block_shape
    1,                 # num_key_value_heads
    actual_seq_lengths=[128],
    actual_seq_lengths_kv=[128],
    q_input_layout="BNSD",
    kv_input_layout="BNSD",
    scale_value=1.0 / math.sqrt(128),
)
```

也可用 `torch.ops.llm_ops.block_sparse_attention_grad(...)`。

`flash_attention_score_grad`：`input_layout` 为 `TND` 时走 `aclnnFlashAttentionUnpaddingScoreGrad`，否则走 `aclnnFlashAttentionScoreGrad`。

```python
# BNSD
dq, dk, dv = llm_ops.flash_attention_score_grad(
    query, key, value, dy, softmax_max, softmax_sum, attention_in,
    1,  # head_num
    atten_mask=atten_mask,
    scale_value=1.0 / math.sqrt(128),
    input_layout="BNSD",
)

# TND / varlen：actual_seq_* 为各序列长度的累加和
dq, dk, dv = llm_ops.flash_attention_score_grad(
    query, key, value, dy, softmax_max, softmax_sum, attention_in,
    1,
    actual_seq_qlen=[128, 256],
    actual_seq_kvlen=[128, 256],
    scale_value=1.0 / math.sqrt(128),
    input_layout="TND",
)
```

当前 `block_sparse_attention_grad` 的最小可跑形状（BNSD、fp16）：

| 张量 | dtype | shape |
|------|--------|--------|
| query / key / value / dout / attention_out | fp16 或 bf16 | `[B, N, S, D]` |
| softmax_lse | fp32 | `[B, N, S]` |
| block_sparse_mask | uint8 | `[B, N, ceil(S/bx), ceil(S_kv/by)]` |
| block_shape | — | `[bx, by]`，例如 `[64, 64]` |

当前 `flash_attention_score_grad` 的最小可跑形状（fp16）：

**BNSD**（`input_layout="BNSD"`）

| 张量 | dtype | shape |
|------|--------|--------|
| query / key / value / dy / attention_in | fp16 或 bf16 | `[B, N, S, D]` |
| softmax_max / softmax_sum | fp32 | `[B, N, S, 8]` |
| atten_mask（可选） | uint8 | `[S, S]` |

**TND**（`input_layout="TND"`，须传 `actual_seq_qlen` / `actual_seq_kvlen`）

| 张量 / 参数 | dtype | shape |
|------|--------|--------|
| query / dy / attention_in | fp16 或 bf16 | `[T_q, N, D]`，`T_q = actual_seq_qlen[-1]` |
| key / value | fp16 或 bf16 | `[T_kv, N, D]` |
| softmax_max / softmax_sum | fp32 | `[T_q, N, 8]` |
| actual_seq_qlen / actual_seq_kvlen | — | 各序列长度的累加和，如 `[128, 256]` |

`all_gather_matmul_v2`（封装 `aclnnAllGatherMatmulV2`，AllGather + MatMul 融合）。910B 上 `comm_mode` 用 `"aiv"`。`group` 是 HCCL 通信域名（`get_hccl_comm_name`）。`world_size` 仅支持 2/4/8，K 轴范围 `[256, 65535)`。

```python
output, gather_out = llm_ops.all_gather_matmul_v2(
    x1, x2, hcom, world_size, comm_mode="aiv",
)
```

当前 `all_gather_matmul_v2` 的最小可跑形状（2 卡、fp16、`gather_index=0`）：

| 张量 / 参数 | dtype | shape |
|------|--------|--------|
| x1 | fp16 / bf16 | `[M, K]`，K≥256 |
| x2 | 同 x1 | `[K, N]` |
| output | 同 x1 | `[M * world_size, N]` |
| gather_out | 同 x1 | `[M * world_size, K]` |

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
5. **SOC 两处要一致：** ops-transformer `--soc=ascend910b` ↔ 本仓 `./build.sh Ascend910B3`。
6. 自定义包安装后可用  
   `grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini`  
   确认 vendor 已写入；没有的话检查 `opp/vendors` 权限后重装。
7. 官方 CANN 已有同名算子时，不装 ops-transformer 也能跑通测试，那只说明封装对了，**不代表**跑的是 ops-transformer 的实现。

---

## 当前算子

| Python API | ACLNN | 本仓 | C++ |
|------------|--------|------|-----|
| `block_sparse_attention_grad` | `aclnnBlockSparseAttentionGrad` | `ops/block_sparse_attention_grad/` | ops-transformer `attention/block_sparse_attention_grad/` |
| `flash_attention_score_grad` | `aclnnFlashAttentionScoreGrad` / `aclnnFlashAttentionUnpaddingScoreGrad`（TND） | `ops/flash_attention_score_grad/` | ops-transformer `attention/flash_attention_score_grad/` |
| `all_gather_matmul_v2` | `aclnnAllGatherMatmulV2` | `ops/all_gather_matmul_v2/` | ops-transformer `mc2/all_gather_matmul_v2/` |
