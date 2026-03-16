# TFLM Dynamic Quantization (Hybrid) 推理支持技术报告

## 1. 概述

本项目为 TensorFlow Lite for Microcontrollers (TFLM) 添加了 **Dynamic Quantization（动态量化，又称 Hybrid 推理）** 的完整支持，包括 reference C 实现和 CMSIS-NN 加速实现，并在 Cortex-M55 FVP 上完成了验证。

**Dynamic Quantization 模型特征**：
- 激活值（input/output）为 `float32`
- 权重（filter）为 `int8`（per-channel 量化）
- 推理时动态量化输入为 int8，执行 int8 矩阵乘法，再反量化输出为 float32

**支持的算子**：`FULLY_CONNECTED`、`CONV_2D`

**加速效果**：在 Cortex-M55 上，CMSIS-NN 加速路径比 reference C 实现快 **11.3 倍**。

## 2. 背景与动机

### 2.1 问题

TFLM 原始代码对 hybrid 模型（float32 input + int8 filter）直接报错：
```
"Hybrid models are not supported on TFLite Micro."
```

这意味着通过 TFLite converter 使用 `optimizations=[tf.lite.Optimize.DEFAULT]`（不指定 representative dataset）生成的 dynamic quantization 模型无法在 TFLM 上运行。

### 2.2 目标

1. 让 hybrid 模型能在 TFLM 上正确运行（reference 实现）
2. 在 CMSIS-NN 构建中利用 `arm_nn_mat_mult_nt_t_s8_s32` 实现 MVE/DSP 加速
3. 在 Cortex-M55 FVP 上验证正确性和性能

## 3. 技术方案

### 3.1 Hybrid 推理流程

```
float32 input → 动态量化为 int8 → int8 matmul → int32 累加 → 反量化为 float32 → 加 bias → 激活函数
```

**动态量化**（per-tensor symmetric）：
```
scale = max(|input|) / 127
input_int8[i] = round(input[i] / scale), clamp to [-127, 127]
```

**反量化**（per-channel）：
```
output_float[ch] = int32_acc[ch] * input_scale * filter_scale[ch] + bias[ch]
```

### 3.2 CMSIS-NN 加速 API

使用 `arm_nn_mat_mult_nt_t_s8_s32`（来自 `arm_nnsupportfunctions.h`）：

```c
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8_s32(
    const int8_t *lhs,           // 量化后的输入
    const int8_t *rhs,           // int8 权重
    int32_t *dst,                // int32 输出（必须预先清零）
    const int32_t lhs_rows,      // batch × spatial
    const int32_t rhs_rows,      // 累加维度
    const int32_t rhs_cols,      // 输出通道数
    const int32_t lhs_offset,    // 输入 zero point（symmetric 为 0）
    const int32_t dst_idx_offset // 输出步长（通常为 1）
);
```

**硬件加速**：Cortex-M55 使用 MVE (Helium) SIMD，Cortex-M4/M7 使用 DSP SMLAD，其他平台使用标量回退。

### 3.3 FullyConnected 参数映射

```c
arm_nn_mat_mult_nt_t_s8_s32(
    input_quantized,   // [batches × accum_depth]
    filter_int8,       // [output_depth × accum_depth]
    int32_output,      // [batches × output_depth]
    batches,           // lhs_rows
    accum_depth,       // rhs_rows (= input_depth)
    output_depth,      // rhs_cols
    0,                 // lhs_offset (symmetric)
    1);                // dst_idx_offset
```

### 3.4 Conv2D 参数映射（per-row im2col）

Conv2D 需要先做 im2col 将卷积转换为矩阵乘法，按输出行处理以节省内存：

```c
// 对每个 output row:
// 1. im2col: [output_width × patch_size]
// 2. matmul:
arm_nn_mat_mult_nt_t_s8_s32(
    im2col,            // [output_width × patch_size]
    filter_int8,       // [output_depth × patch_size]
    int32_output,      // [output_width × output_depth]
    output_width,      // lhs_rows
    patch_size,        // rhs_rows (= filter_h × filter_w × input_depth)
    output_depth,      // rhs_cols
    0,                 // lhs_offset
    1);                // dst_idx_offset
```

## 4. 修改的文件

### 4.1 核心实现文件

| 文件 | 修改内容 |
|------|----------|
| `kernels/fully_connected_common.cc` | 新增 `FullyConnectedEvalHybrid()`，含 `#if defined(CMSIS_NN)` 条件编译 |
| `kernels/conv_common.cc` | 新增 `ConvEvalHybrid()`，含 im2col + CMSIS-NN 加速路径 |
| `kernels/fully_connected.h` | 声明 `FullyConnectedEvalHybrid()` |
| `kernels/conv.h` | 声明 `ConvEvalHybrid()`，新增 `hybrid_im2col_scratch_index` 字段 |

### 4.2 Reference kernel Prepare

| 文件 | 修改内容 |
|------|----------|
| `kernels/fully_connected.cc` | 添加 null check、分配 int32 output scratch |
| `kernels/conv.cc` | 调用 `ConvEvalHybrid()`，移除内联实现 |

### 4.3 CMSIS-NN kernel Prepare

| 文件 | 修改内容 |
|------|----------|
| `kernels/cmsis_nn/fully_connected.cc` | 允许 float32+int8 组合、分配 scratch、调用 hybrid eval |
| `kernels/cmsis_nn/conv.cc` | 允许 float32+int8 组合、分配 im2col/output scratch、调用 hybrid eval |

### 4.4 测试文件

| 文件 | 说明 |
|------|------|
| `kernels/hybrid_model_test.cc` | FVP 端到端测试，嵌入真实 hybrid 模型 |
| `kernels/hybrid_model_test_data.h` | 嵌入的模型数据、输入数据、参考输出 |
| `kernels/Makefile.inc` | 添加 hybrid_model_test 到测试列表 |

## 5. Scratch Buffer 分配

### 5.1 FullyConnected

| Buffer | 大小 | 用途 |
|--------|------|------|
| `hybrid_input_scratch` | `input_size × sizeof(int8_t)` | 量化后的输入（已有） |
| `hybrid_output_scratch` | `batches × output_depth × sizeof(int32_t)` | int32 累加器（新增） |

### 5.2 Conv2D

| Buffer | 大小 | 用途 |
|--------|------|------|
| `hybrid_input_scratch` | `input_size × sizeof(int8_t)` | 量化后的输入（已有） |
| `hybrid_im2col_scratch` | `output_w × patch_size × sizeof(int8_t)` | im2col 缓冲区（新增） |
| `hybrid_output_scratch` | `output_w × output_depth × sizeof(int32_t)` | int32 累加器（新增） |

其中 `patch_size = filter_h × filter_w × input_depth`。

## 6. 性能测试结果

### 6.1 测试环境

- **平台**：Arm Corstone-300 FVP (Cortex-M55, MVE enabled)
- **模型**：MiHr_pnet_aiq_dynamic_skip_stage0_conv.tflite (381KB)
  - 输入：`[1, 1, 256]` float32
  - 输出：`[1, 1, 64]` float32
  - 算子：ADD, AVERAGE_POOL_2D, CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, LOGISTIC, MEAN, MUL, PAD, RESHAPE, TRANSPOSE
- **测量方式**：DWT CYCCNT (32-bit cycle counter)，5 次推理取最小值

### 6.2 性能对比

| 指标 | Reference C | CMSIS-NN (MVE) | 加速比 |
|------|------------|----------------|--------|
| **Cycles/inference** | 40,404,270 | 3,573,019 | **11.3×** |
| 总指令数 (7次推理) | 2,589M | 232M | 11.2× |

### 6.3 正确性验证

| 测试 | 平台 | 结果 |
|------|------|------|
| `fully_connected_test` | x86 (bazel) | PASSED |
| `conv_test` | x86 (bazel) | PASSED |
| `fully_connected_test` | Cortex-M55 FVP | 12/12 PASSED |
| `conv_test` | Cortex-M55 FVP | 21/21 PASSED |
| `hybrid_model_test` | x86 | max_abs_diff=0.000000 |
| `hybrid_model_test` | Cortex-M55 FVP | max_abs_diff=0.000000 |
| `depthwise_conv_test` | Cortex-M55 FVP | PASSED (回归) |
| `pooling_test` | Cortex-M55 FVP | PASSED (回归) |
| `softmax_test` | Cortex-M55 FVP | PASSED (回归) |
| `add_test` | Cortex-M55 FVP | PASSED (回归) |

## 7. 构建方式

### 7.1 x86 构建与测试

```bash
# Reference kernel 测试
bazel test //tensorflow/lite/micro/kernels:fully_connected_test
bazel test //tensorflow/lite/micro/kernels:conv_test
```

### 7.2 Cortex-M55 FVP 构建与测试

```bash
# 构建 CMSIS-NN 加速版本
make -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=cmsis_nn \
  TARGET=cortex_m_corstone_300 \
  TARGET_ARCH=cortex-m55 \
  test_kernel_fully_connected_test

# 构建 hybrid 模型端到端测试
make -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=cmsis_nn \
  TARGET=cortex_m_corstone_300 \
  TARGET_ARCH=cortex-m55 \
  test_kernel_hybrid_model_test
```

## 8. 设计决策与权衡

### 8.1 为什么用 `#if defined(CMSIS_NN)` 而不是虚函数

- TFLM 设计原则是零动态分配、最小运行时开销
- 编译时条件编译是 TFLM 中 CMSIS-NN 优化的标准模式
- 保持与现有 int8/int16 kernel 的一致性

### 8.2 为什么 Conv2D 按行做 im2col

- 全量 im2col 需要 `output_h × output_w × patch_size` 字节，对大特征图内存开销过大
- 按行处理只需 `output_w × patch_size` 字节，内存节省 `output_h` 倍
- 每行仍然可以利用 CMSIS-NN 的 SIMD 加速

### 8.3 为什么 hybrid eval 放在 `_common.cc`

- `fully_connected_common.cc` 和 `conv_common.cc` 被 reference 和 CMSIS-NN 两个 kernel 共享
- hybrid eval 函数只需实现一次，两个 kernel 的 Eval 都调用它
- CMSIS-NN 加速通过 `#if defined(CMSIS_NN)` 在编译时选择

## 9. 已知限制

1. **仅支持 symmetric quantization**：输入量化使用 zero_point=0，与 TFLite converter 的 dynamic quantization 输出一致
2. **不支持 per-tensor filter quantization**：仅支持 per-channel（这是 TFLite 的默认行为）
3. **不支持 DEPTHWISE_CONV_2D hybrid**：当前仅实现了 CONV_2D 和 FULLY_CONNECTED
4. **FVP linker warning**：大模型测试时 linker 会报 `section '.data' can't be allocated in segment 1` 警告，但不影响运行

## 10. 后续工作

1. **DEPTHWISE_CONV_2D hybrid 支持**：类似 CONV_2D 的 im2col + matmul 方案
2. **其他算子 hybrid 支持**：如 TRANSPOSE_CONV 等
3. **per-batch 动态量化**：当前是 per-tensor 量化整个输入，可以改为 per-batch 提高精度
4. **真实硬件验证**：在实际 Cortex-M55 开发板上验证（当前仅 FVP）
5. **上游贡献**：整理代码风格后向 tflite-micro 上游提交 PR
6. **Benchmark 工具集成**：将 hybrid 模型性能测试集成到 TFLM 的 benchmark 框架中
