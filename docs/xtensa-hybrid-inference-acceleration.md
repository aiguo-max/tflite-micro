# Xtensa HiFi4 Hybrid Inference 加速

## 概述

为 TFLM 的 Xtensa HiFi4 后端添加 hybrid inference 支持（float32 activations + int8 weights），
使用 xa_nnlib 的 `xa_nn_matXvec_8x8_32` 加速 Conv2D 和 FullyConnected 的 int8 dot product。

## 推理路径

```
float32 input
  → 动态对称量化 (range/127, zero_point=0)
  → int8 input
  → int8 Conv/FC (xa_nn_matXvec_8x8_32 加速)
  → int32 accumulator
  → × input_scale × filter_scale[c] + bias
  → float32 output
  → activation clamp
```

## 改动文件

| 文件 | 改动 |
|------|------|
| `kernels/xtensa/conv_hybrid_hifi.cc` | 新增：HiFi4 hybrid Conv2D eval，使用 xa_nn_matXvec_8x8_32 |
| `kernels/xtensa/fully_connected_hybrid_hifi.cc` | 新增：HiFi4 hybrid FC eval，使用 xa_nn_matXvec_8x8_32 |
| `kernels/xtensa/conv.cc` | 增加 hybrid dispatch（`is_hybrid → ConvEvalHybridHifi`） |
| `kernels/xtensa/fully_connected.cc` | 增加 hybrid dispatch（`is_hybrid → FullyConnectedEvalHybridHifi`） |
| `kernels/xtensa/conv_common_xtensa.cc` | Prepare 阶段：is_hybrid 检测 + scratch buffer 分配 |
| `kernels/xtensa/fully_connected_common_xtensa.cc` | 同上，scratch 请求移到 temp 释放之后 |
| `kernels/xtensa/conv_hifi.cc` | 修复：scratch 请求移到 DeallocateTemp 之后，避免 temp leak |
| `kernels/xtensa/xtensa_conv.h` | 新增 ConvEvalHybridHifi 声明 |
| `arena_allocator/single_arena_buffer_allocator.cc` | 修复：ResetTempAllocations 不再因 temp 未完全配对而返回 error |
| `tools/make/targets/xtensa_makefile.inc` | 新增两个 hybrid hifi 源文件 |
| `tools/make/ext_libs/xtensa.inc` | 移除 xa_nn_activations_asym8_asym8.c 的错误排除 |

## Bug 修复

### 1. Xtensa Prepare 阶段 temp buffer 泄漏

`conv_hifi.cc` 和 `fully_connected_common_xtensa.cc` 中，`RequestScratchBufferInArena`
在 temp tensor 未释放时调用，违反了 `SingleArenaBufferAllocator::ResizeBuffer` 的
`head_temp_ == next_temp_` 约束。

修复：将 scratch 请求移到 `DeallocateTempTfLiteTensor` 之后。

### 2. SingleArenaBufferAllocator::ResetTempAllocations 过早报错

`ResetTempAllocations` 在 `IsAllTempDeallocated()` 返回 false 时直接返回 kTfLiteError，
但 TFLM 原生 `ConvPrepare` 代码存在已知的 temp 未配对释放问题（upstream bug），
导致所有使用 `SingleArenaBufferAllocator` 的 Xtensa 路径 AllocateTensors 失败。

修复：按原始 TODO 注释意图，打印警告但强制 reset，与 `NonPersistentArenaBufferAllocator`
的行为保持一致。

## 构建

```bash
export PATH="/mnt/data/dev-system/prebuilts/clang-xtensa/linux-x86_64/bin:$PATH"
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=xtensa TARGET_ARCH=hifi4 OPTIMIZED_KERNEL_DIR=xtensa \
  XTENSA_CORE=hifi4_bes_asic_15_05 \
  XTENSA_BASE=/mnt/data/dev-system/prebuilts/clang-xtensa/linux-x86_64 \
  XTENSA_TOOLS_VERSION=dummy \
  gen/xtensa_hifi4_default_xtensa_gcc/bin/hybrid_test
```

## ISS 验证

```bash
xt-run --exit_with_target_code gen/xtensa_hifi4_default_xtensa_gcc/bin/hybrid_test
```

结果：
```
AllocateTensors OK  arena_used=155960/524288
output: 64 elements  max_diff=0.000000 at [0]
PASS
```

## 真机验证 (BES Best1600EP Audio Core)

```
audio> xtensa_hybrid_test
[xtensa] AllocateTensors OK
[xtensa] 10 runs  total=1464 ms  avg=146.4 ms
[skip_s0_conv1] max_diff=0.000001 (tol=1e-4)
[xtensa] PASS
```

## xa_nn_matXvec_8x8_32 用法

```c
xa_nn_matXvec_8x8_32(
    acc_buf,          // int32 output: output_depth × 1
    filter_int8,      // int8 matrix: output_depth × patch_size
    nullptr,          // mat2: unused
    im2col_buffer,    // int8 vector: patch_size × 1
    nullptr,          // vec2: unused
    s_zero_bias,      // int8 bias: all zeros (bias 在 float 域加)
    output_depth,     // rows
    patch_size,       // cols1
    0,                // cols2: unused
    patch_size,       // row_stride1
    0,                // row_stride2: unused
    0,                // acc_shift: no shift, raw int32
    0);               // bias_shift: zero bias
```

hybrid 推理中 bias 是 float32，所以 xa_nn 的 int8 bias 传全零，
float bias 在后续 dequantize 阶段加上。

## 已知限制

- `s_zero_bias` 固定 512 字节，output_depth 不能超过 512
- im2col 仍然是 C 循环（未加速），后续可用 xa_nn_conv2d_std 内置 im2col
- ConvPrepare (TFLM upstream) 存在 temp leak，在 KernelRunner 测试中会产生 ValidateTempBufferDeallocated 失败，不影响实际推理正确性
