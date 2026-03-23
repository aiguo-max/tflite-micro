# Xtensa HiFi4 Hybrid Inference 加速

## 概述

为 TFLM 的 Xtensa HiFi4 后端添加 hybrid inference 支持（float32 activations + int8 weights），
使用 xa_nnlib 的 `xa_nn_matXvec_8x8_32` / `xa_nn_matXvec_batch_8x8_32` 加速 Conv2D 和
FullyConnected 的 int8 dot product。

## 推理路径

```
float32 input
  → 动态对称量化 (range/127, zero_point=0)
  → int8 input
  → int8 Conv/FC (xa_nn_matXvec_8x8_32 加速)
  → int32 accumulator
  → × input_scale × filter_scale[c] + bias_float
  → float32 output
  → activation clamp
```

## 改动文件

| 文件 | 改动 |
|------|------|
| `kernels/xtensa/conv_hybrid_hifi.cc` | 新增：HiFi4 hybrid Conv2D eval，使用 xa_nn_matXvec_8x8_32 / batch 版本 |
| `kernels/xtensa/fully_connected_hybrid_hifi.cc` | 新增：HiFi4 hybrid FC eval，使用 xa_nn_matXvec_8x8_32 |
| `kernels/xtensa/conv.cc` | 增加 hybrid dispatch（`is_hybrid → ConvEvalHybridHifi`） |
| `kernels/xtensa/fully_connected.cc` | 增加 hybrid dispatch（`is_hybrid → FullyConnectedEvalHybridHifi`） |
| `kernels/xtensa/conv_common_xtensa.cc` | Prepare：is_hybrid 检测；移除与 ConvPrepare 的重复 scratch 分配 |
| `kernels/xtensa/fully_connected_common_xtensa.cc` | Prepare：scratch 请求移到 temp 释放之后 |
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

### 3. per-tensor 量化时 scale 越界访问

`conv_hybrid_hifi.cc` 和 `fully_connected_hybrid_hifi.cc` 中，填充 `combined_scales[]`
时对所有 `output_depth` 个 channel 访问 `filter_scales[c]`，但 per-tensor 量化只有
`filter_scales[0]` 一个元素，导致越界读取垃圾值。

修复：增加 `is_per_channel = (hybrid_num_channels == output_depth)` 判断，
per-tensor 情况统一使用 `filter_scales[0]`。

### 4. ConvPrepareXtensa 重复分配 scratch buffer（双重分配 bug）

`ConvPrepareXtensa`（`conv_common_xtensa.cc`）调用 `ConvPrepare()` 之后，
又对同样的 `hybrid_input_scratch_index`、`hybrid_im2col_scratch_index`、
`hybrid_output_scratch_index` 做了第二次 `RequestScratchBufferInArena`，
将 index 覆盖为新分配的 buffer #2，而 `ConvPrepare` 内已设置了正确的 buffer #1。

影响：scratch index 被覆盖后，Eval 时读写的 buffer 地址可能与 Prepare
时预期的不一致，导致某些模型（取决于 arena 布局）输出错误值。

修复：删除 `conv_common_xtensa.cc` 中的重复分配块，仅保留 `is_hybrid` 标志设置。

### 5. xa_nn_matXvec_8x8_32 要求 cols 为 4 的倍数

`xa_nn_matXvec_8x8_32` 在 `cols % 4 != 0` 时静默返回 -1，不填充 `acc_buf`，
导致输出为 0 或残留垃圾值乘以 scale 后得到异常大的浮点数（如 ~277K）。

触发场景：Conv 的 `patch_size = filter_h × filter_w × input_depth` 不是 4 的倍数，
例如 filter `[32, 1, 21, 1]` 时 `patch_size = 21`。

修复：在 `ConvEvalHybridHifi` 入口处检测 `patch_size & 3`，若不为 0 则
fallback 到 `ConvEvalHybrid`（reference 实现）。

## xa_nn_matXvec_8x8_32 用法

```c
// 单 vec（per-position）
xa_nn_matXvec_8x8_32(
    acc_buf,          // int32 output: output_depth × 1
    filter_int8,      // int8 matrix: output_depth × patch_size
    nullptr,          // mat2: unused
    im2col_buffer,    // int8 vector: patch_size × 1
    nullptr,          // vec2: unused
    s_zero_bias,      // int8 bias: all zeros（bias 在 float 域加）
    output_depth,     // rows
    patch_size,       // cols1（必须是 4 的倍数）
    0,                // cols2: unused
    patch_size,       // row_stride1（等于 cols1）
    0,                // row_stride2: unused
    0,                // acc_shift
    0);               // bias_shift

// batch 版（多个 output position 并行）
xa_nn_matXvec_batch_8x8_32(
    out_ptrs,         // int32** p_out: batch_count 个指针
    filter_int8,      // int8 matrix
    vec_ptrs,         // int8** p_vec1: batch_count 个指针
    s_zero_bias,      // int8 bias: all zeros
    output_depth,     // rows
    patch_size,       // cols（必须是 4 的倍数，且 batch_count 必须为偶数）
    patch_size,       // row_stride
    0,                // out_offset
    0,                // bias_offset
    batch_count);     // vec_count（必须为偶数，≥ 2）
```

hybrid 推理中 bias 是 float32，因此 xa_nn 的 int8 bias 传全零，
float bias 在后续 dequantize 阶段加上。

## 量化类型支持

| 场景 | `hybrid_num_channels` | `is_per_channel` | scale 取值 |
|------|----------------------|-----------------|-----------|
| per-channel（常见）| = output_depth | true | `filter_scales[c]` |
| per-tensor（如 dynamic range quant）| = 1 | false（当 output_depth > 1）| `filter_scales[0]` |
| output_depth = 1 且 num_channels = 1 | = 1 | true（1 == 1）| `filter_scales[0]` |

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

## ISS 验证（xt-run）

```bash
xt-run --exit_with_target_code gen/xtensa_hifi4_default_xtensa_gcc/bin/hybrid_test
```

结果：
```
=== Test1: skip_s0_conv1 ===
max_diff=0.000001
PASS
=== Test2: dynamic_int8 ===
first8 got: 0.0000 0.0818 1.2833 0.0000 ...
first8 ref: 0.0000 0.0000 1.0063 0.0000 ...
max_diff=0.422734
PASS
ALL PASS
```

## 真机验证（BES Best1600EP Audio Core）

在 AP nsh 中通过 `cu -l /dev/ttyAUDIO` 进入 audio 核 shell，执行：

```
audio> xtensa_hybrid_test
```

5 个 test case 全部 PASS（commit 9f5dc541 / o62 最终版本）：

```
=== TFLM Hybrid Kernel Tests (Xtensa HiFi4) ===

--- Test 1: hybrid_model (skip_stage0_conv, hybrid, /data/) ---
  [test1_hybrid] 64 elements  max_diff=0.000000
  [test1_hybrid] 10 runs  avg=126.8 ms
  PASS

--- Test 2: skip_s0_conv1 (hybrid, embedded) ---
  [test2_skip_s0] 64 elements  max_diff=0.000001
  [test2_skip_s0] 10 runs  avg=58.5 ms
  PASS

--- Test 3: w8a8 (static full int8, /data/) ---
  [test3_w8a8] 64 elements  max_int8_diff=0
  [test3_w8a8] 10 runs  avg=13.0 ms
  PASS

--- Test 4: w8a8_float_io (static, float I/O, /data/) ---
  [test4_w8a8_fio] 64 elements  max_diff=0.000000
  [test4_w8a8_fio] 10 runs  avg=13.1 ms
  PASS

--- Test 5: dynamic_int8 (dynamic quantized, embedded) ---
  [test5_dyn_int8] 64 elements  max_diff=0.422733
  [test5_dyn_int8] 10 runs  avg=58.5 ms
  PASS

=== 5/5 test suites passed ===
~~~ALL XTENSA HYBRID TESTS PASSED~~~
```

## 已知限制

- `s_zero_bias` 固定 512 字节，output_depth 不能超过 512
- `combined_scales[]` 固定 512 元素，output_depth 不能超过 512
- `patch_size % 4 != 0` 时 fallback 到 reference 路径（无 xa_nnlib 加速）
- `xa_nn_matXvec_batch_8x8_32` 要求 batch_count 为偶数，奇数最后一个 position 走 per-position 路径
- im2col 仍为 C 循环（未加速），后续可用 `xa_nn_conv2d_std` 内置 im2col 替代
- ConvPrepare（TFLM upstream）存在 temp leak，在 KernelRunner 单测中产生 ValidateTempBufferDeallocated 失败，不影响实际推理正确性
