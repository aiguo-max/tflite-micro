# TFLM Hybrid Inference Verification Guide

Reusable step-by-step guide for verifying hybrid inference
(float32 activations + int8 weights) correctness on two platforms:

- **x86-64** — fast reference baseline using the host TFLM build
- **Corstone-300 FVP** — functional verification on a real Cortex-M55 core
  with CMSIS-NN MVE acceleration

---

## Prerequisites

### Toolchain & FVP

| Item | Path / Command |
|------|---------------|
| ARM GCC cross-compiler | auto-downloaded by `make` into `tools/make/downloads/gcc_embedded/` |
| Corstone-300 FVP binary | `/mnt/data2/FVP_Corstone_SSE-300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55` |
| FVP must be on `PATH` | `export PATH=/mnt/data2/FVP_Corstone_SSE-300/models/Linux64_GCC-9.3:$PATH` |

Check FVP is reachable:

```bash
FVP_Corstone_SSE-300_Ethos-U55 --version
```

### Working directory

All commands below are run from the repo root:

```bash
cd /home/xielin3/data/opensources/tflite-micro
```

---

## Step 1 — Prepare Input and Golden Reference Output

Choose a fixed random input so results are reproducible across runs.

```python
# gen_test_input.py
import numpy as np
np.random.seed(42)
inp = np.random.randn(1, 1, 256).astype(np.float32)   # adjust shape to match your model
inp.tofile("/tmp/hybrid_test_input.bin")
print("input shape:", inp.shape, "bytes:", inp.nbytes)
```

```bash
python3 gen_test_input.py
```

Build the **reference** `tflm_infer` tool (x86-64, **no** CMSIS-NN) and run it
to produce the golden output:

```bash
# Build reference lib (x86-64, reference kernels)
make -f tensorflow/lite/micro/tools/make/Makefile -j$(nproc) microlite

# Link tflm_infer against reference lib
g++ -O2 -std=c++17 \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON \
  -I. \
  -Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -Itensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -Itensorflow/lite/micro/tools/make/downloads/ruy \
  -Itensorflow/lite/micro/tools/make/downloads/kissfft \
  tensorflow/lite/micro/tools/benchmarking/tflm_infer.cc \
  gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a \
  -o /tmp/tflm_infer_ref \
  -ldl -lpthread -lm

# Produce golden output
/tmp/tflm_infer_ref <model.tflite> /tmp/hybrid_test_input.bin /tmp/hybrid_golden_output.bin
```

> **Why reference (not CMSIS-NN) for golden?**
> The reference kernels use plain C arithmetic — no quantization approximations.
> All CMSIS-NN and FVP results are compared against this baseline.

---

## Step 2 — x86-64 CMSIS-NN Verification

Build the CMSIS-NN lib and `tflm_infer` for x86-64 and compare against golden.

```bash
# Build CMSIS-NN lib
make -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=cmsis_nn -j$(nproc) microlite

# Link tflm_infer against CMSIS-NN lib
g++ -O2 -std=c++17 \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DCMSIS_NN \
  -I. \
  -Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -Itensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -Itensorflow/lite/micro/tools/make/downloads/ruy \
  -Itensorflow/lite/micro/tools/make/downloads/kissfft \
  -Itensorflow/lite/micro/tools/make/downloads/cmsis_nn \
  -Itensorflow/lite/micro/tools/make/downloads/cmsis_nn/Include \
  tensorflow/lite/micro/tools/benchmarking/tflm_infer.cc \
  gen/linux_x86_64_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a \
  -o /tmp/tflm_infer_cmsis \
  -ldl -lpthread -lm

# Run CMSIS-NN inference
/tmp/tflm_infer_cmsis <model.tflite> /tmp/hybrid_test_input.bin /tmp/hybrid_cmsis_output.bin

# Compare outputs
python3 - <<'EOF'
import numpy as np
ref   = np.frombuffer(open("/tmp/hybrid_golden_output.bin", "rb").read(), dtype=np.float32)
got   = np.frombuffer(open("/tmp/hybrid_cmsis_output.bin", "rb").read(), dtype=np.float32)
diff  = np.abs(ref - got)
print(f"elements  : {len(ref)}")
print(f"max diff  : {diff.max():.8f}")
print(f"mean diff : {diff.mean():.8f}")
print(f"bit-exact : {np.array_equal(ref, got)}")
print("PASS" if diff.max() < 1e-4 else "FAIL")
EOF
```

> **Note:** On x86-64, `arm_nn_mat_mult_nt_t_s8_s32` runs the **scalar fallback**
> path — no MVE instructions.  This step validates functional correctness of the
> hybrid code path; MVE acceleration is validated in Step 3.

---

## Step 3 — Corstone-300 FVP Verification (Cortex-M55 + MVE)

### 3a. Generate the C test-data header

The FVP test embeds the model and I/O directly in flash as C arrays.
Use the following script to convert `.tflite`, `input.bin`, `golden_output.bin`
into a header file:

```python
# gen_test_data_header.py
import sys

def to_c_array(name, size_name, data, section):
    lines = [f"constexpr unsigned int {size_name} = {len(data)};"]
    lines.append(
        f'const unsigned char {name}[] '
        f'__attribute__((aligned(16), section("{section}"))) = {{'
    )
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines[-1] = lines[-1].rstrip(",")   # remove trailing comma on last row
    lines.append("};")
    return "\n".join(lines)

prefix   = sys.argv[1]          # e.g. "my_model"
model    = open(sys.argv[2], "rb").read()
inp      = open(sys.argv[3], "rb").read()
out      = open(sys.argv[4], "rb").read()
dst      = sys.argv[5]          # output header path

guard = f"{prefix.upper()}_TEST_DATA_H_"
with open(dst, "w") as f:
    f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <cstdint>\n\n")
    f.write(to_c_array(f"g_{prefix}_model_data",      f"g_{prefix}_model_data_size",
                       model, f".rodata.g_{prefix}_model_data"))
    f.write("\n\n")
    f.write(to_c_array(f"g_{prefix}_input_data",      f"g_{prefix}_input_data_size",
                       inp,   f".rodata.g_{prefix}_input_data"))
    f.write("\n\n")
    f.write(to_c_array(f"g_{prefix}_ref_output_data", f"g_{prefix}_ref_output_data_size",
                       out,   f".rodata.g_{prefix}_ref_output_data"))
    f.write(f"\n\n#endif  // {guard}\n")

print(f"Written {dst}  (model={len(model)}B  input={len(inp)}B  output={len(out)}B)")
```

```bash
python3 gen_test_data_header.py \
  my_model \
  path/to/model.tflite \
  /tmp/hybrid_test_input.bin \
  /tmp/hybrid_golden_output.bin \
  tensorflow/lite/micro/kernels/my_model_test_data.h
```

### 3b. Write the test source file

Create `tensorflow/lite/micro/kernels/my_model_test.cc`:

```cpp
#include <cmath>
#include <cstring>

#include "tensorflow/lite/micro/kernels/my_model_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kTensorArenaSize = 256 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MyModelHybridInference) {
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_my_model_model_data);
  TF_LITE_MICRO_EXPECT(model != nullptr);

  tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddAdd();
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddExpandDims();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddPad();
  resolver.AddReshape();
  resolver.AddTranspose();

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input_tensor(0);
  TF_LITE_MICRO_EXPECT_EQ(input->bytes,
                           static_cast<size_t>(g_my_model_input_data_size));
  memcpy(input->data.raw, g_my_model_input_data, input->bytes);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  TfLiteTensor* output = interpreter.output_tensor(0);
  TF_LITE_MICRO_EXPECT_EQ(output->bytes,
                           static_cast<size_t>(g_my_model_ref_output_data_size));

  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(g_my_model_ref_output_data);
  const int n = output->bytes / sizeof(float);

  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
  }
  MicroPrintf("max_abs_diff=%.6f", static_cast<double>(max_diff));
  TF_LITE_MICRO_EXPECT_LE(max_diff, 1e-4f);
}

TF_LITE_MICRO_TESTS_END
```

### 3c. Register the test in the build system

Add one line to `tensorflow/lite/micro/kernels/Makefile.inc`
(alphabetical order, inside the `*_TEST_SRCS` block):

```makefile
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/my_model_test.cc \
```

### 3d. Build and run on FVP

```bash
# Ensure FVP is on PATH
export PATH=/mnt/data2/FVP_Corstone_SSE-300/models/Linux64_GCC-9.3:$PATH

make -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=cmsis_nn \
  TARGET=cortex_m_corstone_300 \
  TARGET_ARCH=cortex-m55 \
  test_kernel_my_model_test
```

The make target name follows the pattern:
`test_kernel_<filename_without_.cc>`

Expected output:

```
Testing MyModelHybridInference
max_abs_diff=0.000000
1/1 tests passed
~~~ALL TESTS PASSED~~~
Application exit code: 0.
```

---

## Step 4 — Interpreting Results

| max_abs_diff | Meaning |
|-------------|---------|
| `0.000000` | Bit-exact — CMSIS-NN MVE path produces identical results to reference C |
| `< 1e-4` | Within tolerance — acceptable for hybrid symmetric quantization |
| `>= 1e-4` | **FAIL** — investigate scratch buffer sizing, memset in Init(), or type check in Eval |

---

## Troubleshooting

**`Hybrid models are not supported on TFLite Micro.` in Eval**

The CMSIS-NN `Eval()` function has a type check separate from `Prepare()`.
Both must allow `(input==float32, filter==int8)`.
Search for `"Hybrid models are not supported"` in `cmsis_nn/conv.cc` and
`cmsis_nn/fully_connected.cc` — there are typically two occurrences per file
(one in Prepare, one in Eval).

**MVE not active — `arm_nn_mat_mult_nt_t_s8_s32` runs scalar fallback**

GCC only defines `__ARM_FEATURE_MVE` when `-mcpu=cortex-m55` is passed, not
with `-march=armv8.1-m.main+mve.fp` alone. For TFLM Makefile builds the
`TARGET_ARCH=cortex-m55` flag handles this. For CMake-based builds (e.g. trunk
repos), add `-mcpu=cortex-m55` explicitly to the CMSIS-NN target:

```cmake
target_compile_options(cmsis_nn PRIVATE -mcpu=cortex-m55)
```

**Stale `.o` files after header changes**

TFLM Makefile has no header dependency tracking. After modifying `conv.h` or
`fully_connected.h`, clean kernel objects before rebuilding:

```bash
rm -rf gen/linux_x86_64_*/obj/kernels
rm -rf gen/cortex_m_corstone_300_*/obj/kernels
```

**Arena too small**

If `AllocateTensors()` fails, increase `kTensorArenaSize` in the test file.
256 KB covers the MiHr_pnet models; larger models may need more.

---

## Existing Tests in This Repo

| Test source | Model | What it verifies |
|-------------|-------|-----------------|
| `kernels/hybrid_model_test.cc` | `MiHr_pnet_aiq_dynamic_skip_stage0_conv` | original hybrid port, reference vs CMSIS-NN |
| `kernels/skip_s0_conv1_test.cc` | `MiHr_pnet_aiq_dynamic_skip_s0_conv1` | s0 conv1 model, CMSIS-NN MVE on Cortex-M55 FVP |

Both tests use `seed=42` random float32 input and compare against reference TFLM output.

---

## Quick Reference — Build Commands

```bash
# x86-64 reference lib
make -f tensorflow/lite/micro/tools/make/Makefile -j$(nproc) microlite

# x86-64 CMSIS-NN lib
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn -j$(nproc) microlite

# Cortex-M55 FVP — run a single test
export PATH=/mnt/data2/FVP_Corstone_SSE-300/models/Linux64_GCC-9.3:$PATH
make -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=cmsis_nn \
  TARGET=cortex_m_corstone_300 \
  TARGET_ARCH=cortex-m55 \
  test_kernel_<test_name>

# Clean stale kernel objects (after header changes)
rm -rf gen/linux_x86_64_*/obj/kernels
rm -rf gen/cortex_m_corstone_300_*/obj/kernels
```
