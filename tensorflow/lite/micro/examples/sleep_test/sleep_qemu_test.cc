// Sleep model TFLM test — embedded float model + file-based hybrid model.
// On NuttX: input/ref from /data/, hybrid model also from /data/.
// On ISS: everything embedded via headers.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__NuttX__)
#include <time.h>
#endif

#include "tensorflow/lite/micro/examples/sleep_test/sleep_model_data.h"
#if !defined(__NuttX__)
#include "tensorflow/lite/micro/examples/sleep_test/sleep_test_data.h"
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
extern TFLMRegistration Register_BIDIRECTIONAL_SEQUENCE_GRU();
extern TFLMRegistration Register_LAYER_NORM();
}  // namespace tflite

constexpr size_t kArenaSize = 1280 * 1024;
#if !defined(__NuttX__)
alignas(16) static uint8_t g_arena[kArenaSize];
#endif

#if defined(__NuttX__)
static uint8_t* ReadFile(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("Cannot open %s\n", path); return nullptr; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* buf = static_cast<uint8_t*>(malloc(sz));
  if (!buf) { printf("malloc(%ld) failed for %s\n", sz, path); fclose(f); return nullptr; }
  size_t rd = fread(buf, 1, sz, f);
  fclose(f);
  if (rd != static_cast<size_t>(sz)) {
    printf("Read %zu/%ld from %s\n", rd, sz, path);
    free(buf); return nullptr;
  }
  *out_size = static_cast<size_t>(sz);
  return buf;
}

static int64_t now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}
#endif

static int RunModel(const char* tag, const unsigned char* model_data,
                    const uint8_t* input_data, size_t input_size,
                    const uint8_t* ref_data, size_t ref_size,
                    float tolerance) {
#if defined(__NuttX__)
  int64_t t_start = now_us();
#endif

  const tflite::Model* model = tflite::GetModel(model_data);
  if (!model) { printf("[%s] GetModel FAILED\n", tag); return 1; }

#if defined(__NuttX__)
  int64_t t_getmodel = now_us();
  uint8_t* arena_raw = static_cast<uint8_t*>(malloc(kArenaSize + 15));
  if (!arena_raw) { printf("[%s] malloc arena FAILED\n", tag); return 1; }
  uint8_t* tensor_arena = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(arena_raw) + 15) & ~static_cast<uintptr_t>(15));
  int64_t t_arena = now_us();
#else
  uint8_t* arena_raw = nullptr;
  uint8_t* tensor_arena = g_arena;
#endif

  tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddElu();
  resolver.AddSoftmax();
  TFLMRegistration bigru_reg = tflite::Register_BIDIRECTIONAL_SEQUENCE_GRU();
  resolver.AddCustom("BIDIRECTIONAL_SEQUENCE_GRU", &bigru_reg);
  TFLMRegistration ln_reg = tflite::Register_LAYER_NORM();
  resolver.AddCustom("LAYER_NORM", &ln_reg);

#if defined(__NuttX__)
  int64_t t_resolver = now_us();
#endif

  tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);

#if defined(__NuttX__)
  int64_t t_interp = now_us();
#endif

  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("[%s] AllocateTensors FAILED\n", tag);
    free(arena_raw); return 1;
  }

#if defined(__NuttX__)
  int64_t t_alloc = now_us();
#endif
  printf("[%s] AllocateTensors OK  arena=%d/%d\n", tag,
         (int)interp.arena_used_bytes(), (int)kArenaSize);

  TfLiteTensor* input = interp.input_tensor(0);
  if (!input || input->bytes != input_size) {
    printf("[%s] input mismatch: got=%d expected=%zu\n", tag,
           input ? (int)input->bytes : -1, input_size);
    free(arena_raw); return 1;
  }
  memcpy(input->data.raw, input_data, input->bytes);

#if defined(__NuttX__)
  int64_t t_memcpy = now_us();
#endif

  if (interp.Invoke() != kTfLiteOk) {
    printf("[%s] Invoke FAILED\n", tag);
    free(arena_raw); return 1;
  }

#if defined(__NuttX__)
  int64_t t_invoke = now_us();
#endif

  const TfLiteTensor* output = interp.output_tensor(0);
  if (output->bytes != ref_size) {
    printf("[%s] output size mismatch: got=%d expected=%zu\n", tag,
           (int)output->bytes, ref_size);
    free(arena_raw); return 1;
  }

  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(ref_data);
  const int n = output->bytes / sizeof(float);

  float max_diff = 0, sum_diff = 0;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
    sum_diff += d;
  }

#if defined(__NuttX__)
  int64_t t_compare = now_us();
#endif

  printf("[%s] max_diff=%.6e mean_diff=%.6e\n", tag,
         static_cast<double>(max_diff), static_cast<double>(sum_diff / n));
  printf("[%s] out[0:4]: %.6f %.6f %.6f %.6f\n", tag,
         static_cast<double>(got[0]), static_cast<double>(got[1]),
         static_cast<double>(got[2]), static_cast<double>(got[3]));

  free(arena_raw);

#if defined(__NuttX__)
  int64_t t_end = now_us();

  // Phase timing summary (all in ms)
  printf("\n[%s] === Profile Summary ===\n", tag);
  printf("[%s]   GetModel:         %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_getmodel - t_start), (double)(t_getmodel - t_start) / 1000.0);
  printf("[%s]   Arena malloc:     %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_arena - t_getmodel), (double)(t_arena - t_getmodel) / 1000.0);
  printf("[%s]   OpResolver:       %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_resolver - t_arena), (double)(t_resolver - t_arena) / 1000.0);
  printf("[%s]   InterpreterCtor:  %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_interp - t_resolver), (double)(t_interp - t_resolver) / 1000.0);
  printf("[%s]   AllocateTensors:  %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_alloc - t_interp), (double)(t_alloc - t_interp) / 1000.0);
  printf("[%s]   Input memcpy:     %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_memcpy - t_alloc), (double)(t_memcpy - t_alloc) / 1000.0);
  printf("[%s]   Invoke:           %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_invoke - t_memcpy), (double)(t_invoke - t_memcpy) / 1000.0);
  printf("[%s]   Output compare:   %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_compare - t_invoke), (double)(t_compare - t_invoke) / 1000.0);
  printf("[%s]   Cleanup/free:     %6lld us  (%5.1f ms)\n", tag,
         (long long)(t_end - t_compare), (double)(t_end - t_compare) / 1000.0);
  printf("[%s]   -----------------------------------\n", tag);
  printf("[%s]   TOTAL (RunModel):  %5.1f ms\n", tag,
         (double)(t_end - t_start) / 1000.0);
  printf("[%s] ========================\n", tag);
#endif

  if (max_diff < tolerance) {
    printf("[%s] PASS\n", tag);
    return 0;
  } else {
    printf("[%s] FAIL (max_diff=%.6e > %.6e)\n", tag,
           static_cast<double>(max_diff), static_cast<double>(tolerance));
    return 1;
  }
}

extern "C" int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  int failures = 0;

#if defined(__NuttX__)
  int64_t t_main_start = now_us();

  size_t input_size = 0;
  uint8_t* input_buf = ReadFile("/data/sleep_input.bin", &input_size);
  if (!input_buf) return 1;

  size_t ref_size = 0;
  uint8_t* ref_buf = ReadFile("/data/sleep_ref.bin", &ref_size);
  if (!ref_buf) { free(input_buf); return 1; }

  int64_t t_fileio_float = now_us();
  printf("File I/O (input+ref): %.1f ms\n",
         (double)(t_fileio_float - t_main_start) / 1000.0);

  printf("=== Test 1: float model (embedded) ===\n");
  failures += RunModel("float", g_sleep_model, input_buf, input_size,
                        ref_buf, ref_size, 1e-4f);
  free(ref_buf);

  int64_t t_after_float = now_us();

  size_t qmodel_size = 0;
  uint8_t* qmodel_buf = ReadFile("/data/sleep_quant_model.tflite", &qmodel_size);
  if (qmodel_buf) {
    size_t qref_size = 0;
    uint8_t* qref_buf = ReadFile("/data/sleep_quant_ref.bin", &qref_size);
    if (qref_buf) {
      int64_t t_fileio_hybrid = now_us();
      printf("File I/O (hybrid model+ref): %.1f ms\n",
             (double)(t_fileio_hybrid - t_after_float) / 1000.0);

      printf("=== Test 2: hybrid int8 model (from file) ===\n");
      failures += RunModel("hybrid", qmodel_buf, input_buf, input_size,
                            qref_buf, qref_size, 1e-2f);
      free(qref_buf);
    }
    free(qmodel_buf);
  } else {
    printf("No hybrid model at /data/sleep_quant_model.tflite, skipping\n");
  }

  free(input_buf);

  int64_t t_main_end = now_us();
  printf("\n=== Overall Profile ===\n");
  printf("  Total wall time: %.1f ms\n",
         (double)(t_main_end - t_main_start) / 1000.0);
  printf("=======================\n");
#else
  printf("=== Test 1: float model (embedded) ===\n");
  failures += RunModel("float", g_sleep_model,
                        g_sleep_input, g_sleep_input_len,
                        g_sleep_ref, g_sleep_ref_len, 1e-2f);
#endif

  if (failures == 0) {
    printf("~~~ALL TESTS PASSED~~~\n");
    return 0;
  }
  printf("%d test(s) FAILED\n", failures);
  return 1;
}
