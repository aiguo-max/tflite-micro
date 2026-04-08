// Sleep model TFLM test for QEMU (embedded data, no filesystem).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/examples/sleep_test/sleep_model_data.h"
#include "tensorflow/lite/micro/examples/sleep_test/sleep_test_data.h"
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

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_sleep_model);
  if (!model) {
    printf("GetModel FAILED\n");
    return 1;
  }
  printf("Model version: %lu, opcodes: %lu, subgraphs: %lu\n",
         static_cast<unsigned long>(model->version()),
         static_cast<unsigned long>(model->operator_codes()->size()),
         static_cast<unsigned long>(model->subgraphs()->size()));

#if defined(__NuttX__)
  uint8_t* arena_raw = static_cast<uint8_t*>(malloc(kArenaSize + 15));
  if (!arena_raw) {
    printf("malloc arena (%zu) FAILED\n", kArenaSize);
    return 1;
  }
  uint8_t* tensor_arena = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(arena_raw) + 15) & ~static_cast<uintptr_t>(15));
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

  tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors FAILED\n");
    return 1;
  }
  printf("AllocateTensors OK  arena=%d/%d\n", (int)interp.arena_used_bytes(),
         (int)kArenaSize);

  TfLiteTensor* input = interp.input_tensor(0);
  if (!input || input->bytes != g_sleep_input_len) {
    printf("input mismatch: got=%d expected=%u\n",
           input ? (int)input->bytes : -1, g_sleep_input_len);
    return 1;
  }
  memcpy(input->data.raw, g_sleep_input, input->bytes);

  printf("Running inference...\n");
  if (interp.Invoke() != kTfLiteOk) {
    printf("Invoke FAILED\n");
    return 1;
  }
  printf("Invoke OK\n");

  const TfLiteTensor* output = interp.output_tensor(0);
  if (output->bytes != g_sleep_ref_len) {
    printf("output size mismatch: got=%d expected=%u\n", (int)output->bytes,
           g_sleep_ref_len);
    return 1;
  }

  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(g_sleep_ref);
  const int n = output->bytes / sizeof(float);

  float max_diff = 0, sum_diff = 0;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
    sum_diff += d;
  }

  printf("max_diff=%.6e mean_diff=%.6e\n", static_cast<double>(max_diff),
         static_cast<double>(sum_diff / n));
  printf("out[0:4]: %.6f %.6f %.6f %.6f\n", static_cast<double>(got[0]),
         static_cast<double>(got[1]), static_cast<double>(got[2]),
         static_cast<double>(got[3]));
  printf("ref[0:4]: %.6f %.6f %.6f %.6f\n", static_cast<double>(ref[0]),
         static_cast<double>(ref[1]), static_cast<double>(ref[2]),
         static_cast<double>(ref[3]));

  constexpr float kTolerance = 1e-2f;
  if (max_diff < kTolerance) {
    printf("~~~ALL TESTS PASSED~~~\n");
    free(arena_raw);
    return 0;
  } else {
    printf("FAIL (max_diff=%.6e > %.6e)\n", static_cast<double>(max_diff),
           static_cast<double>(kTolerance));
    free(arena_raw);
    return 1;
  }
}
