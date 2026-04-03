// Sleep model TFLM test for QEMU (embedded data, no filesystem).

#include <cmath>
#include <cstdio>
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
constexpr size_t kArenaSize = 2 * 1024 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_sleep_model);
  if (!model) {
    printf("GetModel FAILED\n");
    return 1;
  }

  // Create minimal op resolver with only the ops needed for sleep model
  tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddElu();

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

  if (max_diff < 1e-4f) {
    printf("~~~ALL TESTS PASSED~~~\n");
    return 0;
  } else {
    printf("FAIL (max_diff=%.6e > 1e-4)\n", static_cast<double>(max_diff));
    return 1;
  }
}
