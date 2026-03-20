/* Xtensa HiFi4 hybrid inference test
 * Validates skip_s0_conv1 (hybrid: float32 activations + int8 weights)
 * on ISS via xt-run.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/examples/hybrid_test/skip_s0_conv1_model_data.h"
#include "tensorflow/lite/micro/examples/hybrid_test/skip_s0_conv1_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kArenaSize = 512 * 1024;

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  uint8_t* arena = static_cast<uint8_t*>(malloc(kArenaSize));
  if (!arena) {
    printf("arena malloc failed\n");
    return 1;
  }

  const tflite::Model* model = tflite::GetModel(g_skip_s0_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("schema version mismatch\n");
    free(arena);
    return 1;
  }

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

  tflite::MicroInterpreter interp(model, resolver, arena, kArenaSize);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors FAILED (arena_used=%d/%d)\n",
           (int)interp.arena_used_bytes(), (int)kArenaSize);
    free(arena);
    return 1;
  }
  printf("AllocateTensors OK  arena_used=%d/%d\n",
         (int)interp.arena_used_bytes(), (int)kArenaSize);

  TfLiteTensor* input = interp.input_tensor(0);
  if (!input || input->bytes != g_skip_s0_input_data_size) {
    printf("input mismatch: got=%d expected=%d\n",
           input ? (int)input->bytes : -1,
           (int)g_skip_s0_input_data_size);
    free(arena);
    return 1;
  }
  memcpy(input->data.raw, g_skip_s0_input_data, input->bytes);

  if (interp.Invoke() != kTfLiteOk) {
    printf("Invoke FAILED\n");
    free(arena);
    return 1;
  }

  TfLiteTensor* output = interp.output_tensor(0);
  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(g_skip_s0_ref_output_data);
  const int n = output->bytes / sizeof(float);

  float max_diff = 0.0f;
  int max_idx = 0;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) { max_diff = d; max_idx = i; }
  }

  printf("output: %d elements  max_diff=%.6f at [%d]\n",
         n, (double)max_diff, max_idx);

  free(arena);

  constexpr float kTolerance = 1e-4f;
  if (max_diff > kTolerance) {
    printf("FAIL: max_diff %.6f > tolerance %.6f\n",
           (double)max_diff, (double)kTolerance);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
