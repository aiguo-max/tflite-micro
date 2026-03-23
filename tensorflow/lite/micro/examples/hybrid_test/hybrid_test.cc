/* Xtensa HiFi4 hybrid inference test — skip_s0_conv1 + dynamic_int8 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/examples/hybrid_test/skip_s0_conv1_model_data.h"
#include "tensorflow/lite/micro/examples/hybrid_test/skip_s0_conv1_test_data.h"
#include "tensorflow/lite/micro/examples/hybrid_test/dynamic_int8_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kArenaSize = 512 * 1024;

static int run_skip_s0(uint8_t* arena) {
  printf("=== Test1: skip_s0_conv1 ===\n");
  const tflite::Model* model = tflite::GetModel(g_skip_s0_model);

  tflite::MicroMutableOpResolver<11> resolver;
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
    printf("AllocateTensors FAILED\n");
    return 1;
  }

  TfLiteTensor* input = interp.input_tensor(0);
  memcpy(input->data.raw, g_skip_s0_input_data, input->bytes);

  if (interp.Invoke() != kTfLiteOk) {
    printf("Invoke FAILED\n");
    return 1;
  }

  TfLiteTensor* output = interp.output_tensor(0);
  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(g_skip_s0_ref_output_data);
  const int n = output->bytes / sizeof(float);

  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
  }
  printf("max_diff=%.6f\n", (double)max_diff);
  if (max_diff > 1e-4f) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

static int run_dynamic_int8(uint8_t* arena) {
  printf("=== Test2: dynamic_int8 ===\n");
  const tflite::Model* model = tflite::GetModel(g_dynamic_int8_model);
  if (!model) {
    printf("GetModel FAILED\n");
    return 1;
  }

  tflite::MicroMutableOpResolver<11> resolver;
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
    return 1;
  }
  printf("AllocateTensors OK  arena_used=%d/%d\n",
         (int)interp.arena_used_bytes(), (int)kArenaSize);

  TfLiteTensor* input = interp.input_tensor(0);
  if (!input || input->bytes != (size_t)g_dynamic_int8_input_size) {
    printf("input mismatch: %d vs %d\n",
           input ? (int)input->bytes : -1, g_dynamic_int8_input_size);
    return 1;
  }
  memcpy(input->data.raw, g_dynamic_int8_input, input->bytes);

  if (interp.Invoke() != kTfLiteOk) {
    printf("Invoke FAILED\n");
    return 1;
  }

  TfLiteTensor* output = interp.output_tensor(0);
  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(g_dynamic_int8_ref_output);
  const int n = output->bytes / sizeof(float);

  printf("first8 got:");
  for (int i = 0; i < 8 && i < n; ++i) printf(" %.4f", (double)got[i]);
  printf("\nfirst8 ref:");
  for (int i = 0; i < 8 && i < n; ++i) printf(" %.4f", (double)ref[i]);
  printf("\n");

  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
  }
  printf("max_diff=%.6f\n", (double)max_diff);

  if (max_diff > 1.5f) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  uint8_t* arena = static_cast<uint8_t*>(malloc(kArenaSize));
  if (!arena) {
    printf("arena malloc failed\n");
    return 1;
  }

  int ret = 0;
  ret += run_skip_s0(arena);
  ret += run_dynamic_int8(arena);

  free(arena);
  printf("%s\n", ret == 0 ? "ALL PASS" : "SOME FAIL");
  return ret;
}
