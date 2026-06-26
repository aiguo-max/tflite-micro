#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/examples/hr_pnet_test/pnet_v2_skip_s0_conv1_model_data.h"
#include "tensorflow/lite/micro/examples/hr_pnet_test/pnet_v2_skip_s0_conv1_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kArenaSize = 256 * 1024;

static int run_one(uint8_t* arena, const char* label,
                   const uint8_t* model_data,
                   const uint8_t* input_bytes, int input_size,
                   const uint8_t* ref_output_bytes, int ref_output_size,
                   float tol) {
  printf("=== %s ===\n", label);
  const tflite::Model* model = tflite::GetModel(model_data);
  if (!model) { printf("GetModel FAILED\n"); return 1; }

  tflite::MicroMutableOpResolver<13> resolver;
  resolver.AddAdd();
  resolver.AddAveragePool2D();
  resolver.AddConcatenation();
  resolver.AddConv2D();
  resolver.AddExpandDims();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddPad();
  resolver.AddReshape();
  resolver.AddStridedSlice();
  resolver.AddTranspose();

  tflite::MicroInterpreter interp(model, resolver, arena, kArenaSize);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors FAILED (arena=%d/%d)\n",
           (int)interp.arena_used_bytes(), (int)kArenaSize);
    return 1;
  }
  printf("AllocateTensors OK  arena=%d/%d\n",
         (int)interp.arena_used_bytes(), (int)kArenaSize);

  TfLiteTensor* input = interp.input_tensor(0);
  if (!input || input->bytes != (size_t)input_size) {
    printf("input mismatch: got=%d expected=%d\n",
           input ? (int)input->bytes : -1, input_size);
    return 1;
  }
  memcpy(input->data.raw, input_bytes, input->bytes);

  if (interp.Invoke() != kTfLiteOk) { printf("Invoke FAILED\n"); return 1; }

  TfLiteTensor* output = interp.output_tensor(0);
  if (!output || output->bytes != (size_t)ref_output_size) {
    printf("output mismatch: got=%d expected=%d\n",
           output ? (int)output->bytes : -1, ref_output_size);
    return 1;
  }
  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(ref_output_bytes);
  const int n = output->bytes / sizeof(float);

  printf("first4 got:");
  for (int i = 0; i < 4 && i < n; ++i) printf(" %.4f", (double)got[i]);
  printf("\nfirst4 ref:");
  for (int i = 0; i < 4 && i < n; ++i) printf(" %.4f", (double)ref[i]);
  printf("\n");

  float max_diff = 0.0f;
  double sum_abs = 0.0;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
    sum_abs += static_cast<double>(d);
  }
  double mean_diff = sum_abs / n;
  printf("max_diff=%.6f mean_diff=%.6f (tol=%.6f)\n",
         (double)max_diff, mean_diff, (double)tol);
  if (max_diff > tol) { printf("FAIL\n"); return 1; }
  printf("PASS\n");
  return 0;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  uint8_t* arena = static_cast<uint8_t*>(malloc(kArenaSize));
  if (!arena) { printf("arena malloc failed\n"); return 1; }

  int ret = 0;
  ret += run_one(arena,
      "Test1: pnet_v2_skip_s0_conv1 (hybrid, float32 I/O, ref=XNNPACK)",
      g_pnet_v2_skip_s0_conv1_model,
      g_pnet_v2_skip_s0_conv1_input, g_pnet_v2_skip_s0_conv1_input_size,
      g_pnet_v2_skip_s0_conv1_ref_output_bytes,
      g_pnet_v2_skip_s0_conv1_ref_output_size, 0.01f);

  free(arena);
  if (ret == 0) {
    printf("~~~ALL TESTS PASSED~~~\n");
  } else {
    printf("SOME FAIL\n");
  }
  return ret;
}
