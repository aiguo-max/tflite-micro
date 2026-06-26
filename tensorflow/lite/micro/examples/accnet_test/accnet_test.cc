#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/examples/accnet_test/accnet_int8_model_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_int8_test_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_v3_int8_model_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_v3_int8_test_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_skip_stage0_conv_model_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_skip_stage0_conv_test_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_skip_s0_conv1_model_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_skip_s0_conv1_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kArenaSize = 512 * 1024;

static int run_accnet_int8(uint8_t* arena, const char* label,
                            const uint8_t* model_data,
                            const int8_t* input_bytes, int input_size,
                            const int8_t* ref_output, int ref_output_size,
                            int tol) {
  printf("=== %s ===\n", label);
  const tflite::Model* model = tflite::GetModel(model_data);
  if (!model) { printf("GetModel FAILED\n"); return 1; }

  tflite::MicroMutableOpResolver<13> resolver;
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
  resolver.AddSqueeze();
  resolver.AddQuantize();

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
  const int8_t* got = reinterpret_cast<const int8_t*>(output->data.raw);
  const int n = output->bytes;

  printf("first8 got:");
  for (int i = 0; i < 8 && i < n; ++i) printf(" %d", (int)got[i]);
  printf("\nfirst8 ref:");
  for (int i = 0; i < 8 && i < n; ++i) printf(" %d", (int)ref_output[i]);
  printf("\n");

  int max_diff = 0;
  for (int i = 0; i < n; ++i) {
    int d = static_cast<int>(got[i]) - static_cast<int>(ref_output[i]);
    if (d < 0) d = -d;
    if (d > max_diff) max_diff = d;
  }
  printf("max_int8_diff=%d (tol=%d)\n", max_diff, tol);
  if (max_diff > tol) { printf("FAIL\n"); return 1; }
  printf("PASS\n");
  return 0;
}

static int run_accnet_hybrid(uint8_t* arena, const char* label,
                              const uint8_t* model_data,
                              const uint8_t* input_bytes, int input_size,
                              const uint8_t* ref_output_bytes, int ref_output_size,
                              float tol) {
  printf("=== %s ===\n", label);
  const tflite::Model* model = tflite::GetModel(model_data);
  if (!model) { printf("GetModel FAILED\n"); return 1; }

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
  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref = reinterpret_cast<const float*>(ref_output_bytes);
  const int n = output->bytes / sizeof(float);

  printf("first4 got:");
  for (int i = 0; i < 4 && i < n; ++i) printf(" %.4f", (double)got[i]);
  printf("\nfirst4 ref:");
  for (int i = 0; i < 4 && i < n; ++i) printf(" %.4f", (double)ref[i]);
  printf("\n");

  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = std::fabs(got[i] - ref[i]);
    if (d > max_diff) max_diff = d;
  }
  printf("max_diff=%.6f (tol=%.6f)\n", (double)max_diff, (double)tol);
  if (max_diff > tol) { printf("FAIL\n"); return 1; }
  printf("PASS\n");
  return 0;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  uint8_t* arena = static_cast<uint8_t*>(malloc(kArenaSize));
  if (!arena) { printf("arena malloc failed\n"); return 1; }

  int ret = 0;
  ret += run_accnet_int8(arena, "Test1: accnet v2 int8 (full int8, int8 I/O)",
      g_accnet_int8_model,
      g_accnet_int8_input, g_accnet_int8_input_size,
      g_accnet_int8_ref_output, g_accnet_int8_ref_output_size,
      /*tol=*/2);
  ret += run_accnet_int8(arena, "Test1b: accnet v3 int8 (full int8, int8 I/O, real PyTorch sample 0)",
      g_accnet_v3_int8_model,
      g_accnet_v3_int8_input, g_accnet_v3_int8_input_size,
      g_accnet_v3_int8_ref_output_bytes, g_accnet_v3_int8_ref_output_size,
      /*tol=*/2);
  ret += run_accnet_hybrid(
      arena, "Test2: accnet_skip_stage0_conv (hybrid, float32 I/O)",
      g_accnet_skip_stage0_conv_model,
      g_accnet_skip_stage0_conv_input, g_accnet_skip_stage0_conv_input_size,
      g_accnet_skip_stage0_conv_ref_output_bytes,
      g_accnet_skip_stage0_conv_ref_output_size,
      /*tol=*/0.05f);
  ret += run_accnet_hybrid(
      arena, "Test3: accnet_skip_s0_conv1 (hybrid, float32 I/O)",
      g_accnet_skip_s0_conv1_model,
      g_accnet_skip_s0_conv1_input, g_accnet_skip_s0_conv1_input_size,
      g_accnet_skip_s0_conv1_ref_output_bytes,
      g_accnet_skip_s0_conv1_ref_output_size,
      /*tol=*/0.05f);

  free(arena);
  printf("%s\n", ret == 0 ? "ALL PASS" : "SOME FAIL");
  return ret;
}
