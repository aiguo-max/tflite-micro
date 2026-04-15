// HR model TFLM validation: 3 models from /data/ (NuttX) or embedded (x86)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__NuttX__)
#include <time.h>
#else
#include "tensorflow/lite/micro/examples/hr_test/hr_model_data.h"
#include "tensorflow/lite/micro/examples/hr_test/hr_hybrid_gru_model_data.h"
#include "tensorflow/lite/micro/examples/hr_test/hr_hybrid_full_model_data.h"
#endif
#include "tensorflow/lite/micro/examples/hr_test/hr_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
extern TFLMRegistration Register_LAYER_NORM();
extern TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_GRU();
extern TFLMRegistration Register_MULTI_HEAD_ATTENTION();
}  // namespace tflite

constexpr size_t kArenaSize = 512 * 1024;
#if !defined(__NuttX__)
alignas(16) static uint8_t g_arena[kArenaSize];
#endif

static uint8_t* ReadFile(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("Cannot open %s\n", path); return nullptr; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* buf = static_cast<uint8_t*>(malloc(sz));
  if (!buf) { printf("malloc(%ld) failed\n", sz); fclose(f); return nullptr; }
  size_t rd = fread(buf, 1, sz, f);
  fclose(f);
  if (rd != static_cast<size_t>(sz)) { free(buf); return nullptr; }
  *out_size = static_cast<size_t>(sz);
  return buf;
}

#if defined(__NuttX__)
static int64_t now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}
#endif

static int RunModel(const char* tag, const unsigned char* model_data,
                    const uint8_t* input_data, size_t input_size) {
#if defined(__NuttX__)
  int64_t t0 = now_us();
  uint8_t* arena_raw = static_cast<uint8_t*>(malloc(kArenaSize + 15));
  if (!arena_raw) { printf("[%s] malloc arena FAILED\n", tag); return 1; }
  uint8_t* tensor_arena = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(arena_raw) + 15) &
      ~static_cast<uintptr_t>(15));
#else
  uint8_t* arena_raw = nullptr;
  uint8_t* tensor_arena = g_arena;
#endif

  const tflite::Model* model = tflite::GetModel(model_data);
  if (!model) { printf("[%s] GetModel FAILED\n", tag); free(arena_raw); return 1; }

  tflite::MicroMutableOpResolver<20> resolver;
  resolver.AddFullyConnected();
  resolver.AddPrelu();
  resolver.AddReshape();
  resolver.AddStridedSlice();
  resolver.AddConcatenation();
  resolver.AddAdd();
  resolver.AddMean();
  resolver.AddSoftmax();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddSquaredDifference();
  resolver.AddSum();
  resolver.AddSqrt();
  resolver.AddRsqrt();
  resolver.AddDiv();
  resolver.AddNeg();
  TFLMRegistration ln_reg = tflite::Register_LAYER_NORM();
  resolver.AddCustom("LAYER_NORM", &ln_reg);
  TFLMRegistration gru_reg = tflite::Register_UNIDIRECTIONAL_SEQUENCE_GRU();
  resolver.AddCustom("UNIDIRECTIONAL_SEQUENCE_GRU", &gru_reg);
  TFLMRegistration mha_reg = tflite::Register_MULTI_HEAD_ATTENTION();
  resolver.AddCustom("MULTI_HEAD_ATTENTION", &mha_reg);

  tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("[%s] AllocateTensors FAILED\n", tag);
    free(arena_raw); return 1;
  }
  printf("[%s] arena=%d/%d\n", tag,
         (int)interp.arena_used_bytes(), (int)kArenaSize);

  size_t offset = 0;
  for (size_t i = 0; i < interp.inputs_size(); ++i) {
    TfLiteTensor* t = interp.input_tensor(i);
    if (offset + t->bytes > input_size) {
      printf("[%s] input overflow\n", tag); free(arena_raw); return 1;
    }
    memcpy(t->data.raw, input_data + offset, t->bytes);
    offset += t->bytes;
  }

#if defined(__NuttX__)
  int64_t t1 = now_us();
#endif

  if (interp.Invoke() != kTfLiteOk) {
    printf("[%s] Invoke FAILED\n", tag); free(arena_raw); return 1;
  }

#if defined(__NuttX__)
  int64_t t2 = now_us();
  printf("[%s] Invoke: %.2f ms\n", tag, (double)(t2 - t1) / 1000.0);
#endif

  for (size_t i = 0; i < interp.outputs_size(); ++i) {
    const TfLiteTensor* t = interp.output_tensor(i);
    const float* out = reinterpret_cast<const float*>(t->data.raw);
    printf("[%s] Output[%zu](%d): %.6f\n", tag, i,
           (int)(t->bytes / sizeof(float)), static_cast<double>(out[0]));
  }

  free(arena_raw);
  printf("[%s] DONE\n", tag);
  return 0;
}

extern "C" int main(int argc, char* argv[]) {
  (void)argc; (void)argv;
  tflite::InitializeTarget();

#if defined(__NuttX__)
  int64_t t_start = now_us();

  struct { const char* tag; const char* path; } models[] = {
    {"f32",      "/data/attengru_v2_fused.tflite"},
    {"gru_hyb",  "/data/attengru_v2_fused_hybrid_gru_only.tflite"},
    {"full_hyb", "/data/attengru_v2_fused_hybrid.tflite"},
  };

  for (int m = 0; m < 3; ++m) {
    size_t model_size = 0;
    uint8_t* model_buf = ReadFile(models[m].path, &model_size);
    if (!model_buf) { printf("Skipping %s\n", models[m].tag); continue; }
    printf("\n=== %s (%zu KB) ===\n", models[m].tag, model_size / 1024);
    RunModel(models[m].tag, model_buf, g_hr_input, g_hr_input_len);
    free(model_buf);
  }

  int64_t t_end = now_us();
  printf("\nTotal: %.1f ms\n", (double)(t_end - t_start) / 1000.0);

#else
  const uint8_t* input_ptr = g_hr_input;
  size_t input_len = g_hr_input_len;
  uint8_t* file_input = nullptr;
  if (argc >= 2) {
    size_t fsize = 0;
    file_input = ReadFile(argv[1], &fsize);
    if (file_input) { input_ptr = file_input; input_len = fsize; }
  }
  printf("=== f32 ===\n");
  RunModel("f32", g_hr_model, input_ptr, input_len);
  printf("\n=== gru_hyb ===\n");
  RunModel("gru_hyb", g_hr_hybrid_gru_model, input_ptr, input_len);
  printf("\n=== full_hyb ===\n");
  RunModel("full_hyb", g_hr_hybrid_full_model, input_ptr, input_len);
  free(file_input);
#endif

  printf("~~~ALL TESTS PASSED~~~\n");
  return 0;
}
