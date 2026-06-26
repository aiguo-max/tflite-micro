#include <cstdio>
#include <cstdint>
#include <cstring>
#include "tensorflow/lite/micro/examples/accnet_test/accnet_int8_model_data.h"
#include "tensorflow/lite/micro/examples/accnet_test/accnet_int8_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kArenaSize = 128 * 1024;
alignas(16) static uint8_t s_arena[kArenaSize];

int main() {
  tflite::InitializeTarget();
  printf("=== accnet_int8 QEMU CMSIS-NN test ===\n");

  const tflite::Model* model = tflite::GetModel(g_accnet_int8_model);
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

  tflite::MicroInterpreter interp(model, resolver, s_arena, kArenaSize);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors failed\n");
    return 1;
  }
  printf("arena_used=%d/%d\n", (int)interp.arena_used_bytes(), (int)kArenaSize);

  TfLiteTensor* input = interp.input_tensor(0);
  memcpy(input->data.raw, g_accnet_int8_input, input->bytes);

  if (interp.Invoke() != kTfLiteOk) {
    printf("ERROR: Invoke failed\n");
    return 1;
  }

  TfLiteTensor* output = interp.output_tensor(0);
  const int8_t* got = reinterpret_cast<const int8_t*>(output->data.raw);
  const int8_t* ref = g_accnet_int8_ref_output;
  int n = output->bytes;
  int max_diff = 0;
  for (int i = 0; i < n; ++i) {
    int d = static_cast<int>(got[i]) - static_cast<int>(ref[i]);
    if (d < 0) d = -d;
    if (d > max_diff) max_diff = d;
  }
  printf("first8 got: %d %d %d %d %d %d %d %d\n",
         got[0],got[1],got[2],got[3],got[4],got[5],got[6],got[7]);
  printf("first8 ref: %d %d %d %d %d %d %d %d\n",
         ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
  printf("max_int8_diff=%d\n", max_diff);
  if (max_diff > 2) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
