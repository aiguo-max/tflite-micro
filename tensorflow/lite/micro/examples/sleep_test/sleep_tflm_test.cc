#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
extern TFLMRegistration Register_BIDIRECTIONAL_SEQUENCE_GRU();
extern TFLMRegistration Register_LAYER_NORM();
}  // namespace tflite
constexpr size_t kTensorArenaSize = 5 * 1024 * 1024;

static uint8_t* ReadFile(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return nullptr;
  struct stat st;
  fstat(fileno(f), &st);
  *size = st.st_size;
  uint8_t* buf = new uint8_t[*size];
  size_t n = fread(buf, 1, *size, f);
  (void)n;
  fclose(f);
  return buf;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s sleep_full.tflite input.bin reference.bin\n",
            argv[0]);
    return 1;
  }

  tflite::InitializeTarget();

  size_t model_size;
  uint8_t* model_data = ReadFile(argv[1], &model_size);
  if (!model_data) {
    fprintf(stderr, "Failed to read model\n");
    return 1;
  }

  size_t input_size;
  uint8_t* input_data = ReadFile(argv[2], &input_size);
  if (!input_data) {
    fprintf(stderr, "Failed to read input\n");
    return 1;
  }

  size_t ref_size;
  uint8_t* ref_data = ReadFile(argv[3], &ref_size);
  if (!ref_data) {
    fprintf(stderr, "Failed to read reference\n");
    return 1;
  }

  const tflite::Model* model = tflite::GetModel(model_data);

  // Create minimal op resolver with only the ops needed for sleep model
  tflite::MicroMutableOpResolver<10> op_resolver;
  op_resolver.AddFullyConnected();
  op_resolver.AddSoftmax();
  op_resolver.AddReshape();
  op_resolver.AddQuantize();
  op_resolver.AddDequantize();
  op_resolver.AddElu();

  TFLMRegistration bigru_reg = tflite::Register_BIDIRECTIONAL_SEQUENCE_GRU();
  op_resolver.AddCustom("BIDIRECTIONAL_SEQUENCE_GRU", &bigru_reg);
  TFLMRegistration ln_reg = tflite::Register_LAYER_NORM();
  op_resolver.AddCustom("LAYER_NORM", &ln_reg);

  uint8_t* arena = new uint8_t[kTensorArenaSize];
  tflite::MicroInterpreter interpreter(model, op_resolver, arena,
                                       kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "AllocateTensors failed\n");
    return 1;
  }
  fprintf(stderr, "AllocateTensors OK  arena=%d/%d\n",
          (int)interpreter.arena_used_bytes(), (int)kTensorArenaSize);

  TfLiteTensor* input_tensor = interpreter.input_tensor(0);
  fprintf(stderr, "Input: dims=%d, bytes=%zu, expected=%zu\n",
          input_tensor->dims->size, input_tensor->bytes, input_size);

  if (input_size != input_tensor->bytes) {
    fprintf(stderr, "Input size mismatch\n");
    return 1;
  }
  memcpy(input_tensor->data.raw, input_data, input_size);

  fprintf(stderr, "Running inference...\n");
  if (interpreter.Invoke() != kTfLiteOk) {
    fprintf(stderr, "Invoke FAILED\n");
    return 1;
  }
  fprintf(stderr, "Invoke OK\n");

  const TfLiteTensor* output_tensor = interpreter.output_tensor(0);
  if (ref_size != output_tensor->bytes) {
    fprintf(stderr, "Output size mismatch: ref=%zu tensor=%zu\n", ref_size,
            output_tensor->bytes);
    return 1;
  }

  const float* output = reinterpret_cast<const float*>(output_tensor->data.raw);
  const float* ref = reinterpret_cast<const float*>(ref_data);
  int n_elements = ref_size / sizeof(float);

  float max_diff = 0, sum_diff = 0;
  for (int i = 0; i < n_elements; ++i) {
    float diff = std::fabs(output[i] - ref[i]);
    if (diff > max_diff) max_diff = diff;
    sum_diff += diff;
  }
  float mean_diff = sum_diff / n_elements;

  printf("Max  abs diff: %e\n", max_diff);
  printf("Mean abs diff: %e\n", mean_diff);

  printf("Output[0:4]:  ");
  for (int i = 0; i < 4; ++i) printf("%.6f ", output[i]);
  printf("\nRef[0:4]:     ");
  for (int i = 0; i < 4; ++i) printf("%.6f ", ref[i]);
  printf("\nOutput[-4:]:  ");
  for (int i = n_elements - 4; i < n_elements; ++i) printf("%.6f ", output[i]);
  printf("\nRef[-4:]:     ");
  for (int i = n_elements - 4; i < n_elements; ++i) printf("%.6f ", ref[i]);
  printf("\n");

  if (max_diff < 1e-4f) {
    printf("PASS\n");
    return 0;
  } else {
    printf("FAIL (max_diff=%e > 1e-4)\n", max_diff);
    return 1;
  }
}
