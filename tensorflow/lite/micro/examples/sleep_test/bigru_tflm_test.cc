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
}

constexpr size_t kTensorArenaSize = 2 * 1024 * 1024;

static uint8_t* ReadFile(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return nullptr;
  struct stat st;
  fstat(fileno(f), &st);
  *size = st.st_size;
  uint8_t* buf = new uint8_t[*size];
  fread(buf, 1, *size, f);
  fclose(f);
  return buf;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s bigru1_test.tflite input.bin reference.bin\n",
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

  tflite::MicroMutableOpResolver<1> op_resolver;
  TFLMRegistration bigru_reg = tflite::Register_BIDIRECTIONAL_SEQUENCE_GRU();
  if (op_resolver.AddCustom("BIDIRECTIONAL_SEQUENCE_GRU", &bigru_reg) !=
      kTfLiteOk) {
    fprintf(stderr, "Failed to register BiGRU op\n");
    return 1;
  }

  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "AllocateTensors failed\n");
    return 1;
  }

  TfLiteTensor* input_tensor = interpreter.input_tensor(0);
  fprintf(stderr, "Input tensor: %d dims, bytes=%zu\n",
          input_tensor->dims->size, input_tensor->bytes);

  if (input_size != input_tensor->bytes) {
    fprintf(stderr, "Input size mismatch: file=%zu tensor=%zu\n", input_size,
            input_tensor->bytes);
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
  printf("Output[0:5]:  ");
  for (int i = 0; i < 5; ++i) printf("%.6f ", output[i]);
  printf("\nRef[0:5]:     ");
  for (int i = 0; i < 5; ++i) printf("%.6f ", ref[i]);
  printf("\n");

  if (max_diff < 1e-4f) {
    printf("PASS\n");
    return 0;
  } else {
    printf("FAIL\n");
    return 1;
  }
}
