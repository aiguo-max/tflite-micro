#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/tools/benchmarking/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kTensorArenaSize = 5 * 1024 * 1024;

static uint8_t* ReadFile(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return nullptr;
  struct stat st;
  fstat(fileno(f), &st);
  *size = st.st_size;
  uint8_t* buf = new uint8_t[*size];
  size_t bytes_read = fread(buf, 1, *size, f);
  (void)bytes_read;
  fclose(f);
  return buf;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s model.tflite input.bin output.bin\n", argv[0]);
    return 1;
  }

  tflite::InitializeTarget();

  size_t model_size;
  uint8_t* model_data = ReadFile(argv[1], &model_size);
  if (!model_data) {
    fprintf(stderr, "Failed to read model: %s\n", argv[1]);
    return 1;
  }

  size_t input_size;
  uint8_t* input_data = ReadFile(argv[2], &input_size);
  if (!input_data) {
    fprintf(stderr, "Failed to read input: %s\n", argv[2]);
    return 1;
  }

  const tflite::Model* model = tflite::GetModel(model_data);

  tflite::TflmOpResolver op_resolver;
  if (tflite::CreateOpResolver(op_resolver) != kTfLiteOk) {
    fprintf(stderr, "CreateOpResolver failed\n");
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
  if (input_size != input_tensor->bytes) {
    fprintf(stderr, "Input size mismatch: file=%zu tensor=%zu\n", input_size,
            input_tensor->bytes);
    return 1;
  }
  memcpy(input_tensor->data.raw, input_data, input_size);

  if (interpreter.Invoke() != kTfLiteOk) {
    fprintf(stderr, "Invoke failed\n");
    return 1;
  }

  TfLiteTensor* output_tensor = interpreter.output_tensor(0);
  FILE* fout = fopen(argv[3], "wb");
  if (!fout) {
    fprintf(stderr, "Failed to open output: %s\n", argv[3]);
    return 1;
  }
  fwrite(output_tensor->data.raw, 1, output_tensor->bytes, fout);
  fclose(fout);

  return 0;
}
