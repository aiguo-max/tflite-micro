/* measure_arena.cc — 加载 .tflite 文件，打印 arena_used_bytes()
 *
 * 用法： ./measure_arena model.tflite [model2.tflite ...]
 *
 * 构建（在 TFLM 根目录）：
 *   make -f tensorflow/lite/micro/tools/make/Makefile \
 *     measure_arena -j$(nproc)
 *
 * 输出二进制：
 *   tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/bin/measure_arena
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static std::vector<uint8_t> read_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open %s\n", path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  size_t read = fread(buf.data(), 1, sz, f);
  fclose(f);
  if ((long)read != sz) {
    fprintf(stderr, "ERROR: short read on %s\n", path);
    exit(1);
  }
  return buf;
}

static constexpr int kMaxOps = 25;
using OpResolver = tflite::MicroMutableOpResolver<kMaxOps>;

static void setup_resolver(OpResolver& r) {
  r.AddAdd();
  r.AddAveragePool2D();
  r.AddConcatenation();
  r.AddConv2D();
  r.AddDequantize();
  r.AddExpandDims();
  r.AddFullyConnected();
  r.AddLogistic();
  r.AddMean();
  r.AddMul();
  r.AddPad();
  r.AddQuantize();
  r.AddReshape();
  r.AddStridedSlice();
  r.AddTranspose();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s model.tflite [model2.tflite ...]\n", argv[0]);
    return 1;
  }

  tflite::InitializeTarget();

  constexpr int kArenaSize = 512 * 1024;
  alignas(16) uint8_t tensor_arena[kArenaSize];

  for (int i = 1; i < argc; i++) {
    printf("=== %s ===\n", argv[i]);

    auto model_data = read_file(argv[i]);
    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (!model) {
      printf("  GetModel FAILED\n");
      continue;
    }

    OpResolver resolver;
    setup_resolver(resolver);

    constexpr int kInterpBufSize = sizeof(tflite::MicroInterpreter);
    alignas(16) uint8_t interp_buf[kInterpBufSize];

    tflite::MicroInterpreter* interp = new (interp_buf)
        tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);

    TfLiteStatus status = interp->AllocateTensors();
    if (status != kTfLiteOk) {
      printf("  AllocateTensors FAILED\n");
    } else {
      size_t used = interp->arena_used_bytes();
      printf("  arena_used_bytes = %zu  (%.1f KB)\n", used, (double)used / 1024.0);
      printf("  arena_provided   = %d  (%.1f KB)\n", kArenaSize, (double)kArenaSize / 1024.0);
      printf("  utilization      = %.1f%%\n", 100.0 * used / kArenaSize);
    }

    interp->~MicroInterpreter();
  }

  return 0;
}
