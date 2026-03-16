/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"); */

#include <cmath>
#include <cstring>

#include "tensorflow/lite/micro/kernels/hybrid_model_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr size_t kTensorArenaSize = 256 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(HybridModelInference) {
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_hybrid_model_data);
  TF_LITE_MICRO_EXPECT(model != nullptr);

  // Register ops needed by the hybrid model (MobileNetV1 variant)
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

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());

  // Copy input data
  TfLiteTensor* input = interpreter.input_tensor(0);
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(input->bytes,
                           static_cast<size_t>(g_hybrid_input_data_size));
  memcpy(input->data.raw, g_hybrid_input_data, input->bytes);

  // Run inference
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  // Compare output against reference
  TfLiteTensor* output = interpreter.output_tensor(0);
  TF_LITE_MICRO_EXPECT(output != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(output->bytes,
                           static_cast<size_t>(g_hybrid_ref_output_data_size));

  const float* got = reinterpret_cast<const float*>(output->data.raw);
  const float* ref =
      reinterpret_cast<const float*>(g_hybrid_ref_output_data);
  const int num_elements = output->bytes / sizeof(float);

  float max_abs_diff = 0.0f;
  for (int i = 0; i < num_elements; ++i) {
    float diff = std::fabs(got[i] - ref[i]);
    if (diff > max_abs_diff) max_abs_diff = diff;
  }

  MicroPrintf("Hybrid model output: %d elements, max_abs_diff=%.6f",
              num_elements, static_cast<double>(max_abs_diff));

  // Tolerance: hybrid quantization introduces small differences
  constexpr float kTolerance = 1e-4f;
  TF_LITE_MICRO_EXPECT_LE(max_abs_diff, kTolerance);
}

TF_LITE_MICRO_TESTS_END
