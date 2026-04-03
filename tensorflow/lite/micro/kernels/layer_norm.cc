/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

struct LayerNormOpData {
  float epsilon;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  auto* data = static_cast<LayerNormOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(LayerNormOpData)));

  if (buffer != nullptr && length >= sizeof(float)) {
    const float* params = reinterpret_cast<const float*>(buffer);
    data->epsilon = params[0];
  } else {
    data->epsilon = 1e-3f;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, input->type == kTfLiteFloat32);
  TF_LITE_ENSURE(context, input->dims->size >= 2);

  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* data = static_cast<const LayerNormOpData*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  const int ndims = input->dims->size;
  const int features = input->dims->data[ndims - 1];
  int outer_size = 1;
  for (int i = 0; i < ndims - 1; ++i) {
    outer_size *= input->dims->data[i];
  }

  const float eps = data->epsilon;
  const float inv_features = 1.0f / static_cast<float>(features);

  for (int i = 0; i < outer_size; ++i) {
    const float* row = in_data + i * features;
    float* out_row = out_data + i * features;

    float sum = 0.0f;
    for (int j = 0; j < features; ++j) {
      sum += row[j];
    }
    const float mean = sum * inv_features;

    float var_sum = 0.0f;
    for (int j = 0; j < features; ++j) {
      float diff = row[j] - mean;
      var_sum += diff * diff;
    }
    const float inv_std = 1.0f / std::sqrt(var_sum * inv_features + eps);

    for (int j = 0; j < features; ++j) {
      out_row[j] = (row[j] - mean) * inv_std;
    }
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_LAYER_NORM() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
