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

// customOptions layout: [eps: float32, type: uint8]
// type=0: standard LN (1 input)
// type=1: bessel-corrected LN with a_2/b_2 weights (3 inputs)
struct LayerNormOpData {
  float epsilon;
  uint8_t type;  // 0=standard, 1=bessel
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  auto* data = static_cast<LayerNormOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(LayerNormOpData)));

  if (buffer != nullptr && length >= sizeof(float)) {
    const float* params = reinterpret_cast<const float*>(buffer);
    data->epsilon = params[0];
    data->type = (length > sizeof(float))
                     ? static_cast<uint8_t>(buffer[sizeof(float)])
                     : 0;
  } else {
    data->epsilon = 1e-3f;
    data->type = 0;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs == 1 || num_inputs == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, input->type == kTfLiteFloat32);

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

  const bool has_affine = (node->inputs->size >= 3);
  const float* gamma = nullptr;
  const float* beta = nullptr;
  if (has_affine) {
    gamma = tflite::micro::GetTensorData<float>(
        tflite::micro::GetEvalInput(context, node, 1));
    beta = tflite::micro::GetTensorData<float>(
        tflite::micro::GetEvalInput(context, node, 2));
  }

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

    if (data->type == 0) {
      const float inv_std = 1.0f / std::sqrt(var_sum * inv_features + eps);
      if (gamma) {
        for (int j = 0; j < features; ++j) {
          out_row[j] = gamma[j] * (row[j] - mean) * inv_std + beta[j];
        }
      } else {
        for (int j = 0; j < features; ++j) {
          out_row[j] = (row[j] - mean) * inv_std;
        }
      }
    } else {
      const float n_minus_1 = static_cast<float>(features - 1);
      const float std_val = std::sqrt(var_sum / n_minus_1) + eps;
      if (gamma) {
        for (int j = 0; j < features; ++j) {
          out_row[j] = gamma[j] * (row[j] - mean) / std_val + beta[j];
        }
      } else {
        for (int j = 0; j < features; ++j) {
          out_row[j] = (row[j] - mean) / std_val;
        }
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_LAYER_NORM() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
