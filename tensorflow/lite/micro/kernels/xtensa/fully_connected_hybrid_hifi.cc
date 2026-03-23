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

#if defined(HIFI4) || defined(HIFI5) || defined(XTENSA)

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "xa_nnlib_api.h"

namespace tflite {

TfLiteStatus FullyConnectedEvalHybridHifi(
    TfLiteContext* context,
    const TfLiteFullyConnectedParams& params,
    const OpDataFullyConnected& data,
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  const int8_t* filter_int8 = tflite::micro::GetTensorData<int8_t>(filter);
  const float* input_data = tflite::micro::GetTensorData<float>(input);
  const float* bias_float = tflite::micro::GetOptionalTensorData<float>(bias);
  float* output_data = tflite::micro::GetTensorData<float>(output);

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);

  const int input_size = input_shape.FlatSize();
  int8_t* input_quantized = static_cast<int8_t*>(
      context->GetScratchBuffer(context, data.hybrid_input_scratch_index));

  // Dynamic quantization: symmetric quantize (range = max(|min|,|max|), scale = range/127, zero_point=0)
  float min_val = input_data[0];
  float max_val = input_data[0];
  for (int i = 1; i < input_size; ++i) {
    if (input_data[i] < min_val) min_val = input_data[i];
    if (input_data[i] > max_val) max_val = input_data[i];
  }

  const float range = std::max(std::abs(min_val), std::abs(max_val));
  float input_scale;
  if (range == 0.0f) {
    input_scale = 1.0f;
    memset(input_quantized, 0, input_size * sizeof(int8_t));
  } else {
    input_scale = range / 127.0f;
    const float inv_scale = 127.0f / range;
    for (int i = 0; i < input_size; ++i) {
      int32_t v = static_cast<int32_t>(input_data[i] * inv_scale +
                  (input_data[i] >= 0.0f ? 0.5f : -0.5f));
      v = v < -127 ? -127 : (v > 127 ? 127 : v);
      input_quantized[i] = static_cast<int8_t>(v);
    }
  }

  const int output_depth = filter_shape.Dims(0);
  const int accum_depth = filter_shape.Dims(1);
  const int batches = input_size / accum_depth;

  FullyConnectedParams op_params = FullyConnectedParamsFloat(params.activation);
  const float act_min = op_params.float_activation_min;
  const float act_max = op_params.float_activation_max;

  int32_t* acc_buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.hybrid_output_scratch_index));

  static int8_t s_zero_bias[512] = {};

  for (int b = 0; b < batches; ++b) {
    const int8_t* batch_input = input_quantized + b * accum_depth;

    xa_nn_matXvec_8x8_32(
        acc_buf,
        const_cast<int8_t*>(filter_int8),
        nullptr,
        const_cast<int8_t*>(batch_input),
        nullptr,
        s_zero_bias,
        output_depth,
        accum_depth,
        0,
        accum_depth,
        0,
        0,
        0);

    const bool is_per_channel = (data.hybrid_num_channels == output_depth);
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      const float filter_scale = is_per_channel ? data.hybrid_filter_scales[out_c]
                                                : data.hybrid_filter_scales[0];
      float float_acc = static_cast<float>(acc_buf[out_c]) *
                        input_scale * filter_scale;
      if (bias_float) {
        float_acc += bias_float[out_c];
      }
      
      // Activation clamp
      float_acc = float_acc < act_min ? act_min : float_acc;
      float_acc = float_acc > act_max ? act_max : float_acc;
      output_data[b * output_depth + out_c] = float_acc;
    }
  }

  return kTfLiteOk;
}

}  // namespace tflite

#endif  // defined(HIFI4) || defined(HIFI5) || defined(XTENSA)