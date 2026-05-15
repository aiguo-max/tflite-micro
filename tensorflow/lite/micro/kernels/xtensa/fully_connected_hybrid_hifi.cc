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

#include <algorithm>
#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "xa_nnlib_api.h"

namespace tflite {

TfLiteStatus FullyConnectedEvalHybridHifi(
    TfLiteContext* context, const TfLiteFullyConnectedParams& params,
    const OpDataFullyConnected& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
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

  const int output_depth = filter_shape.Dims(0);
  const int accum_depth = filter_shape.Dims(1);
  const int batches = input_size / accum_depth;

  // Asymmetric per-batch activation quantization, matching TFLite's
  // PortableAsymmetricQuantizeFloats + HybridConvPerChannel semantics:
  //   scale = (max(0, rmax) - min(0, rmin)) / 255
  //   zp chosen as the endpoint with smaller rounding error
  //   acc_real = (q - zp) * w = q*w - zp*sum(w)
  float* input_scales = static_cast<float*>(
      context->GetScratchBuffer(context, data.hybrid_scales_scratch_index));
  int32_t* input_zps = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.hybrid_zero_points_scratch_index));

  for (int b = 0; b < batches; ++b) {
    const float* row_data = input_data + b * accum_depth;
    int8_t* row_quant = input_quantized + b * accum_depth;

    float min_val = row_data[0];
    float max_val = row_data[0];
    for (int i = 1; i < accum_depth; ++i) {
      if (row_data[i] < min_val) min_val = row_data[i];
      if (row_data[i] > max_val) max_val = row_data[i];
    }

    const double rmin = std::min(0.0, static_cast<double>(min_val));
    const double rmax = std::max(0.0, static_cast<double>(max_val));
    if (rmin == rmax) {
      input_scales[b] = 1.0f;
      input_zps[b] = 0;
      memset(row_quant, 0, accum_depth * sizeof(int8_t));
    } else {
      const double scale = (rmax - rmin) / 255.0;
      const double zp_from_min = -128.0 - rmin / scale;
      const double zp_from_max = 127.0 - rmax / scale;
      const double err_min = std::abs(-128.0) + std::abs(rmin / scale);
      const double err_max = std::abs(127.0) + std::abs(rmax / scale);
      const double zp_double =
          err_min < err_max ? zp_from_min : zp_from_max;
      int32_t zp = static_cast<int32_t>(std::round(zp_double));
      if (zp < -128) zp = -128;
      if (zp > 127) zp = 127;
      input_scales[b] = static_cast<float>(scale);
      input_zps[b] = zp;
      const float inv_scale = static_cast<float>(1.0 / scale);
      for (int i = 0; i < accum_depth; ++i) {
        int32_t v = static_cast<int32_t>(std::round(
            static_cast<float>(zp) + row_data[i] * inv_scale));
        v = v < -128 ? -128 : (v > 127 ? 127 : v);
        row_quant[i] = static_cast<int8_t>(v);
      }
    }
  }

  FullyConnectedParams op_params = FullyConnectedParamsFloat(params.activation);
  const float act_min = op_params.float_activation_min;
  const float act_max = op_params.float_activation_max;

  TF_LITE_ENSURE(context, output_depth <= 512);

  int32_t* acc_buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.hybrid_output_scratch_index));

  alignas(8) int8_t s_zero_bias[512] = {};
  const int32_t* row_sums = data.hybrid_row_sums;

  for (int b = 0; b < batches; ++b) {
    const int8_t* batch_input = input_quantized + b * accum_depth;

    if ((accum_depth & 3) == 0) {
      xa_nn_matXvec_8x8_32(acc_buf, const_cast<int8_t*>(filter_int8), nullptr,
                           const_cast<int8_t*>(batch_input), nullptr,
                           s_zero_bias, output_depth, accum_depth, 0,
                           accum_depth, 0, 0, 0);
    } else {
      for (int r = 0; r < output_depth; ++r) {
        int32_t acc = 0;
        for (int d = 0; d < accum_depth; ++d) {
          acc += static_cast<int32_t>(batch_input[d]) *
                 static_cast<int32_t>(filter_int8[r * accum_depth + d]);
        }
        acc_buf[r] = acc;
      }
    }

    const float row_scale = input_scales[b];
    const int32_t row_zp = input_zps[b];
    const bool is_per_channel = (data.hybrid_num_channels == output_depth);
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      const float filter_scale = is_per_channel
                                     ? data.hybrid_filter_scales[out_c]
                                     : data.hybrid_filter_scales[0];
      const int32_t acc = acc_buf[out_c] - row_zp * row_sums[out_c];
      float float_acc = static_cast<float>(acc) * row_scale * filter_scale;
      if (bias_float) {
        float_acc += bias_float[out_c];
      }
      float_acc = float_acc < act_min ? act_min : float_acc;
      float_acc = float_acc > act_max ? act_max : float_acc;
      output_data[b * output_depth + out_c] = float_acc;
    }
  }

  return kTfLiteOk;
}

}  // namespace tflite

#endif  // defined(HIFI4) || defined(HIFI5) || defined(XTENSA)