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

#include <algorithm>
#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

#if defined(CMSIS_NN)
#include "Include/arm_nnsupportfunctions.h"
#endif

namespace tflite {

TfLiteStatus FullyConnectedEvalHybrid(TfLiteContext* context,
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

  const int output_depth = filter_shape.Dims(0);
  const int accum_depth = filter_shape.Dims(1);
  const int batches = input_size / accum_depth;

  // Per-row symmetric quantization: each batch row gets its own scale.
  // This matches LiteRT's BatchQuantizeFloats (symmetric mode).
  float* input_scales = static_cast<float*>(
      context->GetScratchBuffer(context, data.hybrid_scales_scratch_index));

  for (int b = 0; b < batches; ++b) {
    const float* row_data = input_data + b * accum_depth;
    int8_t* row_quant = input_quantized + b * accum_depth;

    float min_val = row_data[0];
    float max_val = row_data[0];
    for (int i = 1; i < accum_depth; ++i) {
      if (row_data[i] < min_val) min_val = row_data[i];
      if (row_data[i] > max_val) max_val = row_data[i];
    }

    const float range = std::max(std::abs(min_val), std::abs(max_val));
    if (range == 0.0f) {
      input_scales[b] = 1.0f;
      memset(row_quant, 0, accum_depth * sizeof(int8_t));
    } else {
      input_scales[b] = range / 127.0f;
      const float inv_scale = 127.0f / range;
      for (int i = 0; i < accum_depth; ++i) {
        int32_t v = static_cast<int32_t>(row_data[i] * inv_scale +
                                         (row_data[i] >= 0.0f ? 0.5f : -0.5f));
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        row_quant[i] = static_cast<int8_t>(v);
      }
    }
  }

  const bool is_per_channel = (data.hybrid_num_channels == output_depth);

  FullyConnectedParams op_params = FullyConnectedParamsFloat(params.activation);
  const float act_min = op_params.float_activation_min;
  const float act_max = op_params.float_activation_max;

#if defined(CMSIS_NN)
  int32_t* int32_output = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.hybrid_output_scratch_index));

  for (int b = 0; b < batches; ++b) {
    memset(int32_output, 0, output_depth * sizeof(int32_t));

    arm_nn_mat_mult_nt_t_s8_s32(input_quantized + b * accum_depth, filter_int8,
                                int32_output, 1, accum_depth, output_depth, 0,
                                1);

    const float row_scale = input_scales[b];
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      const float filter_scale = is_per_channel
                                     ? data.hybrid_filter_scales[out_c]
                                     : data.hybrid_filter_scales[0];
      float float_acc = int32_output[out_c] * row_scale * filter_scale;
      if (bias_float) {
        float_acc += bias_float[out_c];
      }
      float_acc = float_acc < act_min ? act_min : float_acc;
      float_acc = float_acc > act_max ? act_max : float_acc;
      output_data[b * output_depth + out_c] = float_acc;
    }
  }
#else
  for (int b = 0; b < batches; ++b) {
    const float row_scale = input_scales[b];
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        acc += static_cast<int32_t>(input_quantized[b * accum_depth + d]) *
               static_cast<int32_t>(filter_int8[out_c * accum_depth + d]);
      }
      const float filter_scale = is_per_channel
                                     ? data.hybrid_filter_scales[out_c]
                                     : data.hybrid_filter_scales[0];
      float float_acc = acc * row_scale * filter_scale;
      if (bias_float) {
        float_acc += bias_float[out_c];
      }
      float_acc = float_acc < act_min ? act_min : float_acc;
      float_acc = float_acc > act_max ? act_max : float_acc;
      output_data[b * output_depth + out_c] = float_acc;
    }
  }
#endif  // defined(CMSIS_NN)
  return kTfLiteOk;
}

const int kFullyConnectedInputTensor = 0;
const int kFullyConnectedWeightsTensor = 1;
const int kFullyConnectedBiasTensor = 2;
const int kFullyConnectedOutputTensor = 0;

FullyConnectedParams FullyConnectedParamsQuantized(
    const OpDataFullyConnected& op_data) {
  FullyConnectedParams op_params;
  op_params.input_offset = -op_data.input_zero_point;
  op_params.weights_offset = -op_data.filter_zero_point;
  op_params.output_offset = op_data.output_zero_point;
  op_params.output_multiplier = op_data.output_multiplier;
  op_params.output_shift = op_data.output_shift;
  op_params.quantized_activation_min = op_data.output_activation_min;
  op_params.quantized_activation_max = op_data.output_activation_max;
  return op_params;
}

FullyConnectedParams FullyConnectedParamsFloat(
    TfLiteFusedActivation activation) {
  FullyConnectedParams op_params;
  CalculateActivationRange(activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

TfLiteStatus CalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
#ifndef HEXAGON
  data->is_per_channel = false;
#endif

  if (data_type == kTfLiteFloat32) {
    return kTfLiteOk;
  }

  bool is_per_channel = false;
  if (filter->quantization.type == kTfLiteAffineQuantization &&
      filter->quantization.params != nullptr) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    is_per_channel = affine_quantization->scale->size > 1;
  }

  if (is_per_channel) {
// Hexagon currently does not support per-channel fully connected, and the
// existing hexagon support library is intolerant of data members being added to
// OpDataFullyConnected. As such, we have to be careful not to reference newer
// data members. This is why we use a local variable is_per_channel in common
// code, and only reference the data->is_per_channel in non-HEXAGON code.
#ifdef HEXAGON
    TF_LITE_ENSURE_MSG(
        context, !is_per_channel,
        "FullyConnected per-channel quantization not yet supported on Hexagon. "
        "Please set converter._experimental_disable_per_channel_quantization_"
        "for_dense_layers = True.");
#else
    data->is_per_channel = is_per_channel;
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const int per_channel_quantization_size = affine_quantization->scale->size;

    //  Currently only Int8/Int16 are supported for per channel quantization.
    TF_LITE_ENSURE(
        context,
        (input->type == kTfLiteInt8 && filter->type != kTfLiteInt4) ||
            (input->type == kTfLiteInt16 && filter->type != kTfLiteInt4));

    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      per_channel_quantization_size);

    TF_LITE_ENSURE_EQ(
        context, per_channel_quantization_size,
        filter->dims->data[affine_quantization->quantized_dimension]);

    data->per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, per_channel_quantization_size * sizeof(int32_t)));
    data->per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, per_channel_quantization_size * sizeof(int32_t)));

    // Populate multiplier and shift using affine quantization.
    const float input_scale = input->params.scale;
    const float output_scale = output->params.scale;
    const float* filter_scales = affine_quantization->scale->data;

    for (int i = 0; i < per_channel_quantization_size; ++i) {
      const float scale = filter_scales[i];
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t significand;
      int channel_shift;
      QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
      data->per_channel_output_multiplier[i] = significand;
      data->per_channel_output_shift[i] = channel_shift;
    }
#endif
  } else {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  // Filter weights will always be symmetric quantized since we only support
  // int8 quantization. See
  // https://github.com/tensorflow/tensorflow/issues/44912 for additional
  // context.
  TFLITE_DCHECK(filter->params.zero_point == 0);

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  return CalculateActivationRangeQuantized(context, activation, output,
                                           &data->output_activation_min,
                                           &data->output_activation_max);
}

}  // namespace tflite
