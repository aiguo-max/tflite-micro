/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      if (data.is_hybrid) {
        const int8_t* filter_int8 = tflite::micro::GetTensorData<int8_t>(filter);
        const float* input_data = tflite::micro::GetTensorData<float>(input);
        const float* bias_float = tflite::micro::GetOptionalTensorData<float>(bias);
        float* output_data = tflite::micro::GetTensorData<float>(output);

        const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
        const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
        const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);

        const int input_size = input_shape.FlatSize();
        int8_t* input_quantized = static_cast<int8_t*>(
            context->GetScratchBuffer(context, data.hybrid_input_scratch_index));

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

        const int batches = MatchingDim(input_shape, 0, output_shape, 0);
        const int input_depth = input_shape.Dims(3);
        const int output_depth = filter_shape.Dims(0);
        const int input_height = input_shape.Dims(1);
        const int input_width = input_shape.Dims(2);
        const int filter_height = filter_shape.Dims(1);
        const int filter_width = filter_shape.Dims(2);
        const int output_height = output_shape.Dims(1);
        const int output_width = output_shape.Dims(2);

        const int stride_width = params.stride_width;
        const int stride_height = params.stride_height;
        const int dilation_width_factor = params.dilation_width_factor;
        const int dilation_height_factor = params.dilation_height_factor;
        const int pad_width = data.padding.width;
        const int pad_height = data.padding.height;

        ConvParams op_params = ConvParamsFloat(params, data);
        const float act_min = op_params.float_activation_min;
        const float act_max = op_params.float_activation_max;

        for (int batch = 0; batch < batches; ++batch) {
          for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
              for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                const int in_y_origin = (out_y * stride_height) - pad_height;

                int32_t acc = 0;

                for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                      const int in_x = in_x_origin + dilation_width_factor * filter_x;
                      const int in_y = in_y_origin + dilation_height_factor * filter_y;

                      if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height)) {
                        int input_idx = Offset(input_shape, batch, in_y, in_x, in_channel);
                        int filter_idx = Offset(filter_shape, out_channel, filter_y, filter_x, in_channel);
                        acc += static_cast<int32_t>(input_quantized[input_idx]) *
                               static_cast<int32_t>(filter_int8[filter_idx]);
                      }
                    }
                  }
                }

                float float_acc = acc * input_scale * data.hybrid_filter_scales[out_channel];
                if (bias_float) {
                  float_acc += bias_float[out_channel];
                }
                
                float_acc = float_acc < act_min ? act_min : float_acc;
                float_acc = float_acc > act_max ? act_max : float_acc;
                
                int out_idx = Offset(output_shape, batch, out_y, out_x, out_channel);
                output_data[out_idx] = float_acc;
              }
            }
          }
        }
      } else {
        tflite::reference_ops::Conv(
            ConvParamsFloat(params, data),
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<float>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<float>(micro_context, filter,
                                                weights_comp_td,
                                                data.weights_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<float>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<float>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<float>(output),
            tflite::micro::GetTensorShape(nullptr), nullptr);
      }
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 weights_comp_td,
                                                 data.weights_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<int32_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else if (bias->type == kTfLiteInt64) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 weights_comp_td,
                                                 data.weights_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetTensorData<int64_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetTensorData<std::int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                   weights_comp_td,
                                                   data.weights_scratch_index),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(
                  micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare, ConvEval);
}

}  // namespace tflite
