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

#include <algorithm>
#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#if defined(CMSIS_NN)
#include "Include/arm_nnsupportfunctions.h"
#endif

namespace tflite {

TfLiteStatus ConvEvalHybrid(TfLiteContext* context,
                             const TfLiteConvParams& params,
                             const OpDataConv& data,
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

#if defined(CMSIS_NN)
  // CMSIS-NN accelerated path: im2col + matmul per output row
  const int patch_size = filter_height * filter_width * input_depth;
  int8_t* im2col = static_cast<int8_t*>(
      context->GetScratchBuffer(context, data.hybrid_im2col_scratch_index));
  int32_t* int32_output = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.hybrid_output_scratch_index));

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      // Fill im2col for this output row
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int fy = 0; fy < filter_height; ++fy) {
          for (int fx = 0; fx < filter_width; ++fx) {
            const int in_y = in_y_origin + dilation_height_factor * fy;
            const int in_x = in_x_origin + dilation_width_factor * fx;
            int8_t* dst = im2col + out_x * patch_size +
                          (fy * filter_width + fx) * input_depth;
            if (in_y >= 0 && in_y < input_height &&
                in_x >= 0 && in_x < input_width) {
              memcpy(dst,
                     input_quantized + Offset(input_shape, batch, in_y, in_x, 0),
                     input_depth);
            } else {
              memset(dst, 0, input_depth);
            }
          }
        }
      }

      // Matmul: [output_w × patch_size] × [output_depth × patch_size]^T
      memset(int32_output, 0, output_width * output_depth * sizeof(int32_t));
      arm_nn_mat_mult_nt_t_s8_s32(
          im2col, filter_int8, int32_output,
          output_width, patch_size, output_depth, 0, 1);

      // Dequantize int32 → float32
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
          float float_acc = int32_output[out_x * output_depth + out_c] *
                            input_scale * data.hybrid_filter_scales[out_c];
          if (bias_float) {
            float_acc += bias_float[out_c];
          }
          float_acc = float_acc < act_min ? act_min : float_acc;
          float_acc = float_acc > act_max ? act_max : float_acc;
          output_data[Offset(output_shape, batch, out_y, out_x, out_c)] =
              float_acc;
        }
      }
    }
  }
#else
  // Reference C fallback
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
#endif  // defined(CMSIS_NN)
  return kTfLiteOk;
}

const int kConvInputTensor = 0;
const int kConvWeightsTensor = 1;
const int kConvBiasTensor = 2;
const int kConvOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
const int kConvQuantizedDimension = 0;

// Returns a ConvParams struct with all the parameters needed for a
// float computation.
ConvParams ConvParamsFloat(const TfLiteConvParams& params,
                           const OpDataConv& data) {
  ConvParams op_params;
  CalculateActivationRange(params.activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  op_params.padding_type = tflite::micro::RuntimePaddingType(params.padding);
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  return op_params;
}

// Returns a ConvParams struct with all the parameters needed for a
// quantized computation.
ConvParams ConvParamsQuantized(const TfLiteConvParams& params,
                               const OpDataConv& data) {
  ConvParams op_params;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = -data.filter_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier = data.output_multiplier;
  op_params.output_shift = -data.output_shift;
  op_params.padding_type = tflite::micro::RuntimePaddingType(params.padding);
  op_params.padding_values.height = data.padding.height;
  op_params.padding_values.width = data.padding.width;
  op_params.stride_height = params.stride_height;
  op_params.stride_width = params.stride_width;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;
  return op_params;
}

void* ConvInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* raw = context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
  memset(raw, 0, sizeof(OpDataConv));
  return raw;
}

TfLiteStatus CalculateOpDataConv(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteConvParams& params, int width,
                                 int height, int filter_width,
                                 int filter_height, int out_width,
                                 int out_height, const TfLiteType data_type,
                                 OpDataConv* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params.padding;
  data->padding = ComputePaddingHeightWidth(
      params.stride_height, params.stride_width, params.dilation_height_factor,
      params.dilation_width_factor, height, width, filter_height, filter_width,
      padding, &out_height, &out_width);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kConvBiasTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params.activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        output_channels));
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpDataConv* data = static_cast<OpDataConv*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  data->is_hybrid =
      (input->type == kTfLiteFloat32 && filter->type == kTfLiteInt8);
  TF_LITE_ENSURE_MSG(
      context,
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteFloat32 || filter->type == kTfLiteInt8)) ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 &&
           (filter->type == kTfLiteInt4 || filter->type == kTfLiteInt8)),
      "Unsupported input/filter type combination.");

  const int input_width = input->dims->data[2];
  const int input_height = input->dims->data[1];
  const int filter_width = filter->dims->data[2];
  const int filter_height = filter->dims->data[1];
  const int output_width = output->dims->data[2];
  const int output_height = output->dims->data[1];

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
    TFLITE_DCHECK(affine_quantization->zero_point != nullptr);

    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, data));

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  if (data->is_hybrid) {
    const auto* aq =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TF_LITE_ENSURE(context, aq != nullptr);
    TF_LITE_ENSURE(context, aq->scale != nullptr);

    int input_size =
        RuntimeShape(input->dims->size,
                     reinterpret_cast<const int32_t*>(input->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(
        context, input_size * sizeof(int8_t), &data->hybrid_input_scratch_index);

    data->hybrid_num_channels = aq->scale->size;
    float* scales_copy = static_cast<float*>(context->AllocatePersistentBuffer(
        context, aq->scale->size * sizeof(float)));
    memcpy(scales_copy, aq->scale->data, aq->scale->size * sizeof(float));
    data->hybrid_filter_scales = scales_copy;
    data->hybrid_row_sums = nullptr;

    // Allocate im2col scratch: one output row of patches
    const int filter_h = filter->dims->data[1];
    const int filter_w = filter->dims->data[2];
    const int input_depth = input->dims->data[3];
    const int output_w = output->dims->data[2];
    const int patch_size = filter_h * filter_w * input_depth;
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, output_w * patch_size * sizeof(int8_t),
        &data->hybrid_im2col_scratch_index));

    // Allocate int32 output scratch: one output row
    const int output_depth = filter->dims->data[0];
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, output_w * output_depth * sizeof(int32_t),
        &data->hybrid_output_scratch_index));
  }

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  if (micro_context->IsTensorCompressed(node, kConvWeightsTensor) &&
      filter->type == kTfLiteInt4) {
    MicroPrintf("Compression not supported with INT4 tensors");
    return kTfLiteError;
  }
  data->weights_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node,
                                                        kConvWeightsTensor);
  data->bias_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}
}  // namespace tflite
