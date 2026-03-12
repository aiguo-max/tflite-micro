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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* FullyConnectedInit(TfLiteContext* context, const char* buffer,
                         size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* raw = context->AllocatePersistentBuffer(context,
                                                sizeof(OpDataFullyConnected));
  memset(raw, 0, sizeof(OpDataFullyConnected));
  return raw;
}

TfLiteStatus FullyConnectedPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  data->is_hybrid =
      (input->type == kTfLiteFloat32 && filter->type == kTfLiteInt8);

  if ((input->type == kTfLiteFloat32 &&
       filter->type != kTfLiteFloat32 && filter->type != kTfLiteInt8) ||
      (input->type == kTfLiteInt8 &&
       (filter->type != kTfLiteInt8 && filter->type != kTfLiteInt4)) ||
      (input->type == kTfLiteInt16 && filter->type != kTfLiteInt8)) {
    MicroPrintf("Input type: %s with filter type: %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(filter->type));
    return kTfLiteError;
  }

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  if (data->is_hybrid) {
    int input_size =
        RuntimeShape(input->dims->size,
                     reinterpret_cast<const int32_t*>(input->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(
        context, input_size * sizeof(int8_t), &data->hybrid_input_scratch_index);

    const auto* aq =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    data->hybrid_num_channels = aq->scale->size;
    float* scales_copy = static_cast<float*>(context->AllocatePersistentBuffer(
        context, aq->scale->size * sizeof(float)));
    memcpy(scales_copy, aq->scale->data, aq->scale->size * sizeof(float));
    data->hybrid_filter_scales = scales_copy;

    const int output_depth = filter->dims->data[0];
    const int accum_depth = filter->dims->data[1];
    int32_t* row_sums = static_cast<int32_t*>(context->AllocatePersistentBuffer(
        context, output_depth * sizeof(int32_t)));
    const int8_t* filter_data = GetTensorData<int8_t>(filter);
    for (int oc = 0; oc < output_depth; ++oc) {
      int32_t sum = 0;
      const int8_t* row = filter_data + oc * accum_depth;
      for (int i = 0; i < accum_depth; ++i) {
        sum += row[i];
      }
      row_sums[oc] = sum;
    }
    data->hybrid_row_sums = row_sums;
  }

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  if (micro_context->IsTensorCompressed(node, kFullyConnectedWeightsTensor) &&
      filter->type == kTfLiteInt4) {
    MicroPrintf("Compression not supported with INT4 tensors");
    return kTfLiteError;
  }
  data->weights_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(
          node, kFullyConnectedWeightsTensor);
  data->bias_scratch_index = micro_context->AllocateDecompressionScratchBuffer(
      node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kFullyConnectedWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
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

        const int output_depth = filter_shape.Dims(0);
        const int accum_depth = filter_shape.Dims(1);
        const int batches = input_size / accum_depth;

        FullyConnectedParams op_params = FullyConnectedParamsFloat(params->activation);
        const float act_min = op_params.float_activation_min;
        const float act_max = op_params.float_activation_max;

        for (int b = 0; b < batches; ++b) {
          for (int out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
              acc += static_cast<int32_t>(input_quantized[b * accum_depth + d]) *
                     static_cast<int32_t>(filter_int8[out_c * accum_depth + d]);
            }
            float float_acc = acc * input_scale * data.hybrid_filter_scales[out_c];
            if (bias_float) {
              float_acc += bias_float[out_c];
            }
            float_acc = float_acc < act_min ? act_min : float_acc;
            float_acc = float_acc > act_max ? act_max : float_acc;
            output_data[b * output_depth + out_c] = float_acc;
          }
        }
      } else {
        tflite::reference_ops::FullyConnected(
            FullyConnectedParamsFloat(params->activation),
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
            tflite::micro::GetTensorData<float>(output));
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
          tflite::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
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
          data.is_per_channel
              ? tflite::reference_integer_ops::FullyConnectedPerChannel(
                    FullyConnectedParamsQuantized(data),
                    data.per_channel_output_multiplier,
                    reinterpret_cast<const int*>(data.per_channel_output_shift),
                    tflite::micro::GetTensorShape(input),
                    tflite::micro::GetTensorData<int8_t>(input),
                    tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(
                        micro_context, filter, weights_comp_td,
                        data.weights_scratch_index),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(
                        micro_context, bias, bias_comp_td,
                        data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(filter),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorShape(output),
                    tflite::micro::GetTensorData<int8_t>(output))
              : tflite::reference_integer_ops::FullyConnected(
                    FullyConnectedParamsQuantized(data),
                    tflite::micro::GetTensorShape(input),
                    tflite::micro::GetTensorData<int8_t>(input),
                    tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(
                        micro_context, filter, weights_comp_td,
                        data.weights_scratch_index),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(
                        micro_context, bias, bias_comp_td,
                        data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(filter),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorShape(output),
                    tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          if (bias == nullptr || bias->type == kTfLiteInt32) {
            data.is_per_channel
                ? tflite::reference_integer_ops::FullyConnectedPerChannel(
                      FullyConnectedParamsQuantized(data),
                      data.per_channel_output_multiplier,
                      reinterpret_cast<const int*>(
                          data.per_channel_output_shift),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output))
                : tflite::reference_integer_ops::FullyConnected(
                      FullyConnectedParamsQuantized(data),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
          } else if (bias->type == kTfLiteInt64) {
            data.is_per_channel
                ? tflite::reference_integer_ops::FullyConnectedPerChannel(
                      FullyConnectedParamsQuantized(data),
                      data.per_channel_output_multiplier,
                      reinterpret_cast<const int*>(
                          data.per_channel_output_shift),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output))
                : tflite::reference_integer_ops::FullyConnected(
                      FullyConnectedParamsQuantized(data),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
          }
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedInit, FullyConnectedPrepare,
                                   FullyConnectedEval);
}

TFLMInferenceRegistration RegisterInference_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedEval);
}

}  // namespace tflite
