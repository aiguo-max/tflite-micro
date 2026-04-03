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

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

void* XtensaInitFullyConnected(TfLiteContext* context, const char* buffer,
                               size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if !defined(VISION_P6)
  void* raw =
      context->AllocatePersistentBuffer(context, sizeof(OpDataFullyConnected));
  memset(raw, 0, sizeof(OpDataFullyConnected));
  return raw;
#else
  void* data = context->AllocatePersistentBuffer(
      context, sizeof(XtensaFullyConnectedOpData));
  memset(data, 0, sizeof(XtensaFullyConnectedOpData));
#if !defined(HIFIMINI)
  if (InitXtensaContext()) {
    return nullptr;
  }
#endif
  return data;
#endif  // defined(VISION_P6)
}

TfLiteStatus XtensaCalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
#if defined(HIFIMINI)
    if (input->type == kTfLiteInt8) {
      QuantizeMultiplierForInt24(real_multiplier, &data->output_multiplier,
                                 &data->output_shift);
    } else {
      QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                         &data->output_shift);
    }
#else
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
#endif

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
  return kTfLiteOk;
}

TfLiteStatus XtensaPrepareFullyConnected(TfLiteContext* context,
                                         TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

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

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  // Stash hybrid scratch sizes before deallocating temp tensors.
  // RequestScratchBufferInArena requires temp allocations to be clean
  // (head_temp_ == next_temp_), so scratch requests must happen after
  // DeallocateTempTfLiteTensor calls below.
  int hybrid_input_size = 0;
  int hybrid_output_size = 0;
  int hybrid_num_channels = 0;
  int hybrid_batches = 0;
  float* hybrid_filter_scales = nullptr;
  if (data->is_hybrid) {
    const auto* aq =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TF_LITE_ENSURE(context, aq != nullptr);
    TF_LITE_ENSURE(context, aq->scale != nullptr);
    hybrid_num_channels = aq->scale->size;
    hybrid_input_size =
        RuntimeShape(input->dims->size,
                     reinterpret_cast<const int32_t*>(input->dims->data))
            .FlatSize();
    hybrid_output_size =
        RuntimeShape(output->dims->size,
                     reinterpret_cast<const int32_t*>(output->dims->data))
            .FlatSize();
    const int accum_depth = filter->dims->data[1];
    hybrid_batches = hybrid_input_size / accum_depth;
    hybrid_filter_scales =
        static_cast<float*>(context->AllocatePersistentBuffer(
            context, aq->scale->size * sizeof(float)));
    memcpy(hybrid_filter_scales, aq->scale->data,
           aq->scale->size * sizeof(float));
  }

  TFLITE_DCHECK_GE(GetTensorShape(output).DimensionsCount(), 1);

  TF_LITE_ENSURE_OK(context, XtensaCalculateOpDataFullyConnected(
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

  if (data->is_hybrid) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, hybrid_input_size * sizeof(int8_t),
        &data->hybrid_input_scratch_index));
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, hybrid_batches * sizeof(float),
        &data->hybrid_scales_scratch_index));
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, hybrid_output_size * sizeof(int32_t),
        &data->hybrid_output_scratch_index));
    data->hybrid_num_channels = hybrid_num_channels;
    data->hybrid_filter_scales = hybrid_filter_scales;
    data->hybrid_row_sums = nullptr;
  }

#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, FullyConnectedPrepareVision(context, node));
#endif  // defined(VISION_P6)

  return kTfLiteOk;
}

}  // namespace tflite
