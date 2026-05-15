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
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "xa_nnlib_api.h"

namespace tflite {

TfLiteStatus ConvEvalHybridHifi(
    TfLiteContext* context,
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

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = filter_shape.Dims(0);

  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int dilation_height_factor = params.dilation_height_factor;
  const int dilation_width_factor = params.dilation_width_factor;
  const int pad_height = data.padding.height;
  const int pad_width = data.padding.width;

  const int input_size = input_shape.FlatSize();
  int8_t* input_quantized = static_cast<int8_t*>(
      context->GetScratchBuffer(context, data.hybrid_input_scratch_index));
  int8_t* im2col_buffer = static_cast<int8_t*>(
      context->GetScratchBuffer(context, data.hybrid_im2col_scratch_index));

  // Asymmetric quantization matching TFLite HybridConvPerChannel semantics:
  //   scale = (max(0, rmax) - min(0, rmin)) / 255
  //   zp chosen as the endpoint with smaller rounding error
  // The xa_nn_matXvec API has no lhs_offset parameter, so zp compensation
  // happens externally via row_sums after the matmul.
  float min_val = input_data[0];
  float max_val = input_data[0];
  for (int i = 1; i < input_size; ++i) {
    if (input_data[i] < min_val) min_val = input_data[i];
    if (input_data[i] > max_val) max_val = input_data[i];
  }

  const double rmin = std::min(0.0, static_cast<double>(min_val));
  const double rmax = std::max(0.0, static_cast<double>(max_val));
  float input_scale;
  int32_t input_zp;
  if (rmin == rmax) {
    input_scale = 1.0f;
    input_zp = 0;
    memset(input_quantized, 0, input_size * sizeof(int8_t));
  } else {
    const double scale = (rmax - rmin) / 255.0;
    const double zp_from_min = -128.0 - rmin / scale;
    const double zp_from_max = 127.0 - rmax / scale;
    const double err_min = std::abs(-128.0) + std::abs(rmin / scale);
    const double err_max = std::abs(127.0) + std::abs(rmax / scale);
    const double zp_double = err_min < err_max ? zp_from_min : zp_from_max;
    int32_t zp = static_cast<int32_t>(std::round(zp_double));
    if (zp < -128) zp = -128;
    if (zp > 127) zp = 127;
    input_scale = static_cast<float>(scale);
    input_zp = zp;
    const float inv_scale = static_cast<float>(1.0 / scale);
    for (int i = 0; i < input_size; ++i) {
      int32_t v = static_cast<int32_t>(std::round(
          static_cast<float>(zp) + input_data[i] * inv_scale));
      v = v < -128 ? -128 : (v > 127 ? 127 : v);
      input_quantized[i] = static_cast<int8_t>(v);
    }
  }

  ConvParams op_params = ConvParamsFloat(params, data);
  const float act_min = op_params.float_activation_min;
  const float act_max = op_params.float_activation_max;

  const int patch_size = filter_height * filter_width * input_depth;

  if ((patch_size & 3) != 0) {
    return ConvEvalHybrid(context, params, data, input, filter, bias, output);
  }

  TF_LITE_ENSURE(context, output_depth <= 512);
  TF_LITE_ENSURE(context, output_width <= 256);

  int8_t s_zero_bias[512] = {};

  float combined_scales[512];
  const bool is_per_channel = (data.hybrid_num_channels == output_depth);
  for (int c = 0; c < output_depth; ++c) {
    const float filter_scale = is_per_channel ? data.hybrid_filter_scales[c]
                                              : data.hybrid_filter_scales[0];
    combined_scales[c] = input_scale * filter_scale;
  }

  const int tile_w = kHybridConvTileWidth;
  const int32_t* row_sums = data.hybrid_row_sums;

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* batch_input = input_quantized + batch * input_height * input_width * input_depth;

    for (int out_y = 0; out_y < output_height; ++out_y) {
      float* row_out = output_data +
          (batch * output_height + out_y) * output_width * output_depth;

      for (int tile_start = 0; tile_start < output_width; tile_start += tile_w) {
        const int cur_tile = (tile_start + tile_w <= output_width)
                                 ? tile_w
                                 : (output_width - tile_start);

        for (int t = 0; t < cur_tile; ++t) {
          const int out_x = tile_start + t;
          int8_t* patch = im2col_buffer + t * patch_size;

          for (int fy = 0; fy < filter_height; ++fy) {
            const int in_y = out_y * stride_height - pad_height +
                             fy * dilation_height_factor;
            if (in_y >= 0 && in_y < input_height) {
              for (int fx = 0; fx < filter_width; ++fx) {
                const int in_x = out_x * stride_width - pad_width +
                                 fx * dilation_width_factor;
                if (in_x >= 0 && in_x < input_width) {
                  memcpy(patch + (fy * filter_width + fx) * input_depth,
                         batch_input +
                             (in_y * input_width + in_x) * input_depth,
                         input_depth * sizeof(int8_t));
                } else {
                  // Padded positions represent 0.0 in float; quantized value of
                  // 0.0 is zp, so (q - zp) = 0 contributes nothing to acc.
                  memset(patch + (fy * filter_width + fx) * input_depth,
                         static_cast<int8_t>(input_zp),
                         input_depth * sizeof(int8_t));
                }
              }
            } else {
              memset(patch + fy * filter_width * input_depth,
                     static_cast<int8_t>(input_zp),
                     filter_width * input_depth * sizeof(int8_t));
            }
          }
        }

        // Reuse the output float buffer for this tile as int32 accumulator:
        // sizeof(float) == sizeof(int32_t) == 4. The post-loop converts each
        // int32 acc back to float in-place (forward iteration is safe because
        // each 4-byte slot is read once as int32 and written once as float).
        int32_t* tile_acc = reinterpret_cast<int32_t*>(
            row_out + tile_start * output_depth);

        int8_t* vec_ptrs[kHybridConvTileWidth];
        int32_t* out_ptrs[kHybridConvTileWidth];
        for (int t = 0; t < cur_tile; ++t) {
          vec_ptrs[t] = im2col_buffer + t * patch_size;
          out_ptrs[t] = tile_acc + t * output_depth;
        }

        int vec_done = 0;
        while (vec_done < cur_tile) {
          int batch_count = cur_tile - vec_done;
          if (batch_count >= 2 && (batch_count & 1)) batch_count--;
          if (batch_count >= 2) {
            xa_nn_matXvec_batch_8x8_32(
                out_ptrs + vec_done,
                const_cast<int8_t*>(filter_int8),
                vec_ptrs + vec_done,
                s_zero_bias,
                output_depth,
                patch_size,
                patch_size,
                0,
                0,
                batch_count);
            vec_done += batch_count;
          } else {
            xa_nn_matXvec_8x8_32(
                out_ptrs[vec_done],
                const_cast<int8_t*>(filter_int8),
                nullptr,
                vec_ptrs[vec_done],
                nullptr,
                s_zero_bias,
                output_depth,
                patch_size,
                0,
                patch_size,
                0,
                0,
                0);
            vec_done++;
          }
        }

        for (int t = 0; t < cur_tile; ++t) {
          float* out_ptr = row_out + (tile_start + t) * output_depth;
          for (int c = 0; c < output_depth; ++c) {
            int32_t acc_int32;
            memcpy(&acc_int32, &out_ptr[c], sizeof(int32_t));
            const int32_t acc_compensated = acc_int32 - input_zp * row_sums[c];
            float v = static_cast<float>(acc_compensated) * combined_scales[c];
            if (bias_float) v += bias_float[c];
            v = v < act_min ? act_min : (v > act_max ? act_max : v);
            out_ptr[c] = v;
          }
        }
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace tflite

#endif  // defined(HIFI4) || defined(HIFI5) || defined(XTENSA)