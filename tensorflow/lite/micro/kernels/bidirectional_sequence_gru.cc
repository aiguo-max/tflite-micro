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
#include <cstring>
#include <limits>

#if defined(CMSIS_NN)
#include "Include/arm_nnsupportfunctions.h"
#elif defined(XTENSA) && defined(HIFI4)
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#endif

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

// GRU equations (reset-after variant, Keras default):
//   z = sigmoid(xw[0:h] + hw[0:h])
//   r = sigmoid(xw[h:2h] + hw[h:2h])
//   n = tanh(xw[2h:3h] + r * hw[2h:3h])
//   h_new = (1 - z) * n + z * h_prev
//
// Weight layout: [3*hidden, dim] with z/r/n gates stacked vertically.

#if !(defined(XTENSA) && defined(HIFI4))
static inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
#endif

static inline size_t AlignUp4(size_t v) { return (v + 3) & ~static_cast<size_t>(3); }

void FloatMatVec(int rows, int cols, const float* matrix, const float* vec,
                 const float* bias, float* output) {
  for (int i = 0; i < rows; ++i) {
    float sum = bias[i];
    for (int j = 0; j < cols; ++j) {
      sum += vec[j] * matrix[i * cols + j];
    }
    output[i] = sum;
  }
}

void SymmetricQuantize(const float* input, int size, int8_t* output,
                       float* scale) {
  float min_val = input[0];
  float max_val = input[0];
  for (int i = 1; i < size; ++i) {
    if (input[i] < min_val) min_val = input[i];
    if (input[i] > max_val) max_val = input[i];
  }
  const float range = std::max(std::abs(min_val), std::abs(max_val));
  if (range == 0.0f) {
    *scale = 1.0f;
    std::memset(output, 0, size * sizeof(int8_t));
    return;
  }
  *scale = range / 127.0f;
  const float inv_scale = 127.0f / range;
  for (int i = 0; i < size; ++i) {
    int32_t v = static_cast<int32_t>(input[i] * inv_scale +
                                     (input[i] >= 0.0f ? 0.5f : -0.5f));
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    output[i] = static_cast<int8_t>(v);
  }
}

void HybridMatVec(int rows, int cols, const int8_t* matrix, float filter_scale,
                  const float* vec, const float* bias, float* output,
                  int8_t* quantized_vec, int32_t* acc_buf) {
  float input_scale;
  SymmetricQuantize(vec, cols, quantized_vec, &input_scale);

#if defined(CMSIS_NN)
  std::memset(acc_buf, 0, rows * sizeof(int32_t));
  arm_nn_mat_mult_nt_t_s8_s32(quantized_vec, matrix, acc_buf, 1, cols, rows, 0,
                              1);
  const float combined_scale = input_scale * filter_scale;
  for (int i = 0; i < rows; ++i) {
    output[i] = static_cast<float>(acc_buf[i]) * combined_scale + bias[i];
  }
#elif defined(XTENSA) && defined(HIFI4)
  if ((cols & 3) == 0) {
    alignas(8) int8_t zero_bias[192] = {};
    xa_nn_matXvec_8x8_32(
        acc_buf,
        const_cast<int8_t*>(matrix), nullptr,
        quantized_vec, nullptr,
        zero_bias,
        rows, cols, 0, cols, 0, 0, 0);
    const float combined_scale = input_scale * filter_scale;
    for (int i = 0; i < rows; ++i) {
      output[i] = static_cast<float>(acc_buf[i]) * combined_scale + bias[i];
    }
  } else {
    for (int i = 0; i < rows; ++i) {
      int32_t acc = 0;
      for (int j = 0; j < cols; ++j) {
        acc += static_cast<int32_t>(quantized_vec[j]) *
               static_cast<int32_t>(matrix[i * cols + j]);
      }
      output[i] = static_cast<float>(acc) * input_scale * filter_scale + bias[i];
    }
  }
#else
  for (int i = 0; i < rows; ++i) {
    int32_t acc = 0;
    for (int j = 0; j < cols; ++j) {
      acc += static_cast<int32_t>(quantized_vec[j]) *
             static_cast<int32_t>(matrix[i * cols + j]);
    }
    output[i] = static_cast<float>(acc) * input_scale * filter_scale + bias[i];
  }
#endif
}

void ApplyGruGates(int n_output, const float* scratch_xw,
                   const float* scratch_hw, const float* h_prev,
                   float* h_new) {
#if defined(XTENSA) && defined(HIFI4)
  constexpr int kMax = 64;
  float tmp[kMax];
  float z_vec[kMax];
  float r_vec[kMax];
  float n_vec[kMax];
  // z = sigmoid(xw[0:h] + hw[0:h])
  for (int i = 0; i < n_output; ++i)
    tmp[i] = scratch_xw[i] + scratch_hw[i];
  xa_nn_vec_sigmoid_f32_f32(z_vec, tmp, n_output);
  // r = sigmoid(xw[h:2h] + hw[h:2h])
  for (int i = 0; i < n_output; ++i)
    tmp[i] = scratch_xw[n_output + i] + scratch_hw[n_output + i];
  xa_nn_vec_sigmoid_f32_f32(r_vec, tmp, n_output);
  // n = tanh(xw[2h:3h] + r * hw[2h:3h])
  for (int i = 0; i < n_output; ++i)
    tmp[i] = scratch_xw[2 * n_output + i] +
             r_vec[i] * scratch_hw[2 * n_output + i];
  xa_nn_vec_tanh_f32_f32(n_vec, tmp, n_output);
  // h_new = (1 - z) * n + z * h_prev
  for (int i = 0; i < n_output; ++i)
    h_new[i] = (1.0f - z_vec[i]) * n_vec[i] + z_vec[i] * h_prev[i];
#else
  for (int i = 0; i < n_output; ++i) {
    float z = Sigmoid(scratch_xw[i] + scratch_hw[i]);
    float r = Sigmoid(scratch_xw[n_output + i] + scratch_hw[n_output + i]);
    float n = std::tanh(scratch_xw[2 * n_output + i] +
                        r * scratch_hw[2 * n_output + i]);
    h_new[i] = (1.0f - z) * n + z * h_prev[i];
  }
#endif
}

void GruCellStep(int n_input, int n_output, const float* input,
                 const float* h_prev, const float* input_kernel,
                 const float* input_bias, const float* recurrent_kernel,
                 const float* recurrent_bias, float* h_new, float* scratch_xw,
                 float* scratch_hw) {
  const int n3 = 3 * n_output;
  FloatMatVec(n3, n_input, input_kernel, input, input_bias, scratch_xw);
  FloatMatVec(n3, n_output, recurrent_kernel, h_prev, recurrent_bias,
              scratch_hw);
  ApplyGruGates(n_output, scratch_xw, scratch_hw, h_prev, h_new);
}

void GruCellStepHybrid(int n_input, int n_output, const float* input,
                       const float* h_prev, const int8_t* input_kernel,
                       float ik_scale, const float* input_bias,
                       const int8_t* recurrent_kernel, float rk_scale,
                       const float* recurrent_bias, float* h_new,
                       float* scratch_xw, float* scratch_hw, int8_t* quant_buf,
                       int32_t* acc_buf) {
  const int n3 = 3 * n_output;
  HybridMatVec(n3, n_input, input_kernel, ik_scale, input, input_bias,
               scratch_xw, quant_buf, acc_buf);
  HybridMatVec(n3, n_output, recurrent_kernel, rk_scale, h_prev,
               recurrent_bias, scratch_hw, quant_buf, acc_buf);
  ApplyGruGates(n_output, scratch_xw, scratch_hw, h_prev, h_new);
}

// Bidirectional GRU custom op for TFLM.
// Input tensors:
//   0: input               [batch, seq_len, input_dim] or [seq_len, input_dim]
//   1-4: fw_input_kernel, fw_input_bias, fw_recurrent_kernel, fw_recurrent_bias
//   5-8: bw_input_kernel, bw_input_bias, bw_recurrent_kernel, bw_recurrent_bias
// Output:
//   0: [seq_len, 2*hidden] — forward in left half, backward in right half

constexpr int kInputTensor = 0;
constexpr int kFwInputKernel = 1;
constexpr int kFwInputBias = 2;
constexpr int kFwRecurrentKernel = 3;
constexpr int kFwRecurrentBias = 4;
constexpr int kBwInputKernel = 5;
constexpr int kBwInputBias = 6;
constexpr int kBwRecurrentKernel = 7;
constexpr int kBwRecurrentBias = 8;
constexpr int kNumInputs = 9;
constexpr int kOutputTensor = 0;
constexpr int kNumOutputs = 1;

struct BiGruOpData {
  int scratch_index;
  bool is_hybrid;
  int hybrid_quant_offset;
  int hybrid_acc_offset;
  float fw_ik_scale;
  float fw_rk_scale;
  float bw_ik_scale;
  float bw_rk_scale;
};

void* BiGruInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(BiGruOpData));
}

TfLiteStatus BiGruPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), kNumInputs);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), kNumOutputs);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, input->dims->size == 2 || input->dims->size == 3);

  TfLiteTensor* fw_ik =
      micro_context->AllocateTempInputTensor(node, kFwInputKernel);
  TF_LITE_ENSURE(context, fw_ik != nullptr);

  TfLiteTensor* fw_rk =
      micro_context->AllocateTempInputTensor(node, kFwRecurrentKernel);
  TF_LITE_ENSURE(context, fw_rk != nullptr);
  const int n_output = fw_rk->dims->data[1];

  const int ndims = input->dims->size;
  const int n_input =
      (ndims == 3) ? input->dims->data[2] : input->dims->data[1];
  const int seq_len =
      (ndims == 3) ? input->dims->data[1] : input->dims->data[0];

  BiGruOpData* op_data = reinterpret_cast<BiGruOpData*>(node->user_data);
  op_data->is_hybrid = (fw_ik->type == kTfLiteInt8);

  // Scratch layout (all float region first, then hybrid int8/int32):
  //   [h_state: n_output] [scratch_xw: 3*n_output] [scratch_hw: 3*n_output]
  //   [fw_output: seq_len*n_output]
  //   --- hybrid only (4-byte aligned) ---
  //   [quant_buf: max(n_input, n_output) int8, padded to 4-byte]
  //   [acc_buf: 3*n_output int32]
  size_t float_bytes =
      static_cast<size_t>(7 * n_output + seq_len * n_output) * sizeof(float);
  size_t scratch_size = float_bytes;

  if (op_data->is_hybrid) {
    const int max_vec = std::max(n_input, n_output);
    size_t quant_bytes = AlignUp4(max_vec * sizeof(int8_t));
    op_data->hybrid_quant_offset = static_cast<int>(float_bytes);
    op_data->hybrid_acc_offset =
        static_cast<int>(float_bytes + quant_bytes);
    scratch_size = float_bytes + quant_bytes + 3 * n_output * sizeof(int32_t);

    auto get_scale = [](TfLiteTensor* t) -> float {
      const auto* aq =
          static_cast<TfLiteAffineQuantization*>(t->quantization.params);
      return aq->scale->data[0];
    };

    op_data->fw_ik_scale = get_scale(fw_ik);
    op_data->fw_rk_scale = get_scale(fw_rk);

    TfLiteTensor* bw_ik =
        micro_context->AllocateTempInputTensor(node, kBwInputKernel);
    TfLiteTensor* bw_rk =
        micro_context->AllocateTempInputTensor(node, kBwRecurrentKernel);
    op_data->bw_ik_scale = get_scale(bw_ik);
    op_data->bw_rk_scale = get_scale(bw_rk);
    micro_context->DeallocateTempTfLiteTensor(bw_ik);
    micro_context->DeallocateTempTfLiteTensor(bw_rk);
  }

  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, scratch_size, &op_data->scratch_index));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(fw_ik);
  micro_context->DeallocateTempTfLiteTensor(fw_rk);

  return kTfLiteOk;
}

TfLiteStatus BiGruEval(TfLiteContext* context, TfLiteNode* node) {
  const BiGruOpData* op_data =
      reinterpret_cast<const BiGruOpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* fw_ik =
      tflite::micro::GetEvalInput(context, node, kFwInputKernel);
  const TfLiteEvalTensor* fw_ib =
      tflite::micro::GetEvalInput(context, node, kFwInputBias);
  const TfLiteEvalTensor* fw_rk =
      tflite::micro::GetEvalInput(context, node, kFwRecurrentKernel);
  const TfLiteEvalTensor* fw_rb =
      tflite::micro::GetEvalInput(context, node, kFwRecurrentBias);
  const TfLiteEvalTensor* bw_ik =
      tflite::micro::GetEvalInput(context, node, kBwInputKernel);
  const TfLiteEvalTensor* bw_ib =
      tflite::micro::GetEvalInput(context, node, kBwInputBias);
  const TfLiteEvalTensor* bw_rk =
      tflite::micro::GetEvalInput(context, node, kBwRecurrentKernel);
  const TfLiteEvalTensor* bw_rb =
      tflite::micro::GetEvalInput(context, node, kBwRecurrentBias);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const int ndims = input->dims->size;
  const int seq_len =
      (ndims == 3) ? input->dims->data[1] : input->dims->data[0];
  const int n_input =
      (ndims == 3) ? input->dims->data[2] : input->dims->data[1];
  const int n_output = fw_rk->dims->data[1];

  const float* input_data = tflite::micro::GetTensorData<float>(input);
  float* output_data = tflite::micro::GetTensorData<float>(output);

  uint8_t* raw_scratch = reinterpret_cast<uint8_t*>(
      context->GetScratchBuffer(context, op_data->scratch_index));
  float* h_state = reinterpret_cast<float*>(raw_scratch);
  float* scratch_xw = h_state + n_output;
  float* scratch_hw = scratch_xw + 3 * n_output;
  float* fw_output = scratch_hw + 3 * n_output;

  int8_t* quant_buf = nullptr;
  int32_t* acc_buf = nullptr;
  if (op_data->is_hybrid) {
    quant_buf = reinterpret_cast<int8_t*>(
        raw_scratch + op_data->hybrid_quant_offset);
    acc_buf = reinterpret_cast<int32_t*>(
        raw_scratch + op_data->hybrid_acc_offset);
  }

  // --- Forward pass → fw_output[seq_len, n_output] ---
  std::memset(h_state, 0, n_output * sizeof(float));
  for (int t = 0; t < seq_len; ++t) {
    const float* x_t = input_data + t * n_input;
    float* out_t = fw_output + t * n_output;
    if (op_data->is_hybrid) {
      GruCellStepHybrid(
          n_input, n_output, x_t, h_state,
          tflite::micro::GetTensorData<int8_t>(fw_ik), op_data->fw_ik_scale,
          tflite::micro::GetTensorData<float>(fw_ib),
          tflite::micro::GetTensorData<int8_t>(fw_rk), op_data->fw_rk_scale,
          tflite::micro::GetTensorData<float>(fw_rb), out_t, scratch_xw,
          scratch_hw, quant_buf, acc_buf);
    } else {
      GruCellStep(n_input, n_output, x_t, h_state,
                  tflite::micro::GetTensorData<float>(fw_ik),
                  tflite::micro::GetTensorData<float>(fw_ib),
                  tflite::micro::GetTensorData<float>(fw_rk),
                  tflite::micro::GetTensorData<float>(fw_rb), out_t,
                  scratch_xw, scratch_hw);
    }
    std::memcpy(h_state, out_t, n_output * sizeof(float));
  }

  // Copy forward results to left half of output [seq_len, 2*n_output]
  for (int t = 0; t < seq_len; ++t) {
    std::memcpy(output_data + t * 2 * n_output, fw_output + t * n_output,
                n_output * sizeof(float));
  }

  // --- Backward pass → right half of output ---
  std::memset(h_state, 0, n_output * sizeof(float));
  for (int t = 0; t < seq_len; ++t) {
    int src_t = seq_len - 1 - t;
    const float* x_t = input_data + src_t * n_input;
    float* out_t = output_data + src_t * 2 * n_output + n_output;
    if (op_data->is_hybrid) {
      GruCellStepHybrid(
          n_input, n_output, x_t, h_state,
          tflite::micro::GetTensorData<int8_t>(bw_ik), op_data->bw_ik_scale,
          tflite::micro::GetTensorData<float>(bw_ib),
          tflite::micro::GetTensorData<int8_t>(bw_rk), op_data->bw_rk_scale,
          tflite::micro::GetTensorData<float>(bw_rb), out_t, scratch_xw,
          scratch_hw, quant_buf, acc_buf);
    } else {
      GruCellStep(n_input, n_output, x_t, h_state,
                  tflite::micro::GetTensorData<float>(bw_ik),
                  tflite::micro::GetTensorData<float>(bw_ib),
                  tflite::micro::GetTensorData<float>(bw_rk),
                  tflite::micro::GetTensorData<float>(bw_rb), out_t,
                  scratch_xw, scratch_hw);
    }
    std::memcpy(h_state, out_t, n_output * sizeof(float));
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_BIDIRECTIONAL_SEQUENCE_GRU() {
  return tflite::micro::RegisterOp(BiGruInit, BiGruPrepare, BiGruEval);
}

}  // namespace tflite
