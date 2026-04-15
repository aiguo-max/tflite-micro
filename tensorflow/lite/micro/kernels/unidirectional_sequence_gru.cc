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

// Unidirectional GRU custom op for TFLM (reset-after variant).
// Supports float32 and hybrid int8 (int8 weights, float32 activations).
//
// Input tensors:
//   0: sequence        [batch, seq_len, input_dim]
//   1: h_init          [batch, hidden_dim]
//   2: input_kernel    [3*hidden, input_dim]  (float32 or int8)
//   3: input_bias      [3*hidden]
//   4: recurrent_kernel [3*hidden, hidden_dim] (float32 or int8)
//   5: recurrent_bias  [3*hidden]
// Output tensors:
//   0: output_sequence [batch, seq_len, hidden_dim]
//   1: h_final         [batch, hidden_dim]

#include <cmath>
#include <cstring>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#if defined(CMSIS_NN)
#include "Include/arm_nnsupportfunctions.h"
#endif

namespace tflite {
namespace {

static inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline size_t AlignUp4(size_t v) {
  return (v + 3) & ~static_cast<size_t>(3);
}

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

void HybridMatVec(int rows, int cols, const int8_t* matrix,
                  const float* filter_scales, const float* vec,
                  const float* bias, float* output, int8_t* quantized_vec,
                  int32_t* acc_buf) {
  float input_scale;
  SymmetricQuantize(vec, cols, quantized_vec, &input_scale);

#if defined(CMSIS_NN)
  std::memset(acc_buf, 0, rows * sizeof(int32_t));
  arm_nn_mat_mult_nt_t_s8_s32(quantized_vec, matrix, acc_buf, 1, cols, rows, 0,
                              1);
  for (int i = 0; i < rows; ++i) {
    output[i] =
        static_cast<float>(acc_buf[i]) * input_scale * filter_scales[i] +
        bias[i];
  }
#else
  for (int i = 0; i < rows; ++i) {
    int32_t acc = 0;
    for (int j = 0; j < cols; ++j) {
      acc += static_cast<int32_t>(quantized_vec[j]) *
             static_cast<int32_t>(matrix[i * cols + j]);
    }
    output[i] =
        static_cast<float>(acc) * input_scale * filter_scales[i] + bias[i];
  }
#endif
}

// GRU cell: reset-after variant (Keras default)
//   z = sigmoid(xw[0:h] + hw[0:h])
//   r = sigmoid(xw[h:2h] + hw[h:2h])
//   n = tanh(xw[2h:3h] + r * hw[2h:3h])
//   h_new = (1 - z) * n + z * h_prev
void ApplyGruGates(int n_output, const float* scratch_xw,
                   const float* scratch_hw, const float* h_prev,
                   float* h_new) {
  for (int i = 0; i < n_output; ++i) {
    float z = Sigmoid(scratch_xw[i] + scratch_hw[i]);
    float r = Sigmoid(scratch_xw[n_output + i] + scratch_hw[n_output + i]);
    float n = std::tanh(scratch_xw[2 * n_output + i] +
                        r * scratch_hw[2 * n_output + i]);
    h_new[i] = (1.0f - z) * n + z * h_prev[i];
  }
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
                       const float* ik_scales, const float* input_bias,
                       const int8_t* recurrent_kernel,
                       const float* rk_scales, const float* recurrent_bias,
                       float* h_new, float* scratch_xw, float* scratch_hw,
                       int8_t* quant_buf, int32_t* acc_buf) {
  const int n3 = 3 * n_output;
  HybridMatVec(n3, n_input, input_kernel, ik_scales, input, input_bias,
               scratch_xw, quant_buf, acc_buf);
  HybridMatVec(n3, n_output, recurrent_kernel, rk_scales, h_prev,
               recurrent_bias, scratch_hw, quant_buf, acc_buf);
  ApplyGruGates(n_output, scratch_xw, scratch_hw, h_prev, h_new);
}

constexpr int kSeqInput = 0;
constexpr int kHInit = 1;
constexpr int kInputKernel = 2;
constexpr int kInputBias = 3;
constexpr int kRecurrentKernel = 4;
constexpr int kRecurrentBias = 5;
constexpr int kSeqOutput = 0;
constexpr int kHFinal = 1;

struct UniGruOpData {
  int scratch_index;
  bool is_hybrid;
  int hybrid_quant_offset;
  int hybrid_acc_offset;
  const float* ik_scales;
  const float* rk_scales;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(UniGruOpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 6);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* ik_t =
      micro_context->AllocateTempInputTensor(node, kInputKernel);
  TfLiteTensor* rk_t =
      micro_context->AllocateTempInputTensor(node, kRecurrentKernel);
  TfLiteTensor* seq_t =
      micro_context->AllocateTempInputTensor(node, kSeqInput);
  TF_LITE_ENSURE(context, ik_t != nullptr);
  TF_LITE_ENSURE(context, rk_t != nullptr);
  TF_LITE_ENSURE(context, seq_t != nullptr);

  const int n_output = rk_t->dims->data[1];
  const int ndims = seq_t->dims->size;
  const int n_input = seq_t->dims->data[ndims - 1];

  UniGruOpData* op_data = reinterpret_cast<UniGruOpData*>(node->user_data);
  op_data->is_hybrid = (ik_t->type == kTfLiteInt8);

  // Scratch: [h_state: n_output] [scratch_xw: 3*n_output] [scratch_hw: 3*n_output]
  // Hybrid adds: [quant_buf: max(n_input, n_output) int8, aligned] [acc_buf: 3*n_output int32]
  size_t float_bytes = static_cast<size_t>(7 * n_output) * sizeof(float);
  size_t scratch_size = float_bytes;

  if (op_data->is_hybrid) {
    const int max_vec = std::max(n_input, n_output);
    size_t quant_bytes = AlignUp4(max_vec * sizeof(int8_t));
    op_data->hybrid_quant_offset = static_cast<int>(float_bytes);
    op_data->hybrid_acc_offset =
        static_cast<int>(float_bytes + quant_bytes);
    scratch_size = float_bytes + quant_bytes + 3 * n_output * sizeof(int32_t);

    auto get_scales = [](TfLiteTensor* t) -> const float* {
      const auto* aq =
          static_cast<TfLiteAffineQuantization*>(t->quantization.params);
      return aq->scale->data;
    };
    op_data->ik_scales = get_scales(ik_t);
    op_data->rk_scales = get_scales(rk_t);
  }

  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, scratch_size, &op_data->scratch_index));

  micro_context->DeallocateTempTfLiteTensor(ik_t);
  micro_context->DeallocateTempTfLiteTensor(rk_t);
  micro_context->DeallocateTempTfLiteTensor(seq_t);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const UniGruOpData* op_data =
      reinterpret_cast<const UniGruOpData*>(node->user_data);

  const TfLiteEvalTensor* seq_input =
      tflite::micro::GetEvalInput(context, node, kSeqInput);
  const TfLiteEvalTensor* h_init =
      tflite::micro::GetEvalInput(context, node, kHInit);
  const TfLiteEvalTensor* ik =
      tflite::micro::GetEvalInput(context, node, kInputKernel);
  const TfLiteEvalTensor* ib =
      tflite::micro::GetEvalInput(context, node, kInputBias);
  const TfLiteEvalTensor* rk =
      tflite::micro::GetEvalInput(context, node, kRecurrentKernel);
  const TfLiteEvalTensor* rb =
      tflite::micro::GetEvalInput(context, node, kRecurrentBias);
  TfLiteEvalTensor* seq_output =
      tflite::micro::GetEvalOutput(context, node, kSeqOutput);
  TfLiteEvalTensor* h_final =
      tflite::micro::GetEvalOutput(context, node, kHFinal);

  const int ndims = seq_input->dims->size;
  const int seq_len = (ndims == 3) ? seq_input->dims->data[1] : 1;
  const int n_input = seq_input->dims->data[ndims - 1];
  const int n_output = rk->dims->data[1];

  const float* input_data = tflite::micro::GetTensorData<float>(seq_input);
  float* output_data = tflite::micro::GetTensorData<float>(seq_output);
  float* h_final_data = tflite::micro::GetTensorData<float>(h_final);

  uint8_t* raw_scratch = reinterpret_cast<uint8_t*>(
      context->GetScratchBuffer(context, op_data->scratch_index));
  float* h_state = reinterpret_cast<float*>(raw_scratch);
  float* scratch_xw = h_state + n_output;
  float* scratch_hw = scratch_xw + 3 * n_output;

  std::memcpy(h_state, tflite::micro::GetTensorData<float>(h_init),
              n_output * sizeof(float));

  if (op_data->is_hybrid) {
    int8_t* quant_buf = reinterpret_cast<int8_t*>(
        raw_scratch + op_data->hybrid_quant_offset);
    int32_t* acc_buf = reinterpret_cast<int32_t*>(
        raw_scratch + op_data->hybrid_acc_offset);
    const int8_t* ik_data = tflite::micro::GetTensorData<int8_t>(ik);
    const int8_t* rk_data = tflite::micro::GetTensorData<int8_t>(rk);

    for (int t = 0; t < seq_len; ++t) {
      const float* x_t = input_data + t * n_input;
      float* out_t = output_data + t * n_output;
      GruCellStepHybrid(n_input, n_output, x_t, h_state, ik_data,
                        op_data->ik_scales,
                        tflite::micro::GetTensorData<float>(ib), rk_data,
                        op_data->rk_scales,
                        tflite::micro::GetTensorData<float>(rb), out_t,
                        scratch_xw, scratch_hw, quant_buf, acc_buf);
      std::memcpy(h_state, out_t, n_output * sizeof(float));
    }
  } else {
    const float* ik_data = tflite::micro::GetTensorData<float>(ik);
    const float* rk_data = tflite::micro::GetTensorData<float>(rk);

    for (int t = 0; t < seq_len; ++t) {
      const float* x_t = input_data + t * n_input;
      float* out_t = output_data + t * n_output;
      GruCellStep(n_input, n_output, x_t, h_state, ik_data,
                  tflite::micro::GetTensorData<float>(ib), rk_data,
                  tflite::micro::GetTensorData<float>(rb), out_t, scratch_xw,
                  scratch_hw);
      std::memcpy(h_state, out_t, n_output * sizeof(float));
    }
  }

  std::memcpy(h_final_data, h_state, n_output * sizeof(float));
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_GRU() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
