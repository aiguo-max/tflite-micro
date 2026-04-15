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

// Multi-Head Self-Attention custom op for TFLM.
// Supports float32 and hybrid int8 (int8 weights, float32 activations).
//
// Input tensors:
//   0: input       [batch, seq, d_model]
//   1: q_weight    [d_model, d_model]  (float32 or int8)
//   2: q_bias      [d_model]
//   3: k_weight    [d_model, d_model]  (float32 or int8)
//   4: k_bias      [d_model]
//   5: v_weight    [d_model, d_model]  (float32 or int8)
//   6: v_bias      [d_model]
//   7: merge_weight [d_model, d_model] (float32 or int8)
//   8: merge_bias  [d_model]
//   9: scale       [] scalar (1/sqrt(head_dim))
// Output:
//   0: output      [batch, seq, d_model]
// customOptions: [num_heads: uint8, head_dim: uint8]

#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

#if defined(CMSIS_NN)
#include "Include/arm_nnsupportfunctions.h"
#endif

namespace tflite {
namespace {

struct MhaOpData {
  int num_heads;
  int head_dim;
  int scratch_index;
  bool is_hybrid;
  int hybrid_extra_offset;
  const float* q_scales;
  const float* k_scales;
  const float* v_scales;
  const float* merge_scales;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  auto* data = static_cast<MhaOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(MhaOpData)));
  if (buffer != nullptr && length >= 2) {
    data->num_heads = static_cast<int>(static_cast<uint8_t>(buffer[0]));
    data->head_dim = static_cast<int>(static_cast<uint8_t>(buffer[1]));
  } else {
    data->num_heads = 4;
    data->head_dim = 16;
  }
  return data;
}

static void FloatMatMulAddBias(int rows, int cols_in, int cols_out,
                               const float* input, const float* weight,
                               const float* bias, float* output) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols_out; ++c) {
      float sum = bias[c];
      for (int k = 0; k < cols_in; ++k) {
        sum += input[r * cols_in + k] * weight[c * cols_in + k];
      }
      output[r * cols_out + c] = sum;
    }
  }
}

static void SymmetricQuantize(const float* input, int size, int8_t* output,
                              float* scale) {
  float min_val = input[0], max_val = input[0];
  for (int i = 1; i < size; ++i) {
    if (input[i] < min_val) min_val = input[i];
    if (input[i] > max_val) max_val = input[i];
  }
  const float range = std::max(std::abs(min_val), std::abs(max_val));
  if (range == 0.0f) {
    *scale = 1.0f;
    std::memset(output, 0, size);
    return;
  }
  *scale = range / 127.0f;
  const float inv = 127.0f / range;
  for (int i = 0; i < size; ++i) {
    int32_t v = static_cast<int32_t>(input[i] * inv +
                                     (input[i] >= 0.0f ? 0.5f : -0.5f));
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    output[i] = static_cast<int8_t>(v);
  }
}

// Hybrid matmul: quantize each input row, int8 matmul, per-channel dequant
static void HybridMatMulAddBias(int rows, int cols_in, int cols_out,
                                const float* input, const int8_t* weight,
                                const float* weight_scales, const float* bias,
                                float* output, int8_t* quant_buf,
                                int32_t* acc_buf) {
  for (int r = 0; r < rows; ++r) {
    float input_scale;
    SymmetricQuantize(input + r * cols_in, cols_in, quant_buf, &input_scale);

#if defined(CMSIS_NN)
    std::memset(acc_buf, 0, cols_out * sizeof(int32_t));
    arm_nn_mat_mult_nt_t_s8_s32(quant_buf, weight, acc_buf, 1, cols_in,
                                cols_out, 0, 1);
    for (int c = 0; c < cols_out; ++c) {
      output[r * cols_out + c] =
          static_cast<float>(acc_buf[c]) * input_scale * weight_scales[c] +
          bias[c];
    }
#else
    for (int c = 0; c < cols_out; ++c) {
      int32_t acc = 0;
      for (int k = 0; k < cols_in; ++k) {
        acc += static_cast<int32_t>(quant_buf[k]) *
               static_cast<int32_t>(weight[c * cols_in + k]);
      }
      output[r * cols_out + c] =
          static_cast<float>(acc) * input_scale * weight_scales[c] + bias[c];
    }
#endif
  }
}

static void SoftmaxRow(float* data, int len) {
  float max_val = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > max_val) max_val = data[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < len; ++i) {
    data[i] = std::exp(data[i] - max_val);
    sum += data[i];
  }
  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < len; ++i) {
    data[i] *= inv_sum;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 10);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* q_w_t = micro_context->AllocateTempInputTensor(node, 1);
  TF_LITE_ENSURE(context, input != nullptr && q_w_t != nullptr);

  const int ndims = input->dims->size;
  const int seq_len =
      (ndims == 3) ? input->dims->data[1] : input->dims->data[0];
  const int d_model = input->dims->data[ndims - 1];

  MhaOpData* data = reinterpret_cast<MhaOpData*>(node->user_data);
  const int H = data->num_heads;
  const int hd = d_model / H;
  data->is_hybrid = (q_w_t->type == kTfLiteInt8);

  size_t float_bytes =
      (3 * seq_len * d_model + H * seq_len * seq_len +
       H * seq_len * hd) *
      sizeof(float);
  size_t scratch_size = float_bytes;

  if (data->is_hybrid) {
    size_t qbuf = ((d_model + 3) & ~3) * sizeof(int8_t);
    data->hybrid_extra_offset = static_cast<int>(float_bytes);
    scratch_size = float_bytes + qbuf + d_model * sizeof(int32_t);

    auto get_scales = [](TfLiteTensor* t) -> const float* {
      return static_cast<TfLiteAffineQuantization*>(t->quantization.params)
          ->scale->data;
    };
    data->q_scales = get_scales(q_w_t);
    TfLiteTensor* k_w_t = micro_context->AllocateTempInputTensor(node, 3);
    data->k_scales = get_scales(k_w_t);
    TfLiteTensor* v_w_t = micro_context->AllocateTempInputTensor(node, 5);
    data->v_scales = get_scales(v_w_t);
    TfLiteTensor* m_w_t = micro_context->AllocateTempInputTensor(node, 7);
    data->merge_scales = get_scales(m_w_t);
    micro_context->DeallocateTempTfLiteTensor(k_w_t);
    micro_context->DeallocateTempTfLiteTensor(v_w_t);
    micro_context->DeallocateTempTfLiteTensor(m_w_t);
  }

  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, scratch_size, &data->scratch_index));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(q_w_t);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = static_cast<const MhaOpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* q_w = tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor* q_b = tflite::micro::GetEvalInput(context, node, 2);
  const TfLiteEvalTensor* k_w = tflite::micro::GetEvalInput(context, node, 3);
  const TfLiteEvalTensor* k_b = tflite::micro::GetEvalInput(context, node, 4);
  const TfLiteEvalTensor* v_w = tflite::micro::GetEvalInput(context, node, 5);
  const TfLiteEvalTensor* v_b = tflite::micro::GetEvalInput(context, node, 6);
  const TfLiteEvalTensor* merge_w =
      tflite::micro::GetEvalInput(context, node, 7);
  const TfLiteEvalTensor* merge_b =
      tflite::micro::GetEvalInput(context, node, 8);
  const TfLiteEvalTensor* scale_t =
      tflite::micro::GetEvalInput(context, node, 9);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  const int ndims = input->dims->size;
  const int seq = (ndims == 3) ? input->dims->data[1] : input->dims->data[0];
  const int d_model = input->dims->data[ndims - 1];
  const int H = op_data->num_heads;
  const int hd = d_model / H;
  const float scale = tflite::micro::GetTensorData<float>(scale_t)[0];

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  uint8_t* raw_scratch = reinterpret_cast<uint8_t*>(
      context->GetScratchBuffer(context, op_data->scratch_index));
  float* Q = reinterpret_cast<float*>(raw_scratch);
  float* K = Q + seq * d_model;
  float* V = K + seq * d_model;
  float* scores = V + seq * d_model;
  float* attn_v = scores + H * seq * seq;

  if (op_data->is_hybrid) {
    int8_t* quant_buf = reinterpret_cast<int8_t*>(
        raw_scratch + op_data->hybrid_extra_offset);
    int32_t* acc_buf = reinterpret_cast<int32_t*>(
        quant_buf + ((d_model + 3) & ~3));

    HybridMatMulAddBias(seq, d_model, d_model, in_data,
                        tflite::micro::GetTensorData<int8_t>(q_w),
                        op_data->q_scales,
                        tflite::micro::GetTensorData<float>(q_b), Q,
                        quant_buf, acc_buf);
    HybridMatMulAddBias(seq, d_model, d_model, in_data,
                        tflite::micro::GetTensorData<int8_t>(k_w),
                        op_data->k_scales,
                        tflite::micro::GetTensorData<float>(k_b), K,
                        quant_buf, acc_buf);
    HybridMatMulAddBias(seq, d_model, d_model, in_data,
                        tflite::micro::GetTensorData<int8_t>(v_w),
                        op_data->v_scales,
                        tflite::micro::GetTensorData<float>(v_b), V,
                        quant_buf, acc_buf);
  } else {
    FloatMatMulAddBias(seq, d_model, d_model, in_data,
                       tflite::micro::GetTensorData<float>(q_w),
                       tflite::micro::GetTensorData<float>(q_b), Q);
    FloatMatMulAddBias(seq, d_model, d_model, in_data,
                       tflite::micro::GetTensorData<float>(k_w),
                       tflite::micro::GetTensorData<float>(k_b), K);
    FloatMatMulAddBias(seq, d_model, d_model, in_data,
                       tflite::micro::GetTensorData<float>(v_w),
                       tflite::micro::GetTensorData<float>(v_b), V);
  }

  // Attention: scores = Q_h @ K_h^T * scale → softmax → attn_v = attn @ V_h
  for (int h = 0; h < H; ++h) {
    float* s = scores + h * seq * seq;
    for (int qi = 0; qi < seq; ++qi) {
      for (int ki = 0; ki < seq; ++ki) {
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) {
          dot += Q[qi * d_model + h * hd + d] * K[ki * d_model + h * hd + d];
        }
        s[qi * seq + ki] = dot * scale;
      }
      SoftmaxRow(s + qi * seq, seq);
    }
    float* av = attn_v + h * seq * hd;
    for (int qi = 0; qi < seq; ++qi) {
      for (int d = 0; d < hd; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < seq; ++j) {
          sum += s[qi * seq + j] * V[j * d_model + h * hd + d];
        }
        av[qi * hd + d] = sum;
      }
    }
  }

  // Merge heads → merge FC
  float* merged = V;
  for (int i = 0; i < seq; ++i) {
    for (int h = 0; h < H; ++h) {
      std::memcpy(merged + i * d_model + h * hd,
                  attn_v + h * seq * hd + i * hd, hd * sizeof(float));
    }
  }

  if (op_data->is_hybrid) {
    int8_t* quant_buf = reinterpret_cast<int8_t*>(
        raw_scratch + op_data->hybrid_extra_offset);
    int32_t* acc_buf = reinterpret_cast<int32_t*>(
        quant_buf + ((d_model + 3) & ~3));
    HybridMatMulAddBias(seq, d_model, d_model, merged,
                        tflite::micro::GetTensorData<int8_t>(merge_w),
                        op_data->merge_scales,
                        tflite::micro::GetTensorData<float>(merge_b), out_data,
                        quant_buf, acc_buf);
  } else {
    FloatMatMulAddBias(seq, d_model, d_model, merged,
                       tflite::micro::GetTensorData<float>(merge_w),
                       tflite::micro::GetTensorData<float>(merge_b), out_data);
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_MULTI_HEAD_ATTENTION() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
