#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

constexpr int SEQ_LEN = 1800;
constexpr int INPUT_DIM = 49;
constexpr int HIDDEN = 32;
constexpr int N_CLASSES = 4;
constexpr float LN_EPS = 0.001f;

static float buf_a[SEQ_LEN * 64];
static float buf_b[SEQ_LEN * 64];
static float buf_out[SEQ_LEN * N_CLASSES];
static float gru_xw[3 * HIDDEN];
static float gru_hw[3 * HIDDEN];

static float* load_bin(const std::string& path, int expected_floats) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path.c_str());
    exit(1);
  }
  float* data = new float[expected_floats];
  size_t n = fread(data, sizeof(float), expected_floats, f);
  fclose(f);
  if ((int)n != expected_floats) {
    fprintf(stderr, "%s: expected %d floats, got %zu\n", path.c_str(),
            expected_floats, n);
    exit(1);
  }
  return data;
}

static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void layer_norm(const float* in, float* out, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float* row = in + r * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; c++) mean += row[c];
    mean /= cols;
    float var = 0.0f;
    for (int c = 0; c < cols; c++) {
      float d = row[c] - mean;
      var += d * d;
    }
    var /= cols;
    float inv = 1.0f / sqrtf(var + LN_EPS);
    float* o = out + r * cols;
    for (int c = 0; c < cols; c++) o[c] = (row[c] - mean) * inv;
  }
}

static void dense(const float* in, float* out, int rows, int in_dim,
                  int out_dim, const float* W, const float* b) {
  for (int r = 0; r < rows; r++) {
    const float* x = in + r * in_dim;
    float* o = out + r * out_dim;
    for (int j = 0; j < out_dim; j++) {
      float sum = b[j];
      const float* wj = W + j * in_dim;
      for (int k = 0; k < in_dim; k++) sum += x[k] * wj[k];
      o[j] = sum;
    }
  }
}

static void elu_inplace(float* data, int n) {
  for (int i = 0; i < n; i++)
    if (data[i] < 0.0f) data[i] = expf(data[i]) - 1.0f;
}

static void gru_cell(const float* x_t, const float* h_prev, float* h_out,
                     int in_dim, const float* ik, const float* ib,
                     const float* rk, const float* rb) {
  const int H = HIDDEN;
  for (int j = 0; j < 3 * H; j++) {
    float s = ib[j];
    const float* wj = ik + j * in_dim;
    for (int k = 0; k < in_dim; k++) s += x_t[k] * wj[k];
    gru_xw[j] = s;
  }
  for (int j = 0; j < 3 * H; j++) {
    float s = rb[j];
    const float* wj = rk + j * H;
    for (int k = 0; k < H; k++) s += h_prev[k] * wj[k];
    gru_hw[j] = s;
  }
  for (int i = 0; i < H; i++) {
    float z = sigmoid(gru_xw[i] + gru_hw[i]);
    float r = sigmoid(gru_xw[H + i] + gru_hw[H + i]);
    float n = tanhf(gru_xw[2 * H + i] + r * gru_hw[2 * H + i]);
    h_out[i] = (1.0f - z) * n + z * h_prev[i];
  }
}

static void bigru(const float* in, float* out, int in_dim, const float* fw_ik,
                  const float* fw_ib, const float* fw_rk, const float* fw_rb,
                  const float* bw_ik, const float* bw_ib, const float* bw_rk,
                  const float* bw_rb) {
  static float h_fw[HIDDEN], h_bw[HIDDEN], h_tmp[HIDDEN];
  memset(h_fw, 0, sizeof(h_fw));
  for (int t = 0; t < SEQ_LEN; t++) {
    gru_cell(in + t * in_dim, h_fw, h_tmp, in_dim, fw_ik, fw_ib, fw_rk, fw_rb);
    memcpy(h_fw, h_tmp, sizeof(h_fw));
    memcpy(out + t * 2 * HIDDEN, h_fw, HIDDEN * sizeof(float));
  }
  memset(h_bw, 0, sizeof(h_bw));
  for (int t = SEQ_LEN - 1; t >= 0; t--) {
    gru_cell(in + t * in_dim, h_bw, h_tmp, in_dim, bw_ik, bw_ib, bw_rk, bw_rb);
    memcpy(h_bw, h_tmp, sizeof(h_bw));
    memcpy(out + t * 2 * HIDDEN + HIDDEN, h_bw, HIDDEN * sizeof(float));
  }
}

static void softmax(float* data, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float* row = data + r * cols;
    float mx = row[0];
    for (int c = 1; c < cols; c++)
      if (row[c] > mx) mx = row[c];
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
      row[c] = expf(row[c] - mx);
      sum += row[c];
    }
    for (int c = 0; c < cols; c++) row[c] /= sum;
  }
}

struct Weights {
  float *d1_w, *d1_b;
  float *b1_fw_ik, *b1_fw_ib, *b1_fw_rk, *b1_fw_rb;
  float *b1_bw_ik, *b1_bw_ib, *b1_bw_rk, *b1_bw_rb;
  float *b2_fw_ik, *b2_fw_ib, *b2_fw_rk, *b2_fw_rb;
  float *b2_bw_ik, *b2_bw_ib, *b2_bw_rk, *b2_bw_rb;
  float *d2_w, *d2_b;
  float *d3_w, *d3_b;
};

static Weights load_weights(const std::string& dir) {
  Weights w;
  auto p = [&](const char* name) { return dir + "/" + name; };
  w.d1_w = load_bin(p("dense1_kernel.bin"), 32 * 49);
  w.d1_b = load_bin(p("dense1_bias.bin"), 32);
  w.b1_fw_ik = load_bin(p("bigru1_fw_ik.bin"), 96 * 32);
  w.b1_fw_ib = load_bin(p("bigru1_fw_ib.bin"), 96);
  w.b1_fw_rk = load_bin(p("bigru1_fw_rk.bin"), 96 * 32);
  w.b1_fw_rb = load_bin(p("bigru1_fw_rb.bin"), 96);
  w.b1_bw_ik = load_bin(p("bigru1_bw_ik.bin"), 96 * 32);
  w.b1_bw_ib = load_bin(p("bigru1_bw_ib.bin"), 96);
  w.b1_bw_rk = load_bin(p("bigru1_bw_rk.bin"), 96 * 32);
  w.b1_bw_rb = load_bin(p("bigru1_bw_rb.bin"), 96);
  w.b2_fw_ik = load_bin(p("bigru2_fw_ik.bin"), 96 * 64);
  w.b2_fw_ib = load_bin(p("bigru2_fw_ib.bin"), 96);
  w.b2_fw_rk = load_bin(p("bigru2_fw_rk.bin"), 96 * 32);
  w.b2_fw_rb = load_bin(p("bigru2_fw_rb.bin"), 96);
  w.b2_bw_ik = load_bin(p("bigru2_bw_ik.bin"), 96 * 64);
  w.b2_bw_ib = load_bin(p("bigru2_bw_ib.bin"), 96);
  w.b2_bw_rk = load_bin(p("bigru2_bw_rk.bin"), 96 * 32);
  w.b2_bw_rb = load_bin(p("bigru2_bw_rb.bin"), 96);
  w.d2_w = load_bin(p("dense2_kernel.bin"), 32 * 64);
  w.d2_b = load_bin(p("dense2_bias.bin"), 32);
  w.d3_w = load_bin(p("dense3_kernel.bin"), 4 * 32);
  w.d3_b = load_bin(p("dense3_bias.bin"), 4);
  return w;
}

static void run_pipeline(const float* input, float* output, const Weights& w) {
  layer_norm(input, buf_a, SEQ_LEN, INPUT_DIM);
  dense(buf_a, buf_b, SEQ_LEN, INPUT_DIM, HIDDEN, w.d1_w, w.d1_b);
  elu_inplace(buf_b, SEQ_LEN * HIDDEN);
  bigru(buf_b, buf_a, HIDDEN, w.b1_fw_ik, w.b1_fw_ib, w.b1_fw_rk, w.b1_fw_rb,
        w.b1_bw_ik, w.b1_bw_ib, w.b1_bw_rk, w.b1_bw_rb);
  bigru(buf_a, buf_b, 2 * HIDDEN, w.b2_fw_ik, w.b2_fw_ib, w.b2_fw_rk,
        w.b2_fw_rb, w.b2_bw_ik, w.b2_bw_ib, w.b2_bw_rk, w.b2_bw_rb);
  layer_norm(buf_b, buf_a, SEQ_LEN, 2 * HIDDEN);
  dense(buf_a, buf_b, SEQ_LEN, 2 * HIDDEN, HIDDEN, w.d2_w, w.d2_b);
  elu_inplace(buf_b, SEQ_LEN * HIDDEN);
  dense(buf_b, output, SEQ_LEN, HIDDEN, N_CLASSES, w.d3_w, w.d3_b);
  softmax(output, SEQ_LEN, N_CLASSES);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <weights_dir> <input.bin> <reference.bin>\n",
            argv[0]);
    return 1;
  }
  const char* weights_dir = argv[1];
  const char* input_path = argv[2];
  const char* ref_path = argv[3];

  printf("Loading weights from %s\n", weights_dir);
  Weights w = load_weights(weights_dir);

  printf("Loading input from %s\n", input_path);
  float* input = load_bin(input_path, SEQ_LEN * INPUT_DIM);

  printf("Loading reference from %s\n", ref_path);
  float* ref = load_bin(ref_path, SEQ_LEN * N_CLASSES);

  printf("Running pipeline...\n");
  run_pipeline(input, buf_out, w);

  float max_diff = 0.0f, sum_diff = 0.0f;
  int total = SEQ_LEN * N_CLASSES;
  for (int i = 0; i < total; i++) {
    float d = fabsf(buf_out[i] - ref[i]);
    if (d > max_diff) max_diff = d;
    sum_diff += d;
  }
  float mean_diff = sum_diff / total;

  printf("Max  abs diff: %.6e\n", max_diff);
  printf("Mean abs diff: %.6e\n", mean_diff);

  if (max_diff < 1e-4f) {
    printf("PASS\n");
  } else {
    printf("FAIL (max_diff >= 1e-4)\n");
    int printed = 0;
    for (int i = 0; i < total && printed < 10; i++) {
      float d = fabsf(buf_out[i] - ref[i]);
      if (d > 1e-5f) {
        printf("  [%d] got=%.6f ref=%.6f diff=%.6e\n", i, buf_out[i], ref[i],
               d);
        printed++;
      }
    }
    delete[] input;
    delete[] ref;
    return 1;
  }

  delete[] input;
  delete[] ref;
  return 0;
}
