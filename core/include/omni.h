/*
 * Omni Core - C API Header
 *
 * Minimal, clean C interface for Rust FFI
 */

#ifndef OMNI_H
#define OMNI_H

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Data types
enum omni_dtype {
  OMNI_F32 = 0,
  OMNI_F16 = 1,
  OMNI_Q8_0 = 3,
  OMNI_Q4_K = 10,
};

// Tensor structure
struct omni_tensor {
  std::string name;
  std::vector<int64_t> shape;
  omni_dtype dtype;
  uint64_t offset;
  uint64_t size;
  void *data;
};

// Model structure
struct omni_model {
  int fd;
  void *mapped_data;
  size_t mapped_size;

  uint32_t version;
  uint32_t architecture;
  uint32_t alignment;
  uint64_t n_tensors;
  uint64_t n_kv;
  size_t data_offset;

  std::unordered_map<std::string, std::string> metadata;
  std::unordered_map<std::string, omni_tensor> tensors;

  // Config
  int hidden_size;
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int vocab_size;
  int max_seq_len;
  float rope_theta;
  float norm_eps;
};

// Context structure
struct omni_context {
  omni_model *model;

  // Working buffers
  std::vector<float> hidden;
  std::vector<float> residual;
  std::vector<float> q_buf;
  std::vector<float> k_buf;
  std::vector<float> v_buf;
  std::vector<float> attn_out;
  std::vector<float> ffn_buf;
  std::vector<float> logits;
  std::vector<float> weight_buf;

  // KV Cache for autoregressive generation
  // cache_k[layer][head][pos][head_dim]
  // cache_v[layer][head][pos][head_dim]
  std::vector<std::vector<float>>
      cache_k; // [n_layers][max_seq_len * n_kv_heads * head_dim]
  std::vector<std::vector<float>>
      cache_v;   // [n_layers][max_seq_len * n_kv_heads * head_dim]
  int cache_pos; // Current position in cache

  // Conv Cache for causal convolution
  // Caches Bx (after gating B*x) for each channel - matches PyTorch
  // Lfm2HybridConvCache.conv_cache Layout: cache_conv[layer][hidden *
  // kernel_size] where kernel_size=3
  std::vector<std::vector<float>>
      cache_conv; // [n_layers][hidden * kernel_size]

  // Weight cache (dequantized weights) - trade RAM for speed
  std::unordered_map<std::string, std::vector<float>> weight_cache;
};

// Model loading
omni_model *omni_load_model(const char *path);
void omni_free_model(omni_model *model);

// Context creation
omni_context *omni_create_context(omni_model *model, int max_seq_len);
void omni_free_context(omni_context *ctx);
void omni_reset_kv_cache(omni_context *ctx); // Reset KV cache for new sequence

// Tensor access
const omni_tensor *omni_get_tensor(omni_model *model, const char *name);
float *omni_get_tensor_f32(omni_model *model, const char *name, float *buf);

// Kernels
void omni_rms_norm(float *out, const float *x, const float *weight, int n,
                   float eps);
void omni_silu(float *out, const float *x, int n);
void omni_rope(float *q, float *k, int seq_len, int n_heads, int head_dim,
               const int *positions, float theta);
void omni_softmax(float *x, int n);
void omni_matmul(float *C, const float *A, const float *B, int M, int N, int K);
void omni_matmul_q8_0(float *C, const float *A, const void *B_q8, int M, int N,
                      int K);
void omni_matmul_q8_0_asm(float *C, const float *A, const void *B_q8, int M,
                          int N, int K);
void omni_gemv_q8_0_neon(float *out, const float *x, const void *W_q8, int N,
                         int K);

// Quantization
void omni_dequant_q8_0(float *out, const void *data, int n);
void omni_dequant_q4_k(float *out, const void *data, int n);

// Assembly kernels (if available)
#ifdef __cplusplus
extern "C" {
#endif
void omni_dequant_q8_0_asm(float *out, const void *data, int n_blocks);
float omni_gemv_q8_0_row_asm(const float *A, const void *B, int n_blocks);
void omni_gemv_q8_0_4row_asm(float *out, const float *A, const void *B,
                             int n_blocks, int stride);
void omni_gemv_q8_s8_4row_asm(float *out, const int8_t *A_q, const void *B,
                              int n_blocks, int stride);
void omni_attn_score_asm(float *scores, const float *q, const float *k_cache,
                         int n_pos, int head_dim, int kv_stride);
void omni_attn_value_asm(float *out, const float *scores, const float *v_cache,
                         int n_pos, int head_dim, int kv_stride);
void omni_rms_norm_asm(float *out, const float *x, const float *w, int n,
                       float eps);
#ifdef __cplusplus
}
#endif

// Tensor access with optional dequantization
const omni_tensor *omni_get_tensor(omni_model *model, const char *name);
float *omni_get_tensor_f32(omni_model *model, const char *name, float *buf);
const void *omni_get_tensor_data(omni_model *model, const char *name);

// Inference
void omni_forward(omni_context *ctx, const int *tokens, int n_tokens,
                  float *logits);

// Generation
int omni_sample(const float *logits, int vocab_size, float temperature,
                float top_p);
int omni_generate(omni_context *ctx, const int *prompt, int prompt_len,
                  int *output, int max_tokens, float temperature);

// Info
void omni_print_info(omni_model *model);
const char *omni_get_metadata(omni_model *model, const char *key);

#ifdef __cplusplus
}
#endif

#endif // OMNI_H
