/*
 * Omni Core - SIMD Kernels (NEON Intrinsics)
 *
 * Optimized operations for Apple Silicon using ARM NEON
 */

#include "../include/omni.h"
#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

// RMS Normalization with NEON
void omni_rms_norm(float *out, const float *x, const float *weight, int n,
                   float eps) {
#if defined(__ARM_NEON)
  // Compute sum of squares
  float32x4_t sum_sq = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);
    sum_sq = vfmaq_f32(sum_sq, vx, vx); // FMA: sum_sq += vx * vx
  }

  // Horizontal sum
  float sum = vaddvq_f32(sum_sq);
  for (; i < n; i++) {
    sum += x[i] * x[i];
  }

  // RMS = 1 / sqrt(mean + eps)
  float rms = 1.0f / sqrtf(sum / n + eps);
  float32x4_t vrms = vdupq_n_f32(rms);

  // Normalize and scale
  i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);
    float32x4_t vw = vld1q_f32(weight + i);
    float32x4_t vout = vmulq_f32(vmulq_f32(vx, vrms), vw);
    vst1q_f32(out + i, vout);
  }
  for (; i < n; i++) {
    out[i] = x[i] * rms * weight[i];
  }
#else
  // Scalar fallback
  float sum_sq = 0.0f;
  for (int i = 0; i < n; i++) {
    sum_sq += x[i] * x[i];
  }
  float rms = 1.0f / sqrtf(sum_sq / n + eps);
  for (int i = 0; i < n; i++) {
    out[i] = x[i] * rms * weight[i];
  }
#endif
}

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
void omni_silu(float *out, const float *x, int n) {
#if defined(__ARM_NEON)
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);

    // Compute sigmoid: 1 / (1 + exp(-x))
    // For NEON, we compute element-wise
    float vals[4];
    vst1q_f32(vals, vx);
    for (int j = 0; j < 4; j++) {
      float sig = 1.0f / (1.0f + expf(-vals[j]));
      vals[j] = vals[j] * sig;
    }
    vst1q_f32(out + i, vld1q_f32(vals));
  }
  for (; i < n; i++) {
    out[i] = x[i] / (1.0f + expf(-x[i]));
  }
#else
  for (int i = 0; i < n; i++) {
    out[i] = x[i] / (1.0f + expf(-x[i]));
  }
#endif
}

// RoPE (Rotary Position Embedding) - single buffer version
// Uses official transformers rotate_half pattern:
// q_embed = (q * cos) + (rotate_half(q) * sin)
// where rotate_half(x) = cat([-x[half:], x[:half]])
void omni_rope_inplace(float *x, int seq_len, int n_heads, int head_dim,
                       const int *positions, float theta) {
  int half_dim = head_dim / 2;

  for (int pos_idx = 0; pos_idx < seq_len; pos_idx++) {
    int pos = positions[pos_idx];

    for (int h = 0; h < n_heads; h++) {
      float *head = x + pos_idx * n_heads * head_dim + h * head_dim;

      // Compute cos/sin for all dimensions
      float cos_vals[64], sin_vals[64]; // Max head_dim = 64
      for (int i = 0; i < head_dim; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * (i % half_dim)) / head_dim);
        float angle = pos * freq;
        cos_vals[i] = cosf(angle);
        sin_vals[i] = sinf(angle);
      }

      // Apply: result = (x * cos) + (rotate_half(x) * sin)
      // rotate_half: [-x[half:], x[:half]]
      float temp[64];
      for (int i = 0; i < head_dim; i++) {
        temp[i] = head[i];
      }

      for (int i = 0; i < half_dim; i++) {
        // First half: x[i] * cos[i] + (-x[i+half]) * sin[i]
        head[i] = temp[i] * cos_vals[i] - temp[i + half_dim] * sin_vals[i];
      }
      for (int i = half_dim; i < head_dim; i++) {
        // Second half: x[i] * cos[i] + x[i-half] * sin[i]
        head[i] = temp[i] * cos_vals[i] + temp[i - half_dim] * sin_vals[i];
      }
    }
  }
}

// RoPE (Rotary Position Embedding) - original version for compatibility
// Uses official transformers rotate_half pattern
void omni_rope(float *q, float *k, int seq_len, int n_heads, int head_dim,
               const int *positions, float theta) {
  // If q and k are the same, use inplace version
  if (q == k) {
    omni_rope_inplace(q, seq_len, n_heads, head_dim, positions, theta);
    return;
  }

  int half_dim = head_dim / 2;

  for (int pos_idx = 0; pos_idx < seq_len; pos_idx++) {
    int pos = positions[pos_idx];

    for (int h = 0; h < n_heads; h++) {
      float *q_head = q + pos_idx * n_heads * head_dim + h * head_dim;
      float *k_head = k + pos_idx * n_heads * head_dim + h * head_dim;

      // Compute cos/sin for all dimensions
      float cos_vals[64], sin_vals[64];
      for (int i = 0; i < head_dim; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * (i % half_dim)) / head_dim);
        float angle = pos * freq;
        cos_vals[i] = cosf(angle);
        sin_vals[i] = sinf(angle);
      }

      // Apply to Q: result = (q * cos) + (rotate_half(q) * sin)
      float q_temp[64];
      for (int i = 0; i < head_dim; i++)
        q_temp[i] = q_head[i];

      for (int i = 0; i < half_dim; i++) {
        q_head[i] =
            q_temp[i] * cos_vals[i] - q_temp[i + half_dim] * sin_vals[i];
      }
      for (int i = half_dim; i < head_dim; i++) {
        q_head[i] =
            q_temp[i] * cos_vals[i] + q_temp[i - half_dim] * sin_vals[i];
      }

      // Apply to K: result = (k * cos) + (rotate_half(k) * sin)
      float k_temp[64];
      for (int i = 0; i < head_dim; i++)
        k_temp[i] = k_head[i];

      for (int i = 0; i < half_dim; i++) {
        k_head[i] =
            k_temp[i] * cos_vals[i] - k_temp[i + half_dim] * sin_vals[i];
      }
      for (int i = half_dim; i < head_dim; i++) {
        k_head[i] =
            k_temp[i] * cos_vals[i] + k_temp[i - half_dim] * sin_vals[i];
      }
    }
  }
}

// Softmax
void omni_softmax(float *x, int n) {
#if defined(__ARM_NEON)
  // Find max
  float32x4_t vmax = vld1q_f32(x);
  int i = 4;
  for (; i + 4 <= n; i += 4) {
    vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
  }
  float max_val = vmaxvq_f32(vmax);
  for (; i < n; i++) {
    if (x[i] > max_val)
      max_val = x[i];
  }

  // Compute exp(x - max) and sum
  float32x4_t vsum = vdupq_n_f32(0.0f);
  i = 0;
  for (; i + 4 <= n; i += 4) {
    float vals[4];
    float32x4_t vx = vld1q_f32(x + i);
    vst1q_f32(vals, vsubq_f32(vx, vdupq_n_f32(max_val)));

    for (int j = 0; j < 4; j++) {
      vals[j] = expf(vals[j]);
    }
    float32x4_t vexp = vld1q_f32(vals);
    vst1q_f32(x + i, vexp);
    vsum = vaddq_f32(vsum, vexp);
  }

  float sum = vaddvq_f32(vsum);
  for (; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // Normalize
  float inv_sum = 1.0f / sum;
  float32x4_t vinv = vdupq_n_f32(inv_sum);
  i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
  }
  for (; i < n; i++) {
    x[i] *= inv_sum;
  }
#else
  // Scalar fallback
  float max_val = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max_val)
      max_val = x[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  float inv_sum = 1.0f / sum;
  for (int i = 0; i < n; i++) {
    x[i] *= inv_sum;
  }
#endif
}

// Matrix multiplication: C[M,N] = A[M,K] @ B[K,N]^T
// Uses Apple Accelerate for large matrices
void omni_matmul(float *C, const float *A, const float *B, int M, int N,
                 int K) {
#if defined(__APPLE__)
  // Use Apple Accelerate BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B,
              K, 0.0f, C, N);
#else
  // Simple tiled implementation
  memset(C, 0, M * N * sizeof(float));

  const int TILE = 32;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;

#if defined(__ARM_NEON)
      float32x4_t vsum = vdupq_n_f32(0.0f);
      int k = 0;
      for (; k + 4 <= K; k += 4) {
        float32x4_t va = vld1q_f32(A + i * K + k);
        float32x4_t vb = vld1q_f32(B + j * K + k);
        vsum = vfmaq_f32(vsum, va, vb);
      }
      sum = vaddvq_f32(vsum);
      for (; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
#else
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
#endif
      C[i * N + j] = sum;
    }
  }
#endif
}

// Q8_0 block structure (32 elements per block)
struct block_q8_0_kernel {
  uint16_t scale; // f16 scale
  int8_t qs[32];  // quantized values
};

// F16 to F32 conversion (inline for speed)
static inline float fp16_to_fp32_fast(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;

  if (exp == 0) {
    return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2, -14);
  } else if (exp == 31) {
    return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
  } else {
    uint32_t f32 = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
  }
}

// Quantized matmul: C[M,N] = A[M,K] @ B_q8[N,K]^T
// A is F32, B is Q8_0, C is F32
// Strategy: Dequantize B on-the-fly during matmul (fused dequant+matmul)
void omni_matmul_q8_0(float *C, const float *A, const void *B_q8, int M, int N,
                      int K) {
  const block_q8_0_kernel *blocks = (const block_q8_0_kernel *)B_q8;
  int n_blocks_per_row = K / 32;

  memset(C, 0, M * N * sizeof(float));

#if defined(__ARM_NEON)
  // NEON optimized: dequantize and accumulate in one pass
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float32x4_t sum_vec = vdupq_n_f32(0.0f);

      for (int b = 0; b < n_blocks_per_row; b++) {
        const block_q8_0_kernel *block = &blocks[j * n_blocks_per_row + b];
        const float *a_ptr = A + i * K + b * 32;

        // Get scale
        float scale = fp16_to_fp32_fast(block->scale);
        float32x4_t vscale = vdupq_n_f32(scale);

        // Process 32 elements in chunks of 4
        for (int k = 0; k < 32; k += 4) {
          // Load A values
          float32x4_t va = vld1q_f32(a_ptr + k);

          // Load and convert int8 to float
          int8x8_t vb_i8 = vld1_s8(block->qs + k);
          int16x4_t vb_i16 = vget_low_s16(vmovl_s8(vb_i8));
          int32x4_t vb_i32 = vmovl_s16(vb_i16);
          float32x4_t vb = vcvtq_f32_s32(vb_i32);

          // Dequantize and accumulate: sum += a * (b * scale)
          sum_vec = vfmaq_f32(sum_vec, va, vmulq_f32(vb, vscale));
        }
      }

      C[i * N + j] = vaddvq_f32(sum_vec);
    }
  }
#else
  // Scalar fallback
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;

      for (int b = 0; b < n_blocks_per_row; b++) {
        const block_q8_0_kernel *block = &blocks[j * n_blocks_per_row + b];
        const float *a_block = A + i * K + b * 32;

        float scale = fp16_to_fp32_fast(block->scale);

        // Fused dequant + dot product
        for (int k = 0; k < 32; k++) {
          sum += a_block[k] * (block->qs[k] * scale);
        }
      }

      C[i * N + j] = sum;
    }
  }
#endif
}
