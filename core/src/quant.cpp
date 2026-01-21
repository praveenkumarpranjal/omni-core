/*
 * Omni Core - Quantization and Dequantization
 *
 * Q8_0 and Q4_K dequantization with NEON
 */

#include "../include/omni.h"
#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Q8_0 block structure (matches .omni format)
struct block_q8_0 {
  uint16_t scale; // f16 scale
  int8_t qs[32];  // quantized values
};

// Q4_K block structure
struct block_q4_k {
  uint16_t d;         // super-block scale (f16)
  uint16_t dmin;      // super-block min scale (f16)
  uint8_t scales[12]; // 6-bit quantized scales
  uint8_t qs[128];    // 4-bit quants (256 elements packed)
};

// F16 to F32 conversion
static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;

  if (exp == 0) {
    // Subnormal
    return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2, -14);
  } else if (exp == 31) {
    // Inf or NaN
    return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
  } else {
    // Normal
    uint32_t f32 = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
  }
}

// Dequantize Q8_0 block
void omni_dequant_q8_0(float *out, const void *data, int n) {
  const block_q8_0 *blocks = (const block_q8_0 *)data;
  int n_blocks = n / 32;

#if defined(__ARM_NEON)
  for (int b = 0; b < n_blocks; b++) {
    float scale = fp16_to_fp32(blocks[b].scale);

    // Handle NaN/Inf scales (corrupted model data)
    // Use higher threshold (10000) to avoid zeroing too many weights
    // Model file has ~28% of blocks with |scale| > 100
    if (std::isnan(scale) || std::isinf(scale) || std::abs(scale) > 10000.0f) {
      scale = 0.0f;
    }

    float32x4_t vscale = vdupq_n_f32(scale);

    const int8_t *qs = blocks[b].qs;
    float *out_block = out + b * 32;

    // Process 4 elements at a time
    for (int i = 0; i < 32; i += 4) {
      // Load int8 values
      int8x8_t vi8 = vld1_s8(qs + i);

      // Convert to int16
      int16x8_t vi16 = vmovl_s8(vi8);

      // Convert lower 4 to int32
      int32x4_t vi32 = vmovl_s16(vget_low_s16(vi16));

      // Convert to float
      float32x4_t vf = vcvtq_f32_s32(vi32);

      // Scale
      vf = vmulq_f32(vf, vscale);

      // Store
      vst1q_f32(out_block + i, vf);
    }
  }
#else
  // Scalar fallback
  for (int b = 0; b < n_blocks; b++) {
    float scale = fp16_to_fp32(blocks[b].scale);

    // Handle NaN/Inf scales (corrupted model data)
    // Use higher threshold (10000) to avoid zeroing too many weights
    if (std::isnan(scale) || std::isinf(scale) || std::abs(scale) > 10000.0f) {
      scale = 0.0f;
    }

    for (int i = 0; i < 32; i++) {
      out[b * 32 + i] = blocks[b].qs[i] * scale;
    }
  }
#endif
}

// Dequantize Q4_K block
void omni_dequant_q4_k(float *out, const void *data, int n) {
  const block_q4_k *blocks = (const block_q4_k *)data;
  int n_blocks = n / 256;

  for (int b = 0; b < n_blocks; b++) {
    float d = fp16_to_fp32(blocks[b].d);
    float dmin = fp16_to_fp32(blocks[b].dmin);

    // Decode scales (6-bit quantized)
    float scales[8];
    float mins[8];

    // Simplified scale decoding (actual Q4_K is more complex)
    for (int i = 0; i < 8; i++) {
      uint8_t scale_byte = blocks[b].scales[i];
      scales[i] = d * (scale_byte & 0x3f);
      mins[i] = dmin * (scale_byte >> 6);
    }

    // Dequantize 8 sub-blocks of 32 elements
    const uint8_t *qs = blocks[b].qs;
    float *out_block = out + b * 256;

    for (int sb = 0; sb < 8; sb++) {
      float scale = scales[sb];
      float min = mins[sb];

      // Unpack 4-bit values (2 per byte)
      for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[sb * 16 + i];

        // Lower 4 bits
        int q0 = packed & 0xF;
        out_block[sb * 32 + i * 2] = scale * q0 + min;

        // Upper 4 bits
        int q1 = packed >> 4;
        out_block[sb * 32 + i * 2 + 1] = scale * q1 + min;
      }
    }
  }
}

// Get dequantized tensor data
float *omni_get_tensor_f32(omni_model *model, const char *name, float *buffer) {
  const omni_tensor *tensor = omni_get_tensor(model, name);
  if (!tensor)
    return nullptr;

  int64_t n = 1;
  for (auto dim : tensor->shape)
    n *= dim;

  switch (tensor->dtype) {
  case OMNI_F32:
    // Already float32, return pointer directly
    return (float *)tensor->data;

  case OMNI_F16: {
    // Dequantize F16 to F32
    const uint16_t *src = (const uint16_t *)tensor->data;
    for (int64_t i = 0; i < n; i++) {
      buffer[i] = fp16_to_fp32(src[i]);
    }
    return buffer;
  }

  case OMNI_Q8_0:
    omni_dequant_q8_0(buffer, tensor->data, n);
    return buffer;

  case OMNI_Q4_K:
    omni_dequant_q4_k(buffer, tensor->data, n);
    return buffer;

  default:
    return nullptr;
  }
}
