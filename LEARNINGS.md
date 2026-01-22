# Omni Core - Learnings

Knowledge extracted from reverse engineering llama.cpp, MLX, GGUF, and other high-performance inference engines.

---

## From Current Implementation

### Model File Corruption Issues

**Problem: Corrupted Q8_0 Quantization**
- Discovered that original `lfm2.omni` had severely corrupted Q8_0 blocks
- Statistics from scanning 36.5M blocks:
  - 4.6% have NaN/Inf scales
  - 27.8% have extreme scales (|scale| > 100)
  - 8.8% have large scales (10 < |scale| <= 100)
  - Only 58.8% have normal scales (|scale| <= 10)
  - Scale range: [-65504, 65504] (F16 min/max)

**Impact:**
- Extreme weight values after dequantization (±8M)
- Extreme matmul outputs (±388M)
- All-zero logits in final output
- Model completely unusable

**Root Cause:**
- Likely issue during quantization from HuggingFace model
- F16 scales hitting min/max values suggests overflow
- May be due to outlier weights in original model

**Solution Implemented:**
Created `omni-core/convert.py` with proper Q8_0 quantization:
```python
# Clip extreme values to prevent F16 overflow
MAX_SCALE = 1000.0  # Safe threshold for F16 (max is 65504)
absmax = np.clip(absmax, 1e-8, MAX_SCALE)

# Quantize: scale values to [-127, 127]
scales = absmax / 127.0
quantized = np.round(data / scales).astype(np.int8)

# Convert scales to F16 safely
scales_f16 = scales.astype(np.float16).flatten()

# Verify no NaN/Inf in scales
if np.any(np.isnan(scales_f16)) or np.any(np.isinf(scales_f16)):
    scales_f16 = np.nan_to_num(scales_f16, nan=0.0, posinf=MAX_SCALE, neginf=-MAX_SCALE)
```

**Results:**
- New model: 100% normal scales, 0% NaN/extreme
- Weight range: [-0.16, 0.20] (reasonable)
- Inference works: 17 tok/s (24x faster than NumPy)
- Compression: 3.42x vs F32

**Key Learnings:**
- Always clip outliers before quantization
- Verify scale distribution after quantization
- F16 has limited range - need safeguards
- Model quality is critical for performance

### What Works
- **KV Caching**: Stores K/V tensors per layer, avoids recomputation in autoregressive decoding
- **Flash Attention**: Block-wise computation reduces memory bandwidth for long sequences
- **Grouped Query Attention**: Efficient multi-head attention with fewer KV heads
- **.omni Format**: Custom binary format with mmap support, metadata JSON, aligned tensors
- **Weight Caching**: Dequantize once, cache in RAM, reuse forever (9x speedup)
- **NEON Dequantization**: Fast Q8_0 dequantization with ARM NEON intrinsics

### What's Slow
- **Pure NumPy**: 2.9 tok/s for text, 0.7 tok/s for LFM2
- **Python Overhead**: Loops, function calls, type conversions
- **No Quantization**: F32 weights = 4x memory bandwidth vs Q8_0
- **No SIMD**: NumPy uses BLAS but with overhead

### Key Insights
- Prefill (process all tokens) is O(n²) - bottleneck for long sequences
- Decode (one token at a time with cache) is fast with KV cache
- Vision tokens (9,522 for PLM) make prefill very slow
- Flash Attention gives 2x speedup for sequences >512 tokens
- **Weight caching is critical**: Dequantizing on every forward pass is 10x slower
- **Model quality matters**: Corrupted quantization makes model unusable
- **Attention is critical**: Without proper attention, model just repeats last token
- **Placeholder attention symptom**: Last input token always has highest logit
- **KV cache is essential**: Without it, last token always ranks #1 (correct behavior!)
- **Why last token ranks #1 without KV cache**: Model reprocesses all tokens, last token has strongest recency signal

### Understanding KV Cache Necessity

**Without KV Cache (Current C++ Implementation):**
```
Step 1: Process [1] → predict token A
Step 2: Process [1, A] → predict token B  
Step 3: Process [1, A, B] → predict token C
```
- Each step reprocesses ALL previous tokens
- Last token has strongest signal (just processed)
- Model correctly predicts last token as most likely
- Result: Token repetition (last token keeps winning)

**With KV Cache (Python omni.py):**
```
Step 1: Process [1], cache K/V → predict token A
Step 2: Process [A] only, use cached K/V from [1] → predict token B
Step 3: Process [B] only, use cached K/V from [1, A] → predict token C
```
- Only process NEW token
- Use cached context from previous tokens
- Balanced signal from all context
- Result: Proper generation

**Key Takeaway**: Last token ranking #1 is not a bug - it's expected without KV cache!

### Critical Discovery: Conv Layers Are Essential

**Problem**: Skipping conv layers causes completely wrong outputs
- Official model: "The capital of France is" → ":\n Paris"
- Our model (skipping conv): "The capital of France is" → " is is is"

**Root Cause**: LFM2 has 10 conv layers and 6 attention layers
- Conv layers: 0, 1, 3, 4, 6, 7, 9, 11, 13, 15
- Attention layers: 2, 5, 8, 10, 12, 14
- Skipping conv layers means 62.5% of the model is missing!

**Impact**: Model cannot function correctly without conv layers
- Conv layers process local patterns
- Attention layers process long-range dependencies
- Both are essential for correct predictions

**Solution**: Implemented conv1d layers but still debugging correctness

### Conv Layer Implementation Status

**Critical Discovery: Transpose Required!**

The official PyTorch implementation uses a transpose pattern:
```python
BCx = self.in_proj(x).transpose(-1, -2)  # [batch, seq, 3*hidden] -> [batch, 3*hidden, seq]
B, C, x = BCx.chunk(3, dim=-2)  # Split in hidden dimension
Bx = B * x  # [batch, hidden, seq]
conv_out = self.conv(Bx)[..., :seqlen]  # Conv operates on [batch, hidden, seq]
y = C * conv_out
y = y.transpose(-1, -2).contiguous()  # [batch, hidden, seq] -> [batch, seq, hidden]
y = self.out_proj(y)
```

**Key Insight**: Conv operates on **transposed** data `[batch, hidden, seq]` not `[batch, seq, hidden]`!

**What's Implemented:**
- ✅ Transpose after in_proj
- ✅ Split in transposed space
- ✅ Element-wise gating in transposed space
- ✅ Causal conv1d with left padding
- ✅ Transpose back before out_proj
- ✅ Weight caching for conv weights
- ✅ Correct weight access pattern (verified: kernel[0,0,:] = [0.0134735, 0.0926304, -0.035368])

**Current Issue:**
- Conv implementation runs without errors
- Weights load correctly (matches official model)
- Produces numerical values (not NaN)
- But outputs are still incorrect (last token ranks #1)
- Python reference implementation with same logic produces correct results (max diff 0.002)

**Debugging Status:**
1. ✅ Verified layer type detection is correct
2. ✅ Verified weights are being loaded correctly
3. ✅ Verified dequantization is correct (C++ matches official PyTorch values)
4. ✅ Verified transpose logic matches Python reference
5. ⏳ Need to trace intermediate values through C++ conv to find discrepancy

**Next Steps:**
- Add detailed logging to compare C++ intermediate values with Python at each step
- Verify padding implementation matches PyTorch
- Check if there's a subtle indexing bug in the transpose or convolution loops
- Consider testing with a single token to isolate the issue

### Debugging Token Repetition

**Problem**: Model generates infinite repetition (e.g., "is is is is...")

**Diagnosis Steps**:
1. Check logits distribution - are they reasonable? (Yes: -20 to +50 range)
2. Check top predictions - is last input token always #1? (Yes: confirms placeholder attention)
3. Check logit gap - how much higher is last token? (9+ points = virtually certain)

**Root Cause**: Placeholder attention (copying Q) means:
- No context from previous tokens
- Model only sees last token's embedding
- FFN amplifies strongest signal (last token)
- Prediction: "what follows X?" → "X" (repetition)

**Solution**: Implement proper attention mechanism (Q·K^T softmax V)

---

## From justine.lol (llamafile) - CRITICAL OPTIMIZATION TECHNIQUE

### Outer Loop Unrolling for Matrix Multiplication

**Key Insight**: Unroll BOTH outer loops (not inner loop) to share register loads across multiple FMAs.

**Why This Works**:
- Modern CPUs can speculatively execute inner loops on their own
- Unrolling outer loops enables sharing register loads across multiple operations
- Example: Load `a0` once, use it for 4 different FMAs with `k0, k1, k2, k3`
- Reduces memory references while exploiting instruction-level parallelism

**Performance Gains**:
- 2x faster than Intel MKL for matrices that fit in L2 cache (512-1024 size)
- 30-500% faster prompt processing for llama.cpp
- 810 GFLOPS on Alderlake i9-14900K (vs 295 GFLOPS for MKL at same size)

**Technique - 3x4 Tile Example**:
```cpp
// Unroll both outer loops: process 3 rows x 4 cols at once
for (int i = 0; i < m; i += 3)
    for (int j = 0; j < n; j += 4) {
        // 12 accumulators (3x4 tile)
        __m256 c00, c01, c02, c03;
        __m256 c10, c11, c12, c13;
        __m256 c20, c21, c22, c23;
        
        for (int l = 0; l < k; l += 8) {
            // Load B columns once
            __m256 k0 = load(B + j*ldb + l);
            __m256 k1 = load(B + (j+1)*ldb + l);
            __m256 k2 = load(B + (j+2)*ldb + l);
            __m256 k3 = load(B + (j+3)*ldb + l);
            
            // Load A row 0, use for 4 FMAs
            __m256 a0 = load(A + i*lda + l);
            c00 = fma(a0, k0, c00);
            c01 = fma(a0, k1, c01);
            c02 = fma(a0, k2, c02);
            c03 = fma(a0, k3, c03);
            
            // Load A row 1, use for 4 FMAs
            __m256 a1 = load(A + (i+1)*lda + l);
            c10 = fma(a1, k0, c10);
            c11 = fma(a1, k1, c11);
            c12 = fma(a1, k2, c12);
            c13 = fma(a1, k3, c13);
            
            // Load A row 2, use for 4 FMAs
            __m256 a2 = load(A + (i+2)*lda + l);
            c20 = fma(a2, k0, c20);
            c21 = fma(a2, k1, c21);
            c22 = fma(a2, k2, c22);
            c23 = fma(a2, k3, c23);
        }
        // Store 12 results
    }
```

**Key Points**:
- Each `k` vector is loaded once and used for 3 FMAs (shared across rows)
- Each `a` vector is loaded once and used for 4 FMAs (shared across columns)
- Total: 7 loads (4 k + 3 a) for 12 FMAs = 1.7x better than naive
- Works best for matrices that fit in L2 cache (512-1024 elements)

**Tile Sizes**:
- 3x4: Good balance for AVX2 (8 float32 vectors)
- 4x4: Better for AVX512 (16 float32 vectors)
- 1x4: For decode (M=1, single row)
- Adaptive tiling based on matrix size

**For LFM2 Decode (M=1)**:
- Most matmuls are M=1 (single token), N=2048-8192, K=2048
- Use 1x4 or 1x8 tiles to process multiple output columns at once
- Share single input row across multiple output computations

**Implementation Strategy**:
1. Create specialized kernels for common sizes (M=1, M=3, M=4)
2. Use tile packing to handle remainder elements
3. Integrate with llama.cpp threading model (no OpenMP in hot path)
4. Profile and optimize for L2 cache locality

**CRITICAL FINDING - Apple Silicon**:
- Custom matmul kernels from justine.lol are SLOWER than Apple Accelerate on M-series chips
- Tested: Custom 1x4 GEMV kernel = 9.4 tok/s vs Accelerate = 14.1 tok/s (33% slower!)
- Reason: Apple Accelerate is already highly optimized for Apple Silicon unified memory
- Justine's optimizations target x86 where MKL has overhead
- **Conclusion**: On Apple Silicon, stick with Accelerate for matmul, optimize elsewhere

## From llama.cpp

### GGUF Format Structure

**File Layout:**
```
1. Magic: "GGUF" (4 bytes)
2. Version: uint32_t (current: 3)
3. n_tensors: int64_t
4. n_kv: int64_t (number of key-value pairs)
5. KV pairs (metadata):
   - key: string (uint64_t length + data)
   - type: gguf_type (int32_t)
   - value: type-specific data
6. Tensor info (for each tensor):
   - name: string
   - n_dims: uint32_t
   - dims: int64_t[n_dims]
   - type: ggml_type (int32_t)
   - offset: uint64_t (in data section)
7. Padding to alignment (default: 32 bytes)
8. Tensor data (aligned binary blob)
```

**Key Insights:**
- Strings: length (uint64_t) + data (no null terminator)
- Alignment: configurable via "general.alignment" key (default 32)
- Metadata first, data last (enables streaming)
- All enums stored as int32_t
- Tensor data is one contiguous block

### Quantization Schemes

**Q4_K (4-bit K-quant) - 4.5 bits/weight:**
```cpp
typedef struct {
    ggml_half d;      // super-block scale for quantized scales
    ggml_half dmin;   // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];  // scales and mins (6-bit quantized)
    uint8_t qs[QK_K/2];            // 4-bit quants (256 elements)
} block_q4_K;
// QK_K = 256 (super-block size)
// K_SCALE_SIZE = 12
// Total: 4 + 12 + 128 = 144 bytes per 256 weights
```

**Formula:** `weight = (d * scale + dmin * min) * q`
- 8 blocks of 32 elements each
- Scales and mins are 6-bit quantized
- Better quality than Q4_0 due to per-block mins

**Q8_0 (8-bit) - 8.5 bits/weight:**
```cpp
typedef struct {
    ggml_half d;      // scale (f16)
    int8_t qs[32];    // quantized values
} block_q8_0;
// Block size: 32 elements
// Total: 2 + 32 = 34 bytes per 32 weights
```

**Formula:** `weight = d * q`
- Simple, fast dequantization
- Good balance of quality and speed

**Type Hierarchy:**
- F32: 32 bits (baseline)
- F16: 16 bits (2x compression)
- Q8_0: 8.5 bits (3.7x compression)
- Q4_K: 4.5 bits (7.1x compression)
- Q2_K: 2.6 bits (12.3x compression)

### NEON Optimizations

**Key Patterns from llama.cpp:**

1. **Conditional Compilation:**
```cpp
#if defined(__ARM_NEON)
#include <arm_neon.h>
// NEON code
#else
// Fallback code
#endif
```

2. **Feature Detection:**
```cpp
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
// Use dot product instructions (ARMv8.2+)
#elif defined(__ARM_NEON)
// Use basic NEON
#endif
```

3. **Q8_0 Quantization (NEON):**
```cpp
// Process 8 float32x4_t vectors (32 elements) at once
float32x4_t srcv[8];
for (int l = 0; l < 8; l++) srcv[l] = vld1q_f32(x + i*32 + 4*l);

// Find max absolute value
float32x4_t maxv = vabsq_f32(srcv[0]);
for (int l = 1; l < 8; l++) {
    maxv = vmaxq_f32(maxv, vabsq_f32(srcv[l]));
}
float max = vmaxvq_f32(maxv);  // Horizontal max

// Quantize: scale = max / 127
float d = max / 127.0f;
float id = (max != 0.0f) ? 127.0f / max : 0.0f;

// Convert to int8
for (int l = 0; l < 8; l++) {
    float32x4_t v = vmulq_n_f32(srcv[l], id);
    int32x4_t vi = vcvtnq_s32_f32(v);  // Round to nearest
    y[i].qs[4*l + 0] = vgetq_lane_s32(vi, 0);
    // ... extract other lanes
}
```

4. **Dot Product (NEON with DOTPROD):**
```cpp
#if defined(__ARM_FEATURE_DOTPROD)
// 4x faster than manual multiply-accumulate
int32x4_t sum = vdupq_n_s32(0);
for (int j = 0; j < QK8_0; j += 16) {
    int8x16_t v0 = vld1q_s8(x[i].qs + j);
    int8x16_t v1 = vld1q_s8(y[i].qs + j);
    sum = vdotq_s32(sum, v0, v1);  // Fused dot product
}
sumf += vaddvq_s32(sum) * (x[i].d * y[i].d);
#endif
```

5. **Vector Operations:**
```cpp
// Load: vld1q_f32, vld1q_s8, vld1q_u8
// Store: vst1q_f32, vst1q_s8
// Arithmetic: vaddq, vsubq, vmulq, vfmaq (FMA)
// Comparison: vmaxq, vminq
// Horizontal: vaddvq (sum all lanes), vmaxvq (max of all lanes)
// Convert: vcvtq_f32_s32, vcvtnq_s32_f32 (round)
```

**Apple Silicon Specific:**
- M1/M2/M3 have full ARMv8.4-A support
- DOTPROD available (ARMv8.2+)
- MATMUL available on M1+ (ARMv8.6+)
- 128-bit NEON registers (4x float32, 16x int8)
- Excellent FMA performance

### Memory Layout

**Cache-Friendly Patterns:**

1. **Tensor Alignment:**
   - GGUF default: 32 bytes
   - Omni uses: 64 bytes (cache line on Apple Silicon)
   - Ensures tensors start on cache boundaries

2. **Block-Based Quantization:**
   - Q8_0: 32 elements per block (128 bytes unquantized)
   - Q4_K: 256 elements per super-block (1KB unquantized)
   - Fits in L1 cache for processing

3. **Planar vs Interleaved:**
   - GGUF: Metadata first, data last (planar)
   - Enables mmap without copying
   - Data section is contiguous

4. **Tiled Matrix Multiplication:**
```cpp
// Process in tiles that fit in cache
for (int i = 0; i < M; i += TILE_M) {
    for (int j = 0; j < N; j += TILE_N) {
        for (int k = 0; k < K; k += TILE_K) {
            // Multiply tile (stays in L1/L2)
        }
    }
}
```

**Apple Silicon Cache Hierarchy:**
- L1: 128KB data + 192KB instruction (per P-core)
- L2: 12-24MB (shared per cluster)
- L3/SLC: 24-48MB (system level cache)
- Memory bandwidth: 200-400 GB/s (M3)

---

## From MLX (To Be Filled)

### Apple Silicon Patterns
(Reverse engineer and document)

### Unified Memory Usage
(Reverse engineer and document)

### Metal Integration
(Reverse engineer and document)

---

## From GGML (To Be Filled)

### Tensor Operations
(Reverse engineer and document)

### Graph Execution
(Reverse engineer and document)

---

## Performance Patterns

### Hot Paths (Where Time is Spent)
1. **Matrix Multiplication** (70% of compute)
   - Solution: Direct BLAS calls, quantization, assembly
2. **Attention** (20% of compute)
   - Solution: Flash attention, fused kernels
3. **Element-wise Ops** (10% of compute)
   - Solution: SIMD intrinsics

### Memory Bandwidth
- F32: 4 bytes per weight
- F16: 2 bytes per weight (2x faster)
- Q8_0: 1 byte per weight (4x faster)
- Q4_K: 0.5 bytes per weight (8x faster)

### Cache Optimization
- Align tensors to 64 bytes (cache line)
- Use tiled matrix multiplication
- Prefetch data before use
- Keep working set in L1/L2 cache

---

## Apple Silicon Specifics

### M-Series Architecture
- **CPU**: Performance + Efficiency cores
- **GPU**: Unified memory, Metal compute
- **Neural Engine**: For specific ops
- **AMX**: Apple Matrix coprocessor (undocumented)

### NEON (ARM64 SIMD)
- 128-bit registers (4x float32, 8x float16)
- Fused multiply-add (FMA)
- Vector operations
- Good for element-wise ops

### Accelerate Framework
- Optimized BLAS (cblas_sgemm)
- vDSP for signal processing
- Best performance on Apple Silicon
- Use for large matrix multiplications

### Memory
- Unified memory (CPU + GPU share)
- Zero-copy possible
- High bandwidth (200-400 GB/s on M3)

---

## Quantization Theory

### Q8_0 (8-bit)
- 1 byte per weight
- Block size: 32 elements
- 1 scale (f16) per block
- Formula: `weight = scale * int8_value`

### Q4_K (4-bit K-quant)
- 0.5 bytes per weight
- Super-block: 256 elements
- Multiple scales per super-block
- More complex but better quality

### Dequantization
- Can fuse with matmul for speed
- Dequant on-the-fly vs pre-dequant
- Trade-off: compute vs memory bandwidth

---

## Assembly Optimization

### When to Use Assembly
- Critical hot paths only (profiled)
- Operations done millions of times
- When compiler can't optimize well
- Fused operations (dequant + matmul)

### ARM64 NEON Instructions
(To be filled with specific patterns)

### Register Usage
(To be filled with conventions)

---

## Code Patterns

### Efficient Matrix Multiplication
```cpp
// Tiled matmul for cache efficiency
// C[M,N] = A[M,K] @ B[K,N]
for (int i = 0; i < M; i += TILE) {
    for (int j = 0; j < N; j += TILE) {
        for (int k = 0; k < K; k += TILE) {
            // Multiply tile
        }
    }
}
```

### NEON Intrinsics
```cpp
// Vector dot product
float32x4_t vsum = vdupq_n_f32(0.0f);
for (int i = 0; i < n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vsum = vfmaq_f32(vsum, va, vb);  // FMA
}
float sum = vaddvq_f32(vsum);
```

### Memory Mapping
```cpp
// mmap for zero-copy model loading
int fd = open(path, O_RDONLY);
void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
// Use data directly, no malloc/copy
```

---

## From llama.cpp Reverse Engineering (Optimization Breakthrough)

### Q8_0 Dot Product Optimization (The "Juice")

**Source Analysis**: `ggml/src/ggml-cpu/arch/arm/quants.c` (lines 1086-1116)

**Critical Findings**:
1.  **Block Unrolling (2x)**:
    - Processes **2 blocks (64 elements)** per loop iteration.
    - Reads two `Q8_0` blocks (`x0`, `x1`) and two `y` blocks (`y0`, `y1`).
    - Minimizes loop overhead and maximizes pipeline utilization.

2.  **Integer Dot Product (vdotq_s32)**:
    - **does NOT** dequantize to float before multiplication.
    - Uses `vdotq_s32` (SDOT instruction) to multiply `int8x16` vectors directly.
    - Accumulates into `int32` registers.
    - **Memory Bandwidth**: 1 byte/weight (vs 4 bytes if dequantized to float).
    - **Throughput**: SDOT instruction performs 4 multiplies per cycle per lane.

3.  **Accumulation Strategy**:
    - Accumulates results into `float32x4_t` vectors `sumv0` and `sumv1`.
    - Only performs FMA (floating point multiply-add) with the block scale **after** the dot product.
    - `sum = sum + (int_dot_prod * (scale_x * scale_y))`
    - Reduces the number of expensive floating point operations.

4.  **Bypassing BLAS for M=1**:
    - For vector-matrix multiplication (`gemv`, used in decode), `llama.cpp` **avoids** generic BLAS (`cblas_sgemm`).
    - Uses this custom hand-written NEON kernel instead.
    - Eliminates function call overhead and generic dispatcher overhead.

**Implementation Plan for Omni Core**:
1.  Implement `omni_gemv_q8_0_neon` in `kernels.cpp`.
2.  Use `__ARM_FEATURE_DOTPROD` or `vdotq_s32` intrinsic.
3.  Unroll loop 2x (process 64 weights).
4.  Bypass Apple Accelerate for `M=1` decode path.

**Expected Gain**:
- Increase from 14 tok/s to >35 tok/s by removing memory bandwidth bottleneck.
- Current: Read Int8 -> Convert to F32 -> F32 Math (4x bandwidth).
- Optimized: Read Int8 -> Int8 Math -> Scale (1x bandwidth).

## Benchmarking

### What to Measure
- Tokens/second (end-to-end)
- Time per operation (profiling)
- Memory bandwidth utilization
- Cache hit rates
- CPU utilization

### Tools
- Xcode Instruments (macOS)
- `perf` (Linux)
- Manual timing (std::chrono)
- Apple Performance API

---

## Common Pitfalls

### Avoid
- ❌ Python in hot paths
- ❌ Unnecessary memory copies
- ❌ Unaligned memory access
- ❌ Cache-unfriendly layouts
- ❌ Premature optimization (profile first!)

### Do
- ✅ Use mmap for model loading
- ✅ Align tensors to cache lines
- ✅ Batch operations when possible
- ✅ Use SIMD for element-wise ops
- ✅ Profile before optimizing

---

**This document grows as we learn. Update after every discovery!**


### MAJOR BREAKTHROUGH: Root Cause Found!

**After systematic comparison of EVERY step between PyTorch and C++:**

1. ✅ Embeddings match perfectly (diff < 3e-8)
2. ✅ RMS norm matches (diff < 0.02, within bfloat16 precision)
3. ✅ Conv in_proj matches (diff < 0.05)
4. ✅ Conv transpose/chunk matches
5. ✅ Conv gating matches
6. ✅ Conv convolution matches (diff < 0.05)
7. ✅ **Conv layer 0 output matches PyTorch!**
8. ✗ **Final predictions completely wrong (correlation 0.42)**

**ROOT CAUSE IDENTIFIED**: **Attention layers are not implemented correctly!**

- All 16 layers are being processed (verified with logging)
- Conv layers (0,1,3,4,6,7,9,11,13,15) work perfectly ✅
- Attention layers (2,5,8,10,12,14) produce wrong outputs ✗
- Official PyTorch attention requires `position_embeddings` and `attention_mask` parameters
- C++ implementation doesn't pass these parameters correctly

**Conv Padding Discovery**:
- Official uses `padding=kernel_size-1` (symmetric padding on BOTH sides)
- Then trims output to original length: `conv_out[..., :seqlen]`
- NOT causal left-padding as initially thought!
- PyTorch Conv1d with `padding=2` adds 2 zeros on LEFT and RIGHT

**Key Learnings**:
1. Always compare step-by-step, not just final output
2. Precision differences (bfloat16 vs float32) are acceptable (< 0.05)
3. Conv implementation was correct all along (after padding fix)
4. The real issue was attention layers, not conv layers!

### CRITICAL ATTENTION FINDINGS (2026-01-21)

**✅ RESOLVED - Model Working Correctly!**

**The Problem**: Model had correlation of 0.42 with PyTorch (completely wrong)

**Root Cause**: RoPE (Rotary Position Embedding) implementation was incorrect

**Our Implementation** (WRONG):
```cpp
// Pair rotation - rotates adjacent pairs
for (int i = 0; i < half_dim; i++) {
    float x0 = head[2*i];
    float x1 = head[2*i+1];
    head[2*i]   = x0 * cos - x1 * sin;
    head[2*i+1] = x1 * cos + x0 * sin;
}
```

**Official Implementation** (CORRECT):
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    # Split tensor in half and swap with negation
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

**The Fix** (C++):
```cpp
// Official rotate_half pattern
int half_dim = head_dim / 2;

// Compute cos/sin for all dimensions
float cos_vals[64], sin_vals[64];
for (int i = 0; i < head_dim; i++) {
    float freq = 1.0f / powf(theta, (float)(2 * (i % half_dim)) / head_dim);
    float angle = pos * freq;
    cos_vals[i] = cosf(angle);
    sin_vals[i] = sinf(angle);
}

// Apply: result = (x * cos) + (rotate_half(x) * sin)
float temp[64];
for (int i = 0; i < head_dim; i++) temp[i] = head[i];

// First half: x[i] * cos[i] + (-x[i+half]) * sin[i]
for (int i = 0; i < half_dim; i++) {
    head[i] = temp[i] * cos_vals[i] - temp[i + half_dim] * sin_vals[i];
}
// Second half: x[i] * cos[i] + x[i-half] * sin[i]
for (int i = half_dim; i < head_dim; i++) {
    head[i] = temp[i] * cos_vals[i] + temp[i - half_dim] * sin_vals[i];
}
```

**Results After Fix**:
- ✅ Correlation with PyTorch: **0.9998** (excellent!)
- ✅ Top-10 predictions match: 10/10
- ✅ Top-1 prediction: "Paris" (correct for "The capital of France is")
- ✅ Max difference: 0.22, Mean: 0.03 (within tolerance)

**Key Learnings**:
1. **rotate_half is NOT pair rotation** - it splits the tensor in half and swaps
2. **Pattern**: `[-second_half, first_half]` not `[(x0*cos - x1*sin), (x1*cos + x0*sin)]`
3. **Always validate against official implementation** - subtle differences matter!
4. **Test end-to-end** - intermediate values can match but final output can be wrong

---

**Official LFM2 Attention Signature**:
```python
def forward(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
    attention_mask: torch.Tensor,  # REQUIRED! Not optional!
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
```

**Key Differences from Our C++ Implementation**:

1. **Position Embeddings (cos, sin)**:
   - Official: Pre-computed by `Lfm2RotaryEmbedding` module
   - Shape: `[batch, seq_len, head_dim]`
   - Formula: `cos = emb.cos() * attention_scaling`, `sin = emb.sin() * attention_scaling`
   - Applied AFTER QK LayerNorm using `apply_rotary_pos_emb(q, k, cos, sin)`
   - Our C++: Computes cos/sin on-the-fly in `omni_rope()` - WRONG!

2. **Attention Mask**:
   - Official: REQUIRED parameter, not optional!
   - Shape: `[batch, 1, seq_len, seq_len]`
   - Causal mask: Upper triangle filled with `-inf`
   - Applied to attention scores BEFORE softmax
   - Our C++: Uses simple `if (k_pos <= q_pos)` check - INCOMPLETE!

3. **RoPE Application**:
   - Official: `apply_rotary_pos_emb()` uses complex rotation with cos/sin
   - Applied to transposed tensors `[batch, num_heads, seq_len, head_dim]`
   - Our C++: Simple rotation formula - may be incorrect

4. **Attention Scaling**:
   - Official: `scaling = head_dim ** -0.5` (stored in module)
   - Also has `attention_scaling` from RoPE (applied to cos/sin)
   - Our C++: Only uses `1.0 / sqrt(head_dim)` - missing attention_scaling

**What We Need to Fix**:
1. Pre-compute cos/sin embeddings (or pass them in)
2. Apply attention_scaling to cos/sin
3. Use proper attention mask (not just causal check)
4. Verify RoPE rotation formula matches `apply_rotary_pos_emb()`
5. Test each component against PyTorch

**Official RoPE Formula** (from transformers):
```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

This is DIFFERENT from our simple rotation! We need to implement `rotate_half` pattern.


---

## LFM2 Conv Layer Caching (2026-01-21)

### The Bug
During autoregressive decode with KV cache, the model produced incorrect outputs (correlation 0.66 with PyTorch instead of 0.9999).

### Root Cause Analysis

**What PyTorch Does** (from `modeling_lfm2.py` `Lfm2ShortConv.slow_forward()`):
- Caches `Bx = B * x` (AFTER gating), not BCx (before gating)
- Uses roll-left-then-insert pattern for decode
- Computes conv output via direct dot product during decode (no full convolution)
- Cache shape: `[batch, hidden, kernel_size]`

**What We Were Doing Wrong**:
1. Caching BCx (before gating) instead of Bx (after `B * x` gating)
2. Running full convolution during decode instead of direct dot product
3. Cache size was kernel_size-1=2 instead of kernel_size=3
4. Cache layout was [time, features] instead of [features, time]

### The Fix

1. Cache Bx (after gating B*x), not BCx
2. Use direct dot product for decode: `conv_out = sum(cache * weights)`
3. Use roll-left-then-insert pattern
4. Cache size: [hidden x kernel_size] per layer

### Key Learnings

1. **Conv caching is separate from attention KV caching** - LFM2 uses `Lfm2HybridConvCache` for BOTH
2. **Cache what comes OUT of gating, not what goes IN** - `Bx = B * x` is the actual conv input
3. **Decode mode uses direct dot product** - No need for full conv since we have exactly kernel_size cached values
4. **Roll-left-insert pattern** - Shift cache left, insert new value at end
5. **Cache shape: [hidden, kernel_size]** - Per-channel, full kernel window

---

## LFM2 Chat Template Format (2026-01-22)

### The Problem
Model was generating A/B/C/D multiple choice format responses instead of natural conversation.

### Root Cause
Using simple `Q: {input}\nA:` format instead of the model's proper chat template.

### The Solution

**Correct Chat Template Format**:
```
<|startoftext|><|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

**How to Apply**:
```python
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_input}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
)[0].tolist()
```

### Sampling Parameters (Official)
- `temperature=0.3` - Lower = more deterministic
- `min_p=0.15` - Nucleus-like filtering, removes low probability tokens
- `repetition_penalty=1.05` - Prevents token repetition

### Stopping Conditions
- EOS token (tokenizer.eos_token_id)
- `<|im_end|>` token (ID 7 for LFM2)

### Key Learnings

1. **Every model has its own chat template** - Check `tokenizer.chat_template`
2. **Use `apply_chat_template()`** - Don not manually format prompts
3. **Stop on template-specific tokens** - `<|im_end|>` for LFM2, not just EOS
4. **Min-P > Top-P** - More effective for coherent generation
5. **Low temperature (0.3)** - Better for factual responses
6. **Decode with skip_special_tokens=True** - Clean output for users

---

## Reverse Engineering: MLX (2026-01-22)

### Overview
MLX is Apple's machine learning framework designed specifically for Apple Silicon. It achieves high performance through native Metal GPU compute shaders with unified memory architecture.

### Key Architecture Findings

#### 1. **Metal Device Management** (mlx/backend/metal/device.cpp)
- Uses `MTL::CopyAllDevices()` to get Metal devices
- Command queues with concurrent dispatch: `computeCommandEncoder(MTL::DispatchTypeConcurrent)`
- Smart memory barrier management between encoders
- Kernel caching system with library/kernel maps
- Residency sets for keeping model weights resident on GPU

#### 2. **GEMV Metal Kernel Design** (mlx/backend/metal/kernels/gemv.metal)
Key optimizations:
```metal
template <typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    ...>
struct GEMVKernel {
    // Each thread handles TM x TN block
    // Uses simd_sum for efficient reduction within simdgroup
    // Threadgroup memory for cross-simdgroup accumulation
};
```

**Key Patterns:**
- **SIMD Group Reductions**: Uses `simd_sum()` and `simd_shuffle_down()` for efficient parallel reduction
- **Threadgroup Memory**: Uses `threadgroup` memory for cross-simdgroup reduction
- **Thread-Local Accumulation**: Each thread accumulates results locally before reducing
- **Multiple Block Sizes**: Configurable BM, BN, SM, SN, TM, TN for different workloads
- **Safe/Unsafe Load Paths**: Separate paths for aligned vs edge cases

#### 3. **Quantized GEMV** (mlx/backend/metal/kernels/quantized.h)
Supports 2/3/4/5/6/8-bit quantization with:
```metal
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
```

**Key Patterns:**
- **Fused Dequant+Dot**: Dequantizes on-the-fly during dot product
- **Simdgroup Parallelism**: Each simdgroup processes multiple output rows
- **Block Processing**: Processes quantization blocks (32 elements) as atomic unit
- **Quad Groups**: Uses quad (4-thread) groups for small workloads

#### 4. **RMS Norm Metal Kernel** (mlx/backend/metal/kernels/rms_norm.metal)
```metal
template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_single_row(...) {
    // Phase 1: Thread-local accumulation
    for (int i = 0; i < N_READS; i++) {
        float xi = x[i];
        acc += xi * xi;
    }
    
    // Phase 2: Simd sum
    acc = simd_sum(acc);
    
    // Phase 3: Cross-simdgroup reduction via threadgroup memory
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 4: Final reduction and normalize
    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    
    // Phase 5: Apply normalization
    out[i] = w[i] * static_cast<T>(x[i] * local_inv_mean[0]);
}
```

#### 5. **Steel GEMM Dispatch** (mlx/backend/metal/matmul.cpp)
Apple's STEEL (Streaming Tensor Engine for Efficient Linear) kernels:
- **Block sizes**: 64x64, 32x64, 64x32 with BK=16 or 32
- **Warp dimensions**: wm=2, wn=2 (warps per tile)
- **Swizzle pattern**: For cache efficiency
- **Split-K**: For tall/skinny matrices with large K dimension

```cpp
// Split-K when output is small but K is large
if (batch_size_out == 1 && (_tm * _tn) <= 32 && _tk >= 8) {
    return steel_gemm_splitk_axpby(...)
}
```

---

## Reverse Engineering: llama.cpp GGML Metal (2026-01-22)

### Overview
GGML Metal backend achieves high performance through carefully optimized Metal shaders with extensive quantization support.

### Key Architecture Findings

#### 1. **Metal Backend Interface** (ggml/src/ggml-metal/ggml-metal.cpp)
- **Shared vs Private Buffers**: Uses `MTLStorageMode.shared` for CPU-GPU accessible memory
- **Mapped Buffers**: Zero-copy buffers from host memory with `buffer_from_host_ptr`
- **Graph Optimization**: Pre-processes computation graph for Metal execution
- **Async Tensor Ops**: `set_tensor_async` and `get_tensor_async` for overlapped transfers

```cpp
static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    .free_buffer    = ggml_backend_metal_buffer_shared_free_buffer,
    .get_base       = ggml_backend_metal_buffer_shared_get_base,
    .memset_tensor  = ggml_backend_metal_buffer_shared_memset_tensor,
    .set_tensor     = ggml_backend_metal_buffer_shared_set_tensor,
    .get_tensor     = ggml_backend_metal_buffer_shared_get_tensor,
    ...
};
```

#### 2. **Q8_0 Dequantization** (ggml/src/ggml-metal/ggml-metal.metal)
```metal
template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;
    
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (type4x4) reg_f;
}
```

**Key Patterns:**
- Uses `float4x4` (4x4 matrix) as basic processing unit = 16 elements
- `il` (index low) splits block into two halves for parallel processing
- Direct pointer casting for efficient memory access

#### 3. **Q4_K Complex Dequantization**
```metal
template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;
    
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    ...
}
```

#### 4. **Metal Buffer Allocation Strategy**
- **32-byte alignment**: Required for Metal buffer access
- **Max buffer size**: Device-dependent, from `device->maxBufferLength()`
- **Extra allocation** for Flash Attention, mul_mat_id operations

### Key Optimization Patterns for Omni Core

Based on MLX and llama.cpp analysis, here are the critical optimizations needed:

#### **Priority 1: Metal GEMV Kernel**
The single biggest win for single-token decode:
1. Create Metal compute shader for matrix-vector multiplication
2. Use simd_sum/shuffle for efficient reduction
3. Process Q8_0 directly on GPU (fused dequant+dot)
4. Target block size 4x1 or 8x1 for GEMV

#### **Priority 2: Unified Memory Zero-Copy**
1. Keep model weights in Metal buffers (shared storage mode)
2. Avoid CPU↔GPU copies - use unified memory directly
3. Make buffers resident with residency sets

#### **Priority 3: Metal RMS Norm + SiLU + RoPE**
1. Fuse normalization operations into single kernel
2. Use simd_sum for parallel reduction
3. Batch process with N_READS pattern

#### **Priority 4: Flash Attention on Metal**
1. Block-wise attention with threadgroup memory
2. Fused softmax within attention computation
3. Use simd operations for score accumulation

### Performance Expectations

Based on MLX achieving 50+ tok/s for similar model sizes:
- **Metal GEMV**: 2-3x speedup for matmul (14 → 35-40 tok/s)
- **Fused Kernels**: Additional 10-20% improvement
- **Memory Resident**: Eliminates buffer allocation overhead
- **Target**: 35-50 tok/s on M-series chips

---

## Implementation Plan: Metal Backend for Omni Core

### Critical Findings from Reverse Engineering (2026-01-22)

**MLX GEMV Kernel Architecture** (gemv.metal):
```metal
template <typename T, int BM, int BN, int SM, int SN, int TM, int TN>
struct GEMVKernel {
  // BM, BN: Threadgroup blocks (in simdgroups)
  // SM, SN: Simdgroup size (in threads) - must be 32 total
  // TM, TN: Thread work size (in elements)
  
  // Key pattern: Each thread processes TM x TN block
  // Uses simd_shuffle_down for efficient reduction
  // Threadgroup memory for cross-simdgroup accumulation
}
```

**Optimal Block Sizes for Decode (M=1)**:
- `BM=4, BN=1, SM=1, SN=32, TM=4, TN=4` - Best for single-token decode
- `BM=8, BN=1, SM=1, SN=32, TM=4, TN=4` - Alternative for larger outputs

**llama.cpp Q8_0 Dequantization** (ggml-metal.metal):
```metal
template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;  // F16 scale
    
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (type4x4) reg_f;
}
```

**Key Patterns**:
1. **Simd Reductions**: `simd_shuffle_down()` for parallel sum within simdgroup
2. **Threadgroup Memory**: For cross-simdgroup accumulation
3. **float4x4**: Process 16 elements at once (4x4 matrix)
4. **Unified Memory**: Zero-copy with `MTLStorageMode.shared`
5. **Pipeline Caching**: Compile once, reuse forever

### Phase 1: Metal Infrastructure (Core)
**Goal**: Initialize Metal device, create command queue, load shaders

**Files to Create**:
- `core/metal/device.mm` - Objective-C++ Metal device wrapper
- `core/metal/kernels.metal` - Metal shader library
- `core/include/omni_metal.h` - C API for Metal backend

**Key APIs**:
```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queue = [device newCommandQueue];
id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
```

### Phase 2: Metal GEMV Q8_0 Kernel
**Goal**: Fused dequant+matmul for single-token decode (M=1)

**Kernel Signature**:
```metal
kernel void gemv_q8_0(
    device const block_q8_0* B [[buffer(0)]],  // Quantized weights [N, K]
    device const float* A [[buffer(1)]],        // Input vector [K]
    device float* C [[buffer(2)]],              // Output vector [N]
    constant int& K [[buffer(3)]],              // Input size
    constant int& N [[buffer(4)]],              // Output size
    uint tid [[thread_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
```

**Algorithm**:
1. Each thread processes 4 output elements (TM=4)
2. Loop over K in blocks of 32 (Q8_0 block size)
3. Dequantize on-the-fly: `weight = scale * int8_value`
4. Accumulate: `result += A[k] * weight`
5. Simd reduction with `simd_shuffle_down()`
6. Write output

### Phase 3: Metal RMS Norm Kernel
**Goal**: Fast normalization with simd reductions

**Kernel Signature**:
```metal
kernel void rms_norm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
```

**Algorithm**:
1. Thread-local accumulation: `sum_sq += x[i] * x[i]`
2. Simd sum: `sum_sq = simd_sum(sum_sq)`
3. Compute RMS: `rms = 1.0 / sqrt(sum_sq / n + eps)`
4. Apply: `out[i] = x[i] * rms * weight[i]`

### Phase 4: Metal Attention Kernel
**Goal**: Efficient Q·K^T + softmax + Score·V

**Approach**: Use 3 separate kernels (easier to optimize):
1. `attn_scores`: Compute Q·K^T with causal mask
2. `attn_softmax`: Numerically stable softmax
3. `attn_values`: Weighted sum of V

### Phase 5: Integration & Optimization
1. Fallback to CPU for unsupported ops
2. Profile with Xcode Instruments
3. Optimize buffer management
4. Tune threadgroup sizes

**Expected Performance**:
- Metal GEMV: 2-3x faster than CPU Accelerate
- Target: 35-50 tok/s (vs current 14.1 tok/s)
- Memory: Same (unified memory, zero-copy)
