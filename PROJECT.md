# Omni Core - Project Status

**Last Updated**: 2026-01-21 (23:50 IST)

## Current State

### Status: ✅ **CONV CACHE BUG FIXED! ALL TESTS PASSING!**

**Latest Test Results (2026-01-21)**:
- ✅ Prefill correlation: **0.9999** (matches PyTorch)
- ✅ Decode correlation: **0.9999** (matches PyTorch) - **FIXED FROM 0.66!**
- ✅ Top token predictions match PyTorch exactly
- ✅ All-at-once and KV cache modes produce same results
- ✅ Generation: 12-13 tok/s (with proper conv caching)
- ✅ No compilation errors or warnings
- ✅ Coherent text generation verified

**What's Built:**
- ✅ **model.cpp** (284 lines) - Model loader with mmap
- ✅ **kernels.cpp** (382 lines) - NEON kernels + Apple Accelerate BLAS + **FIXED RoPE**
- ✅ **quant.cpp** (189 lines) - Q8_0/Q4_K dequantization with NaN/extreme value handling
- ✅ **inference.cpp** (~800 lines) - LFM2 forward pass with weight caching, GQA attention, **FIXED conv caching**
- ✅ **omni.h** (153 lines) - C API with weight cache + conv cache
- ✅ **dequant_q8_0.s** (97 lines) - ARM64 assembly Q8_0 dequantization kernel
- ✅ **Rust CLI** (332 lines) - Working CLI with FFI bindings (info, run, bench commands)
- ✅ **convert.py** (284 lines) - HuggingFace to .omni converter with proper Q8_0 quantization
- ✅ **omni_bindings.py** (207 lines) - Python FFI bindings
- ✅ **libomni.dylib** - Shared library
- ✅ **Weight caching** - Dequantize once, cache in RAM, reuse forever
- ✅ **GQA attention** - Proper grouped query attention with causal masking and QK LayerNorm
- ✅ **Conv caching** - NOW WORKING! Caches Bx (after gating) with roll-left-insert pattern
- ✅ **RoPE** - Fixed to use official `rotate_half` pattern
- ✅ **Validation** - Correlation with PyTorch: **0.9999** (excellent!)
- ✅ **KV caching** - Both attention K/V AND conv activations cached correctly
- ✅ **chat.py** (235 lines) - Interactive chatbot with Q&A format, working correctly

**Performance:**
- **Prefill**: ~300 tok/s (varies by prompt length)
- **Decode**: 12-13 tok/s (with conv + KV cache)
- **Generation**: ~12 tok/s (end-to-end)
- Memory: ~2.7 GB (1.3GB model + 1.4GB weight cache + KV cache + conv cache)
- **Accuracy**: 0.9999 correlation with PyTorch ✅

**Model Quality:**
- Re-quantized with proper outlier handling
- 100% normal Q8_0 scales (|scale| <= 10)
- 0% NaN/Inf blocks
- 0% extreme scales
- Weight range: [-0.16, 0.20]
- Compression: 3.42x vs F32

**Architecture Understanding (from research):**
- LFM2.5-1.2B: Hybrid model optimized for on-device
- 16 blocks total: 10 conv blocks + 6 GQA attention blocks
- Conv blocks: Double-gated LIV convolutions for local features
- Attention blocks: Grouped Query Attention for long-range dependencies
- **Conv caching**: Stores Bx=[hidden, kernel_size] per layer (matches PyTorch Lfm2HybridConvCache)

**Bug Fixed (2026-01-21):**
- ✅ **CONV CACHE BUG FIXED!**
- **Root Cause**: Was caching BCx (before gating) instead of Bx (after B*x gating)
- **Solution**: 
  1. Cache Bx (after gating B*x), not BCx
  2. Use direct dot product for decode, not full convolution
  3. Use roll-left-then-insert pattern (matches PyTorch)
  4. Cache size: [hidden × kernel_size] per layer
- **Result**: Decode correlation improved from 0.66 to 0.9999!

**Chat Template Fix (2026-01-22):**
- ✅ **PROPER CHAT TEMPLATE IMPLEMENTED!**
- **Problem**: Model was generating A/B/C/D multiple choice format responses
- **Root Cause**: Using simple "Q: ... \nA:" prompt format instead of model's chat template
- **Solution**: Use `tokenizer.apply_chat_template()` which formats as:
  ```
  <|startoftext|><|im_start|>user
  {message}<|im_end|>
  <|im_start|>assistant
  ```
- **Sampling**: Added min_p=0.15, temperature=0.3 (matches official PyTorch example)
- **Stopping**: Stop generation on `<|im_end|>` token (ID 7)
- **Result**: Natural conversational responses with proper code formatting!

## Directory Structure

```
omni-core/
├── RULES.md           # Development rules and principles
├── VISION.md          # What we're building and why
├── PROJECT.md         # This file - current status
├── LEARNINGS.md       # Knowledge from reverse engineering
├── core/              # C++ inference engine
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── omni.h     # C API header
│   ├── src/
│   │   ├── model.cpp
│   │   ├── inference.cpp
│   │   ├── kernels.cpp
│   │   └── quant.cpp
│   └── asm/
│       └── (assembly kernels)
├── cli/               # Rust CLI
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
└── tools/             # Utilities
    └── convert.rs     # GGUF → Omni converter
```

## Target Model

**LFM2 (Liquid Foundation Model 2)**
- File: `/Users/beingthepraveen/Dev/projects/omni/lfm2.omni`
- Size: 1.3B parameters
- Architecture: Hybrid conv + attention
- Current performance: 0.7 tok/s (NumPy baseline)
- Target: 30-50 tok/s

## Next Steps

1. ✅ Model loader - DONE
2. ✅ NEON kernels - DONE
3. ✅ Quantization - DONE
4. ✅ Basic inference - DONE
5. ✅ Weight caching - DONE
6. ✅ GQA dimension fix - DONE
7. ✅ Fixed norm weight name - DONE
8. ✅ **NaN handling in dequantization** - DONE
9. ✅ **Re-quantized model with proper outlier handling** - DONE
10. ✅ **Implement proper GQA attention** - DONE
11. ✅ **Fixed layer type detection** - DONE
12. ✅ **Implement conv1d layers** - DONE
13. ✅ **Fixed RoPE implementation** - DONE
14. ✅ **Implement KV caching** - DONE (13 tok/s generation)
15. ✅ **Rust CLI with FFI** - DONE (info, run, bench commands)
16. ✅ **Assembly Q8_0 dequant** - DONE
17. ⏳ **Optimize decode performance** - Current: 14.6 tok/s, Target: 30+ tok/s
18. ⏳ Profile and optimize hot paths
19. ⏳ Add more assembly kernels (matmul, RoPE)

## Performance Tracking

### Current Performance (2026-01-21 - Weight Caching + Simplified Attention)
```
Model: LFM2 (1.3B params, Q8_0, 1.2GB)
Platform: Apple Silicon M-series
Compiler: Clang 17.0, -O3 -march=native

Generation Speed: 6.3 tok/s
Memory Usage: ~2.7 GB (1.2GB model + 1.5GB cache)
Speedup vs NumPy: 9x
Gap to Target (30 tok/s): 4.7x

Strategy: Weight caching (dequantize once, reuse)
- First forward: slow (dequant all weights ~1.5s)
- Subsequent forwards: fast (cached weights)
- Attention: Placeholder (just copies Q)
- Conv layers: Skipped (not implemented)
```

### Performance History
| Date | Change | Speed | vs Baseline | Notes |
|------|--------|-------|-------------|-------|
| 2026-01-21 | Clean model + caching | 17 tok/s | 24x | Re-quantized model |
| 2026-01-21 | Weight caching (simple) | 6.3 tok/s | 9x | Placeholder attention |
| 2026-01-21 | Quantized matmul | 0.6 tok/s | 0.9x | Too slow - reverted |
| 2026-01-21 | C++ + NEON | 4.75 tok/s | 6.8x | Repeated dequant |
| Baseline | NumPy | 0.7 tok/s | 1x | - |

### Next Optimizations
1. **Implement proper attention** - Currently just copying Q (wrong!)
2. **Implement conv1d layers** - 10 conv blocks skipped
3. **Profile hot paths** - Find remaining bottlenecks
4. **Optimize matmul** - Consider assembly or better BLAS usage

## Components Status

### C++ Core (libomni.dylib)
- ⏳ Model loader
- ⏳ Inference engine
- ⏳ NEON kernels
- ⏳ Quantization
- ⏳ Assembly integration

### Rust CLI (omni)
- ✅ FFI bindings to C++ library (ffi.rs)
- ✅ Command-line interface (clap)
- ✅ `info` command - Show model information
- ✅ `run` command - Run inference with token IDs
- ✅ `bench` command - Benchmark performance
- ✅ Safe Rust wrappers for C++ types
- ✅ Thread-safe design

### Assembly Kernels
- ⏳ Q4_K matmul
- ⏳ Fused RoPE
- ⏳ Attention ops

## Build Instructions

### Prerequisites
```bash
# C++ toolchain
xcode-select --install

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# CMake
brew install cmake
```

### Build C++ Library
```bash
cd core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Produces: build/libomni.dylib
```

### Build Rust CLI
```bash
cd cli
cargo build --release
# Produces: target/release/omni
```

### Usage
```bash
# Run inference
./cli/target/release/omni run lfm2.omni "Hello world"

# Convert model
./cli/target/release/omni convert model.gguf model.omni

# Benchmark
./cli/target/release/omni bench lfm2.omni
```

## Dependencies

### C++
- Apple Accelerate (system framework)
- C++17 standard library

### Rust
- clap (CLI parsing)
- rayon (parallelism)

### System
- macOS 12.0+ (Apple Silicon)
- Xcode Command Line Tools

## Known Issues

**CRITICAL: No KV Caching - Last Token Always Rank #1**
- KV cache not implemented yet
- Without KV cache, we reprocess ALL tokens on each forward pass
- Model sees: [1, 1098, 5706, 803, 4481, 856] → predicts next token
- Last token (856 " is") has strongest signal → ranks #1
- This is CORRECT behavior without KV cache!
- **With KV cache**: Would only process new token, use cached context
- **Impact**: Cannot generate properly without KV cache
- **Solution**: Implement KV cache to maintain context across generation steps

**Note: Attention Implementation is Correct**
- Proper GQA attention with QK LayerNorm
- Causal masking working
- Context-aware when processing multiple tokens at once
- Matches Python implementation (omni.py)
- Just needs KV caching for autoregressive generation

**Minor: Segfault on cleanup**
- Segfault occurs when freeing context/model in Python bindings
- Doesn't affect inference performance
- Likely double-free or memory management issue
- **Solution**: Debug memory management in C++ destructors

## Learnings Applied

### From llama.cpp Reverse Engineering:

**GGUF Format Insights:**
- Magic + version + counts + KV pairs + tensor info + aligned data
- Strings: length prefix (uint64_t) + data (no null terminator)
- Alignment: 32 bytes default (we'll use 64 for Apple Silicon)
- Metadata-first design enables streaming and mmap

**Quantization Schemes:**
- Q8_0: 8.5 bits/weight, simple scale + int8 values, 32-element blocks
- Q4_K: 4.5 bits/weight, super-blocks of 256, hierarchical scales
- Formula: `weight = (d * scale + dmin * min) * q`

**NEON Patterns:**
- Process 32 elements at once (8x float32x4_t)
- Use vmaxvq_f32 for horizontal max
- Use vcvtnq_s32_f32 for rounding
- Use vdotq_s32 for fast dot products (ARMv8.2+)
- Conditional compilation for feature detection

**Memory Layout:**
- 64-byte alignment for cache lines
- Block-based quantization fits in L1 cache
- Tiled matrix multiplication for cache efficiency
- Planar layout (metadata + data) for mmap

## Metrics to Track

- Tokens/second (generation speed)
- Time to first token (latency)
- Memory usage (GB)
- Model loading time (seconds)
- CPU utilization (%)

## Git Commits

(Track major milestones here)

---

**Remember**: Update this file after EVERY change!
