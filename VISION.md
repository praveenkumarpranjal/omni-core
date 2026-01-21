# Omni Core - Vision

## Mission
Build the fastest CPU inference engine for LLMs on Apple Silicon, beating llama.cpp, MLX, PyTorch, vLLM, and ollama.

## Goals

### Primary Goal
**Beat all competitors on Apple Silicon CPU**
- Target: 30-50 tok/s for LFM2 (1.3B params)
- Stretch: 60-100 tok/s
- Current baseline: 0.7 tok/s (NumPy)

### Secondary Goals
- Universal model format (.omni)
- Efficient quantization (Q4_K, Q8_0)
- Low memory footprint
- Fast model loading

## Architecture

### Language Stack
```
Python (optional wrapper)
    ↓
Rust CLI (omni binary)
    ↓ (C FFI)
C++ Library (libomni.dylib)
    ↓ (inline asm)
Assembly Kernels (ARM64 NEON)
```

### Components

**1. Rust CLI (`omni`)**
- Command-line interface
- Model conversion (GGUF → Omni)
- Benchmarking tools
- User-facing features

**2. C++ Core (`libomni.dylib`)**
- Model loading (mmap)
- Inference engine
- Memory management
- SIMD operations (NEON)
- Apple Accelerate integration

**3. Assembly Kernels**
- Q4_K dequantization + matmul fusion
- Fused RoPE
- Critical attention operations

## Target Model

### LFM2 (Liquid Foundation Model 2)
- Size: 1.3B parameters
- Architecture: Hybrid conv + attention
- File: `lfm2.omni`
- Focus: Optimize this ONE model first

### Why LFM2?
- Unique architecture (conv + attention)
- Good test case for optimization
- Real-world model
- Manageable size for iteration

## Optimization Strategy

### Phase 1: Foundation (Week 1)
- Reverse engineer llama.cpp, GGUF, MLX
- Design optimal .omni format v2
- Build C++ model loader
- Implement basic inference (F32)
- Rust CLI wrapper
- **Target: 5-10 tok/s**

### Phase 2: SIMD Optimization (Week 2)
- NEON intrinsics for all ops
- Apple Accelerate BLAS integration
- Optimized memory layout
- **Target: 15-25 tok/s**

### Phase 3: Quantization (Week 3)
- Q8_0 implementation
- Q4_K implementation
- Fused dequant + matmul
- **Target: 30-40 tok/s**

### Phase 4: Assembly (Week 4)
- Hand-written assembly for hot paths
- Fused operations
- Cache optimization
- **Target: 40-60 tok/s**

### Phase 5: Polish (Week 5)
- Profile and optimize bottlenecks
- Memory optimization
- Final tuning
- **Target: 60-100 tok/s**

## Success Metrics

### Performance
- Tokens/second (primary metric)
- Time to first token (latency)
- Memory usage (GB)
- CPU utilization (%)

### Comparison
Beat these on Apple Silicon:
- llama.cpp: ~10-20 tok/s
- MLX: ~15-30 tok/s
- PyTorch: ~2-5 tok/s
- vLLM: ~5-10 tok/s
- ollama: ~10-15 tok/s

### Quality
- Correct output (matches reference)
- Stable performance
- Low memory usage
- Fast loading

## Future Phases

### Phase 6: More Models
- PLM (Llama3-based with vision)
- Standard Llama/Llama3
- Other architectures

### Phase 7: GPU Support
- Metal compute shaders
- Hybrid CPU+GPU execution
- Unified memory optimization

### Phase 8: Other Platforms
- Linux ARM64
- x86-64 (AVX2/AVX-512)
- CUDA (NVIDIA)
- ROCm (AMD)

## Non-Goals (For Now)

- ❌ Multi-platform support (Apple Silicon only)
- ❌ Multiple models (LFM2 only)
- ❌ GPU support (CPU first)
- ❌ Distributed inference
- ❌ Training support
- ❌ Python-first API

## Philosophy

**Extract every drop of performance from Apple Silicon.**

- Use the hardware to its fullest
- No compromises for portability (yet)
- Measure everything
- Optimize hot paths aggressively
- Keep it simple and fast

## End State

A single command:
```bash
omni run lfm2.omni "Hello world"
```

That runs faster than anything else on Mac.
