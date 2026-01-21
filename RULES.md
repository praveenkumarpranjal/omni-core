# Omni Core - Development Rules

## Core Principles

### 1. No Mock Implementations
- Every function must be real, working code
- No placeholders, no TODOs in committed code
- If it's not ready, don't write it yet

### 2. Always Use venv
```bash
# ALWAYS use:
venv/bin/python3
venv/bin/pip

# NEVER use:
python3
pip
```

### 3. Minimal Documentation
- Maximum 4 .md files:
  - RULES.md (this file)
  - PROJECT.md (current state, updated after EVERY change)
  - LEARNINGS.md (knowledge from reverse engineering)
  - VISION.md (what we're building)
- No extensive READMEs
- Code should be self-documenting

### 4. Clean Codebase
- Remove test files after testing
- No commented-out code
- No unused files
- Compact directory structure

### 5. Testing
- **Test everything you build**
- **Test at EACH step and after EVERY change**
- **Never break working functionality**
- Compile after every component
- Run tests with timing measurements
- Verify functionality before moving on
- **Always remove test files after testing**
- No test/ directory in repo
- **If tests fail, revert changes immediately**

### 5. Apple Silicon First
- Optimize for M1/M2/M3/M4 chips
- Use ARM64 NEON intrinsics
- Use Apple Accelerate framework
- Other platforms come later

### 6. Language Split
- **C++ (80%)**: Core engine, inference, memory management
- **Rust (15%)**: CLI, model conversion, thread-safe utilities
- **Assembly (5%)**: Critical hot paths only (Q4_K matmul, RoPE)
- **Python**: Thin wrapper only (optional)

### 7. Architecture
- **Separate binaries**: Rust CLI → C++ library (libomni.dylib)
- Clean C ABI boundary
- No complex FFI in hot paths

### 8. Single Model Focus
- Optimize LFM2 first (lfm2.omni)
- Get to 30-50 tok/s before adding other models
- Measure everything

### 9. Update PROJECT.md
- After EVERY change, update PROJECT.md
- Keep it current with what's built
- Track performance metrics

### 10. Build System
- CMake for C++
- Cargo for Rust
- Simple Makefile for convenience

## Performance Targets

### LFM2
- Current: 0.7 tok/s (NumPy baseline)
- Target: 30-50 tok/s (40-70x speedup)
- Stretch: 60-100 tok/s

### Metrics
- Tokens/second (generation speed)
- Time to first token (latency)
- Memory usage
- CPU utilization

## Development Workflow

1. Reverse engineer (llama.cpp, MLX, GGUF, pytorch, etc.)
2. Document learnings in LEARNINGS.md
3. Build feature
4. Test and measure
5. Remove test files
6. Update PROJECT.md
7. Commit

## Code Style

### C++
- C++17 standard
- Use RAII
- Prefer stack allocation
- Use Apple Accelerate for BLAS
- NEON intrinsics for SIMD
- Inline assembly for critical kernels

### Rust
- 2021 edition
- Use clap for CLI
- Use rayon for parallelism
- Safe FFI bindings
- Zero unsafe unless necessary

### Assembly
- ARM64 only
- NEON instructions
- Document register usage
- Benchmark before/after

## File Naming

- C++ headers: `.h` (not `.hpp`)
- C++ source: `.cpp`
- Assembly: `.s`
- Rust: standard Cargo structure
- No camelCase in filenames

## Git Workflow

- Commit working code only
- Clear commit messages
- Update PROJECT.md in same commit
- No WIP commits

## Testing

- Test locally
- Measure performance
- Remove test files after
- No test/ directory in repo

## Dependencies

### C++
- Apple Accelerate (system)
- No external dependencies if possible

### Rust
- clap (CLI)
- rayon (parallelism)
- Minimal crates only

## What NOT to Do

- ❌ No mock implementations
- ❌ No extensive documentation
- ❌ No test files in repo
- ❌ No commented-out code
- ❌ No TODO comments
- ❌ No platform-specific code for non-Apple (yet)
- ❌ No Python in hot paths
- ❌ No complex FFI patterns
- ❌ No premature optimization (profile first)
- ❌ No adding features before LFM2 is optimized

## Remember

**Quality over quantity. Speed over features. Measure everything.**
