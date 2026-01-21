// ARM64 NEON Assembly Kernels for Omni Core
// 
// Optimized implementations of critical hot paths:
// 1. RMS Norm
// 2. Attention Score Calculation (Q * K^T)
// 3. Attention Value Aggregation (Sum(Score * V))
// 4. GEMV Q8_0 (Block-wise dequantization and dot product)
// 5. GEMV Int8 (Q8_0 blocks with Int8 A) using SDOT

.global _omni_rms_norm_asm
.global _omni_attn_score_asm
.global _omni_attn_value_asm
.global _omni_gemv_q8_0_row_asm
.global _omni_gemv_q8_0_4row_asm
.global _omni_gemv_q8_s8_4row_asm
.align 4

// ----------------------------------------------------------------------------
// void omni_rms_norm_asm(float* out, const float* x, const float* w, int n, float eps)
// ----------------------------------------------------------------------------
_omni_rms_norm_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    fmov    s16, s0
    movi    v0.4s, #0
    mov     x4, x1
    mov     w5, w3
    
.Lrms_sum_loop_retry:
    cmp     w5, #4
    b.lt    .Lrms_sum_scalar
    ld1     {v1.4s}, [x4], #16
    fmla    v0.4s, v1.4s, v1.4s
    sub     w5, w5, #4
    b       .Lrms_sum_loop_retry

.Lrms_sum_scalar:
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    scvtf   s1, w3
    fdiv    s0, s0, s1
    fadd    s0, s0, s16
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0
    dup     v0.4s, v0.s[0]
    
.Lrms_apply_loop:
    ld1     {v1.4s}, [x1], #16
    ld1     {v2.4s}, [x2], #16
    fmul    v1.4s, v1.4s, v0.4s
    fmul    v1.4s, v1.4s, v2.4s
    st1     {v1.4s}, [x0], #16
    subs    w3, w3, #4
    b.gt    .Lrms_apply_loop
    ldp     x29, x30, [sp], #16
    ret

// ----------------------------------------------------------------------------
// void omni_attn_score_asm(float* scores, const float* q, const float* k_cache, ...
// ----------------------------------------------------------------------------
_omni_attn_score_asm:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x19, [sp, #16]
    lsl     x5, x5, #2
    cbz     x3, .Lscore_end
.Lscore_pos_loop:
    mov     x6, x1
    mov     x7, x2
    mov     w8, w4
    movi    v0.4s, #0
.Lscore_dot_loop:
    ld1     {v1.4s}, [x6], #16
    ld1     {v2.4s}, [x7], #16
    fmla    v0.4s, v1.4s, v2.4s
    subs    w8, w8, #4
    b.gt    .Lscore_dot_loop
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    st1     {v0.s}[0], [x0], #4
    add     x2, x2, x5
    subs    x3, x3, #1
    b.gt    .Lscore_pos_loop
.Lscore_end:
    ldr     x19, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ----------------------------------------------------------------------------
// void omni_attn_value_asm(...)
// ----------------------------------------------------------------------------
_omni_attn_value_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    lsl     x5, x5, #2
    cbz     x3, .Lvalue_end
    mov     x6, x0
    mov     w7, w4
    mov     x0, #0
.Lvalue_dim_loop:
    cmp     w7, #0
    ble     .Lvalue_end
    movi    v0.4s, #0
    mov     x8, x1
    mov     x9, x2
    mov     w10, w3
    add     x9, x9, x0
.Lvalue_pos_loop:
    ld1     {v1.s}[0], [x8], #4
    dup     v1.4s, v1.s[0]
    ld1     {v2.4s}, [x9]
    fmla    v0.4s, v2.4s, v1.4s
    add     x9, x9, x5
    subs    w10, w10, #1
    b.gt    .Lvalue_pos_loop
    st1     {v0.4s}, [x6], #16
    add     x0, x0, #16
    sub     w7, w7, #4
    b       .Lvalue_dim_loop
.Lvalue_end:
    ldp     x29, x30, [sp], #16
    ret

// ----------------------------------------------------------------------------
// float omni_gemv_q8_0_row_asm(...)
// ----------------------------------------------------------------------------
_omni_gemv_q8_0_row_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    movi    v0.4s, #0
    cbz     x2, .Lgemv_end
.Lgemv_block_loop:
    ldrh    w3, [x1]
    fmov    s1, w3
    fcvt    h1, s1
    fcvt    s1, h1
    dup     v1.4s, v1.s[0]
    add     x4, x1, #2
    movi    v2.4s, #0
    
    ld1     {v3.8b}, [x4], #8
    ld1     {v4.4s, v5.4s}, [x0], #32
    sxtl    v6.8h, v3.8b
    sxtl    v7.4s, v6.4h
    sxtl2   v8.4s, v6.8h
    scvtf   v7.4s, v7.4s
    scvtf   v8.4s, v8.4s
    fmla    v2.4s, v7.4s, v4.4s
    fmla    v2.4s, v8.4s, v5.4s

    ld1     {v3.8b}, [x4], #8
    ld1     {v4.4s, v5.4s}, [x0], #32
    sxtl    v6.8h, v3.8b
    sxtl    v7.4s, v6.4h
    sxtl2   v8.4s, v6.8h
    scvtf   v7.4s, v7.4s
    scvtf   v8.4s, v8.4s
    fmla    v2.4s, v7.4s, v4.4s
    fmla    v2.4s, v8.4s, v5.4s

    ld1     {v3.8b}, [x4], #8
    ld1     {v4.4s, v5.4s}, [x0], #32
    sxtl    v6.8h, v3.8b
    sxtl    v7.4s, v6.4h
    sxtl2   v8.4s, v6.8h
    scvtf   v7.4s, v7.4s
    scvtf   v8.4s, v8.4s
    fmla    v2.4s, v7.4s, v4.4s
    fmla    v2.4s, v8.4s, v5.4s

    ld1     {v3.8b}, [x4], #8
    ld1     {v4.4s, v5.4s}, [x0], #32
    sxtl    v6.8h, v3.8b
    sxtl    v7.4s, v6.4h
    sxtl2   v8.4s, v6.8h
    scvtf   v7.4s, v7.4s
    scvtf   v8.4s, v8.4s
    fmla    v2.4s, v7.4s, v4.4s
    fmla    v2.4s, v8.4s, v5.4s
    
    fmul    v2.4s, v2.4s, v1.4s
    fadd    v0.4s, v0.4s, v2.4s
    add     x1, x1, #34
    subs    x2, x2, #1
    b.gt    .Lgemv_block_loop
.Lgemv_end:
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    ldp     x29, x30, [sp], #16
    ret

// ----------------------------------------------------------------------------
// void omni_gemv_q8_0_4row_asm(float* out, const float* A, const void* B, ...)
// ----------------------------------------------------------------------------
_omni_gemv_q8_0_4row_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    mov     x5, x2
    add     x6, x5, x4
    add     x7, x6, x4
    add     x8, x7, x4
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    cbz     x3, .Lgemv4_end
.Lgemv4_block_loop:
    ldrh    w9, [x5]
    ldrh    w10, [x6]
    ldrh    w11, [x7]
    ldrh    w12, [x8]
    fmov    s20, w9
    fcvt    h20, s20
    fcvt    s20, h20
    dup     v20.4s, v20.s[0]
    fmov    s21, w10
    fcvt    h21, s21
    fcvt    s21, h21
    dup     v21.4s, v21.s[0]
    fmov    s22, w11
    fcvt    h22, s22
    fcvt    s22, h22
    dup     v22.4s, v22.s[0]
    fmov    s23, w12
    fcvt    h23, s23
    fcvt    s23, h23
    dup     v23.4s, v23.s[0]
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    add     x13, x5, #2
    add     x14, x6, #2
    add     x15, x7, #2
    add     x9, x8, #2
    
    // Unrolled x4
    // Chunk 0
    ld1     {v0.4s, v1.4s}, [x1], #32
    
    ld1     {v2.8b}, [x13], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v24.4s, v4.4s, v0.4s
    fmla    v24.4s, v5.4s, v1.4s
    
    ld1     {v2.8b}, [x14], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v25.4s, v4.4s, v0.4s
    fmla    v25.4s, v5.4s, v1.4s
    
    ld1     {v2.8b}, [x15], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v26.4s, v4.4s, v0.4s
    fmla    v26.4s, v5.4s, v1.4s
    
    ld1     {v2.8b}, [x9], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v27.4s, v4.4s, v0.4s
    fmla    v27.4s, v5.4s, v1.4s

    // Chunk 1
    ld1     {v0.4s, v1.4s}, [x1], #32
    ld1     {v2.8b}, [x13], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v24.4s, v4.4s, v0.4s
    fmla    v24.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x14], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v25.4s, v4.4s, v0.4s
    fmla    v25.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x15], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v26.4s, v4.4s, v0.4s
    fmla    v26.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x9], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v27.4s, v4.4s, v0.4s
    fmla    v27.4s, v5.4s, v1.4s

    // Chunk 2
    ld1     {v0.4s, v1.4s}, [x1], #32
    ld1     {v2.8b}, [x13], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v24.4s, v4.4s, v0.4s
    fmla    v24.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x14], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v25.4s, v4.4s, v0.4s
    fmla    v25.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x15], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v26.4s, v4.4s, v0.4s
    fmla    v26.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x9], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v27.4s, v4.4s, v0.4s
    fmla    v27.4s, v5.4s, v1.4s

    // Chunk 3
    ld1     {v0.4s, v1.4s}, [x1], #32
    ld1     {v2.8b}, [x13], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v24.4s, v4.4s, v0.4s
    fmla    v24.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x14], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v25.4s, v4.4s, v0.4s
    fmla    v25.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x15], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v26.4s, v4.4s, v0.4s
    fmla    v26.4s, v5.4s, v1.4s
    ld1     {v2.8b}, [x9], #8
    sxtl    v3.8h, v2.8b
    sxtl    v4.4s, v3.4h
    sxtl2   v5.4s, v3.8h
    scvtf   v4.4s, v4.4s
    scvtf   v5.4s, v5.4s
    fmla    v27.4s, v4.4s, v0.4s
    fmla    v27.4s, v5.4s, v1.4s
    
    fmul    v24.4s, v24.4s, v20.4s
    fadd    v16.4s, v16.4s, v24.4s
    fmul    v25.4s, v25.4s, v21.4s
    fadd    v17.4s, v17.4s, v25.4s
    fmul    v26.4s, v26.4s, v22.4s
    fadd    v18.4s, v18.4s, v26.4s
    fmul    v27.4s, v27.4s, v23.4s
    fadd    v19.4s, v19.4s, v27.4s
    
    add     x5, x5, #34
    add     x6, x6, #34
    add     x7, x7, #34
    add     x8, x8, #34
    subs    x3, x3, #1
    b.gt    .Lgemv4_block_loop
    
.Lgemv4_end:
    faddp   v16.4s, v16.4s, v16.4s
    faddp   v16.4s, v16.4s, v16.4s
    faddp   v17.4s, v17.4s, v17.4s
    faddp   v17.4s, v17.4s, v17.4s
    faddp   v18.4s, v18.4s, v18.4s
    faddp   v18.4s, v18.4s, v18.4s
    faddp   v19.4s, v19.4s, v19.4s
    faddp   v19.4s, v19.4s, v19.4s
    st1     {v16.s}[0], [x0], #4
    st1     {v17.s}[0], [x0], #4
    st1     {v18.s}[0], [x0], #4
    st1     {v19.s}[0], [x0], #4
    ldp     x29, x30, [sp], #48
    ret

// ----------------------------------------------------------------------------
// void omni_gemv_q8_s8_4row_asm(float* out, const int8_t* A_q, const void* B,
//                               int n_blocks, int stride)
//
// Optimized Q8_0 x Int8 GEMV using SDOT
// x0: out
// x1: A_q (int8*)
// x2: B (row 0)
// x3: n_blocks
// x4: stride
// ----------------------------------------------------------------------------
_omni_gemv_q8_s8_4row_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    mov     x5, x2
    add     x6, x5, x4
    add     x7, x6, x4
    add     x8, x7, x4
    
    // Global accumulators (float)
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    
    cbz     x3, .Ls8_end
    
.Ls8_block_loop:
    // Load scales
    ldrh    w9, [x5]
    ldrh    w10, [x6]
    ldrh    w11, [x7]
    ldrh    w12, [x8]
    fmov    s20, w9
    fcvt    h20, s20
    fcvt    s20, h20
    dup     v20.4s, v20.s[0]
    fmov    s21, w10
    fcvt    h21, s21
    fcvt    s21, h21
    dup     v21.4s, v21.s[0]
    fmov    s22, w11
    fcvt    h22, s22
    fcvt    s22, h22
    dup     v22.4s, v22.s[0]
    fmov    s23, w12
    fcvt    h23, s23
    fcvt    s23, h23
    dup     v23.4s, v23.s[0]
    
    // Block accumulators (int32)
    movi    v24.4s, #0 // acc0
    movi    v25.4s, #0 // acc1
    movi    v26.4s, #0 // acc2
    movi    v27.4s, #0 // acc3
    
    // Data offsets
    add     x13, x5, #2
    add     x14, x6, #2
    add     x15, x7, #2
    add     x9, x8, #2
    
    // Load A (32 int8s) -> v0, v1 (128-bit regs)
    ld1     {v0.16b, v1.16b}, [x1], #32
    
    // Row 0
    ld1     {v2.16b, v3.16b}, [x13], #32 // Load 32 ints
    // SDOT
    .inst 0x4e829418 // sdot v24.4s, v0.16b, v2.16b (v0=A[0-15], v2=B0[0-15])
    .inst 0x4e839438 // sdot v24.4s, v1.16b, v3.16b (v1=A[16-31], v3=B0[16-31])
    
    // Row 1
    ld1     {v4.16b, v5.16b}, [x14], #32
    .inst 0x4e849419 // sdot v25.4s, v0.16b, v4.16b
    .inst 0x4e859439 // sdot v25.4s, v1.16b, v5.16b
    
    // Row 2
    ld1     {v2.16b, v3.16b}, [x15], #32
    .inst 0x4e82941a // sdot v26.4s, v0.16b, v2.16b
    .inst 0x4e83943a // sdot v26.4s, v1.16b, v3.16b
    
    // Row 3
    ld1     {v4.16b, v5.16b}, [x9], #32
    .inst 0x4e84941b // sdot v27.4s, v0.16b, v4.16b
    .inst 0x4e85943b // sdot v27.4s, v1.16b, v5.16b
    
    // Convert accumulated int32 to float
    scvtf   v24.4s, v24.4s
    scvtf   v25.4s, v25.4s
    scvtf   v26.4s, v26.4s
    scvtf   v27.4s, v27.4s
    
    // Multiply by scale
    fmul    v24.4s, v24.4s, v20.4s
    fmul    v25.4s, v25.4s, v21.4s
    fmul    v26.4s, v26.4s, v22.4s
    fmul    v27.4s, v27.4s, v23.4s
    
    // Add to global float accumulators
    fadd    v16.4s, v16.4s, v24.4s
    fadd    v17.4s, v17.4s, v25.4s
    fadd    v18.4s, v18.4s, v26.4s
    fadd    v19.4s, v19.4s, v27.4s
    
    add     x5, x5, #34
    add     x6, x6, #34
    add     x7, x7, #34
    add     x8, x8, #34
    
    subs    x3, x3, #1
    b.gt    .Ls8_block_loop
    
.Ls8_end:
    // H-sums
    faddp   v16.4s, v16.4s, v16.4s
    faddp   v16.4s, v16.4s, v16.4s
    faddp   v17.4s, v17.4s, v17.4s
    faddp   v17.4s, v17.4s, v17.4s
    faddp   v18.4s, v18.4s, v18.4s
    faddp   v18.4s, v18.4s, v18.4s
    faddp   v19.4s, v19.4s, v19.4s
    faddp   v19.4s, v19.4s, v19.4s
    st1     {v16.s}[0], [x0], #4
    st1     {v17.s}[0], [x0], #4
    st1     {v18.s}[0], [x0], #4
    st1     {v19.s}[0], [x0], #4
    ldp     x29, x30, [sp], #48
    ret
