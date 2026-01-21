// ARM64 NEON Assembly: Q8_0 Dequantization
// Dequantizes Q8_0 blocks (32 int8 values + 1 f16 scale) to float32
//
// Function signature:
// void omni_dequant_q8_0_asm(float* out, const void* data, int n_blocks);
//
// Q8_0 block structure (34 bytes):
//   - scale: f16 (2 bytes)
//   - qs[32]: int8 (32 bytes)
//
// Registers:
//   x0: out pointer (float32*)
//   x1: data pointer (q8_0 blocks*)
//   x2: n_blocks (int)

.global _omni_dequant_q8_0_asm
.align 4

_omni_dequant_q8_0_asm:
    // Save frame pointer
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    
    // Check if n_blocks == 0
    cbz     x2, .Lend
    
.Lblock_loop:
    // Load scale (f16 at offset 0)
    ldrh    w3, [x1]                    // Load f16 scale into w3
    
    // Convert f16 to f32
    // Move to NEON register and convert
    fmov    s0, w3                      // Move to NEON
    fcvt    h0, s0                      // Convert to half precision
    fcvt    s0, h0                      // Convert back to single precision
    dup     v0.4s, v0.s[0]              // Broadcast scale to all lanes
    
    // Load 32 int8 values (starting at offset 2)
    add     x3, x1, #2                  // Point to qs array
    
    // Process 32 elements in 4 chunks of 8
    // Chunk 1: elements 0-7
    ld1     {v1.8b}, [x3], #8           // Load 8 int8 values
    sxtl    v2.8h, v1.8b                // Sign-extend to int16
    sxtl    v3.4s, v2.4h                // Sign-extend lower 4 to int32
    sxtl2   v4.4s, v2.8h                // Sign-extend upper 4 to int32
    scvtf   v3.4s, v3.4s                // Convert to float32
    scvtf   v4.4s, v4.4s                // Convert to float32
    fmul    v3.4s, v3.4s, v0.4s         // Multiply by scale
    fmul    v4.4s, v4.4s, v0.4s         // Multiply by scale
    st1     {v3.4s, v4.4s}, [x0], #32   // Store 8 float32 values
    
    // Chunk 2: elements 8-15
    ld1     {v1.8b}, [x3], #8
    sxtl    v2.8h, v1.8b
    sxtl    v3.4s, v2.4h
    sxtl2   v4.4s, v2.8h
    scvtf   v3.4s, v3.4s
    scvtf   v4.4s, v4.4s
    fmul    v3.4s, v3.4s, v0.4s
    fmul    v4.4s, v4.4s, v0.4s
    st1     {v3.4s, v4.4s}, [x0], #32
    
    // Chunk 3: elements 16-23
    ld1     {v1.8b}, [x3], #8
    sxtl    v2.8h, v1.8b
    sxtl    v3.4s, v2.4h
    sxtl2   v4.4s, v2.8h
    scvtf   v3.4s, v3.4s
    scvtf   v4.4s, v4.4s
    fmul    v3.4s, v3.4s, v0.4s
    fmul    v4.4s, v4.4s, v0.4s
    st1     {v3.4s, v4.4s}, [x0], #32
    
    // Chunk 4: elements 24-31
    ld1     {v1.8b}, [x3], #8
    sxtl    v2.8h, v1.8b
    sxtl    v3.4s, v2.4h
    sxtl2   v4.4s, v2.8h
    scvtf   v3.4s, v3.4s
    scvtf   v4.4s, v4.4s
    fmul    v3.4s, v3.4s, v0.4s
    fmul    v4.4s, v4.4s, v0.4s
    st1     {v3.4s, v4.4s}, [x0], #32
    
    // Move to next block (34 bytes)
    add     x1, x1, #34
    
    // Decrement block counter and loop
    subs    x2, x2, #1
    b.ne    .Lblock_loop
    
.Lend:
    // Restore frame pointer and return
    ldp     x29, x30, [sp], #16
    ret
