#!/usr/bin/env python3
"""
Python FFI bindings for Omni Core C++ library
"""

import ctypes
import numpy as np
from pathlib import Path

# Find library (relative to this file in omni-core/)
lib_path = Path(__file__).parent / "core/build/libomni.dylib"
if not lib_path.exists():
    raise FileNotFoundError(f"Could not find libomni.dylib at {lib_path}")

# Load library
lib = ctypes.CDLL(str(lib_path))

# Define C structures (opaque pointers)
class OmniModel(ctypes.Structure):
    pass

class OmniContext(ctypes.Structure):
    pass

# Function signatures
lib.omni_load_model.argtypes = [ctypes.c_char_p]
lib.omni_load_model.restype = ctypes.POINTER(OmniModel)

lib.omni_free_model.argtypes = [ctypes.POINTER(OmniModel)]
lib.omni_free_model.restype = None

lib.omni_create_context.argtypes = [ctypes.POINTER(OmniModel), ctypes.c_int]
lib.omni_create_context.restype = ctypes.POINTER(OmniContext)

lib.omni_free_context.argtypes = [ctypes.POINTER(OmniContext)]
lib.omni_free_context.restype = None

lib.omni_reset_kv_cache.argtypes = [ctypes.POINTER(OmniContext)]
lib.omni_reset_kv_cache.restype = None

lib.omni_forward.argtypes = [
    ctypes.POINTER(OmniContext),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)
]
lib.omni_forward.restype = None

lib.omni_sample.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float
]
lib.omni_sample.restype = ctypes.c_int

lib.omni_generate.argtypes = [
    ctypes.POINTER(OmniContext),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_float
]
lib.omni_generate.restype = ctypes.c_int

lib.omni_print_info.argtypes = [ctypes.POINTER(OmniModel)]
lib.omni_print_info.restype = None


class OmniRuntime:
    """Python wrapper for Omni Core inference engine"""
    
    def __init__(self, model_path: str, max_seq_len: int = 2048):
        """Load model and create inference context"""
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        
        # Load model
        self.model = lib.omni_load_model(model_path.encode('utf-8'))
        if not self.model:
            raise RuntimeError(f"Failed to load model: {model_path}")
        
        # Create context
        self.ctx = lib.omni_create_context(self.model, max_seq_len)
        if not self.ctx:
            lib.omni_free_model(self.model)
            raise RuntimeError("Failed to create context")
        
        print(f"✓ Loaded model: {model_path}")
    
    def print_info(self):
        """Print model information"""
        lib.omni_print_info(self.model)
    
    def reset_kv_cache(self):
        """Reset KV cache for new sequence"""
        lib.omni_reset_kv_cache(self.ctx)
    
    def forward(self, tokens: list[int]) -> np.ndarray:
        """
        Run forward pass on tokens
        Returns logits for next token prediction
        """
        n_tokens = len(tokens)
        
        # Convert to C arrays
        tokens_arr = (ctypes.c_int * n_tokens)(*tokens)
        
        # Get vocab size from model (assume 128256 for LFM2)
        vocab_size = 128256
        logits_arr = (ctypes.c_float * vocab_size)()
        
        # Run forward pass
        lib.omni_forward(self.ctx, tokens_arr, n_tokens, logits_arr)
        
        # Convert to numpy
        logits = np.array(logits_arr, dtype=np.float32)
        return logits
    
    def sample(self, logits: np.ndarray, temperature: float = 1.0, top_p: float = 0.9) -> int:
        """Sample next token from logits"""
        vocab_size = len(logits)
        logits_arr = logits.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        token = lib.omni_sample(logits_arr, vocab_size, temperature, top_p)
        return token
    
    def generate(self, prompt: list[int], max_tokens: int = 100, temperature: float = 1.0) -> list[int]:
        """
        Generate tokens autoregressively
        Returns list of all tokens (prompt + generated)
        """
        prompt_len = len(prompt)
        max_total = prompt_len + max_tokens
        
        # Allocate output buffer
        output_arr = (ctypes.c_int * max_total)()
        
        # Run generation
        total_len = lib.omni_generate(
            self.ctx,
            (ctypes.c_int * prompt_len)(*prompt),
            prompt_len,
            output_arr,
            max_tokens,
            temperature
        )
        
        # Convert to list
        tokens = [output_arr[i] for i in range(total_len)]
        return tokens
    
    def close(self):
        """Free resources"""
        if hasattr(self, 'ctx') and self.ctx:
            lib.omni_free_context(self.ctx)
            self.ctx = None
        if hasattr(self, 'model') and self.model:
            lib.omni_free_model(self.model)
            self.model = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python omni_bindings.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Load model
    runtime = OmniRuntime(model_path)
    runtime.print_info()
    
    # Test forward pass
    tokens = [1, 1098, 5706, 803, 4481, 856]  # "The capital of France is"
    print(f"\nTest tokens: {tokens}")
    
    logits = runtime.forward(tokens)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Get top predictions
    top_k = 10
    top_indices = np.argsort(logits)[-top_k:][::-1]
    print(f"\nTop {top_k} predictions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Token {idx}: {logits[idx]:.2f}")
    
    runtime.close()
    print("\n✓ Test complete")
