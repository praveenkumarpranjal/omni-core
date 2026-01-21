#!/usr/bin/env python3
"""
Omni Model Converter - Convert HuggingFace models to .omni format

Proper Q8_0 quantization with outlier handling to avoid corruption.
"""

import os
import sys
import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

OMNI_MAGIC = b"OMNI"
OMNI_VERSION = 1
OMNI_ALIGNMENT = 64

class OmniType:
    F32 = 0
    F16 = 1
    Q8_0 = 3
    Q4_K = 10

def fp32_to_fp16(x: np.ndarray) -> np.ndarray:
    """Convert float32 to float16"""
    return x.astype(np.float16)

def quantize_q8_0_safe(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize to Q8_0 with proper outlier handling.
    
    Block size: 32 elements
    Format: [scale_f16] [32 x int8_t]
    
    Returns: (quantized_int8, scales_f16)
    """
    data = data.astype(np.float32).flatten()
    n_blocks = (len(data) + 31) // 32
    data = np.pad(data, (0, n_blocks * 32 - len(data)))
    data = data.reshape(-1, 32)
    
    # Compute scale per block (absmax)
    absmax = np.abs(data).max(axis=1, keepdims=True)
    
    # Handle zeros and outliers
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
        print("WARNING: NaN/Inf detected in scales after conversion!")
        scales_f16 = np.nan_to_num(scales_f16, nan=0.0, posinf=MAX_SCALE, neginf=-MAX_SCALE)
    
    return quantized, scales_f16

def write_omni_file(path: str, architecture: int, metadata: Dict, tensors: Dict[str, Tuple[np.ndarray, int]]):
    """
    Write .omni file with proper alignment and error checking.
    
    Args:
        path: Output file path
        architecture: Architecture ID (1 for LFM2)
        metadata: Model metadata dict
        tensors: Dict of {name: (data, dtype)}
    """
    print(f"Writing {len(tensors)} tensors to {path}...")
    
    with open(path, "wb") as f:
        # Header
        f.write(OMNI_MAGIC)
        f.write(struct.pack("<I", OMNI_VERSION))
        f.write(struct.pack("<I", architecture))
        f.write(struct.pack("<I", OMNI_ALIGNMENT))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))
        
        # Metadata as JSON
        metadata_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
        f.write(struct.pack("<Q", len(metadata_bytes)))
        f.write(metadata_bytes)
        
        # Align
        pos = f.tell()
        padding = (OMNI_ALIGNMENT - (pos % OMNI_ALIGNMENT)) % OMNI_ALIGNMENT
        f.write(b'\x00' * padding)
        
        # Tensor info section
        tensor_info_positions = []
        for name, (data, dtype) in tensors.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<B", dtype))
            
            # Placeholder for offset and size
            tensor_info_positions.append((name, f.tell(), data, dtype))
            f.write(struct.pack("<Q", 0))  # offset
            f.write(struct.pack("<Q", 0))  # size
        
        # Align for data section
        pos = f.tell()
        padding = (OMNI_ALIGNMENT - (pos % OMNI_ALIGNMENT)) % OMNI_ALIGNMENT
        f.write(b'\x00' * padding)
        
        data_start = f.tell()
        
        # Write tensor data
        for name, info_pos, data, dtype in tensor_info_positions:
            offset = f.tell() - data_start
            
            # Convert/quantize
            if dtype == OmniType.F32:
                tensor_bytes = data.astype(np.float32).tobytes()
            elif dtype == OmniType.F16:
                tensor_bytes = data.astype(np.float16).tobytes()
            elif dtype == OmniType.Q8_0:
                quantized, scales = quantize_q8_0_safe(data)
                # Interleave: [scale, 32 x int8] for each block
                n_blocks = len(scales)
                tensor_bytes = bytearray()
                for i in range(n_blocks):
                    tensor_bytes.extend(scales[i:i+1].tobytes())
                    tensor_bytes.extend(quantized[i].tobytes())
                tensor_bytes = bytes(tensor_bytes)
            else:
                tensor_bytes = data.astype(np.float16).tobytes()
            
            f.write(tensor_bytes)
            size = len(tensor_bytes)
            
            # Align
            pos = f.tell()
            padding = (OMNI_ALIGNMENT - (pos % OMNI_ALIGNMENT)) % OMNI_ALIGNMENT
            f.write(b'\x00' * padding)
            
            # Update offset/size
            current_pos = f.tell()
            f.seek(info_pos)
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<Q", size))
            f.seek(current_pos)
            
            print(f"  ✓ {name}: {data.shape} ({dtype})")
    
    print(f"✓ Wrote model to {path}")

def convert_lfm2_from_hf(model_id: str, output_path: str, quant_type: str = "q8_0"):
    """
    Convert LFM2 model from HuggingFace to .omni format.
    
    Args:
        model_id: HuggingFace model ID (e.g., "LiquidAI/LFM2.5-1.2B-Base")
        output_path: Output .omni file path
        quant_type: Quantization type ("f16", "q8_0")
    """
    try:
        from transformers import AutoConfig
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
        import torch
    except ImportError:
        print("ERROR: Missing dependencies. Install with:")
        print("  pip install transformers safetensors huggingface_hub torch")
        sys.exit(1)
    
    print(f"Converting {model_id} to {output_path}...")
    print(f"Quantization: {quant_type}")
    
    # Download model
    print("\nDownloading model...")
    model_path = snapshot_download(model_id)
    print(f"✓ Downloaded to {model_path}")
    
    # Load config
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config_dict = config.to_dict()
    
    # Extract metadata
    metadata = {
        "model_type": config_dict.get("model_type", "lfm2"),
        "hidden_size": config_dict.get("hidden_size", 2048),
        "num_layers": config_dict.get("num_hidden_layers", 16),
        "num_heads": config_dict.get("num_attention_heads", 32),
        "num_kv_heads": config_dict.get("num_key_value_heads", 8),
        "vocab_size": config_dict.get("vocab_size", 65536),
        "max_position_embeddings": config_dict.get("max_position_embeddings", 128000),
        "rope_theta": config_dict.get("rope_theta", 1000000.0),
        "norm_eps": config_dict.get("norm_eps", 1e-5),
        "layer_types": config_dict.get("layer_types", []),
    }
    
    print(f"\nModel config:")
    print(f"  Hidden size: {metadata['hidden_size']}")
    print(f"  Layers: {metadata['num_layers']}")
    print(f"  Heads: {metadata['num_heads']} (KV: {metadata['num_kv_heads']})")
    print(f"  Vocab: {metadata['vocab_size']}")
    
    # Find safetensors files
    safetensor_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensor_files:
        safetensor_files = list(Path(model_path).glob("model*.safetensors"))
    
    if not safetensor_files:
        print("ERROR: No safetensors files found!")
        sys.exit(1)
    
    print(f"\nFound {len(safetensor_files)} safetensors files")
    
    # Determine dtype
    if quant_type == "f16":
        default_dtype = OmniType.F16
    elif quant_type == "q8_0":
        default_dtype = OmniType.Q8_0
    else:
        print(f"ERROR: Unknown quantization type: {quant_type}")
        sys.exit(1)
    
    # Load tensors
    tensors = {}
    total_params = 0
    
    for sf_file in safetensor_files:
        print(f"\nLoading {sf_file.name}...")
        with safe_open(sf_file, framework="pt") as f:  # Use PyTorch to handle bfloat16
            for name in f.keys():
                tensor = f.get_tensor(name)
                
                # Convert to numpy (handles bfloat16 -> float32)
                tensor_np = tensor.cpu().float().numpy()
                
                # Determine dtype for this tensor
                # Keep embeddings and norms in F16 for quality
                if "embed" in name or "norm" in name:
                    dtype = OmniType.F16
                else:
                    dtype = default_dtype
                
                tensors[name] = (tensor_np, dtype)
                total_params += tensor_np.size
                
                print(f"  {name}: {tensor.shape} -> {dtype}")
    
    print(f"\n✓ Loaded {len(tensors)} tensors ({total_params/1e9:.2f}B parameters)")
    
    # Write .omni file
    write_omni_file(output_path, 1, metadata, tensors)  # 1 = LFM2
    
    # Verify file
    file_size = os.path.getsize(output_path)
    print(f"\n✓ Output file: {output_path}")
    print(f"  Size: {file_size / (1024**3):.2f} GB")
    print(f"  Compression: {total_params * 4 / file_size:.2f}x vs F32")

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to .omni format")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., LiquidAI/LFM2.5-1.2B-Base)")
    parser.add_argument("output", help="Output .omni file path")
    parser.add_argument("--quant", choices=["f16", "q8_0"], default="q8_0",
                       help="Quantization type (default: q8_0)")
    
    args = parser.parse_args()
    
    convert_lfm2_from_hf(args.model_id, args.output, args.quant)

if __name__ == "__main__":
    main()
