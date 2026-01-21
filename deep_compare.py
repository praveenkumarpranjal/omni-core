#!/usr/bin/env python3
"""
Deep comparison: Check EVERY intermediate value between PyTorch and Omni
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import omni_bindings

print("=" * 80)
print("DEEP COMPARISON: PyTorch vs Omni Core")
print("=" * 80)

model_id = "LiquidAI/LFM2.5-1.2B-Base"

# Load models
print("\nLoading models...")
pytorch_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
omni_runtime = omni_bindings.OmniRuntime("../lfm2_new.omni")

# Test case: Process prompt, then one token
prompt = "Q: What is C. elegans?\nA:"
tokens = tokenizer.encode(prompt, add_special_tokens=True)
print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens} (length: {len(tokens)})\n")

# ============================================================================
# STEP 1: Embeddings
# ============================================================================
print("=" * 80)
print("STEP 1: Check Embeddings")
print("=" * 80)

# PyTorch embeddings
input_ids = torch.tensor([tokens])
with torch.no_grad():
    pytorch_embeds = pytorch_model.model.embed_tokens(input_ids)[0].float().cpu().numpy()

print(f"PyTorch embeddings shape: {pytorch_embeds.shape}")
print(f"PyTorch embeddings[0, :5]: {pytorch_embeds[0, :5]}")
print(f"PyTorch embeddings[-1, :5]: {pytorch_embeds[-1, :5]}")

# Omni embeddings - we can't directly access, but we can check first layer input
print("\n(Omni embeddings not directly accessible, will check layer outputs)")

# ============================================================================
# STEP 2: First Forward Pass (Prefill)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Prefill - Process all prompt tokens")
print("=" * 80)

# PyTorch prefill
print("\nPyTorch prefill...")
with torch.no_grad():
    pytorch_outputs = pytorch_model(input_ids, use_cache=True, output_hidden_states=True)
    pytorch_logits_prefill = pytorch_outputs.logits[0, -1, :].float().cpu().numpy()
    pytorch_past_kv = pytorch_outputs.past_key_values
    pytorch_hidden_states = [h[0].float().cpu().numpy() for h in pytorch_outputs.hidden_states]

print(f"PyTorch logits shape: {pytorch_logits_prefill.shape}")
print(f"PyTorch logits range: [{pytorch_logits_prefill.min():.2f}, {pytorch_logits_prefill.max():.2f}]")
print(f"PyTorch top token: {pytorch_logits_prefill[:65536].argmax()} = {pytorch_logits_prefill[:65536].max():.2f}")

# Check hidden states at each layer
print(f"\nPyTorch hidden states (last token) at each layer:")
for i, hs in enumerate(pytorch_hidden_states[:5]):  # First 5 layers
    print(f"  Layer {i}: {hs[-1, :5]}")

# Omni prefill
print("\nOmni prefill...")
omni_runtime.reset_kv_cache()
omni_logits_prefill = omni_runtime.forward(tokens)

print(f"Omni logits shape: {omni_logits_prefill.shape}")
print(f"Omni logits range: [{omni_logits_prefill.min():.2f}, {omni_logits_prefill.max():.2f}]")
print(f"Omni top token: {omni_logits_prefill.argmax()} = {omni_logits_prefill.max():.2f}")

# Compare prefill logits
print(f"\nPrefill logits comparison:")
omni_logits_trimmed = omni_logits_prefill[:65536]
correlation = np.corrcoef(pytorch_logits_prefill, omni_logits_trimmed)[0, 1]
max_diff = np.abs(pytorch_logits_prefill - omni_logits_trimmed).max()
print(f"  Correlation: {correlation:.6f}")
print(f"  Max diff: {max_diff:.2f}")
print(f"  Top tokens match: {pytorch_logits_prefill[:65536].argmax() == omni_logits_prefill.argmax()}")

# ============================================================================
# STEP 3: Second Forward Pass (Decode with KV cache)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Decode - Process one new token with KV cache")
print("=" * 80)

next_token = 835  # ' A'
print(f"Next token: {next_token} ('{tokenizer.decode([next_token])}')\n")

# PyTorch decode
print("PyTorch decode with KV cache...")
input_ids_next = torch.tensor([[next_token]])
with torch.no_grad():
    pytorch_outputs_decode = pytorch_model(
        input_ids_next, 
        past_key_values=pytorch_past_kv, 
        use_cache=True,
        output_hidden_states=True
    )
    pytorch_logits_decode = pytorch_outputs_decode.logits[0, -1, :].float().cpu().numpy()
    pytorch_hidden_decode = [h[0].float().cpu().numpy() for h in pytorch_outputs_decode.hidden_states]

print(f"PyTorch logits shape: {pytorch_logits_decode.shape}")
print(f"PyTorch logits range: [{pytorch_logits_decode.min():.2f}, {pytorch_logits_decode.max():.2f}]")
print(f"PyTorch top token: {pytorch_logits_decode[:65536].argmax()} = {pytorch_logits_decode[:65536].max():.2f}")

print(f"\nPyTorch hidden states (decode) at each layer:")
for i, hs in enumerate(pytorch_hidden_decode[:5]):
    print(f"  Layer {i}: {hs[-1, :5]}")

# Omni decode
print("\nOmni decode with KV cache...")
omni_logits_decode = omni_runtime.forward([next_token])

print(f"Omni logits shape: {omni_logits_decode.shape}")
print(f"Omni logits range: [{omni_logits_decode.min():.2f}, {omni_logits_decode.max():.2f}]")
print(f"Omni top token: {omni_logits_decode.argmax()} = {omni_logits_decode.max():.2f}")

# Compare decode logits
print(f"\nDecode logits comparison:")
omni_logits_decode_trimmed = omni_logits_decode[:65536]
correlation_decode = np.corrcoef(pytorch_logits_decode, omni_logits_decode_trimmed)[0, 1]
max_diff_decode = np.abs(pytorch_logits_decode - omni_logits_decode_trimmed).max()
print(f"  Correlation: {correlation_decode:.6f}")
print(f"  Max diff: {max_diff_decode:.2f}")
print(f"  Top tokens match: {pytorch_logits_decode[:65536].argmax() == omni_logits_decode.argmax()}")

# ============================================================================
# STEP 4: Compare with processing all tokens together
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Sanity check - Process all 13 tokens together")
print("=" * 80)

all_tokens = tokens + [next_token]

# PyTorch all-at-once
print("\nPyTorch all-at-once...")
input_ids_all = torch.tensor([all_tokens])
with torch.no_grad():
    pytorch_outputs_all = pytorch_model(input_ids_all)
    pytorch_logits_all = pytorch_outputs_all.logits[0, -1, :].float().cpu().numpy()

print(f"PyTorch top token: {pytorch_logits_all[:65536].argmax()} = {pytorch_logits_all[:65536].max():.2f}")

# Omni all-at-once
print("\nOmni all-at-once...")
omni_runtime.reset_kv_cache()
omni_logits_all = omni_runtime.forward(all_tokens)

print(f"Omni top token: {omni_logits_all.argmax()} = {omni_logits_all.max():.2f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nPyTorch:")
print(f"  All-at-once:  {pytorch_logits_all[:65536].argmax()} ('{tokenizer.decode([pytorch_logits_all[:65536].argmax()])}')")
print(f"  With KV cache: {pytorch_logits_decode[:65536].argmax()} ('{tokenizer.decode([pytorch_logits_decode[:65536].argmax()])}')")
print(f"  Match: {pytorch_logits_all[:65536].argmax() == pytorch_logits_decode[:65536].argmax()} ✅")

print("\nOmni:")
print(f"  All-at-once:  {omni_logits_all.argmax()} ('{tokenizer.decode([omni_logits_all.argmax()])}')")
print(f"  With KV cache: {omni_logits_decode.argmax()} ('{tokenizer.decode([omni_logits_decode.argmax()])}')")
print(f"  Match: {omni_logits_all.argmax() == omni_logits_decode.argmax()} {'✅' if omni_logits_all.argmax() == omni_logits_decode.argmax() else '❌'}")

print("\nCross-comparison:")
print(f"  Omni all-at-once vs PyTorch all-at-once: {omni_logits_all.argmax() == pytorch_logits_all[:65536].argmax()} ✅")
print(f"  Omni KV cache vs PyTorch KV cache: {omni_logits_decode.argmax() == pytorch_logits_decode[:65536].argmax()} {'✅' if omni_logits_decode.argmax() == pytorch_logits_decode[:65536].argmax() else '❌'}")

if omni_logits_decode.argmax() != pytorch_logits_decode[:65536].argmax():
    print("\n❌ KV CACHE BUG CONFIRMED")
    print("The issue is in how we use KV cache during decode.")
    print("Need to trace through attention computation step-by-step.")

omni_runtime.close()
