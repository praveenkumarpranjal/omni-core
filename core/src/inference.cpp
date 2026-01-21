/*
 * Omni Core - Inference Engine
 *
 * LFM2 forward pass with hybrid conv + attention
 * Architecture: 16 blocks (10 conv + 6 GQA attention)
 */

#include "../include/omni.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// Helper: Get weight with caching (dequantize once, reuse forever)
static float *get_cached_weight(omni_context *ctx, const char *name) {
  // Check if already cached
  auto it = ctx->weight_cache.find(name);
  if (it != ctx->weight_cache.end()) {
    return it->second.data();
  }

  // Not cached - need to dequantize
  const omni_tensor *tensor = omni_get_tensor(ctx->model, name);
  if (!tensor)
    return nullptr;

  // Calculate size
  size_t n_elements = 1;
  for (auto dim : tensor->shape) {
    n_elements *= dim;
  }

  // Allocate cache
  std::vector<float> &cache = ctx->weight_cache[name];
  cache.resize(n_elements);

  // Dequantize into cache
  omni_get_tensor_f32(ctx->model, name, cache.data());

  return cache.data();
}

// Create inference context
omni_context *omni_create_context(omni_model *model, int max_seq_len) {
  omni_context *ctx = new omni_context();
  ctx->model = model;

  int hidden = model->hidden_size;
  int n_layers = model->num_layers;
  int n_kv_heads = model->num_kv_heads;
  int head_dim = hidden / model->num_heads;

  // Allocate working buffers
  ctx->hidden.resize(max_seq_len * hidden);
  ctx->residual.resize(max_seq_len * hidden);
  ctx->q_buf.resize(max_seq_len * hidden);
  ctx->k_buf.resize(max_seq_len * hidden);
  ctx->v_buf.resize(max_seq_len * hidden);
  ctx->attn_out.resize(max_seq_len * hidden);
  ctx->ffn_buf.resize(max_seq_len * hidden * 4);
  ctx->logits.resize(model->vocab_size);

  // Weight buffer for dequantization
  size_t max_weight = (size_t)model->vocab_size * hidden;
  ctx->weight_buf.resize(max_weight);

  // Allocate KV cache
  // Each layer stores: [max_seq_len, n_kv_heads, head_dim]
  size_t kv_cache_size = max_seq_len * n_kv_heads * head_dim;
  ctx->cache_k.resize(n_layers);
  ctx->cache_v.resize(n_layers);
  for (int i = 0; i < n_layers; i++) {
    ctx->cache_k[i].resize(kv_cache_size, 0.0f);
    ctx->cache_v[i].resize(kv_cache_size, 0.0f);
  }
  ctx->cache_pos = 0;

  // Allocate Conv cache: [hidden, kernel_size] per layer
  // kernel_size = 3 for LFM2, so cache full kernel_size values of Bx
  ctx->cache_conv.resize(n_layers);
  for (int i = 0; i < n_layers; i++) {
    ctx->cache_conv[i].resize(
        hidden * 3, 0.0f); // Cache kernel_size=3 Bx values per channel
  }

  return ctx;
}

// Free context
void omni_free_context(omni_context *ctx) { delete ctx; }

// Reset KV cache for new sequence
void omni_reset_kv_cache(omni_context *ctx) {
  ctx->cache_pos = 0;
  // No need to zero memory, will be overwritten
}

// LFM2 Conv Block with CORRECT caching (matches PyTorch
// Lfm2ShortConv.slow_forward) Key insight: Cache Bx (after gating B*x), NOT BCx
// (before gating) Decode uses direct dot product, NOT full convolution
static void lfm2_conv_block(omni_context *ctx, float *output,
                            const float *input, int n_tokens, int hidden,
                            int layer_idx) {
  // Get weights
  char weight_names[3][256];
  snprintf(weight_names[0], 256, "model.layers.%d.conv.in_proj.weight",
           layer_idx);
  snprintf(weight_names[1], 256, "model.layers.%d.conv.conv.weight", layer_idx);
  snprintf(weight_names[2], 256, "model.layers.%d.conv.out_proj.weight",
           layer_idx);

  float *in_proj = get_cached_weight(ctx, weight_names[0]);
  float *conv_w =
      get_cached_weight(ctx, weight_names[1]); // [hidden, 1, kernel_size]
  float *out_proj = get_cached_weight(ctx, weight_names[2]);

  if (!in_proj || !conv_w || !out_proj) {
    memcpy(output, input, n_tokens * hidden * sizeof(float));
    return;
  }

  // Get conv kernel info
  const omni_tensor *conv_tensor = omni_get_tensor(ctx->model, weight_names[1]);
  int kernel_size = conv_tensor->shape[2]; // L_cache = 3

  // Ensure conv cache is correctly sized: [hidden, kernel_size]
  if (ctx->cache_conv[layer_idx].size() != (size_t)(hidden * kernel_size)) {
    ctx->cache_conv[layer_idx].resize(hidden * kernel_size, 0.0f);
  }
  float *conv_cache = ctx->cache_conv[layer_idx].data();

  int proj_size = hidden * 3;
  bool use_cache = (n_tokens == 1 && ctx->cache_pos > 0); // Decode mode

  // Allocate buffers
  std::vector<float> bcx(n_tokens * proj_size);
  std::vector<float> bx_t(hidden *
                          n_tokens); // [hidden, n_tokens] - Bx after gating
  std::vector<float> c_t(hidden *
                         n_tokens); // [hidden, n_tokens] - C for output gating
  std::vector<float> conv_out_t(hidden * n_tokens);

  // 1. in_proj: [n_tokens, hidden] @ [3*hidden, hidden].T -> [n_tokens,
  // 3*hidden]
  omni_matmul(bcx.data(), input, in_proj, n_tokens, proj_size, hidden);

  // 2. Transpose and split: BCx -> B, C, X, then compute Bx = B * x
  // Layout after transpose: [3*hidden, n_tokens]
  for (int h = 0; h < hidden; h++) {
    for (int t = 0; t < n_tokens; t++) {
      // BCx is [n_tokens, 3*hidden], access as bcx[t * proj_size + offset]
      float b = bcx[t * proj_size + h];              // B[t, h]
      float c = bcx[t * proj_size + hidden + h];     // C[t, h]
      float x = bcx[t * proj_size + 2 * hidden + h]; // X[t, h]

      bx_t[h * n_tokens + t] = b * x; // Bx[h, t] after gating
      c_t[h * n_tokens + t] = c;      // C[h, t] for output gating
    }
  }

  if (use_cache) {
    // ===== DECODE MODE =====
    // PyTorch: roll left, insert new, then direct dot product
    // conv_cache layout: [hidden, kernel_size]

    for (int h = 0; h < hidden; h++) {
      // 1. Roll cache left by 1 position
      for (int k = 0; k < kernel_size - 1; k++) {
        conv_cache[h * kernel_size + k] = conv_cache[h * kernel_size + k + 1];
      }
      // 2. Insert new Bx at the rightmost position
      conv_cache[h * kernel_size + (kernel_size - 1)] =
          bx_t[h]; // n_tokens=1, so bx_t[h*1 + 0]

      // 3. Direct dot product: conv_out = sum(cache * weights)
      float sum = 0.0f;
      for (int k = 0; k < kernel_size; k++) {
        sum += conv_cache[h * kernel_size + k] * conv_w[h * kernel_size + k];
      }
      conv_out_t[h] = sum; // [hidden, 1]
    }
  } else {
    // ===== PREFILL MODE =====
    // Standard causal conv1d with left padding

    int padding = kernel_size - 1; // Causal: pad on left only
    int padded_len = n_tokens + padding;
    std::vector<float> padded_t(hidden * padded_len, 0.0f);

    // Pad Bx on the left with zeros
    for (int h = 0; h < hidden; h++) {
      for (int t = 0; t < n_tokens; t++) {
        padded_t[h * padded_len + padding + t] = bx_t[h * n_tokens + t];
      }
    }

    // Convolve: output length = padded_len - kernel_size + 1 = n_tokens
    for (int h = 0; h < hidden; h++) {
      for (int t = 0; t < n_tokens; t++) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
          sum += padded_t[h * padded_len + t + k] * conv_w[h * kernel_size + k];
        }
        conv_out_t[h * n_tokens + t] = sum;
      }
    }

    // Cache last kernel_size values of Bx (with left-padding if n_tokens <
    // kernel_size) This matches PyTorch: conv_state = F.pad(Bx, (L_cache -
    // Bx.shape[-1], 0))
    for (int h = 0; h < hidden; h++) {
      if (n_tokens >= kernel_size) {
        // Copy last kernel_size values
        for (int k = 0; k < kernel_size; k++) {
          conv_cache[h * kernel_size + k] =
              bx_t[h * n_tokens + (n_tokens - kernel_size + k)];
        }
      } else {
        // Left-pad with zeros, then copy all values
        int pad_len = kernel_size - n_tokens;
        for (int k = 0; k < pad_len; k++) {
          conv_cache[h * kernel_size + k] = 0.0f;
        }
        for (int k = 0; k < n_tokens; k++) {
          conv_cache[h * kernel_size + pad_len + k] = bx_t[h * n_tokens + k];
        }
      }
    }
  }

  // 4. Output gating: y = C * conv_out
  std::vector<float> y_t(hidden * n_tokens);
  for (int h = 0; h < hidden; h++) {
    for (int t = 0; t < n_tokens; t++) {
      y_t[h * n_tokens + t] =
          c_t[h * n_tokens + t] * conv_out_t[h * n_tokens + t];
    }
  }

  // 5. Transpose back: [hidden, n_tokens] -> [n_tokens, hidden]
  std::vector<float> y(n_tokens * hidden);
  for (int h = 0; h < hidden; h++) {
    for (int t = 0; t < n_tokens; t++) {
      y[t * hidden + h] = y_t[h * n_tokens + t];
    }
  }

  // 6. Output projection
  omni_matmul(output, y.data(), out_proj, n_tokens, hidden, hidden);
}

// Check if layer is conv or attention (LFM2 specific)
static bool is_conv_layer(int layer) {
  // LFM2 layer types from config:
  // 0,1: conv
  // 2: attention
  // 3,4: conv
  // 5: attention
  // 6,7: conv
  // 8: attention
  // 9: conv
  // 10: attention
  // 11: conv
  // 12: attention
  // 13: conv
  // 14: attention
  // 15: conv

  // Attention layers: 2, 5, 8, 10, 12, 14
  if (layer == 2 || layer == 5 || layer == 8 || layer == 10 || layer == 12 ||
      layer == 14) {
    return false; // Attention
  }
  return true; // Conv
}

// Forward pass
void omni_forward(omni_context *ctx, const int *tokens, int n_tokens,
                  float *logits) {
  omni_model *model = ctx->model;
  int hidden = model->hidden_size;
  int n_layers = model->num_layers;
  int n_heads = model->num_heads;
  int n_kv_heads = model->num_kv_heads;
  int head_dim = hidden / n_heads;

  // 1. Token embeddings
  float *emb_weights = get_cached_weight(ctx, "model.embed_tokens.weight");
  if (!emb_weights) {
    std::cerr << "Failed to get embeddings\n";
    return;
  }

  for (int i = 0; i < n_tokens; i++) {
    int tok = tokens[i];
    if (tok >= 0 && tok < model->vocab_size) {
      memcpy(&ctx->hidden[i * hidden], &emb_weights[tok * hidden],
             hidden * sizeof(float));
    }
  }

  // 2. Layer loop
  for (int il = 0; il < n_layers; il++) {
    bool is_conv_block = is_conv_layer(il);

    // Save residual BEFORE operator norm
    memcpy(ctx->residual.data(), ctx->hidden.data(),
           n_tokens * hidden * sizeof(float));

    // Input norm
    char norm_name[256];
    snprintf(norm_name, sizeof(norm_name),
             "model.layers.%d.operator_norm.weight", il);
    float *norm_w = get_cached_weight(ctx, norm_name);

    if (norm_w) {
      for (int t = 0; t < n_tokens; t++) {
        omni_rms_norm(&ctx->hidden[t * hidden], &ctx->hidden[t * hidden],
                      norm_w, hidden, model->norm_eps);
      }
    }

    bool is_conv = is_conv_layer(il);

    if (is_conv_block) {
      // Conv layer - use attn_out as temp buffer
      lfm2_conv_block(ctx, ctx->attn_out.data(), ctx->hidden.data(), n_tokens,
                      hidden, il);
      memcpy(ctx->hidden.data(), ctx->attn_out.data(),
             n_tokens * hidden * sizeof(float));
    } else {
      // Attention layer (GQA)
      char qkv_names[3][256];
      snprintf(qkv_names[0], 256, "model.layers.%d.self_attn.q_proj.weight",
               il);
      snprintf(qkv_names[1], 256, "model.layers.%d.self_attn.k_proj.weight",
               il);
      snprintf(qkv_names[2], 256, "model.layers.%d.self_attn.v_proj.weight",
               il);

      float *wq = get_cached_weight(ctx, qkv_names[0]);
      float *wk = get_cached_weight(ctx, qkv_names[1]);
      float *wv = get_cached_weight(ctx, qkv_names[2]);

      if (wq && wk && wv) {
        // GQA: K/V have fewer heads than Q
        int kv_hidden = n_kv_heads * head_dim;

        // QKV projections
        omni_matmul(ctx->q_buf.data(), ctx->hidden.data(), wq, n_tokens, hidden,
                    hidden);
        omni_matmul(ctx->k_buf.data(), ctx->hidden.data(), wk, n_tokens,
                    kv_hidden, hidden);
        omni_matmul(ctx->v_buf.data(), ctx->hidden.data(), wv, n_tokens,
                    kv_hidden, hidden);

        // Apply QK LayerNorm (LFM2 specific)
        char qk_norm_names[2][256];
        snprintf(qk_norm_names[0], 256,
                 "model.layers.%d.self_attn.q_layernorm.weight", il);
        snprintf(qk_norm_names[1], 256,
                 "model.layers.%d.self_attn.k_layernorm.weight", il);

        float *q_norm = get_cached_weight(ctx, qk_norm_names[0]);
        float *k_norm = get_cached_weight(ctx, qk_norm_names[1]);

        if (q_norm && k_norm) {
          // Apply RMS norm to each head
          for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_heads; h++) {
              float *q_head = ctx->q_buf.data() + t * hidden + h * head_dim;
              omni_rms_norm(q_head, q_head, q_norm, head_dim, model->norm_eps);
            }
            for (int h = 0; h < n_kv_heads; h++) {
              float *k_head = ctx->k_buf.data() + t * kv_hidden + h * head_dim;
              omni_rms_norm(k_head, k_head, k_norm, head_dim, model->norm_eps);
            }
          }
        }

        // RoPE - apply with absolute positions
        std::vector<int> positions(n_tokens);
        for (int i = 0; i < n_tokens; i++)
          positions[i] = ctx->cache_pos + i;

        omni_rope(ctx->q_buf.data(), ctx->q_buf.data(), n_tokens, n_heads,
                  head_dim, positions.data(), model->rope_theta);
        omni_rope(ctx->k_buf.data(), ctx->k_buf.data(), n_tokens, n_kv_heads,
                  head_dim, positions.data(), model->rope_theta);

        // KV Caching: Store K/V in cache AFTER RoPE
        float *cache_k_layer = ctx->cache_k[il].data();
        float *cache_v_layer = ctx->cache_v[il].data();

        // Copy current K/V to cache at cache_pos
        for (int t = 0; t < n_tokens; t++) {
          int cache_offset = (ctx->cache_pos + t) * kv_hidden;
          memcpy(cache_k_layer + cache_offset,
                 ctx->k_buf.data() + t * kv_hidden, kv_hidden * sizeof(float));
          memcpy(cache_v_layer + cache_offset,
                 ctx->v_buf.data() + t * kv_hidden, kv_hidden * sizeof(float));
        }

        // Total sequence length including cache
        int total_seq_len = ctx->cache_pos + n_tokens;

        // GQA attention with KV cache
        // Q: [n_tokens, n_heads, head_dim] (only new tokens)
        // K: [total_seq_len, n_kv_heads, head_dim] (from cache)
        // V: [total_seq_len, n_kv_heads, head_dim] (from cache)

        int n_rep = n_heads / n_kv_heads; // Repetition factor (4 for 32/8)
        float scale = 1.0f / sqrtf((float)head_dim);

        // Zero output
        memset(ctx->attn_out.data(), 0, n_tokens * hidden * sizeof(float));

        // Process each query head
        for (int h = 0; h < n_heads; h++) {
          int kv_h = h / n_rep; // Which KV head this Q head uses

          // For each query token (only new tokens)
          for (int q_idx = 0; q_idx < n_tokens; q_idx++) {
            int q_pos = ctx->cache_pos + q_idx; // Absolute position
            float *q_head = ctx->q_buf.data() + q_idx * hidden + h * head_dim;
            float *out_head =
                ctx->attn_out.data() + q_idx * hidden + h * head_dim;

            // Compute attention scores for all cached key positions
            std::vector<float> scores(total_seq_len);
            float max_score = -INFINITY;

            // Attend to all positions up to and including current
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) { // Causal mask
              float *k_head =
                  cache_k_layer + k_pos * kv_hidden + kv_h * head_dim;

              // Q·K^T (K already has RoPE applied when stored)
              float score = 0.0f;
              for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_head[d];
              }
              score *= scale;

              scores[k_pos] = score;
              if (score > max_score)
                max_score = score;
            }

            // Softmax (numerically stable)
            float sum_exp = 0.0f;
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
              scores[k_pos] = expf(scores[k_pos] - max_score);
              sum_exp += scores[k_pos];
            }

            // Normalize
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
              scores[k_pos] /= sum_exp;
            }

            // Weighted sum of values from cache
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
              float *v_head =
                  cache_v_layer + k_pos * kv_hidden + kv_h * head_dim;
              float weight = scores[k_pos];

              for (int d = 0; d < head_dim; d++) {
                out_head[d] += weight * v_head[d];
              }
            }
          }
        }
      }

      // Output projection
      char out_name[256];
      snprintf(out_name, sizeof(out_name),
               "model.layers.%d.self_attn.out_proj.weight", il);
      float *wo = get_cached_weight(ctx, out_name);

      if (wo) {
        omni_matmul(ctx->hidden.data(), ctx->attn_out.data(), wo, n_tokens,
                    hidden, hidden);
      }
    }

    // Add residual
    for (int i = 0; i < n_tokens * hidden; i++) {
      ctx->hidden[i] += ctx->residual[i];
    }

    // Save residual BEFORE FFN norm
    memcpy(ctx->residual.data(), ctx->hidden.data(),
           n_tokens * hidden * sizeof(float));

    // FFN norm
    snprintf(norm_name, sizeof(norm_name), "model.layers.%d.ffn_norm.weight",
             il);
    norm_w = get_cached_weight(ctx, norm_name);

    if (norm_w) {
      for (int t = 0; t < n_tokens; t++) {
        omni_rms_norm(&ctx->hidden[t * hidden], &ctx->hidden[t * hidden],
                      norm_w, hidden, model->norm_eps);
      }
    }

    // FFN
    char ffn_names[3][256];
    snprintf(ffn_names[0], 256, "model.layers.%d.feed_forward.w1.weight", il);
    snprintf(ffn_names[1], 256, "model.layers.%d.feed_forward.w2.weight", il);
    snprintf(ffn_names[2], 256, "model.layers.%d.feed_forward.w3.weight", il);

    float *w1 = get_cached_weight(ctx, ffn_names[0]);
    float *w2 = get_cached_weight(ctx, ffn_names[1]);
    float *w3 = get_cached_weight(ctx, ffn_names[2]);

    if (w1 && w2 && w3) {
      const omni_tensor *w1_tensor = omni_get_tensor(model, ffn_names[0]);
      int ffn_dim = w1_tensor->shape[0];

      // Gate and up projections
      omni_matmul(ctx->ffn_buf.data(), ctx->hidden.data(), w1, n_tokens,
                  ffn_dim, hidden);
      omni_matmul(ctx->ffn_buf.data() + n_tokens * ffn_dim, ctx->hidden.data(),
                  w3, n_tokens, ffn_dim, hidden);

      // SiLU(gate) * up
      for (int t = 0; t < n_tokens; t++) {
        float *gate = ctx->ffn_buf.data() + t * ffn_dim;
        float *up = ctx->ffn_buf.data() + n_tokens * ffn_dim + t * ffn_dim;

        omni_silu(gate, gate, ffn_dim);
        for (int i = 0; i < ffn_dim; i++) {
          gate[i] *= up[i];
        }
      }

      // Down projection
      omni_matmul(ctx->hidden.data(), ctx->ffn_buf.data(), w2, n_tokens, hidden,
                  ffn_dim);
    }

    // Add residual
    for (int i = 0; i < n_tokens * hidden; i++) {
      ctx->hidden[i] += ctx->residual[i];
    }
  }

  // Final norm
  float *final_norm = get_cached_weight(ctx, "model.embedding_norm.weight");
  if (final_norm) {
    omni_rms_norm(&ctx->hidden[(n_tokens - 1) * hidden],
                  &ctx->hidden[(n_tokens - 1) * hidden], final_norm, hidden,
                  model->norm_eps);
  }

  // Output projection (reuse embeddings as output weights)
  omni_matmul(logits, &ctx->hidden[(n_tokens - 1) * hidden], emb_weights, 1,
              model->vocab_size, hidden);

  // Update cache position for next forward pass
  ctx->cache_pos += n_tokens;
}

// Sampling with top-k, top-p, and repetition penalty (matches PyTorch)
int omni_sample_advanced(const float *logits, int vocab_size, float temperature,
                         float top_p, int top_k, const int *generated_tokens,
                         int n_generated, float repetition_penalty) {
  // Apply repetition penalty to logits
  std::vector<float> penalized_logits(logits, logits + vocab_size);

  if (repetition_penalty != 1.0f && n_generated > 0) {
    for (int i = 0; i < n_generated; i++) {
      int token = generated_tokens[i];
      if (token >= 0 && token < vocab_size) {
        // Apply penalty: divide logit if > 0, multiply if < 0
        if (penalized_logits[token] > 0) {
          penalized_logits[token] /= repetition_penalty;
        } else {
          penalized_logits[token] *= repetition_penalty;
        }
      }
    }
  }

  // Apply temperature
  if (temperature > 1e-6f && temperature != 1.0f) {
    for (int i = 0; i < vocab_size; i++) {
      penalized_logits[i] /= temperature;
    }
  }

  // Create index-value pairs for sorting
  std::vector<std::pair<float, int>> logit_pairs;
  logit_pairs.reserve(vocab_size);
  for (int i = 0; i < vocab_size; i++) {
    logit_pairs.push_back({penalized_logits[i], i});
  }

  // Sort by logit value (descending)
  std::sort(logit_pairs.begin(), logit_pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Apply top-k filtering
  int k = (top_k > 0 && top_k < vocab_size) ? top_k : vocab_size;

  // Compute softmax probabilities for top-k
  float max_logit = logit_pairs[0].first;
  std::vector<float> probs(k);
  float sum = 0.0f;

  for (int i = 0; i < k; i++) {
    probs[i] = expf(logit_pairs[i].first - max_logit);
    sum += probs[i];
  }

  // Normalize
  for (int i = 0; i < k; i++) {
    probs[i] /= sum;
  }

  // Apply top-p (nucleus) filtering
  int p_cutoff = k;
  if (top_p < 1.0f) {
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
      cumsum += probs[i];
      if (cumsum >= top_p) {
        p_cutoff = i + 1;
        break;
      }
    }

    // Renormalize after top-p filtering
    sum = 0.0f;
    for (int i = 0; i < p_cutoff; i++) {
      sum += probs[i];
    }
    for (int i = 0; i < p_cutoff; i++) {
      probs[i] /= sum;
    }
  }

  // Sample from filtered distribution
  if (temperature < 1e-6f) {
    // Greedy: return highest probability token
    return logit_pairs[0].second;
  }

  float r = (float)rand() / RAND_MAX;
  float cumsum = 0.0f;
  for (int i = 0; i < p_cutoff; i++) {
    cumsum += probs[i];
    if (cumsum >= r) {
      return logit_pairs[i].second;
    }
  }

  return logit_pairs[0].second;
}

// Sampling with repetition penalty (old version, kept for compatibility)
int omni_sample_with_penalty(const float *logits, int vocab_size,
                             float temperature, const int *generated_tokens,
                             int n_generated, float repetition_penalty) {
  // Apply repetition penalty to logits
  std::vector<float> penalized_logits(logits, logits + vocab_size);

  if (repetition_penalty != 1.0f && n_generated > 0) {
    for (int i = 0; i < n_generated; i++) {
      int token = generated_tokens[i];
      if (token >= 0 && token < vocab_size) {
        // Apply penalty: divide logit if > 0, multiply if < 0
        if (penalized_logits[token] > 0) {
          penalized_logits[token] /= repetition_penalty;
        } else {
          penalized_logits[token] *= repetition_penalty;
        }
      }
    }
  }

  if (temperature < 1e-6f) {
    // Greedy
    int max_idx = 0;
    float max_val = penalized_logits[0];
    for (int i = 1; i < vocab_size; i++) {
      if (penalized_logits[i] > max_val) {
        max_val = penalized_logits[i];
        max_idx = i;
      }
    }
    return max_idx;
  }

  // Temperature sampling
  std::vector<float> probs(vocab_size);
  float max_val = penalized_logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (penalized_logits[i] > max_val)
      max_val = penalized_logits[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    probs[i] = expf((penalized_logits[i] - max_val) / temperature);
    sum += probs[i];
  }

  float r = (float)rand() / RAND_MAX * sum;
  float cumsum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    cumsum += probs[i];
    if (cumsum >= r)
      return i;
  }

  return vocab_size - 1;
}

// Sampling (original, kept for compatibility)
int omni_sample(const float *logits, int vocab_size, float temperature,
                float top_p) {
  if (temperature < 1e-6f) {
    // Greedy
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
      if (logits[i] > max_val) {
        max_val = logits[i];
        max_idx = i;
      }
    }
    return max_idx;
  }

  // Temperature sampling (simplified)
  std::vector<float> probs(vocab_size);
  float max_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val)
      max_val = logits[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    probs[i] = expf((logits[i] - max_val) / temperature);
    sum += probs[i];
  }

  float r = (float)rand() / RAND_MAX * sum;
  float cumsum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    cumsum += probs[i];
    if (cumsum >= r)
      return i;
  }

  return vocab_size - 1;
}

// Generation
int omni_generate(omni_context *ctx, const int *prompt, int prompt_len,
                  int *output, int max_tokens, float temperature) {
  // Reset KV cache for new sequence
  omni_reset_kv_cache(ctx);

  // Process prompt (prefill)
  omni_forward(ctx, prompt, prompt_len, ctx->logits.data());

  // Copy prompt to output
  for (int i = 0; i < prompt_len; i++) {
    output[i] = prompt[i];
  }

  // Generate tokens one at a time (decode)
  // Use PyTorch-like sampling parameters
  float repetition_penalty = 1.05f;
  float top_p = 0.1f; // Nucleus sampling (very restrictive)
  int top_k = 50;     // Top-k filtering

  int n_generated = 0;
  for (int i = 0; i < max_tokens; i++) {
    int next_token = omni_sample_advanced(
        ctx->logits.data(), ctx->model->vocab_size, temperature, top_p, top_k,
        output, // All generated tokens so far (prompt + generated)
        prompt_len + n_generated, // Total tokens
        repetition_penalty);
    output[prompt_len + n_generated] = next_token;
    n_generated++;

    // Check for EOS (token 2)
    if (next_token == 2)
      break;

    // Process single new token (uses KV cache)
    omni_forward(ctx, &next_token, 1, ctx->logits.data());
  }

  return prompt_len + n_generated;
}
