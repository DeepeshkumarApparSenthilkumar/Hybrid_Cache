# Qwen3.5-0.8B — vLLM Hybrid Cache Characterization

## Experimental Configuration

Model: Qwen/Qwen3.5-0.8B  
Serving engine: vLLM 0.30.1rc1.dev396+gac68c3087  
Tensor parallelism: 1  
Serving dtype: FP16  
Mamba SSM cache dtype: FP32  
Mamba cache mode: align  
Prefix caching: enabled  
Maximum model length: 8192  

Observed runtime configuration:

- effective_attention_block_size = 544 tokens
- mamba_block_size = 544 tokens
- num_gpu_blocks = 1661
- kv_cache_size_tokens = 618496

---

## 1. Model Architecture

Qwen3.5-0.8B contains 24 language-model layers.

Layer pattern:

- 18 Gated DeltaNet / linear-attention layers
- 6 full-attention layers
- Pattern: 3 linear-attention layers followed by 1 full-attention layer

Full-attention configuration:

- KV heads = 2
- head dimension = 256
- FP16 cache = 2 bytes/element

Therefore KV memory per token per full-attention layer is:

2 (K,V) × 2 KV heads × 256 × 2 bytes

= 2048 bytes/token/layer
= 2 KiB/token/layer

Across six full-attention layers:

= 12 KiB/token

---

## 2. Gated DeltaNet State

For Qwen3.5:

linear_num_key_heads = 16
linear_num_value_heads = 16
linear_key_head_dim = 128
linear_value_head_dim = 128
linear_conv_kernel_dim = 4

vLLM calculates:

conv_dim =
(128 × 16 × 2) + (128 × 16)
= 6144

For TP=1 and no speculative decoding:

conv state shape =
(6144, 4 - 1)
= (6144, 3)

With FP16:

6144 × 3 × 2
= 36,864 bytes
= 36 KiB

The recurrent/temporal state shape is:

(16, 128, 128)

With FP32:

16 × 128 × 128 × 4
= 1,048,576 bytes
= 1024 KiB

Total real GDN state page:

1,048,576 + 36,864
= 1,085,440 bytes
= 1060 KiB

---

## 3. Attention-Equivalent State Size

One full-attention token requires:

2048 bytes

Therefore:

1,085,440 / 2048
= 530 attention-token equivalents

The raw GDN state therefore corresponds to 530 attention KV tokens.

---

## 4. Backend Alignment

Both relevant vLLM FlashAttention and Triton attention backends require
kernel block sizes that are multiples of 16.

vLLM hybrid-cache sizing effectively computes:

B = A × ceil(M / (A × P))

where:

A = kernel block alignment = 16 tokens
M = GDN state page = 1,085,440 bytes
P = attention KV size = 2048 bytes/token

Therefore:

B = 16 × ceil(1,085,440 / (16 × 2048))

B = 16 × ceil(33.125)

B = 16 × 34

B = 544 tokens

Thus the observed 544-token block size is produced by:

530 raw token-equivalents
→ rounded upward to the next valid 16-token boundary
→ 544 tokens.

---

## 5. Physical Page Padding

The resulting attention page is:

544 × 2048
= 1,114,112 bytes
= 1088 KiB

Real GDN state:

= 1060 KiB

Padding:

1088 - 1060
= 28 KiB

Padding overhead:

28 / 1060 × 100
≈ 2.64%

vLLM therefore pads the GDN state page so that its physical page size
matches the full-attention cache page.

Because mamba_cache_mode=align:

mamba_block_size = attention_block_size = 544 tokens.

---

## 6. Experimental Prefix-Reuse Validation

Observed repeated-prefix experiment:

| Prefix | Predicted Cached | Observed Cached |
|-------:|-----------------:|----------------:|
| 1024 | 544 | 544 |
| 1088 | 1088 | 1088 |
| 2048 | 1632 | 1632 |
| 2176 | 2176 | 2176 |
| 4096 | 3808 | 3808 |
| 4352 | 4352 | 4352 |

The observed relationship is:

Cached(L) = 544 × floor(L / 544)

Every tested prefix matched the predicted full-block reuse boundary.

Aligned prefixes therefore achieved complete prefix reuse, while
unaligned prefixes left the final partial block unreused.

---

## 7. Main Finding

The 544-token cache granularity is not an intrinsic architectural
constant of Qwen3.5.

For this Qwen3.5-0.8B + vLLM configuration, it emerges from the
interaction of:

1. Gated DeltaNet recurrent-state size,
2. full-attention KV bytes per token,
3. serving-engine kernel block alignment, and
4. vLLM hybrid-cache page unification.

Model architecture
→ 1060 KiB GDN state

Attention architecture
→ 2 KiB KV/token/layer

State equivalence
→ 530 tokens

Backend alignment
→ multiple of 16

vLLM cache manager
→ 544-token unified block

Physical page
→ 1088 KiB

GDN padding
→ 28 KiB (2.64%)

Prefix-cache granularity
→ 544 tokens

This connects model architecture directly to serving-engine memory
management and experimentally observable prefix-reuse behavior.

## Scope

These values are configuration-specific and should not be interpreted
as universal Qwen3.5 or universal vLLM constants. Different model sizes,
KV-cache dtypes, tensor parallelism, recurrent-state dtypes, attention
backends, speculative-decoding settings, or serving-engine versions may
produce different cache-page sizes and block granularities.
