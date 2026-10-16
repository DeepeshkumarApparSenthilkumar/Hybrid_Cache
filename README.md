# Hybrid KV/State Cache Characterization in LLM Inference Serving

This repository contains ongoing research on **hybrid KV/state cache behavior in LLM inference serving**, with a focus on understanding how modern hybrid architectures change the memory and cache-management requirements compared with traditional attention-only Transformer models.

The project studies the problem from the **model architecture outward**:

**Model Architecture → Required Inference State → Serving Engine Cache Management → Cache Reuse → Memory & Latency**

The current work focuses primarily on **Qwen3.5, Falcon-H1, and Kimi Linear**, with experiments and source-level analysis using **vLLM**. A comparison with **SGLang**, additional workloads, and further cache optimization experiments are planned as the next phase.

---

## Research Motivation

Traditional Transformer inference relies heavily on the **KV cache**, where key and value representations from previous tokens are stored to avoid recomputation during decoding. This cache grows with context length and can become a significant memory bottleneck for long-context inference.

Newer hybrid models do not use full attention uniformly across all layers.

For example:

- **Qwen3.5** combines Gated DeltaNet layers with full-attention layers.
- **Falcon-H1** combines attention with Mamba-2/state-space components.
- **Kimi Linear** combines Kimi Delta Attention (KDA) with Multi-head Latent Attention (MLA).

These architectures introduce different forms of inference state. Some state grows with the number of tokens, while recurrent or state-space components can maintain bounded state.

Therefore, this project investigates inference memory as **heterogeneous inference state**, rather than treating every layer as having the same conventional KV-cache behavior.

---

## Current Research Questions

The project is currently investigating:

1. How do hybrid LLM architectures change KV-cache and recurrent-state requirements?
2. Which layers require token-growing KV cache and which maintain bounded recurrent state?
3. How do serving engines such as vLLM manage these heterogeneous cache types?
4. How does cache organization affect prefix reuse, memory utilization, and inference latency?
5. Can architecture-aware characterization reveal opportunities for better cache allocation or reuse?

---

## Models Characterized So Far

### Qwen3.5-0.8B

Qwen3.5-0.8B contains **24 text layers**:

- 18 Gated DeltaNet / linear-attention layers
- 6 full-attention layers

The pattern consists primarily of three linear-attention layers followed by one full-attention layer.

The full-attention layers require token-growing KV cache, while the Gated DeltaNet layers maintain recurrent state.

For the full-attention layers, the characterized configuration gives approximately:

**2 KiB/token/full-attention-layer**

Across all six full-attention layers:

**12 KiB/token**

This produces approximately:

| Context Length | Full-Attention KV |
|---:|---:|
| 1,024 | 12 MiB |
| 4,096 | 48 MiB |
| 8,192 | 96 MiB |
| 32,768 | 384 MiB |
| 131,072 | 1.5 GiB |
| 262,144 | 3 GiB |

These values represent architecture-derived KV requirements and do not include allocator or serving-engine overhead.

---

## Qwen3.5 + vLLM Cache Characterization

Qwen3.5-0.8B was served using vLLM on an NVIDIA Quadro RTX 6000.

The observed vLLM hybrid-cache configuration included:

- Prefix caching enabled
- Effective attention block size: **544 tokens**
- Mamba/Gated DeltaNet block size: **544 tokens**
- 1,661 GPU cache blocks
- Maximum model length: 8,192 tokens

Source-level analysis was then used to understand why the unusual **544-token cache-management granularity** appears.

For the examined Qwen Gated DeltaNet configuration:

- Convolution state: approximately **36 KiB**
- Recurrent/temporal state: approximately **1,024 KiB**
- Total useful GDN state: approximately **1,060 KiB**

The full-attention KV representation requires approximately:

**2,048 bytes/token/layer**

Therefore, the GDN state is byte-equivalent to approximately:

**530 attention KV tokens**

The attention backend requires 16-token alignment:

`16 × ceil(530 / 16) = 544`

This results in an aligned page of approximately **1,088 KiB**, compared with approximately **1,060 KiB** of useful GDN state.

The resulting padding is approximately **28 KiB (~2.64%)** for this configuration.

Importantly, **544 does not mean that Gated DeltaNet stores 544 historical token states**. It is a cache-management/alignment granularity resulting from reconciling recurrent-state storage with the block-based attention cache representation.

---

## Prefix Cache Reuse Experiment

The 544-token granularity was also visible experimentally.

Repeated-prefix experiments produced:

| Prefix Length | Cached/Reused Tokens |
|---:|---:|
| 1,024 | 544 |
| 1,088 | 1,088 |
| 2,048 | 1,632 |
| 2,176 | 2,176 |
| 4,096 | 3,808 |
| 4,352 | 4,352 |

For the tested configuration, the observed reusable token count followed:

`Cached(P) = 544 × floor(P / 544)`

Prefixes aligned to the 544-token cache granularity achieved complete reuse in these experiments, while non-aligned prefixes reused complete cache units.

Prefix reuse also produced substantial reductions in **Time to First Token (TTFT)**, particularly for larger prefixes.

This provides an initial connection between:

**Model Architecture → Recurrent-State Size → Engine Cache Granularity → Prefix Reuse → Observed Latency**

---

## Falcon-H1 Characterization

The current Falcon-H1 study uses:

**tiiuae/Falcon-H1-0.5B-Base**

The examined vLLM implementation contains **36 parallel-hybrid decoder layers**.

Each layer combines:

- Multi-head attention
- Mamba-2/state-space processing

For TP=1, the characterized Mamba-2 state includes:

- Convolution state shape: `(1792, 3)`
- Recurrent SSM state shape: `(24, 64, 128)`

The attention component requires approximately:

**512 bytes/token/layer**

Falcon-H1 therefore represents a different hybrid organization from Qwen3.5: instead of alternating recurrent and full-attention layers, attention and state-space computation coexist within the hybrid decoder layers.

The current Falcon results are architecture/source-derived characterizations. Serving-engine allocation and cache granularity still need to be measured experimentally.

---

## Kimi Linear Characterization

The current Kimi Linear analysis uses:

**moonshotai/Kimi-Linear-48B-A3B-Base**

The model contains **27 layers**:

- 20 Kimi Delta Attention (KDA) layers
- 7 MLA/full-attention layers

The architecture follows a predominantly 3:1 KDA-to-MLA pattern.

### KDA State

For the characterized TP=1 baseline:

- Convolution state shape: `(12288, 3)`
- Recurrent state shape: `(32, 128, 128)`
- Useful state per KDA layer: approximately **2.07 MiB**
- Across 20 KDA layers: approximately **41.41 MiB**

### MLA Cache

The persistent MLA cache uses a compressed representation based on:

- KV LoRA rank: 512
- RoPE dimension: 64

giving a persistent width of:

**576 elements/token/layer**

At two bytes per element:

**1,152 bytes/token/MLA layer**

Across seven MLA layers:

**7.875 KiB/token**

This illustrates another hybrid-cache strategy: a relatively larger bounded KDA state combined with a compressed token-growing MLA representation.

These values currently represent architecture-derived useful state rather than measured physical GPU allocation.

---

## Cross-Model Observation

The current characterization suggests that hybrid models can have substantially different inference-state organizations:

| Model | Hybrid Structure | Main State Behavior |
|---|---|---|
| Qwen3.5-0.8B | Gated DeltaNet + Full Attention | Bounded GDN state + token-growing KV |
| Falcon-H1-0.5B | Mamba-2 + Attention | Parallel recurrent state + token-growing KV |
| Kimi Linear | KDA + MLA | Bounded KDA state + compressed token-growing MLA |

One emerging observation is that **KV cache alone is not sufficient to describe inference memory in hybrid architectures**. These models contain heterogeneous combinations of token-indexed and recurrent inference state.

---

## Repository Structure

```text
architecture/
├── qwen35_0.8b/
├── falcon_h1_0.5b/
├── kimi_linear_48b_a3b/
└── gemma3_1b/

scripts/
├── analysis/
└── vllm/

results/
├── analysis/
└── vllm/

plots/
├── architecture/
└── vllm/
