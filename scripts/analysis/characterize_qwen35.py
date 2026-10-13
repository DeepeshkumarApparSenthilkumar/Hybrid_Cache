#!/usr/bin/env python3

from pathlib import Path
from transformers import AutoConfig
import pandas as pd

MODEL = "Qwen/Qwen3.5-0.8B"

BASE = Path.home() / "hybrid-kv-cache"
OUT = BASE / "architecture/qwen35_0.8b"
OUT.mkdir(parents=True, exist_ok=True)

config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
text = config.text_config

print("===== QWEN3.5-0.8B ARCHITECTURE =====")
print("Hidden size:", text.hidden_size)
print("Layers:", text.num_hidden_layers)
print("Full-attention interval:", text.full_attention_interval)
print("Attention heads:", text.num_attention_heads)
print("KV heads:", text.num_key_value_heads)
print("Attention head dim:", text.head_dim)
print("Linear key heads:", text.linear_num_key_heads)
print("Linear value heads:", text.linear_num_value_heads)
print("Linear key head dim:", text.linear_key_head_dim)
print("Linear value head dim:", text.linear_value_head_dim)
print("Linear conv kernel:", text.linear_conv_kernel_dim)
print("SSM/recurrent dtype:", text.mamba_ssm_dtype)

# Build layer map
rows = []

for i, layer_type in enumerate(text.layer_types):
    if layer_type == "full_attention":
        state_type = "KV cache"
        context_scaling = "O(sequence length)"
    elif layer_type == "linear_attention":
        state_type = "recurrent/linear-attention state"
        context_scaling = "bounded recurrent state"
    else:
        state_type = "unknown"
        context_scaling = "unknown"

    rows.append({
        "layer": i,
        "layer_type": layer_type,
        "state_type": state_type,
        "context_scaling": context_scaling,
    })

layer_df = pd.DataFrame(rows)
layer_df.to_csv(OUT / "layer_map.csv", index=False)

print("\n===== LAYER MAP =====")
print(layer_df.to_string(index=False))

print("\n===== COUNTS =====")
print(layer_df["layer_type"].value_counts())

# Full-attention KV calculation
num_full = int((layer_df["layer_type"] == "full_attention").sum())

kv_heads = text.num_key_value_heads
head_dim = text.head_dim
kv_element_bytes = 2  # BF16

kv_bytes_per_token_per_layer = (
    2 * kv_heads * head_dim * kv_element_bytes
)

kv_bytes_per_token_all_full_layers = (
    kv_bytes_per_token_per_layer * num_full
)

print("\n===== FULL ATTENTION KV =====")
print("KV bytes/token/full-attention-layer:",
      kv_bytes_per_token_per_layer)
print("Number of full-attention layers:", num_full)
print("KV bytes/token across full-attention layers:",
      kv_bytes_per_token_all_full_layers)
print("KV KiB/token:",
      kv_bytes_per_token_all_full_layers / 1024)

contexts = [
    1024,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

memory_rows = []

for context in contexts:
    total_bytes = context * kv_bytes_per_token_all_full_layers

    memory_rows.append({
        "context_tokens": context,
        "full_attention_kv_bytes": total_bytes,
        "full_attention_kv_mib": total_bytes / (1024 ** 2),
        "full_attention_kv_gib": total_bytes / (1024 ** 3),
    })

memory_df = pd.DataFrame(memory_rows)

memory_df.to_csv(
    OUT / "full_attention_kv_scaling.csv",
    index=False
)

print("\n===== THEORETICAL FULL-ATTENTION KV SCALING =====")
print(memory_df.round(4).to_string(index=False))

print("\n===== SAVED FILES =====")
print(OUT / "layer_map.csv")
print(OUT / "full_attention_kv_scaling.csv")
