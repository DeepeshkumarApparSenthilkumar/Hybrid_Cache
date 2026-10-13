#!/usr/bin/env python3

import csv
import json
from pathlib import Path

OUT = Path("architecture/kimi_linear_48b_a3b")
OUT.mkdir(parents=True, exist_ok=True)

# Source-derived Kimi Linear MLA configuration
NUM_MLA_LAYERS = 7

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64

# Baseline cache dtype: BF16/FP16 = 2 bytes
CACHE_BYTES_PER_ELEMENT = 2

# Persistent MLA cache:
# [compressed KV latent | RoPE component]
MLA_CACHE_WIDTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM

BYTES_PER_TOKEN_PER_MLA_LAYER = (
    MLA_CACHE_WIDTH * CACHE_BYTES_PER_ELEMENT
)

BYTES_PER_TOKEN_ALL_MLA = (
    BYTES_PER_TOKEN_PER_MLA_LAYER * NUM_MLA_LAYERS
)

# Previously derived KDA useful state
KDA_TOTAL_MIB = 41.40625
KDA_TOTAL_BYTES = int(KDA_TOTAL_MIB * 1024**2)

CONTEXT_LENGTHS = [
    1024,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]


def mib(x):
    return x / (1024**2)


rows = []

for context in CONTEXT_LENGTHS:
    mla_bytes = context * BYTES_PER_TOKEN_ALL_MLA
    total_bytes = KDA_TOTAL_BYTES + mla_bytes

    rows.append({
        "context_tokens": context,
        "kda_bounded_mib": KDA_TOTAL_MIB,
        "mla_cache_mib": mib(mla_bytes),
        "total_useful_state_mib": mib(total_bytes),
    })


with open(OUT / "mla_cache_scaling.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


summary = {
    "model": "moonshotai/Kimi-Linear-48B-A3B-Base",

    "mla": {
        "num_mla_layers": NUM_MLA_LAYERS,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,

        "persistent_cache_width_elements":
            MLA_CACHE_WIDTH,

        "cache_bytes_per_element":
            CACHE_BYTES_PER_ELEMENT,

        "bytes_per_token_per_mla_layer":
            BYTES_PER_TOKEN_PER_MLA_LAYER,

        "kib_per_token_per_mla_layer":
            BYTES_PER_TOKEN_PER_MLA_LAYER / 1024,

        "bytes_per_token_all_mla_layers":
            BYTES_PER_TOKEN_ALL_MLA,

        "kib_per_token_all_mla_layers":
            BYTES_PER_TOKEN_ALL_MLA / 1024,
    },

    "combined_memory_model": {
        "bounded_kda_state_mib": KDA_TOTAL_MIB,
        "formula":
            "M(L) = KDA_bounded_bytes + L * MLA_bytes_per_token_all_layers",
    },

    "note": (
        "MLA persistent cache width is derived as "
        "kv_lora_rank + qk_rope_head_dim. "
        "Values represent useful cache state under a 2-byte "
        "MLA cache dtype and exclude allocator/page overhead."
    ),
}


with open(OUT / "mla_cache_derivation.json", "w") as f:
    json.dump(summary, f, indent=2)


print("===== Kimi Linear MLA Characterization =====")
print()

print(f"MLA layers: {NUM_MLA_LAYERS}")
print(f"KV LoRA rank: {KV_LORA_RANK}")
print(f"RoPE dimension: {QK_ROPE_HEAD_DIM}")
print()

print(
    f"Persistent MLA width: "
    f"{MLA_CACHE_WIDTH} elements/token/layer"
)

print(
    f"Per MLA layer: "
    f"{BYTES_PER_TOKEN_PER_MLA_LAYER} B/token "
    f"({BYTES_PER_TOKEN_PER_MLA_LAYER / 1024:.3f} KiB/token)"
)

print(
    f"Across all MLA layers: "
    f"{BYTES_PER_TOKEN_ALL_MLA} B/token "
    f"({BYTES_PER_TOKEN_ALL_MLA / 1024:.3f} KiB/token)"
)

print()
print("Context scaling:")

for row in rows:
    print(
        f"  {row['context_tokens']:>6,} tokens : "
        f"MLA={row['mla_cache_mib']:>8.2f} MiB | "
        f"KDA+MLA={row['total_useful_state_mib']:>8.2f} MiB"
    )

print()
print("Saved:")
print(f"  {OUT / 'mla_cache_scaling.csv'}")
print(f"  {OUT / 'mla_cache_derivation.json'}")
