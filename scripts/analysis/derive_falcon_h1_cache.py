from pathlib import Path
import json
import csv
import math

OUT = Path("architecture/falcon_h1_0.5b")
OUT.mkdir(parents=True, exist_ok=True)

# Falcon-H1-0.5B configuration
NUM_LAYERS = 36

# Attention
NUM_KV_HEADS = 2
ATTN_HEAD_DIM = 64

# Mamba-2
INTERMEDIATE_SIZE = 1536
N_GROUPS = 1
MAMBA_HEADS = 24
MAMBA_HEAD_DIM = 64
STATE_SIZE = 128
CONV_KERNEL = 4

TP = 1
NUM_SPEC = 0

# ----------------------------
# vLLM Mamba-2 state shapes
# ----------------------------

conv_dim = INTERMEDIATE_SIZE + 2 * N_GROUPS * STATE_SIZE

conv_state_shape = (
    conv_dim // TP,
    CONV_KERNEL - 1 + NUM_SPEC,
)

temporal_state_shape = (
    MAMBA_HEADS // TP,
    MAMBA_HEAD_DIM,
    STATE_SIZE,
)

conv_elements = math.prod(conv_state_shape)
temporal_elements = math.prod(temporal_state_shape)

# ----------------------------
# Attention KV
# ----------------------------

def kv_bytes_per_token(dtype_bytes):
    # K + V
    return (
        2
        * NUM_KV_HEADS
        * ATTN_HEAD_DIM
        * dtype_bytes
    )

# ----------------------------
# State scenarios
# ----------------------------

scenarios = [
    {
        "name": "FP16_conv_FP16_ssm",
        "conv_dtype_bytes": 2,
        "ssm_dtype_bytes": 2,
    },
    {
        "name": "FP16_conv_FP32_ssm",
        "conv_dtype_bytes": 2,
        "ssm_dtype_bytes": 4,
    },
]

results = []

for scenario in scenarios:
    conv_bytes = conv_elements * scenario["conv_dtype_bytes"]
    ssm_bytes = temporal_elements * scenario["ssm_dtype_bytes"]
    total_state_bytes = conv_bytes + ssm_bytes

    kv_token_bytes = kv_bytes_per_token(2)

    equivalent_kv_tokens = total_state_bytes / kv_token_bytes

    results.append({
        "scenario": scenario["name"],
        "conv_state_shape": str(conv_state_shape),
        "ssm_state_shape": str(temporal_state_shape),
        "conv_elements": conv_elements,
        "ssm_elements": temporal_elements,
        "conv_bytes": conv_bytes,
        "conv_kib": conv_bytes / 1024,
        "ssm_bytes": ssm_bytes,
        "ssm_kib": ssm_bytes / 1024,
        "total_state_bytes": total_state_bytes,
        "total_state_kib": total_state_bytes / 1024,
        "attention_kv_bytes_per_token_per_layer": kv_token_bytes,
        "state_equivalent_kv_tokens": equivalent_kv_tokens,
    })

# ----------------------------
# Save CSV
# ----------------------------

csv_path = OUT / "cache_state_derivation.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ----------------------------
# Save JSON
# ----------------------------

summary = {
    "model": "tiiuae/Falcon-H1-0.5B-Base",
    "num_hidden_layers": NUM_LAYERS,
    "tensor_parallel_size": TP,

    "attention": {
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": ATTN_HEAD_DIM,
        "fp16_kv_bytes_per_token_per_layer": kv_bytes_per_token(2),
    },

    "mamba2": {
        "intermediate_size": INTERMEDIATE_SIZE,
        "n_groups": N_GROUPS,
        "num_heads": MAMBA_HEADS,
        "head_dim": MAMBA_HEAD_DIM,
        "state_size": STATE_SIZE,
        "conv_kernel": CONV_KERNEL,
        "conv_dim": conv_dim,
        "conv_state_shape": conv_state_shape,
        "temporal_state_shape": temporal_state_shape,
        "conv_elements": conv_elements,
        "temporal_elements": temporal_elements,
    },

    "dtype_scenarios": results,
}

with (OUT / "cache_state_derivation.json").open("w") as f:
    json.dump(summary, f, indent=2)

# ----------------------------
# Print results
# ----------------------------

print("\nFalcon-H1-0.5B cache-state derivation")
print("=" * 55)

print(f"\nLayers: {NUM_LAYERS}")

print("\nMamba-2:")
print(f"  conv_dim:             {conv_dim}")
print(f"  conv_state_shape:     {conv_state_shape}")
print(f"  temporal_state_shape: {temporal_state_shape}")
print(f"  conv elements:        {conv_elements:,}")
print(f"  SSM elements:         {temporal_elements:,}")

print("\nAttention:")
print(
    "  FP16 KV bytes/token/layer:",
    f"{kv_bytes_per_token(2):,}",
)

for r in results:
    print("\n" + r["scenario"])
    print("-" * 40)
    print(f"Conv state:  {r['conv_kib']:.2f} KiB")
    print(f"SSM state:   {r['ssm_kib']:.2f} KiB")
    print(f"Total state: {r['total_state_kib']:.2f} KiB")
    print(
        "KV-token equivalent:",
        f"{r['state_equivalent_kv_tokens']:.2f}",
    )

print(f"\nSaved: {csv_path}")
print(f"Saved: {OUT / 'cache_state_derivation.json'}")
