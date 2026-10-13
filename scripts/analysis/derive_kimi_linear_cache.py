#!/usr/bin/env python3

import csv
import json
from pathlib import Path

OUT = Path("architecture/kimi_linear_48b_a3b")
OUT.mkdir(parents=True, exist_ok=True)

# Source-confirmed Kimi Linear configuration
NUM_LAYERS = 27
KDA_LAYERS = [
    1,2,3,5,6,7,9,10,11,13,14,15,
    17,18,19,21,22,23,25,26
]
MLA_LAYERS = [4,8,12,16,20,24,27]

NUM_HEADS = 32
HEAD_DIM = 128
CONV_KERNEL = 4

# Baseline serving assumptions
TP = 1
NUM_SPEC = 0

# kda_state_shape() defaults:
NUM_K_HEADS = NUM_HEADS
HEAD_K_DIM = HEAD_DIM

# vLLM dtype behavior:
# conv cache follows BF16/model cache dtype
# KDA recurrent state with mamba_ssm_cache_dtype="auto" -> FP32
CONV_BYTES = 2
RECURRENT_BYTES = 4

# --------------------------------------------------
# Reproduce vLLM kda_state_shape()
# --------------------------------------------------

proj_size = NUM_HEADS * HEAD_DIM
proj_k_size = NUM_K_HEADS * HEAD_K_DIM

conv_dim = proj_size + 2 * proj_k_size

conv_shape = (
    conv_dim // TP,
    CONV_KERNEL - 1 + NUM_SPEC
)

recurrent_shape = (
    NUM_HEADS // TP,
    HEAD_DIM,
    HEAD_DIM
)

conv_elements = conv_shape[0] * conv_shape[1]

recurrent_elements = (
    recurrent_shape[0]
    * recurrent_shape[1]
    * recurrent_shape[2]
)

conv_bytes = conv_elements * CONV_BYTES
recurrent_bytes = recurrent_elements * RECURRENT_BYTES

per_layer_bytes = conv_bytes + recurrent_bytes
all_kda_bytes = per_layer_bytes * len(KDA_LAYERS)


def kib(x):
    return x / 1024


def mib(x):
    return x / (1024 ** 2)


# --------------------------------------------------
# Save layer map
# --------------------------------------------------

with open(OUT / "layer_map.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "layer_type"])

    for layer in range(1, NUM_LAYERS + 1):
        if layer in KDA_LAYERS:
            kind = "KDA"
        elif layer in MLA_LAYERS:
            kind = "MLA"
        else:
            kind = "UNKNOWN"

        writer.writerow([layer, kind])


# --------------------------------------------------
# Save state derivation
# --------------------------------------------------

with open(OUT / "kda_state_derivation.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "state",
        "shape",
        "elements",
        "dtype",
        "bytes",
        "KiB"
    ])

    writer.writerow([
        "conv",
        str(conv_shape),
        conv_elements,
        "bfloat16",
        conv_bytes,
        kib(conv_bytes)
    ])

    writer.writerow([
        "recurrent",
        str(recurrent_shape),
        recurrent_elements,
        "float32",
        recurrent_bytes,
        kib(recurrent_bytes)
    ])

    writer.writerow([
        "total_per_kda_layer",
        "",
        conv_elements + recurrent_elements,
        "",
        per_layer_bytes,
        kib(per_layer_bytes)
    ])


summary = {
    "model": "moonshotai/Kimi-Linear-48B-A3B-Base",

    "architecture": {
        "num_layers": NUM_LAYERS,
        "num_kda_layers": len(KDA_LAYERS),
        "num_mla_layers": len(MLA_LAYERS),
        "kda_layers": KDA_LAYERS,
        "mla_layers": MLA_LAYERS,
    },

    "assumptions": {
        "tensor_parallel_size": TP,
        "num_speculative_tokens": NUM_SPEC,
        "use_kda_recoverssm": False,
        "model_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "auto",
    },

    "kda": {
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "short_conv_kernel_size": CONV_KERNEL,

        "conv_state_shape": conv_shape,
        "conv_elements": conv_elements,
        "conv_dtype": "bfloat16",
        "conv_bytes": conv_bytes,

        "recurrent_state_shape": recurrent_shape,
        "recurrent_elements": recurrent_elements,
        "recurrent_dtype": "float32",
        "recurrent_bytes": recurrent_bytes,

        "bytes_per_kda_layer": per_layer_bytes,
        "mib_per_kda_layer": mib(per_layer_bytes),

        "bytes_all_kda_layers": all_kda_bytes,
        "mib_all_kda_layers": mib(all_kda_bytes),
    },

    "note": (
        "Derived useful KDA state from the vLLM KimiLinearForCausalLM "
        "state-shape and dtype implementations. This is not measured "
        "physical cache allocation."
    ),
}

with open(OUT / "kda_state_derivation.json", "w") as f:
    json.dump(summary, f, indent=2)


print("===== Kimi Linear KDA Characterization =====")
print()
print(f"Total layers : {NUM_LAYERS}")
print(f"KDA layers   : {len(KDA_LAYERS)}")
print(f"MLA layers   : {len(MLA_LAYERS)}")
print()

print("State shapes:")
print(f"  Conv      : {conv_shape}")
print(f"  Recurrent : {recurrent_shape}")
print()

print("State elements:")
print(f"  Conv      : {conv_elements:,}")
print(f"  Recurrent : {recurrent_elements:,}")
print()

print("Memory per KDA layer:")
print(f"  Conv      : {kib(conv_bytes):.2f} KiB")
print(f"  Recurrent : {kib(recurrent_bytes):.2f} KiB")
print(f"  Total     : {kib(per_layer_bytes):.2f} KiB")
print(f"              {mib(per_layer_bytes):.6f} MiB")
print()

print("Across all 20 KDA layers:")
print(f"  {mib(all_kda_bytes):.6f} MiB")
print()

print("Saved:")
print(f"  {OUT / 'layer_map.csv'}")
print(f"  {OUT / 'kda_state_derivation.csv'}")
print(f"  {OUT / 'kda_state_derivation.json'}")
