#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULT_DIR = Path("results/analysis/architecture")
PLOT_DIR = Path("plots/architecture")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXTS = [
    1024,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

# ============================================================
# QWEN3.5-0.8B
# ============================================================
#
# 24 layers:
#   18 Gated DeltaNet
#    6 full attention
#
# Full-attention KV:
#   2048 B/token/full-attention-layer
#
# GDN useful state per layer:
#   1,085,440 B
#
# IMPORTANT:
# This is useful state, not the padded physical vLLM page.
# ============================================================

QWEN_GDN_LAYERS = 18
QWEN_ATTN_LAYERS = 6

QWEN_GDN_STATE_PER_LAYER = 1_085_440
QWEN_ATTN_BYTES_PER_TOKEN_PER_LAYER = 2048

QWEN_BOUNDED_BYTES = (
    QWEN_GDN_LAYERS * QWEN_GDN_STATE_PER_LAYER
)

QWEN_GROWING_BYTES_PER_TOKEN = (
    QWEN_ATTN_LAYERS
    * QWEN_ATTN_BYTES_PER_TOKEN_PER_LAYER
)

# ============================================================
# FALCON-H1-0.5B
# ============================================================
#
# 36 parallel-hybrid layers.
# Every layer contains:
#   attention + Mamba-2
#
# Attention:
#   512 B/token/layer
#
# Two state-dtype scenarios were derived previously.
#
# For this comparison use:
#   FP16 conv + FP32 SSM
#
#   778.5 KiB / layer
#
# This is explicitly an analytical scenario until serving
# configuration confirms actual Falcon cache dtype.
# ============================================================

FALCON_LAYERS = 36

FALCON_STATE_PER_LAYER = int(778.5 * 1024)
FALCON_ATTN_BYTES_PER_TOKEN_PER_LAYER = 512

FALCON_BOUNDED_BYTES = (
    FALCON_LAYERS * FALCON_STATE_PER_LAYER
)

FALCON_GROWING_BYTES_PER_TOKEN = (
    FALCON_LAYERS
    * FALCON_ATTN_BYTES_PER_TOKEN_PER_LAYER
)

# ============================================================
# KIMI LINEAR 48B-A3B
# ============================================================
#
# 27 layers:
#   20 KDA
#    7 MLA
#
# KDA bounded useful state:
#   41.40625 MiB total
#
# MLA:
#   cache width = 512 + 64 = 576 elements
#   BF16/FP16 => 1152 B/token/layer
# ============================================================

KIMI_KDA_LAYERS = 20
KIMI_MLA_LAYERS = 7

KIMI_BOUNDED_BYTES = int(
    41.40625 * 1024 * 1024
)

KIMI_MLA_BYTES_PER_TOKEN_PER_LAYER = 1152

KIMI_GROWING_BYTES_PER_TOKEN = (
    KIMI_MLA_LAYERS
    * KIMI_MLA_BYTES_PER_TOKEN_PER_LAYER
)


def mib(x):
    return x / (1024 ** 2)


models = [
    {
        "model": "Qwen3.5-0.8B",
        "architecture": "18 GDN + 6 full attention",
        "total_layers": 24,
        "bounded_component": "Gated DeltaNet recurrent state",
        "growing_component": "Full-attention KV",
        "bounded_state_mib": mib(QWEN_BOUNDED_BYTES),
        "growing_bytes_per_token": QWEN_GROWING_BYTES_PER_TOKEN,
        "growing_kib_per_token": QWEN_GROWING_BYTES_PER_TOKEN / 1024,
        "status": "source-derived + vLLM measured",
    },
    {
        "model": "Falcon-H1-0.5B",
        "architecture": "36 parallel Attention + Mamba-2 layers",
        "total_layers": 36,
        "bounded_component": "Mamba-2 recurrent state",
        "growing_component": "Attention KV",
        "bounded_state_mib": mib(FALCON_BOUNDED_BYTES),
        "growing_bytes_per_token": FALCON_GROWING_BYTES_PER_TOKEN,
        "growing_kib_per_token": FALCON_GROWING_BYTES_PER_TOKEN / 1024,
        "status": "source-derived analytical dtype scenario",
    },
    {
        "model": "Kimi-Linear-48B-A3B",
        "architecture": "20 KDA + 7 MLA",
        "total_layers": 27,
        "bounded_component": "KDA recurrent state",
        "growing_component": "Compressed MLA cache",
        "bounded_state_mib": mib(KIMI_BOUNDED_BYTES),
        "growing_bytes_per_token": KIMI_GROWING_BYTES_PER_TOKEN,
        "growing_kib_per_token": KIMI_GROWING_BYTES_PER_TOKEN / 1024,
        "status": "source-derived analytical model",
    },
]

# ============================================================
# Save architecture comparison
# ============================================================

comparison_path = RESULT_DIR / "hybrid_architecture_comparison.csv"

with open(comparison_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=models[0].keys()
    )
    writer.writeheader()
    writer.writerows(models)

# ============================================================
# Context scaling
# ============================================================

scaling_rows = []

for model in models:
    bounded_bytes = (
        model["bounded_state_mib"]
        * 1024 ** 2
    )

    for context in CONTEXTS:
        growing_bytes = (
            context
            * model["growing_bytes_per_token"]
        )

        total_bytes = (
            bounded_bytes
            + growing_bytes
        )

        scaling_rows.append({
            "model": model["model"],
            "context_tokens": context,
            "bounded_state_mib": mib(bounded_bytes),
            "growing_cache_mib": mib(growing_bytes),
            "total_useful_state_mib": mib(total_bytes),
        })

scaling_path = RESULT_DIR / "cache_scaling_comparison.csv"

with open(scaling_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=scaling_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(scaling_rows)

# ============================================================
# JSON summary
# ============================================================

summary = {
    "comparison_scope": (
        "Architecture-derived useful inference state. "
        "Physical serving-engine allocation may differ because "
        "of cache pages, alignment, padding, fragmentation, "
        "checkpointing and allocator overhead."
    ),
    "models": models,
    "important_caveats": [
        (
            "Qwen GDN value represents real recurrent-state bytes, "
            "not its padded 544-token vLLM page."
        ),
        (
            "Falcon uses the FP16-conv + FP32-SSM analytical "
            "scenario for this comparison. Actual serving dtype "
            "must be confirmed experimentally."
        ),
        (
            "Kimi KDA is bounded state while MLA is compressed "
            "token-growing cache."
        ),
        (
            "These values compare useful state rather than total "
            "GPU memory consumption."
        ),
    ],
}

with open(
    RESULT_DIR / "characterization_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)

# ============================================================
# Plot 1 — Total useful state vs context
# ============================================================

plt.figure(figsize=(9, 6))

for model in models:
    xs = []
    ys = []

    for row in scaling_rows:
        if row["model"] == model["model"]:
            xs.append(row["context_tokens"])
            ys.append(row["total_useful_state_mib"])

    plt.plot(
        xs,
        ys,
        marker="o",
        label=model["model"]
    )

plt.xscale("log", base=2)

plt.xlabel("Context Length (tokens)")
plt.ylabel("Derived Useful Inference State (MiB)")
plt.title("Hybrid Cache Growth vs Context Length")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "cache_growth_by_context.png",
    dpi=200
)

plt.close()

# ============================================================
# Plot 2 — bounded state
# ============================================================

plt.figure(figsize=(9, 6))

names = [m["model"] for m in models]
bounded = [m["bounded_state_mib"] for m in models]

plt.bar(names, bounded)

plt.ylabel("Bounded Recurrent State (MiB)")
plt.title("Architecture-Derived Bounded State")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "bounded_state_comparison.png",
    dpi=200
)

plt.close()

# ============================================================
# Plot 3 — growing bytes/token
# ============================================================

plt.figure(figsize=(9, 6))

growing = [
    m["growing_kib_per_token"]
    for m in models
]

plt.bar(names, growing)

plt.ylabel("Context-Growing Cache (KiB/token)")
plt.title("Token-Dependent Cache Growth Rate")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "cache_growth_rate_comparison.png",
    dpi=200
)

plt.close()

# ============================================================
# Console report
# ============================================================

print("===== Hybrid Architecture Comparison =====")
print()

for m in models:
    print(m["model"])
    print(f"  Architecture : {m['architecture']}")
    print(
        f"  Bounded state: "
        f"{m['bounded_state_mib']:.3f} MiB"
    )
    print(
        f"  Growth rate  : "
        f"{m['growing_kib_per_token']:.3f} KiB/token"
    )
    print()

print("Saved:")
print(f"  {comparison_path}")
print(f"  {scaling_path}")
print(
    f"  {RESULT_DIR / 'characterization_summary.json'}"
)
print(
    f"  {PLOT_DIR / 'cache_growth_by_context.png'}"
)
print(
    f"  {PLOT_DIR / 'bounded_state_comparison.png'}"
)
print(
    f"  {PLOT_DIR / 'cache_growth_rate_comparison.png'}"
)
