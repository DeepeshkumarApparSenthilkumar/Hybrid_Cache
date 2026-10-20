#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT = Path(
    "results/real_workload/qwen35_vllm/"
    "sharegpt_checkpoint_analysis.csv"
)

OUTPUT = Path(
    "plots/real_workload/qwen35_vllm"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

prompt_tokens = [int(r["prompt_tokens"]) for r in rows]

checkpoint_rows = [
    r for r in rows
    if r["has_reusable_checkpoint"].lower() == "true"
]

coverage = [
    float(r["checkpoint_fraction_pct"])
    for r in checkpoint_rows
]

tails = [
    int(r["tokens_after_checkpoint"])
    for r in checkpoint_rows
]


# --------------------------------------------------
# Figure 1: Prompt length distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    prompt_tokens,
    bins=30,
    edgecolor="black",
    alpha=0.8,
)

for boundary in [544, 1088, 1632]:
    plt.axvline(
        boundary,
        linestyle="--",
        linewidth=1.5,
        label=f"{boundary}-token boundary",
    )

plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Number of Requests")
plt.title(
    "Real ShareGPT Multi-turn Prompt Length Distribution\n"
    "Qwen3.5-0.8B Tokenization"
)
plt.legend()
plt.tight_layout()

path = OUTPUT / "sharegpt_prompt_length_distribution.png"
plt.savefig(path, dpi=300)
plt.close()

print("Saved:", path)


# --------------------------------------------------
# Figure 2: Checkpoint coverage
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    coverage,
    bins=20,
    edgecolor="black",
    alpha=0.8,
)

plt.xlabel("Checkpoint Coverage of Prompt (%)")
plt.ylabel("Number of Requests")
plt.title(
    "Source-derived Reusable SSM Checkpoint Coverage\n"
    "Requests with a Predicted Checkpoint"
)

plt.tight_layout()

path = OUTPUT / "sharegpt_checkpoint_coverage.png"
plt.savefig(path, dpi=300)
plt.close()

print("Saved:", path)


# --------------------------------------------------
# Figure 3: Tail after checkpoint
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    tails,
    bins=20,
    edgecolor="black",
    alpha=0.8,
)

plt.axvline(
    544,
    linestyle="--",
    linewidth=1.5,
    label="544-token cache granularity",
)

plt.xlabel("Tokens After Reusable Checkpoint")
plt.ylabel("Number of Requests")
plt.title(
    "Real-workload Prompt Tail Beyond Reusable SSM Checkpoint"
)

plt.legend()
plt.tight_layout()

path = OUTPUT / "sharegpt_tokens_after_checkpoint.png"
plt.savefig(path, dpi=300)
plt.close()

print("Saved:", path)


print()
print("Generated all real-workload plots.")
