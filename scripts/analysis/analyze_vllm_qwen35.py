#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "hybrid-kv-cache"

SUMMARY_FILE = (
    BASE
    / "results/vllm/qwen35_rtx6000/results"
    / "vllm_qwen35_repeated_summary.csv"
)

OUTPUT_DIR = BASE / "results/analysis/vllm"
PLOT_DIR = BASE / "plots/vllm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 544

# ---------------------------------------------------------
# Load experiment summary
# ---------------------------------------------------------

df = pd.read_csv(SUMMARY_FILE)

print("Loaded:", SUMMARY_FILE)
print("Rows:", len(df))
print("\nColumns:")
for col in df.columns:
    print(" -", col)

# ---------------------------------------------------------
# Separate cold and reuse measurements
# ---------------------------------------------------------

cold = (
    df[df["request_type"] == "cold"]
    .set_index("target_prefix_tokens")
)

reuse = (
    df[df["request_type"] == "reuse"]
    .set_index("target_prefix_tokens")
)

prefixes = sorted(set(cold.index) & set(reuse.index))

rows = []

for prefix in prefixes:

    cold_row = cold.loc[prefix]
    reuse_row = reuse.loc[prefix]

    cold_ttft = float(cold_row["ttft_mean_s"])
    reuse_ttft = float(reuse_row["ttft_mean_s"])

    cold_e2e = float(cold_row["e2e_mean_s"])
    reuse_e2e = float(reuse_row["e2e_mean_s"])

    cached_tokens = float(reuse_row["cached_prompt_tokens_mean"])

    ttft_reduction = (
        (cold_ttft - reuse_ttft) / cold_ttft * 100
    )

    e2e_reduction = (
        (cold_e2e - reuse_e2e) / cold_e2e * 100
    )

    reusable_full_blocks = int(prefix // BLOCK_SIZE)

    theoretical_full_block_tokens = (
        reusable_full_blocks * BLOCK_SIZE
    )

    partial_block_tokens = (
        prefix - theoretical_full_block_tokens
    )

    observed_reuse_pct = (
        cached_tokens / prefix * 100
    )

    unreused_tokens = (
        prefix - cached_tokens
    )

    aligned = (
        prefix % BLOCK_SIZE == 0
    )

    rows.append(
        {
            "prefix_tokens": prefix,
            "block_size": BLOCK_SIZE,
            "aligned_to_block": aligned,
            "full_blocks": reusable_full_blocks,
            "partial_block_tokens": partial_block_tokens,
            "theoretical_full_block_tokens":
                theoretical_full_block_tokens,
            "observed_cached_tokens": cached_tokens,
            "unreused_tokens": unreused_tokens,
            "observed_reuse_pct": observed_reuse_pct,
            "cold_ttft_s": cold_ttft,
            "reuse_ttft_s": reuse_ttft,
            "ttft_reduction_pct": ttft_reduction,
            "cold_e2e_s": cold_e2e,
            "reuse_e2e_s": reuse_e2e,
            "e2e_reduction_pct": e2e_reduction,
        }
    )

analysis = pd.DataFrame(rows)

# ---------------------------------------------------------
# Save numerical analysis
# ---------------------------------------------------------

output_csv = OUTPUT_DIR / "vllm_prefix_reuse_analysis.csv"

analysis.to_csv(output_csv, index=False)

print("\n===== PREFIX REUSE ANALYSIS =====")
print(analysis.round(4).to_string(index=False))

print("\nSaved:")
print(output_csv)

# ---------------------------------------------------------
# Plot 1 — Cold vs reuse TTFT
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    analysis["prefix_tokens"],
    analysis["cold_ttft_s"],
    marker="o",
    label="Cold"
)

plt.plot(
    analysis["prefix_tokens"],
    analysis["reuse_ttft_s"],
    marker="o",
    label="Prefix reuse"
)

plt.xlabel("Target Prefix Length (tokens)")
plt.ylabel("Mean TTFT (seconds)")
plt.title("Qwen3.5-0.8B — vLLM Prefix Reuse TTFT")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "ttft_cold_vs_reuse.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# Plot 2 — TTFT reduction
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    analysis["prefix_tokens"].astype(str),
    analysis["ttft_reduction_pct"]
)

plt.xlabel("Target Prefix Length (tokens)")
plt.ylabel("TTFT Reduction (%)")
plt.title("TTFT Improvement from Prefix Cache Reuse")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "ttft_reduction_pct.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# Plot 3 — Cached-token percentage
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    analysis["prefix_tokens"].astype(str),
    analysis["observed_reuse_pct"]
)

plt.axhline(
    100,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Target Prefix Length (tokens)")
plt.ylabel("Observed Prefix Reuse (%)")
plt.title("Observed Cached Prefix Percentage")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "prefix_reuse_percentage.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# Plot 4 — Expected full-block reuse vs observed cache
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    analysis["prefix_tokens"],
    analysis["theoretical_full_block_tokens"],
    marker="o",
    label="Complete 544-token blocks"
)

plt.plot(
    analysis["prefix_tokens"],
    analysis["observed_cached_tokens"],
    marker="x",
    label="Observed cached tokens"
)

plt.xlabel("Target Prefix Length (tokens)")
plt.ylabel("Reusable/Cached Tokens")
plt.title("vLLM Block Granularity vs Observed Prefix Reuse")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "block_granularity_validation.png",
    dpi=300
)

plt.close()

print("\nGenerated plots:")

for path in sorted(PLOT_DIR.glob("*.png")):
    print(" -", path)
