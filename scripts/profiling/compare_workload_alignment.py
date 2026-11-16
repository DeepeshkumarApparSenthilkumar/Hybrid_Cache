#!/usr/bin/env python3

import csv
from pathlib import Path
from statistics import mean, median

INPUT = Path("results/real_workload/qwen35_vllm/sharegpt_alignment.csv")

MODELS = {
    "Qwen3.5-0.8B": {
        "mechanism": "Gated DeltaNet",
        "block_size": 544,
    },
    "Falcon-H1-0.5B": {
        "mechanism": "Mamba-2",
        "block_size": 800,
    },
}


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0

    k = (len(values) - 1) * p
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    return values[lower] + (values[upper] - values[lower]) * (k - lower)


with INPUT.open() as f:
    rows = list(csv.DictReader(f))

prompt_lengths = [int(row["prompt_tokens"]) for row in rows]

print("=" * 95)
print("SHAREGPT HYBRID CACHE ALIGNMENT COMPARISON")
print(f"Workload: {len(prompt_lengths)} requests")
print("=" * 95)

results = {}

for model_name, cfg in MODELS.items():
    block = cfg["block_size"]

    complete_blocks = [length // block for length in prompt_lengths]
    reusable_tokens = [blocks * block for blocks in complete_blocks]
    tails = [length % block for length in prompt_lengths]

    requests_with_boundary = sum(
        blocks >= 1 for blocks in complete_blocks
    )

    requests_with_two_boundaries = sum(
        blocks >= 2 for blocks in complete_blocks
    )

    requests_with_three_boundaries = sum(
        blocks >= 3 for blocks in complete_blocks
    )

    exact_boundary = sum(
        length > 0 and length % block == 0
        for length in prompt_lengths
    )

    total_prompt_tokens = sum(prompt_lengths)
    total_reusable_tokens = sum(reusable_tokens)

    overall_aligned_coverage = (
        100.0 * total_reusable_tokens / total_prompt_tokens
        if total_prompt_tokens
        else 0
    )

    per_request_coverage = [
        100.0 * reusable / length if length else 0
        for reusable, length in zip(reusable_tokens, prompt_lengths)
    ]

    positive_coverage = [
        coverage
        for coverage, blocks in zip(
            per_request_coverage,
            complete_blocks,
        )
        if blocks >= 1
    ]

    result = {
        "block": block,
        "requests_with_boundary": requests_with_boundary,
        "requests_with_two_boundaries": requests_with_two_boundaries,
        "requests_with_three_boundaries": requests_with_three_boundaries,
        "exact_boundary": exact_boundary,
        "total_complete_blocks": sum(complete_blocks),
        "total_prompt_tokens": total_prompt_tokens,
        "total_reusable_tokens": total_reusable_tokens,
        "overall_aligned_coverage": overall_aligned_coverage,
        "mean_tail": mean(tails),
        "median_tail": median(tails),
        "p90_tail": percentile(tails, 0.90),
        "max_tail": max(tails),
        "mean_coverage_positive": (
            mean(positive_coverage) if positive_coverage else 0
        ),
    }

    results[model_name] = result

    print(f"\nModel: {model_name}")
    print(f"Mechanism: {cfg['mechanism']}")
    print(f"Alignment block: {block} tokens")

    print(
        f"Requests reaching >=1 boundary : "
        f"{requests_with_boundary}/{len(prompt_lengths)} "
        f"({100 * requests_with_boundary / len(prompt_lengths):.2f}%)"
    )

    print(
        f"Requests reaching >=2 boundaries: "
        f"{requests_with_two_boundaries}/{len(prompt_lengths)} "
        f"({100 * requests_with_two_boundaries / len(prompt_lengths):.2f}%)"
    )

    print(
        f"Requests reaching >=3 boundaries: "
        f"{requests_with_three_boundaries}/{len(prompt_lengths)} "
        f"({100 * requests_with_three_boundaries / len(prompt_lengths):.2f}%)"
    )

    print(f"Exact boundary requests          : {exact_boundary}")
    print(f"Total complete aligned blocks    : {sum(complete_blocks)}")

    print(
        f"Total prompt tokens              : "
        f"{total_prompt_tokens:,}"
    )

    print(
        f"Aligned reusable-token potential : "
        f"{total_reusable_tokens:,}"
    )

    print(
        f"Overall aligned coverage         : "
        f"{overall_aligned_coverage:.2f}%"
    )

    print(
        f"Mean coverage for >=1 block reqs : "
        f"{result['mean_coverage_positive']:.2f}%"
    )

    print(f"Mean residual tail               : {mean(tails):.2f}")
    print(f"Median residual tail             : {median(tails):.2f}")
    print(f"P90 residual tail                : {percentile(tails, 0.90):.2f}")
    print(f"Maximum residual tail            : {max(tails)}")


q = results["Qwen3.5-0.8B"]
f = results["Falcon-H1-0.5B"]

print("\n" + "=" * 95)
print("DIRECT QWEN vs FALCON DIFFERENCE")
print("=" * 95)

boundary_difference = (
    q["requests_with_boundary"] - f["requests_with_boundary"]
)

block_difference = (
    q["total_complete_blocks"] - f["total_complete_blocks"]
)

reusable_difference = (
    q["total_reusable_tokens"] - f["total_reusable_tokens"]
)

print(
    f"Requests gaining >=1 aligned boundary with 544 vs 800 : "
    f"{boundary_difference}"
)

print(
    f"Additional complete aligned regions with 544 vs 800   : "
    f"{block_difference}"
)

print(
    f"Difference in aligned reusable-token potential         : "
    f"{reusable_difference:,} tokens"
)

print(
    f"Coverage difference                                   : "
    f"{q['overall_aligned_coverage'] - f['overall_aligned_coverage']:.2f} percentage points"
)

print(
    f"Mean-tail difference                                  : "
    f"{f['mean_tail'] - q['mean_tail']:.2f} tokens "
    f"(Falcon - Qwen)"
)

print("\nNOTE:")
print(
    "These values describe theoretical alignment/reuse opportunity "
    "from request lengths. They are NOT measured runtime prefix-cache "
    "hit rates or GPU throughput."
)

print("=" * 95)
