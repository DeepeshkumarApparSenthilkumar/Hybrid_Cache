#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path


# Qwen3.5-0.8B / TP=1 values derived from the model architecture
# and the vLLM hybrid-cache configuration used in our experiment.
GDN_USEFUL_BYTES = 1_085_440       # 1060 KiB
FULL_ATTN_BYTES_PER_TOKEN = 2_048  # 2 KiB
ALIGNED_TOKENS = 544
ALLOCATED_PAGE_BYTES = ALIGNED_TOKENS * FULL_ATTN_BYTES_PER_TOKEN


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0

    k = (len(values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo

    return values[lo] * (1 - frac) + values[hi] * frac


def checkpoint_position(num_tokens, hash_block_size):
    """
    Mirrors vLLM get_mamba_prefill_checkpoint_position()
    for the non-EAGLE case:

        (num_tokens - 1) // hash_block_size * hash_block_size
    """
    return max(
        ((num_tokens - 1) // hash_block_size) * hash_block_size,
        0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/real_workload/qwen35_vllm/sharegpt_alignment.csv",
    )

    parser.add_argument(
        "--output",
        default="results/real_workload/qwen35_vllm/sharegpt_checkpoint_analysis.csv",
    )

    parser.add_argument(
        "--hash-block-size",
        type=int,
        default=544,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    padding_bytes = ALLOCATED_PAGE_BYTES - GDN_USEFUL_BYTES
    padding_pct = 100.0 * padding_bytes / GDN_USEFUL_BYTES

    results = []

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            prompt_tokens = int(row["prompt_tokens"])

            cp = checkpoint_position(
                prompt_tokens,
                args.hash_block_size,
            )

            has_checkpoint = cp > 0
            tail_after_checkpoint = prompt_tokens - cp if has_checkpoint else prompt_tokens

            results.append(
                {
                    "request_index": int(row["request_index"]),
                    "conversation_id": row["conversation_id"],
                    "turn_id": int(row["turn_id"]),
                    "message_count": int(row["message_count"]),
                    "prompt_tokens": prompt_tokens,
                    "hash_block_size": args.hash_block_size,
                    "checkpoint_position": cp,
                    "has_reusable_checkpoint": has_checkpoint,
                    "tokens_after_checkpoint": tail_after_checkpoint,
                    "checkpoint_fraction_pct": (
                        round(100.0 * cp / prompt_tokens, 4)
                        if has_checkpoint
                        else 0.0
                    ),
                    "gdn_useful_state_bytes": GDN_USEFUL_BYTES,
                    "gdn_useful_state_kib": GDN_USEFUL_BYTES / 1024,
                    "allocated_page_bytes": ALLOCATED_PAGE_BYTES,
                    "allocated_page_kib": ALLOCATED_PAGE_BYTES / 1024,
                    "padding_bytes": padding_bytes,
                    "padding_kib": padding_bytes / 1024,
                    "padding_pct_of_useful_state": padding_pct,
                }
            )

    if not results:
        raise RuntimeError("No input rows found.")

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    prompts = [r["prompt_tokens"] for r in results]
    checkpointed = [r for r in results if r["has_reusable_checkpoint"]]
    tails = [r["tokens_after_checkpoint"] for r in checkpointed]
    fractions = [r["checkpoint_fraction_pct"] for r in checkpointed]

    print("=== QWEN3.5 REAL-WORKLOAD SSM CHECKPOINT ANALYSIS ===")
    print()
    print(f"Requests: {len(results)}")
    print(f"Hash/checkpoint boundary: {args.hash_block_size} tokens")
    print()

    print("=== PROMPT LENGTH ===")
    print(f"Median: {statistics.median(prompts):.2f}")
    print(f"Mean:   {statistics.mean(prompts):.2f}")
    print(f"P90:    {percentile(prompts, 0.90):.2f}")
    print(f"Max:    {max(prompts)}")
    print()

    print("=== REUSABLE PREFILL CHECKPOINT ===")
    print(
        f"Requests with checkpoint: "
        f"{len(checkpointed)}/{len(results)} "
        f"({100.0 * len(checkpointed) / len(results):.2f}%)"
    )

    if checkpointed:
        print(
            f"Mean checkpoint coverage: "
            f"{statistics.mean(fractions):.2f}%"
        )
        print(
            f"Median checkpoint coverage: "
            f"{statistics.median(fractions):.2f}%"
        )
        print(
            f"Mean tokens after checkpoint: "
            f"{statistics.mean(tails):.2f}"
        )
        print(
            f"Median tokens after checkpoint: "
            f"{statistics.median(tails):.2f}"
        )
        print(
            f"P90 tokens after checkpoint: "
            f"{percentile(tails, 0.90):.2f}"
        )

    print()
    print("=== QWEN GDN STATE PAGE ===")
    print(
        f"Useful state:    "
        f"{GDN_USEFUL_BYTES:,} bytes "
        f"({GDN_USEFUL_BYTES / 1024:.2f} KiB)"
    )
    print(
        f"Allocated page:  "
        f"{ALLOCATED_PAGE_BYTES:,} bytes "
        f"({ALLOCATED_PAGE_BYTES / 1024:.2f} KiB)"
    )
    print(
        f"Padding:         "
        f"{padding_bytes:,} bytes "
        f"({padding_bytes / 1024:.2f} KiB)"
    )
    print(
        f"Padding/useful:  "
        f"{padding_pct:.4f}%"
    )

    print()
    print(f"Saved: {output_path}")
    print()
    print(
        "NOTE: Checkpoint positions are source-derived predictions for the "
        "non-EAGLE case. Runtime instrumentation is still required to validate "
        "actual checkpoint materialization."
    )


if __name__ == "__main__":
    main()
