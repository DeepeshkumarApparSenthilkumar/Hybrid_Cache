#!/usr/bin/env python3

"""
Tokenize a prepared ShareGPT multi-turn workload with the Qwen3.5 tokenizer
and characterize request lengths relative to a specified cache/alignment
granularity.

IMPORTANT:
These are analytical boundary calculations. They do NOT claim that vLLM
actually materializes an SSM checkpoint at every boundary. Runtime
checkpoint behavior will be measured separately.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def convert_messages(prompt):
    """Convert our ShareGPT representation to Hugging Face chat roles."""
    role_map = {
        "human": "user",
        "gpt": "assistant",
    }

    messages = []

    for message in prompt:
        role = role_map.get(message["role"])
        if role is None:
            continue

        messages.append({
            "role": role,
            "content": message["content"],
        })

    return messages


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] * (1 - fraction) + values[upper] * fraction


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
    )

    parser.add_argument(
        "--boundary",
        type=int,
        default=544,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("Tokenizer loaded.")
    print(f"Analytical boundary: {args.boundary} tokens")
    print()

    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for request_index, line in enumerate(f, start=1):
            request = json.loads(line)

            messages = convert_messages(request["prompt"])

            # Apply the model's actual chat template.
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            # Qwen3.5 / Transformers returns a BatchEncoding object.
            # Count the actual input token IDs, not the dictionary keys.
            token_ids = encoded["input_ids"]
            prompt_tokens = len(token_ids)

            boundary = args.boundary

            complete_boundaries = prompt_tokens // boundary
            previous_boundary = complete_boundaries * boundary

            tokens_after_previous_boundary = (
                prompt_tokens - previous_boundary
            )

            if tokens_after_previous_boundary == 0:
                distance_to_next_boundary = 0
            else:
                distance_to_next_boundary = (
                    boundary - tokens_after_previous_boundary
                )

            theoretical_reusable_tokens = previous_boundary

            theoretical_reuse_pct = (
                100.0 * theoretical_reusable_tokens / prompt_tokens
                if prompt_tokens > 0
                else 0.0
            )

            aligned_exactly = (
                tokens_after_previous_boundary == 0
            )

            rows.append({
                "request_index": request_index,
                "conversation_id": request["conversation_id"],
                "turn_id": request["turn_id"],
                "message_count": request["message_count"],
                "prompt_tokens": prompt_tokens,
                "boundary_tokens": boundary,
                "complete_boundaries": complete_boundaries,
                "previous_boundary": previous_boundary,
                "tokens_after_previous_boundary":
                    tokens_after_previous_boundary,
                "distance_to_next_boundary":
                    distance_to_next_boundary,
                "aligned_exactly": aligned_exactly,
                "theoretical_reusable_tokens":
                    theoretical_reusable_tokens,
                "theoretical_reuse_pct":
                    round(theoretical_reuse_pct, 4),
            })

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    prompt_lengths = [
        row["prompt_tokens"]
        for row in rows
    ]

    tails = [
        row["tokens_after_previous_boundary"]
        for row in rows
    ]

    aligned_count = sum(
        row["aligned_exactly"]
        for row in rows
    )

    at_least_one_boundary = sum(
        row["complete_boundaries"] >= 1
        for row in rows
    )

    print("=== REAL WORKLOAD ALIGNMENT SUMMARY ===")
    print(f"Requests: {len(rows)}")

    print()
    print("Prompt tokens:")
    print(f"  min:    {min(prompt_lengths)}")
    print(f"  median: {statistics.median(prompt_lengths):.1f}")
    print(f"  mean:   {statistics.mean(prompt_lengths):.1f}")
    print(f"  p90:    {percentile(prompt_lengths, 0.90):.1f}")
    print(f"  p95:    {percentile(prompt_lengths, 0.95):.1f}")
    print(f"  max:    {max(prompt_lengths)}")

    print()
    print(f"Requests >= {args.boundary} tokens:")
    print(
        f"  {at_least_one_boundary}/{len(rows)} "
        f"({100 * at_least_one_boundary / len(rows):.2f}%)"
    )

    print()
    print("Exactly aligned requests:")
    print(
        f"  {aligned_count}/{len(rows)} "
        f"({100 * aligned_count / len(rows):.2f}%)"
    )

    print()
    print("Tokens after previous boundary:")
    print(f"  mean:   {statistics.mean(tails):.1f}")
    print(f"  median: {statistics.median(tails):.1f}")
    print(f"  p90:    {percentile(tails, 0.90):.1f}")

    print()
    print(f"Saved: {output_path}")

    print()
    print(
        "NOTE: Boundary values above are analytical. "
        "They are not measurements of actual vLLM SSM "
        "checkpoint placement."
    )


if __name__ == "__main__":
    main()
