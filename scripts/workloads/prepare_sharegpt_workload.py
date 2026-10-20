#!/usr/bin/env python3

"""
Prepare a reproducible real-world multi-turn workload from ShareGPT.

This script does NOT run vLLM and does NOT make assumptions about actual
SSM checkpoint behavior. It only converts real ShareGPT conversations into
growing multi-turn requests for later tokenization and serving experiments.
"""

import argparse
import csv
import json
import random
from pathlib import Path


def valid_messages(messages):
    """Keep a clean alternating human/GPT prefix."""
    cleaned = []
    expected = "human"

    for msg in messages:
        role = msg.get("from")
        value = msg.get("value", "")

        if role not in {"human", "gpt"}:
            continue

        if not isinstance(value, str) or not value.strip():
            continue

        # Stop when the normal human -> gpt alternation breaks.
        if role != expected:
            break

        cleaned.append({
            "role": role,
            "content": value.strip()
        })

        expected = "gpt" if expected == "human" else "human"

    return cleaned


def build_requests(conversation_id, messages, max_turns):
    """
    Create growing prompts.

    Request for turn N contains all completed previous turns plus the
    current human message, but not the current GPT answer.
    """
    requests = []

    human_turn = 0

    for i, msg in enumerate(messages):
        if msg["role"] != "human":
            continue

        human_turn += 1

        if human_turn > max_turns:
            break

        history = messages[:i + 1]

        requests.append({
            "conversation_id": conversation_id,
            "turn_id": human_turn,
            "message_count": len(history),
            "prompt": history
        })

    return requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--num-conversations", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    jsonl_path = Path(args.output_jsonl)
    csv_path = Path(args.output_csv)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    eligible = []

    for index, record in enumerate(dataset):
        messages = valid_messages(record.get("conversations", []))

        # Require at least two human turns:
        # human -> gpt -> human
        human_count = sum(m["role"] == "human" for m in messages)

        if human_count >= 2:
            conversation_id = record.get("id", f"sharegpt_{index}")
            eligible.append((conversation_id, messages))

    print(f"Total dataset conversations: {len(dataset)}")
    print(f"Eligible multi-turn conversations: {len(eligible)}")

    sample_size = min(args.num_conversations, len(eligible))
    sampled = random.sample(eligible, sample_size)

    all_requests = []

    for conversation_id, messages in sampled:
        all_requests.extend(
            build_requests(
                conversation_id,
                messages,
                args.max_turns
            )
        )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for request in all_requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "turn_id",
                "message_count",
                "character_count"
            ]
        )

        writer.writeheader()

        for request in all_requests:
            character_count = sum(
                len(message["content"])
                for message in request["prompt"]
            )

            writer.writerow({
                "conversation_id": request["conversation_id"],
                "turn_id": request["turn_id"],
                "message_count": request["message_count"],
                "character_count": character_count
            })

    print()
    print("=== WORKLOAD CREATED ===")
    print(f"Sampled conversations: {sample_size}")
    print(f"Generated requests: {len(all_requests)}")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
