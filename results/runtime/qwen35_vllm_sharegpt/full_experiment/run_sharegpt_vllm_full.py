import json
import csv
import time
import urllib.request
import urllib.error
from pathlib import Path

INPUT = Path("workloads/sharegpt_multiturn_100.jsonl")
OUTPUT = Path("results/sharegpt_runtime_full.csv")
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-0.8B"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def convert_messages(prompt):
    messages = []
    for msg in prompt:
        role = msg["role"]

        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        messages.append({
            "role": role,
            "content": msg["content"]
        })

    return messages


def send_request(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0
    }

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        elapsed = time.perf_counter() - start

        usage = result.get("usage", {})

        return {
            "status": "ok",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "e2e_seconds": elapsed,
            "error": ""
        }

    except Exception as e:
        elapsed = time.perf_counter() - start

        return {
            "status": "error",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "e2e_seconds": elapsed,
            "error": str(e)
        }


rows = []

with INPUT.open("r", encoding="utf-8") as f:
    for request_index, line in enumerate(f):

        if request_index >= 466:
            break

        item = json.loads(line)
        messages = convert_messages(item["prompt"])

        print(
            f"[{request_index + 1}/466] "
            f"conversation={item['conversation_id']} "
            f"turn={item['turn_id']} "
            f"messages={item['message_count']}"
        )

        result = send_request(messages)

        row = {
            "request_index": request_index,
            "conversation_id": item["conversation_id"],
            "turn_id": item["turn_id"],
            "message_count": item["message_count"],
            **result
        }

        rows.append(row)

        print(
            f"    status={result['status']} "
            f"prompt_tokens={result['prompt_tokens']} "
            f"completion_tokens={result['completion_tokens']} "
            f"e2e={result['e2e_seconds']:.4f}s"
        )

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "request_index",
            "conversation_id",
            "turn_id",
            "message_count",
            "status",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "e2e_seconds",
            "error"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved: {OUTPUT}")
print(f"Requests completed: {len(rows)}")
