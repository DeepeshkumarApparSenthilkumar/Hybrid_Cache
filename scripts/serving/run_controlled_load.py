#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import statistics
import time
import urllib.request
from pathlib import Path

import aiohttp


DEFAULT_INPUT = Path("workloads/sharegpt/sharegpt_multiturn_100.jsonl")

MODEL_CONFIGS = {
    "qwen": {
        "model": "Qwen/Qwen3.5-0.8B",
        "endpoint": "/v1/chat/completions",
        "mode": "chat",
    },
    "falcon": {
        "model": "tiiuae/Falcon-H1-0.5B-Base",
        "endpoint": "/v1/completions",
        "mode": "completion",
    },
}


def convert_qwen_messages(prompt):
    messages = []

    for msg in prompt:
        role = msg["role"]

        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        messages.append({
            "role": role,
            "content": msg["content"],
        })

    return messages


def convert_falcon_prompt(prompt):
    parts = []

    for msg in prompt:
        role = msg["role"]
        content = msg["content"]

        if role == "human":
            parts.append(f"User: {content}")
        elif role == "gpt":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role}: {content}")

    return "\n\n".join(parts) + "\n\nAssistant:"


def build_payload(item, config, max_tokens):
    if config["mode"] == "chat":
        return {
            "model": config["model"],
            "messages": convert_qwen_messages(item["prompt"]),
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }

    return {
        "model": config["model"],
        "prompt": convert_falcon_prompt(item["prompt"]),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }


def read_workload(path, limit):
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(items) >= limit:
                break

            items.append(json.loads(line))

    return items


def fetch_metrics(base_url):
    url = base_url.rstrip("/") + "/metrics"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read().decode("utf-8")
    except Exception as exc:
        return f"# METRICS_FETCH_ERROR {exc}\n"


async def stream_request(
    session,
    item,
    request_index,
    config,
    base_url,
    semaphore,
    max_tokens,
    timeout,
    experiment_start,
):
    url = base_url.rstrip("/") + config["endpoint"]
    payload = build_payload(item, config, max_tokens)

    submitted = time.perf_counter()
    submitted_offset = submitted - experiment_start

    async with semaphore:
        request_start = time.perf_counter()
        queue_seconds = request_start - submitted

        first_token_time = None
        last_token_time = None

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        generated_text = ""
        chunk_count = 0
        error = ""
        status = "error"

        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)

            async with session.post(
                url,
                json=payload,
                timeout=client_timeout,
            ) as response:

                if response.status != 200:
                    body = await response.text()

                    raise RuntimeError(
                        f"HTTP {response.status}: {body[:500]}"
                    )

                buffer = ""

                async for chunk_bytes in response.content.iter_any():
                    if not chunk_bytes:
                        continue

                    buffer += chunk_bytes.decode(
                        "utf-8",
                        errors="replace",
                    )

                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)

                        for line in message.splitlines():
                            line = line.strip()

                            if not line:
                                continue

                            # SSE comments / keep-alive pings
                            if line.startswith(":"):
                                continue

                            if not line.startswith("data:"):
                                continue

                            chunk = line.removeprefix("data:").strip()

                            if chunk == "[DONE]":
                                continue

                            if not chunk:
                                continue

                            data = json.loads(chunk)

                            choices = data.get("choices")

                            if choices:
                                timestamp = time.perf_counter()

                                if first_token_time is None:
                                    first_token_time = timestamp

                                last_token_time = timestamp
                                chunk_count += 1

                                if config["mode"] == "chat":
                                    delta = choices[0].get(
                                        "delta",
                                        {},
                                    )

                                    content = delta.get("content")

                                else:
                                    content = choices[0].get("text")

                                generated_text += content or ""

                            usage = data.get("usage")

                            if usage:
                                prompt_tokens = usage.get(
                                    "prompt_tokens",
                                    prompt_tokens,
                                )

                                completion_tokens = usage.get(
                                    "completion_tokens",
                                    completion_tokens,
                                )

                                total_tokens = usage.get(
                                    "total_tokens",
                                    total_tokens,
                                )

                if first_token_time is None:
                    raise RuntimeError(
                        "No generated token chunk received; "
                        "TTFT cannot be calculated."
                    )

                status = "ok"

        except Exception as exc:
            error = str(exc)

        request_end = time.perf_counter()

        if first_token_time is not None:
            ttft_seconds = first_token_time - request_start
        else:
            ttft_seconds = None

        if last_token_time is not None:
            e2e_seconds = last_token_time - request_start
        else:
            e2e_seconds = request_end - request_start

        # Match vLLM benchmark semantics:
        # TPOT = (latency - TTFT) / (output_tokens - 1)
        if (
            status == "ok"
            and ttft_seconds is not None
            and completion_tokens is not None
            and completion_tokens > 1
        ):
            tpot_seconds = (
                e2e_seconds - ttft_seconds
            ) / (completion_tokens - 1)
        else:
            tpot_seconds = None

        return {
            "request_index": request_index,
            "conversation_id": item["conversation_id"],
            "turn_id": item["turn_id"],
            "message_count": item["message_count"],
            "submitted_offset_seconds": submitted_offset,
            "client_queue_seconds": queue_seconds,
            "status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stream_chunks": chunk_count,
            "ttft_seconds": ttft_seconds,
            "tpot_seconds": tpot_seconds,
            "e2e_seconds": e2e_seconds,
            "finished_offset_seconds": (
                request_end - experiment_start
            ),
            "generated_text_chars": len(generated_text),
            "error": error,
        }


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def summarize_metric(rows, key):
    values = [
        row[key]
        for row in rows
        if row["status"] == "ok"
        and row[key] is not None
    ]

    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


async def main():
    parser = argparse.ArgumentParser(
        description=(
            "Controlled concurrent serving benchmark for "
            "Qwen3.5 and Falcon-H1 on vLLM."
        )
    )

    parser.add_argument(
        "--model",
        choices=["qwen", "falcon"],
        required=True,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=466,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/runtime/controlled_load"
        ),
    )

    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError(
            "--concurrency must be greater than zero"
        )

    if args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero"
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Workload not found: {args.input}"
        )

    config = MODEL_CONFIGS[args.model]

    workload = read_workload(
        args.input,
        args.limit,
    )

    run_dir = (
        args.output_root
        / args.model
        / f"c{args.concurrency:02d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("CONTROLLED vLLM SERVING EXPERIMENT")
    print("=" * 72)
    print(f"Model       : {config['model']}")
    print(f"Endpoint    : {config['endpoint']}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Requests    : {len(workload)}")
    print(f"Input       : {args.input}")
    print(f"Output      : {run_dir}")
    print("=" * 72)

    metrics_before = fetch_metrics(
        args.base_url
    )

    (
        run_dir / "metrics_before.prom"
    ).write_text(
        metrics_before,
        encoding="utf-8",
    )

    semaphore = asyncio.Semaphore(
        args.concurrency
    )

    experiment_start = time.perf_counter()

    connector = aiohttp.TCPConnector(
        limit=0,
    )

    async with aiohttp.ClientSession(
        connector=connector
    ) as session:

        # Group growing ShareGPT requests by conversation.
        #
        # Requests within one conversation must execute sequentially:
        # turn 1 -> turn 2 -> turn 3 ...
        #
        # Different conversations may execute concurrently. This preserves
        # the prefix-reuse dependency while still applying controlled load.
        conversations = {}

        for request_index, item in enumerate(workload):
            conversation_id = item["conversation_id"]

            conversations.setdefault(
                conversation_id,
                [],
            ).append(
                (request_index, item)
            )

        # Make turn ordering explicit even if the workload file changes.
        for conversation_items in conversations.values():
            conversation_items.sort(
                key=lambda pair: pair[1]["turn_id"]
            )

        rows = []
        progress_lock = asyncio.Lock()

        async def run_conversation(conversation_items):
            conversation_rows = []

            for request_index, item in conversation_items:
                row = await stream_request(
                    session=session,
                    item=item,
                    request_index=request_index,
                    config=config,
                    base_url=args.base_url,
                    semaphore=semaphore,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    experiment_start=experiment_start,
                )

                conversation_rows.append(row)

                ttft = row["ttft_seconds"]
                tpot = row["tpot_seconds"]

                ttft_text = (
                    f"{ttft * 1000:.2f}ms"
                    if ttft is not None
                    else "NA"
                )

                tpot_text = (
                    f"{tpot * 1000:.2f}ms"
                    if tpot is not None
                    else "NA"
                )

                async with progress_lock:
                    rows.append(row)

                    print(
                        f"[{len(rows)}/{len(workload)}] "
                        f"req={row['request_index']} "
                        f"conv={row['conversation_id']} "
                        f"turn={row['turn_id']} "
                        f"status={row['status']} "
                        f"TTFT={ttft_text} "
                        f"TPOT={tpot_text} "
                        f"E2E={row['e2e_seconds']:.4f}s"
                    )

            return conversation_rows

        conversation_tasks = [
            asyncio.create_task(
                run_conversation(conversation_items)
            )
            for conversation_items in conversations.values()
        ]

        await asyncio.gather(*conversation_tasks)

    experiment_seconds = (
        time.perf_counter()
        - experiment_start
    )

    metrics_after = fetch_metrics(
        args.base_url
    )

    (
        run_dir / "metrics_after.prom"
    ).write_text(
        metrics_after,
        encoding="utf-8",
    )

    rows.sort(
        key=lambda row: row["request_index"]
    )

    csv_path = run_dir / "requests.csv"

    fieldnames = [
        "request_index",
        "conversation_id",
        "turn_id",
        "message_count",
        "submitted_offset_seconds",
        "client_queue_seconds",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "stream_chunks",
        "ttft_seconds",
        "tpot_seconds",
        "e2e_seconds",
        "finished_offset_seconds",
        "generated_text_chars",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    successes = len(successful_rows)
    failures = len(rows) - successes

    total_prompt_tokens = sum(
        row["prompt_tokens"] or 0
        for row in successful_rows
    )

    total_completion_tokens = sum(
        row["completion_tokens"] or 0
        for row in successful_rows
    )

    summary = {
        "model_key": args.model,
        "model": config["model"],
        "endpoint": config["endpoint"],
        "concurrency": args.concurrency,
        "requests": len(rows),
        "successes": successes,
        "failures": failures,
        "experiment_seconds": experiment_seconds,
        "request_throughput_rps": (
            successes / experiment_seconds
            if experiment_seconds > 0
            else None
        ),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "prompt_token_throughput_tps": (
            total_prompt_tokens / experiment_seconds
            if experiment_seconds > 0
            else None
        ),
        "output_token_throughput_tps": (
            total_completion_tokens / experiment_seconds
            if experiment_seconds > 0
            else None
        ),
        "ttft_seconds": summarize_metric(
            successful_rows,
            "ttft_seconds",
        ),
        "tpot_seconds": summarize_metric(
            successful_rows,
            "tpot_seconds",
        ),
        "e2e_seconds": summarize_metric(
            successful_rows,
            "e2e_seconds",
        ),
        "client_queue_seconds": summarize_metric(
            successful_rows,
            "client_queue_seconds",
        ),
        "max_tokens": args.max_tokens,
    }

    (
        run_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)
    print(f"Successes : {successes}")
    print(f"Failures  : {failures}")
    print(
        f"Duration  : "
        f"{experiment_seconds:.3f}s"
    )

    if experiment_seconds > 0:
        print(
            "Throughput: "
            f"{successes / experiment_seconds:.3f} req/s"
        )

    if successful_rows:
        ttft_summary = summary["ttft_seconds"]
        tpot_summary = summary["tpot_seconds"]

        if ttft_summary["median"] is not None:
            print(
                "Median TTFT: "
                f"{ttft_summary['median'] * 1000:.2f} ms"
            )

        if tpot_summary["median"] is not None:
            print(
                "Median TPOT: "
                f"{tpot_summary['median'] * 1000:.2f} ms/token"
            )

    print(f"Saved: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
