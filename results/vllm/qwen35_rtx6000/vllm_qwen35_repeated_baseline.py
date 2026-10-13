#!/usr/bin/env python3
"""
Robust vLLM cache-reuse characterization for Qwen3.5-0.8B.

Design:
- 2 untimed warm-up requests
- 5 trials per prefix length
- aligned and non-aligned prefix lengths:
  1024, 1088, 2048, 2176, 4096, 4352
- each trial uses a unique prefix so the first request is cold
- second request reuses the exact same long prefix with a new suffix
- records TTFT, E2E latency, prefix query/hit counters, cached tokens,
  post-request KV usage, and GPU memory

Outputs:
- results/vllm_qwen35_repeated_raw.csv
- results/vllm_qwen35_repeated_summary.csv
"""

import csv
import json
import re
import statistics as stats
import subprocess
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3.5-0.8B"
PREFIX_LENGTHS = [1024, 1088, 2048, 2176, 4096, 4352]
TRIALS = 5
WARMUPS = 2

OUT_RAW = Path("results/vllm_qwen35_repeated_raw.csv")
OUT_SUM = Path("results/vllm_qwen35_repeated_summary.csv")
OUT_RAW.parent.mkdir(exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)

METRICS = {
    "kv_usage": "vllm:kv_cache_usage_perc",
    "prefix_queries": "vllm:prefix_cache_queries_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "cached_prompt_tokens": "vllm:prompt_tokens_cached_total",
}

def prom_metric(text, name):
    m = re.search(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",
        text,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None

def snapshot():
    r = requests.get(BASE_URL + "/metrics", timeout=15)
    r.raise_for_status()
    return {k: prom_metric(r.text, v) for k, v in METRICS.items()}

def gpu_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return float(out)

def make_prefix(target_tokens, tag):
    base = (
        f"{tag}. Controlled benchmark text for hybrid LLM cache characterization. "
        "This passage is intentionally repetitive so token length can be controlled. "
    )
    text = base
    while len(tok.encode(text, add_special_tokens=False)) < target_tokens:
        text += (
            "The serving engine manages attention KV blocks and recurrent state. "
            "A second request will reuse the same long prefix with a different short suffix. "
        )
    ids = tok.encode(text, add_special_tokens=False)[:target_tokens]
    return tok.decode(ids, skip_special_tokens=True)

def run_stream(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    first_token = None
    usage = {}

    with requests.post(
        BASE_URL + "/v1/completions",
        json=payload,
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                if choice.get("text"):
                    if first_token is None:
                        first_token = time.perf_counter()

    t1 = time.perf_counter()
    return {
        "ttft_s": None if first_token is None else first_token - t0,
        "e2e_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }

def delta(a, b, k):
    if a.get(k) is None or b.get(k) is None:
        return None
    return a[k] - b[k]

print("Running warm-up requests...")
for i in range(WARMUPS):
    p = make_prefix(512, f"WARMUP_{time.time_ns()}_{i}")
    run_stream(p + "\nWarm-up question: answer briefly.")
print("Warm-up complete.\n")

rows = []

for n in PREFIX_LENGTHS:
    for trial in range(1, TRIALS + 1):
        tag = f"CTX_{n}_TRIAL_{trial}_{time.time_ns()}"
        prefix = make_prefix(n, tag)

        cases = [
            ("cold", prefix + "\n\nQuestion A: What is this benchmark measuring?"),
            ("reuse", prefix + "\n\nQuestion B: Why can shared-prefix reuse reduce prefill work?"),
        ]

        for kind, prompt in cases:
            before = snapshot()
            gpu_before = gpu_mib()
            result = run_stream(prompt)
            time.sleep(0.25)
            after = snapshot()
            gpu_after = gpu_mib()

            q = delta(after, before, "prefix_queries")
            h = delta(after, before, "prefix_hits")
            c = delta(after, before, "cached_prompt_tokens")
            hit_rate = None if not q else h / q

            row = {
                "target_prefix_tokens": n,
                "trial": trial,
                "request_type": kind,
                "actual_prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "ttft_s": result["ttft_s"],
                "e2e_latency_s": result["e2e_s"],
                "prefix_queries_delta_tokens": q,
                "prefix_hits_delta_tokens": h,
                "prefix_hit_rate": hit_rate,
                "cached_prompt_tokens_delta": c,
                "kv_usage_before": before["kv_usage"],
                "kv_usage_after": after["kv_usage"],
                "gpu_memory_before_mib": gpu_before,
                "gpu_memory_after_mib": gpu_after,
            }
            rows.append(row)
            print(
                f"{n:4d} | trial={trial} | {kind:5s} | "
                f"TTFT={result['ttft_s']:.4f}s | E2E={result['e2e_s']:.4f}s | "
                f"hits={h} / queries={q} | hit_rate={hit_rate}"
            )

with OUT_RAW.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = []
for n in PREFIX_LENGTHS:
    for kind in ("cold", "reuse"):
        subset = [r for r in rows if r["target_prefix_tokens"] == n and r["request_type"] == kind]
        ttfts = [r["ttft_s"] for r in subset if r["ttft_s"] is not None]
        e2es = [r["e2e_latency_s"] for r in subset if r["e2e_latency_s"] is not None]
        hrs = [r["prefix_hit_rate"] for r in subset if r["prefix_hit_rate"] is not None]

        summary.append({
            "target_prefix_tokens": n,
            "request_type": kind,
            "n_trials": len(subset),
            "ttft_mean_s": stats.mean(ttfts),
            "ttft_stdev_s": stats.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            "ttft_median_s": stats.median(ttfts),
            "e2e_mean_s": stats.mean(e2es),
            "e2e_stdev_s": stats.stdev(e2es) if len(e2es) > 1 else 0.0,
            "e2e_median_s": stats.median(e2es),
            "prefix_hit_rate_mean": stats.mean(hrs) if hrs else None,
            "prefix_hits_mean_tokens": stats.mean(
                [r["prefix_hits_delta_tokens"] for r in subset if r["prefix_hits_delta_tokens"] is not None]
            ),
            "cached_prompt_tokens_mean": stats.mean(
                [r["cached_prompt_tokens_delta"] for r in subset if r["cached_prompt_tokens_delta"] is not None]
            ),
        })

with OUT_SUM.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

print("\nSaved:")
print(" ", OUT_RAW)
print(" ", OUT_SUM)
