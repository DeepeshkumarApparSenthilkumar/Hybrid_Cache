#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


COUNTER_METRICS = {
    "prefix_cache_queries": [
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
        "vllm_prefix_cache_queries",
        "vllm_prefix_cache_queries_total",
    ],
    "prefix_cache_hits": [
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_hits_total",
        "vllm_prefix_cache_hits",
        "vllm_prefix_cache_hits_total",
    ],
    "prompt_tokens_cached": [
        "vllm:prompt_tokens_cached",
        "vllm:prompt_tokens_cached_total",
        "vllm_prompt_tokens_cached",
        "vllm_prompt_tokens_cached_total",
    ],
}

GAUGE_METRICS = {
    "kv_cache_usage_perc": [
        "vllm:kv_cache_usage_perc",
        "vllm_kv_cache_usage_perc",
    ],
}


def parse_prometheus(path):
    metrics = {}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            match = re.match(
                r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+'
                r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)$',
                line,
            )

            if not match:
                continue

            name = match.group(1)
            value_text = match.group(2)

            try:
                value = float(value_text)
            except ValueError:
                continue

            metrics.setdefault(name, []).append(value)

    return metrics


def metric_sum(metrics, candidates):
    for name in candidates:
        if name in metrics:
            return sum(metrics[name]), name

    return None, None


def metric_max(metrics, candidates):
    for name in candidates:
        if name in metrics:
            return max(metrics[name]), name

    return None, None


def counter_delta(before, after, candidates):
    before_value, before_name = metric_sum(
        before,
        candidates,
    )

    after_value, after_name = metric_sum(
        after,
        candidates,
    )

    metric_name = after_name or before_name

    if before_value is None or after_value is None:
        return {
            "metric": metric_name,
            "before": before_value,
            "after": after_value,
            "delta": None,
        }

    return {
        "metric": metric_name,
        "before": before_value,
        "after": after_value,
        "delta": after_value - before_value,
    }


def gauge_snapshot(before, after, candidates):
    before_value, before_name = metric_max(
        before,
        candidates,
    )

    after_value, after_name = metric_max(
        after,
        candidates,
    )

    return {
        "metric": after_name or before_name,
        "before": before_value,
        "after": after_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze vLLM cache metrics before/after "
            "a controlled serving experiment."
        )
    )

    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Experiment directory containing "
            "metrics_before.prom and metrics_after.prom"
        ),
    )

    args = parser.parse_args()

    before_path = (
        args.run_dir / "metrics_before.prom"
    )

    after_path = (
        args.run_dir / "metrics_after.prom"
    )

    if not before_path.exists():
        raise FileNotFoundError(
            f"Missing: {before_path}"
        )

    if not after_path.exists():
        raise FileNotFoundError(
            f"Missing: {after_path}"
        )

    before = parse_prometheus(before_path)
    after = parse_prometheus(after_path)

    query_result = counter_delta(
        before,
        after,
        COUNTER_METRICS["prefix_cache_queries"],
    )

    hit_result = counter_delta(
        before,
        after,
        COUNTER_METRICS["prefix_cache_hits"],
    )

    cached_result = counter_delta(
        before,
        after,
        COUNTER_METRICS["prompt_tokens_cached"],
    )

    kv_usage = gauge_snapshot(
        before,
        after,
        GAUGE_METRICS["kv_cache_usage_perc"],
    )

    query_delta = query_result["delta"]
    hit_delta = hit_result["delta"]

    if (
        query_delta is not None
        and query_delta > 0
        and hit_delta is not None
    ):
        prefix_hit_rate = (
            hit_delta / query_delta
        ) * 100.0
    else:
        prefix_hit_rate = None

    result = {
        "run_dir": str(args.run_dir),
        "prefix_cache_queries": query_result,
        "prefix_cache_hits": hit_result,
        "prompt_tokens_cached": cached_result,
        "prefix_cache_token_hit_rate_pct": (
            prefix_hit_rate
        ),
        "kv_cache_usage_perc_snapshot": kv_usage,
    }

    output_path = (
        args.run_dir / "cache_metrics_summary.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("vLLM CACHE METRICS ANALYSIS")
    print("=" * 68)
    print(f"Run: {args.run_dir}")
    print()

    print(
        "Prefix queries delta :",
        query_delta
        if query_delta is not None
        else "NOT FOUND",
    )

    print(
        "Prefix hits delta    :",
        hit_delta
        if hit_delta is not None
        else "NOT FOUND",
    )

    print(
        "Cached tokens delta  :",
        cached_result["delta"]
        if cached_result["delta"] is not None
        else "NOT FOUND",
    )

    if prefix_hit_rate is not None:
        print(
            "Prefix token hit rate:",
            f"{prefix_hit_rate:.2f}%",
        )
    else:
        print(
            "Prefix token hit rate: NOT AVAILABLE"
        )

    print()

    print(
        "KV usage before      :",
        kv_usage["before"]
        if kv_usage["before"] is not None
        else "NOT FOUND",
    )

    print(
        "KV usage after       :",
        kv_usage["after"]
        if kv_usage["after"] is not None
        else "NOT FOUND",
    )

    print()
    print(f"Saved: {output_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
