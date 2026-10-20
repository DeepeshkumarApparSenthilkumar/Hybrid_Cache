import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

TRACE = Path("results/full_sharegpt/ssm_align_trace_full.txt")
RUNTIME = Path("results/full_sharegpt/sharegpt_runtime_full.csv")

pattern = re.compile(
    r"request_id=(\S+) "
    r"prompt_tokens=(\d+) "
    r"computed_tokens=(\d+) "
    r"start=(\d+) "
    r"prefill_end=(\d+) "
    r"block_size=(\d+) "
    r"hash_block_size=(\d+) "
    r"checkpoint_position=(\d+) "
    r"use_internal_checkpoint=(\w+) "
    r"last_cache_position=(\d+) "
    r"next_block_boundary=(\d+) "
    r"tail_boundary=(\d+) "
    r"shared_prefix_boundary=(\d+) "
    r"stops=\(([^)]*)\) "
    r"original_end=(\d+) "
    r"final_end=(\d+) "
    r"scheduled_tokens=(\d+)"
)

records = []

for line in TRACE.read_text().splitlines():
    m = pattern.search(line)
    if not m:
        continue

    (
        request_id,
        prompt_tokens,
        computed_tokens,
        start,
        prefill_end,
        block_size,
        hash_block_size,
        checkpoint_position,
        use_internal_checkpoint,
        last_cache_position,
        next_block_boundary,
        tail_boundary,
        shared_prefix_boundary,
        stops,
        original_end,
        final_end,
        scheduled_tokens,
    ) = m.groups()

    r = {
        "request_id": request_id,
        "prompt_tokens": int(prompt_tokens),
        "computed_tokens": int(computed_tokens),
        "start": int(start),
        "prefill_end": int(prefill_end),
        "block_size": int(block_size),
        "hash_block_size": int(hash_block_size),
        "checkpoint_position": int(checkpoint_position),
        "use_internal_checkpoint": use_internal_checkpoint == "True",
        "last_cache_position": int(last_cache_position),
        "next_block_boundary": int(next_block_boundary),
        "tail_boundary": int(tail_boundary),
        "shared_prefix_boundary": int(shared_prefix_boundary),
        "original_end": int(original_end),
        "final_end": int(final_end),
        "scheduled_tokens": int(scheduled_tokens),
    }

    r["was_clipped"] = r["final_end"] < r["original_end"]
    r["final_end_aligned"] = (
        r["final_end"] > 0 and
        r["final_end"] % r["block_size"] == 0
    )

    records.append(r)

by_request = defaultdict(list)
for r in records:
    by_request[r["request_id"]].append(r)

clipped = [r for r in records if r["was_clipped"]]
aligned_clips = [
    r for r in clipped
    if r["final_end_aligned"]
]

aligned_positions = Counter(
    r["final_end"] for r in aligned_clips
)

initial_records = [
    min(rs, key=lambda x: x["start"])
    for rs in by_request.values()
]

reused_initial = [
    r for r in initial_records if r["start"] > 0
]

tails = []
for rs in by_request.values():
    last = max(rs, key=lambda x: x["final_end"])
    previous_boundary = (
        last["final_end"] // last["block_size"]
    ) * last["block_size"]

    if last["final_end"] % last["block_size"] == 0:
        tail = 0
    else:
        tail = last["final_end"] - previous_boundary

    tails.append(tail)

with RUNTIME.open(newline="") as f:
    runtime_rows = list(csv.DictReader(f))

latencies = [
    float(r["e2e_seconds"])
    for r in runtime_rows
    if r.get("status") == "ok"
]

print("===== FULL SHAREGPT SSM ALIGNMENT ANALYSIS =====")
print(f"Runtime requests: {len(runtime_rows)}")
print(f"Unique traced requests: {len(by_request)}")
print(f"Total scheduler trace records: {len(records)}")
print()

print("===== CONFIGURATION =====")
if records:
    print(f"Block size: {records[0]['block_size']} tokens")
    print(f"Hash block size: {records[0]['hash_block_size']} tokens")
print(
    "Internal checkpoint used:",
    any(r["use_internal_checkpoint"] for r in records)
)
print()

print("===== ALIGNMENT SPLITS =====")
print(f"Clipped scheduler chunks: {len(clipped)}")
print(f"Aligned clipped chunks: {len(aligned_clips)}")
print(
    "Requests with >=1 aligned split:",
    sum(
        any(r["was_clipped"] and r["final_end_aligned"] for r in rs)
        for rs in by_request.values()
    )
)
print()

print("Aligned boundary positions:")
for pos, count in sorted(aligned_positions.items()):
    print(f"  {pos:5d} tokens : {count}")
print()

print("===== PREFIX REUSE =====")
print(f"Requests starting above token 0: {len(reused_initial)}")
if initial_records:
    print(
        "Fraction starting with reused prefix: "
        f"{100 * len(reused_initial) / len(initial_records):.2f}%"
    )

reuse_positions = Counter(r["start"] for r in reused_initial)

print("Reuse starting positions:")
for pos, count in sorted(reuse_positions.items()):
    print(f"  {pos:5d} tokens : {count}")
print()

print("===== FINAL TAIL AFTER LAST ALIGNED BOUNDARY =====")
if tails:
    print(f"Mean tail: {mean(tails):.2f} tokens")
    print(f"Median tail: {median(tails):.2f} tokens")
    print(f"Maximum tail: {max(tails)} tokens")
print()

print("===== LATENCY =====")
if latencies:
    print(f"Mean E2E: {mean(latencies):.4f} s")
    print(f"Median E2E: {median(latencies):.4f} s")
    print(f"Minimum E2E: {min(latencies):.4f} s")
    print(f"Maximum E2E: {max(latencies):.4f} s")

OUT = Path("results/full_sharegpt/ssm_alignment_summary.csv")

with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["boundary_tokens", "aligned_split_count"])
    for pos, count in sorted(aligned_positions.items()):
        writer.writerow([pos, count])

print()
print(f"Saved boundary summary: {OUT}")
