import re
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

ROOT = Path("results/runtime/qwen35_vllm_sharegpt/full_experiment")
TRACE = ROOT / "results/full_sharegpt/ssm_align_trace_full.txt"
OUT = Path("plots/runtime/qwen35_vllm_sharegpt")
OUT.mkdir(parents=True, exist_ok=True)

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

    records.append({
        "request_id": request_id,
        "prompt_tokens": int(prompt_tokens),
        "start": int(start),
        "block_size": int(block_size),
        "original_end": int(original_end),
        "final_end": int(final_end),
    })

if not records:
    raise RuntimeError("No SSM_ALIGN_TRACE records parsed.")

block_size = records[0]["block_size"]

# Figure 1: alignment-induced scheduler split positions
aligned_splits = [
    r["final_end"]
    for r in records
    if r["final_end"] < r["original_end"]
    and r["final_end"] > 0
    and r["final_end"] % block_size == 0
]

split_counts = Counter(aligned_splits)
positions = sorted(split_counts)
counts = [split_counts[p] for p in positions]

plt.figure(figsize=(7.5, 4.8))
bars = plt.bar([str(p) for p in positions], counts)

for bar, value in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        str(value),
        ha="center",
        va="bottom",
    )

plt.xlabel("Aligned token boundary")
plt.ylabel("Observed scheduler split events")
plt.title("Qwen3.5: Runtime SSM Alignment Splits on ShareGPT")
plt.tight_layout()
plt.savefig(OUT / "ssm_alignment_split_positions.png", dpi=300)
plt.close()

# Figure 2: reusable-prefix starting positions
by_request = defaultdict(list)

for r in records:
    by_request[r["request_id"]].append(r)

initial = [
    min(rs, key=lambda x: x["start"])
    for rs in by_request.values()
]

reuse = [r["start"] for r in initial if r["start"] > 0]
reuse_counts = Counter(reuse)

positions = sorted(reuse_counts)
counts = [reuse_counts[p] for p in positions]

plt.figure(figsize=(7.5, 4.8))
bars = plt.bar([str(p) for p in positions], counts)

for bar, value in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        str(value),
        ha="center",
        va="bottom",
    )

plt.xlabel("Token position where computation resumed")
plt.ylabel("Requests")
plt.title("Qwen3.5: Reused Prefix Starting Positions")
plt.tight_layout()
plt.savefig(OUT / "prefix_reuse_positions.png", dpi=300)
plt.close()

# Figure 3: residual prompt-tail distribution
tails = []

for rs in by_request.values():
    prompt_tokens = rs[0]["prompt_tokens"]
    tails.append(prompt_tokens % block_size)

mean_tail = sum(tails) / len(tails)

plt.figure(figsize=(7.5, 4.8))
plt.hist(tails, bins=17, edgecolor="black")
plt.axvline(
    mean_tail,
    linestyle="--",
    label=f"Mean = {mean_tail:.2f}",
)

plt.xlabel("Tokens after last 544-token boundary")
plt.ylabel("Requests")
plt.title("ShareGPT Residual Tail Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "residual_tail_distribution.png", dpi=300)
plt.close()

# Figure 4: physical GDN state-page padding
useful_kib = 1060
physical_kib = 1088
padding_kib = physical_kib - useful_kib

plt.figure(figsize=(7.5, 4.8))

bars = plt.bar(
    ["Useful GDN state", "Aligned physical page"],
    [useful_kib, physical_kib],
)

for bar, value in zip(bars, [useful_kib, physical_kib]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 4,
        f"{value} KiB",
        ha="center",
        va="bottom",
    )

plt.ylabel("Page size (KiB)")
plt.title(
    "Qwen3.5 Mamba/GDN State-Page Alignment\n"
    f"Padding = {padding_kib} KiB ({padding_kib/useful_kib*100:.2f}%)"
)
plt.ylim(0, 1180)
plt.tight_layout()
plt.savefig(OUT / "state_page_padding.png", dpi=300)
plt.close()

print(f"Parsed trace records: {len(records)}")
print(f"Unique requests: {len(by_request)}")
print(f"Alignment-induced splits: {len(aligned_splits)}")
print(f"Requests with reused prefix: {len(reuse)}")
print(f"Mean residual tail: {mean_tail:.2f} tokens")
print()
print("Generated:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p)
