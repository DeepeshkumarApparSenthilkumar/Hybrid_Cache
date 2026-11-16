#!/usr/bin/env python3

from dataclasses import dataclass
from math import ceil


@dataclass
class HybridModel:
    name: str
    mechanism: str
    recurrent_state_bytes: int
    attention_kv_bytes_per_token: int
    kernel_alignment_tokens: int
    observed_runtime_block_size: int


MODELS = [
    HybridModel(
        name="Qwen3.5-0.8B",
        mechanism="Gated DeltaNet",
        recurrent_state_bytes=1_085_440,
        attention_kv_bytes_per_token=2_048,
        kernel_alignment_tokens=16,
        observed_runtime_block_size=544,
    ),
    HybridModel(
        name="Falcon-H1-0.5B",
        mechanism="Mamba-2",
        recurrent_state_bytes=403_968,
        attention_kv_bytes_per_token=512,
        kernel_alignment_tokens=16,
        observed_runtime_block_size=800,
    ),
]


def profile(model: HybridModel):
    raw_tokens = (
        model.recurrent_state_bytes
        / model.attention_kv_bytes_per_token
    )

    aligned_block = (
        model.kernel_alignment_tokens
        * ceil(raw_tokens / model.kernel_alignment_tokens)
    )

    physical_page_bytes = (
        aligned_block * model.attention_kv_bytes_per_token
    )

    padding_bytes = physical_page_bytes - model.recurrent_state_bytes

    padding_pct = (
        100.0 * padding_bytes / model.recurrent_state_bytes
    )

    extra_token_slots = aligned_block - raw_tokens

    return {
        "raw_tokens": raw_tokens,
        "aligned_block": aligned_block,
        "physical_page_bytes": physical_page_bytes,
        "padding_bytes": padding_bytes,
        "padding_pct": padding_pct,
        "extra_token_slots": extra_token_slots,
    }


print("=" * 90)
print("vLLM HYBRID KV / RECURRENT-STATE ALIGNMENT PROFILE")
print("=" * 90)

for model in MODELS:
    r = profile(model)

    print(f"\nModel: {model.name}")
    print(f"Mechanism: {model.mechanism}")

    print(
        f"Useful recurrent state : "
        f"{model.recurrent_state_bytes:,} B "
        f"({model.recurrent_state_bytes / 1024:.2f} KiB)"
    )

    print(
        f"Attention KV/token     : "
        f"{model.attention_kv_bytes_per_token:,} B"
    )

    print(
        f"Raw token equivalent   : "
        f"{r['raw_tokens']:.3f} tokens"
    )

    print(
        f"Kernel alignment       : "
        f"{model.kernel_alignment_tokens} tokens"
    )

    print(
        f"Calculated block size  : "
        f"{r['aligned_block']} tokens"
    )

    print(
        f"Observed runtime block : "
        f"{model.observed_runtime_block_size} tokens"
    )

    print(
        f"Runtime match          : "
        f"{r['aligned_block'] == model.observed_runtime_block_size}"
    )

    print(
        f"Physical shared page   : "
        f"{r['physical_page_bytes']:,} B "
        f"({r['physical_page_bytes'] / 1024:.2f} KiB)"
    )

    print(
        f"Padding overhead       : "
        f"{r['padding_bytes']:,} B "
        f"({r['padding_bytes'] / 1024:.2f} KiB)"
    )

    print(
        f"Padding percentage     : "
        f"{r['padding_pct']:.4f}%"
    )

    print(
        f"Extra token-equivalent : "
        f"{r['extra_token_slots']:.3f}"
    )

print("\n" + "=" * 90)
