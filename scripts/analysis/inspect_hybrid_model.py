#!/usr/bin/env python3

"""
Generic Hybrid LLM Architecture Inspector

Purpose
-------
Inspect a Hugging Face model configuration and extract architecture fields
that are relevant to hybrid KV/state-cache characterization.

This script is intentionally architecture-agnostic so that it can be reused
for models such as:

- Qwen3.5
- Gemma 3
- Falcon-H1
- Kimi Linear

Outputs
-------
architecture/<name>/
    config.json
    text_config.json
    interesting_fields.json
    architecture_summary.json
    layer_types.csv        (when layer_types exists)
    layer_types.txt        (when layer_types exists)

Example
-------
python scripts/analysis/inspect_hybrid_model.py \
    Qwen/Qwen3.5-0.8B \
    --name qwen35_0.8b_test
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from transformers import AutoConfig


# ---------------------------------------------------------------------------
# Fields/keywords relevant to cache and hybrid architecture characterization
# ---------------------------------------------------------------------------

INTERESTING_WORDS = (
    "layer",
    "attention",
    "attn",
    "head",
    "kv",
    "key",
    "value",
    "mamba",
    "ssm",
    "linear",
    "delta",
    "window",
    "cache",
    "conv",
    "rope",
    "position",
    "sliding",
    "local",
    "global",
)


SUMMARY_KEYS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "sliding_window",
    "sliding_window_pattern",
    "full_attention_interval",
    "attention_bias",
    "attention_dropout",
    "attn_output_gate",
    "partial_rotary_factor",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "mamba_ssm_dtype",
    "mtp_num_hidden_layers",
    "use_cache",
    "dtype",
    "torch_dtype",
    "vocab_size",
)


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert objects into JSON-safe representations.

    Hugging Face configurations are normally JSON serializable, but this
    helper makes the inspector more robust across different model families.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    # Handle torch dtype-like and other configuration objects safely.
    return str(value)


def find_interesting_fields(
    obj: Any,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Recursively search a configuration for fields related to attention,
    KV cache, sliding-window attention, linear attention, Mamba/SSM,
    convolution state, and other hybrid-cache mechanisms.

    Dictionary keys are explicitly converted to strings because some model
    configurations may contain integer keys.
    """

    results: Dict[str, Any] = {}

    if isinstance(obj, dict):

        for key, value in obj.items():

            # Important:
            # Not every nested dictionary key is guaranteed to be a string.
            key_str = str(key)

            if prefix:
                path = f"{prefix}.{key_str}"
            else:
                path = key_str

            key_lower = key_str.lower()

            if any(
                word in key_lower
                for word in INTERESTING_WORDS
            ):
                results[path] = make_json_safe(value)

            if isinstance(value, (dict, list, tuple)):
                results.update(
                    find_interesting_fields(
                        value,
                        path,
                    )
                )

    elif isinstance(obj, (list, tuple)):

        for index, value in enumerate(obj):

            if prefix:
                path = f"{prefix}[{index}]"
            else:
                path = f"[{index}]"

            if isinstance(value, (dict, list, tuple)):
                results.update(
                    find_interesting_fields(
                        value,
                        path,
                    )
                )

    return results


def get_text_config(
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return the language/text configuration.

    Some models such as Qwen3.5 and Gemma multimodal variants wrap the
    language model configuration inside text_config.
    """

    text_config = config_dict.get("text_config")

    if isinstance(text_config, dict):
        return text_config

    return config_dict


def create_architecture_summary(
    model_id: str,
    config_dict: Dict[str, Any],
    text_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a compact machine-readable architecture summary.
    """

    summary: Dict[str, Any] = {
        "model_id": model_id,
        "top_level_model_type": config_dict.get("model_type"),
        "architectures": config_dict.get("architectures"),
        "uses_nested_text_config": (
            isinstance(config_dict.get("text_config"), dict)
        ),
    }

    for key in SUMMARY_KEYS:
        if key in text_config:
            summary[key] = make_json_safe(
                text_config[key]
            )

    layer_types = text_config.get("layer_types")

    if isinstance(layer_types, (list, tuple)):

        layer_types = [
            str(layer_type)
            for layer_type in layer_types
        ]

        counts = Counter(layer_types)

        summary["layer_type_count"] = len(layer_types)
        summary["layer_type_counts"] = dict(counts)

    return summary


def save_json(
    path: Path,
    data: Any,
) -> None:
    """
    Save formatted JSON.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(data),
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


def save_layer_types(
    output_dir: Path,
    layer_types: Any,
) -> None:
    """
    Save layer mapping in both CSV and simple text formats.
    """

    if not isinstance(layer_types, (list, tuple)):
        return

    # CSV version
    csv_path = output_dir / "layer_types.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "layer_index",
                "layer_type",
            ]
        )

        for index, layer_type in enumerate(layer_types):
            writer.writerow(
                [
                    index,
                    str(layer_type),
                ]
            )

    # Simple text version
    txt_path = output_dir / "layer_types.txt"

    with txt_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for index, layer_type in enumerate(layer_types):
            file.write(
                f"{index},{str(layer_type)}\n"
            )


def print_section(title: str) -> None:
    """
    Print a readable terminal section heading.
    """

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_architecture_summary(
    model_id: str,
    config_dict: Dict[str, Any],
    text_config: Dict[str, Any],
) -> None:
    """
    Print the most useful architecture information.
    """

    print_section("MODEL")

    print(f"Model ID: {model_id}")
    print(
        "Top-level model type:",
        config_dict.get("model_type", "NOT PRESENT"),
    )
    print(
        "Architectures:",
        config_dict.get("architectures", "NOT PRESENT"),
    )

    if isinstance(config_dict.get("text_config"), dict):
        print("Nested text_config: YES")
    else:
        print("Nested text_config: NO")

    print_section("TEXT ARCHITECTURE")

    for key in SUMMARY_KEYS:

        if key in text_config:

            value = text_config[key]

            # Avoid dumping huge structures in terminal.
            if isinstance(
                value,
                (str, int, float, bool),
            ) or value is None:

                print(f"{key}: {value}")

    layer_types = text_config.get("layer_types")

    if isinstance(layer_types, (list, tuple)):

        counts = Counter(
            str(layer_type)
            for layer_type in layer_types
        )

        print_section("LAYER TYPE COUNTS")

        print(
            f"Total layer_types entries: "
            f"{len(layer_types)}"
        )

        for layer_type, count in counts.items():
            print(
                f"{layer_type}: {count}"
            )

        print_section("LAYER PATTERN")

        for index, layer_type in enumerate(layer_types):
            print(
                f"Layer {index:02d}: {layer_type}"
            )

    else:

        print_section("LAYER TYPE COUNTS")

        print(
            "No explicit layer_types list was found."
        )

        print(
            "This does NOT mean the model is not hybrid."
        )

        print(
            "The hybrid pattern may instead be represented "
            "through intervals, sliding-window settings, "
            "or model-specific configuration fields."
        )


def print_interesting_scalar_fields(
    interesting: Dict[str, Any],
) -> None:
    """
    Print only scalar interesting fields so terminal output remains readable.
    """

    print_section(
        "CACHE / HYBRID RELATED SCALAR FIELDS"
    )

    scalar_count = 0

    for path in sorted(interesting):

        value = interesting[path]

        if isinstance(
            value,
            (str, int, float, bool),
        ) or value is None:

            print(
                f"{path}: {value}"
            )

            scalar_count += 1

    if scalar_count == 0:
        print(
            "No scalar cache/hybrid fields were found."
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect Hugging Face model configurations "
            "for hybrid KV/state-cache architecture."
        )
    )

    parser.add_argument(
        "model",
        help=(
            "Hugging Face model ID, for example "
            "Qwen/Qwen3.5-0.8B"
        ),
    )

    parser.add_argument(
        "--name",
        required=True,
        help=(
            "Local directory name under architecture/"
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    print()
    print(
        f"Loading configuration: {args.model}"
    )
    print()

    # ---------------------------------------------------------
    # Load Hugging Face configuration
    # ---------------------------------------------------------

    try:

        config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
        )

    except Exception as error:

        print(
            "ERROR: Could not load model configuration."
        )
        print()
        print(
            f"Model: {args.model}"
        )
        print(
            f"Reason: {error}"
        )

        raise

    # ---------------------------------------------------------
    # Convert configuration to dictionary
    # ---------------------------------------------------------

    config_dict = make_json_safe(
        config.to_dict()
    )

    if not isinstance(config_dict, dict):
        raise TypeError(
            "Expected Hugging Face configuration "
            "to convert to a dictionary."
        )

    text_config = get_text_config(
        config_dict
    )

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    output_dir = (
        Path("architecture")
        / args.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Save raw configurations
    # ---------------------------------------------------------

    save_json(
        output_dir / "config.json",
        config_dict,
    )

    save_json(
        output_dir / "text_config.json",
        text_config,
    )

    # ---------------------------------------------------------
    # Extract interesting hybrid/cache fields
    # ---------------------------------------------------------

    interesting = find_interesting_fields(
        config_dict
    )

    save_json(
        output_dir / "interesting_fields.json",
        interesting,
    )

    # ---------------------------------------------------------
    # Create compact architecture summary
    # ---------------------------------------------------------

    summary = create_architecture_summary(
        args.model,
        config_dict,
        text_config,
    )

    save_json(
        output_dir / "architecture_summary.json",
        summary,
    )

    # ---------------------------------------------------------
    # Save layer map when explicitly provided
    # ---------------------------------------------------------

    layer_types = text_config.get(
        "layer_types"
    )

    save_layer_types(
        output_dir,
        layer_types,
    )

    # ---------------------------------------------------------
    # Print useful terminal information
    # ---------------------------------------------------------

    print_architecture_summary(
        args.model,
        config_dict,
        text_config,
    )

    print_interesting_scalar_fields(
        interesting
    )

    # ---------------------------------------------------------
    # Final output summary
    # ---------------------------------------------------------

    print_section("OUTPUT FILES")

    for path in sorted(output_dir.iterdir()):

        if path.is_file():

            size_bytes = path.stat().st_size

            print(
                f"{path.name:<30} "
                f"{size_bytes:>10} bytes"
            )

    print()
    print(
        f"Architecture data saved to: "
        f"{output_dir}"
    )

    print()
    print(
        "Inspection completed successfully."
    )
    print()


if __name__ == "__main__":
    main()
