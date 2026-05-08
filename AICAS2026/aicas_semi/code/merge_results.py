"""
Merge AICAS 2026 GC Hardware semi-finals evaluation outputs into a single
submission JSON.

Reads the output files of:
  - acc_eval.py                  (per-sample list with 'result' field)
  - throughput_eval.py           (prefill_speed_tps / decode_speed_tps)
  - energy_eval.py               (tokens_per_joule, average_power_w, ...)
  - ttft_eval_multiprompt.py     (per-image linear fit + aggregate)

Usage:
    python merge_results.py \\
        --acc        acc_eval_results.json \\
        --throughput throughput_metrics.json \\
        --energy     energy_metrics.json \\
        --ttft       ttft_eval_results.json \\
        -o           aicas_submission.json
"""

import os
import sys
import json
import argparse


def load_json(path):
    """Read a JSON file, raising a friendly error on failure."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_acc(path):
    """Parse acc_eval.py output (a list of OCRBench items) into total accuracy."""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(
            f"acc result expected to be a list of items, got {type(data).__name__}"
        )

    n_total = len(data)
    n_correct = sum(1 for item in data if item.get("result") == 1)
    ratio = n_correct / n_total if n_total > 0 else 0.0

    return {
        "ratio_accuracy": ratio,
        "n_correct": n_correct,
        "n_total": n_total
    }


def parse_throughput(path):
    """Parse throughput_eval.py output."""
    data = load_json(path)
    return {
        "prefill_speed_tps": data.get("prefill_speed_tps"),
        "decode_speed_tps": data.get("decode_speed_tps")
    }


def parse_energy(path):
    """Parse energy_eval.py output. Keeps the audit fields needed to verify
    that tokens_per_joule was not fabricated (numerator, denominator, sampling
    coverage)."""
    data = load_json(path)
    return {
        "tokens_per_joule": data.get("tokens_per_joule"),
        "average_power_w": data.get("average_power_w"),
        "total_energy_j": data.get("total_energy_j"),
        "completion_tokens": data.get("completion_tokens"),
        "inference_duration_s": data.get("inference_duration_s"),
        "num_samples": data.get("num_samples")
    }


def parse_ttft(path):
    """Parse ttft_eval_multiprompt.py output. Returns aggregate slope/intercept
    at the top level (used directly for scheme-B scoring) plus a per-image
    abbreviated fit list as evidence."""
    data = load_json(path)
    fit = data.get("linear_fit") or {}
    agg = fit.get("aggregate") or {}

    per_image = []
    for img_fit in fit.get("per_image", []):
        per_image.append({
            "image": img_fit.get("image"),
            "slope_ms_per_char": img_fit.get("slope_ms_per_char"),
            "intercept_ms": img_fit.get("intercept_ms")
        })

    return {
        "slope_ms_per_char": agg.get("slope_ms_per_char"),
        "intercept_ms": agg.get("intercept_ms"),
        "per_image": per_image
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge AICAS evaluation results into a single submission JSON."
    )
    parser.add_argument(
        "--acc",
        help="Path to accuracy result JSON (acc_eval.py output)."
    )
    parser.add_argument(
        "--throughput",
        help="Path to throughput result JSON (throughput_eval.py output)."
    )
    parser.add_argument(
        "--energy",
        help="Path to energy result JSON (energy_eval.py output)."
    )
    parser.add_argument(
        "--ttft",
        help="Path to ttft result JSON (ttft_eval_multiprompt.py output)."
    )
    parser.add_argument(
        "-o", "--output",
        default="aicas_submission.json",
        help="Path to write the merged submission JSON."
    )
    args = parser.parse_args()

    merged = {}

    # --- accuracy ---
    if args.acc:
        if not os.path.exists(args.acc):
            print(f"Error: accuracy file not found: {args.acc}")
            sys.exit(1)
        merged["accuracy"] = parse_acc(args.acc)
        a = merged["accuracy"]
        print(f"[ok] accuracy:   ratio = {a['ratio_accuracy']:.4f}  "
              f"({a['n_correct']}/{a['n_total']})")

    # --- throughput ---
    if args.throughput:
        if not os.path.exists(args.throughput):
            print(f"Error: throughput file not found: {args.throughput}")
            sys.exit(1)
        merged["throughput"] = parse_throughput(args.throughput)
        t = merged["throughput"]
        prefill = t.get("prefill_speed_tps")
        decode = t.get("decode_speed_tps")
        prefill_str = f"{prefill:.2f}" if prefill is not None else "n/a"
        decode_str = f"{decode:.2f}" if decode is not None else "n/a"
        print(f"[ok] throughput: prefill = {prefill_str} t/s, decode = {decode_str} t/s")

    # --- energy ---
    if args.energy:
        if not os.path.exists(args.energy):
            print(f"Error: energy file not found: {args.energy}")
            sys.exit(1)
        merged["energy"] = parse_energy(args.energy)
        e = merged["energy"]
        eff = e.get("tokens_per_joule")
        pwr = e.get("average_power_w")
        eff_str = f"{eff:.4f}" if eff is not None else "n/a"
        pwr_str = f"{pwr:.2f}" if pwr is not None else "n/a"
        print(f"[ok] energy:     {eff_str} tokens/J  ({pwr_str} W avg)")

    # --- ttft ---
    if args.ttft:
        if not os.path.exists(args.ttft):
            print(f"Error: ttft file not found: {args.ttft}")
            sys.exit(1)
        merged["ttft"] = parse_ttft(args.ttft)
        t = merged["ttft"]
        if t.get("slope_ms_per_char") is not None:
            print(f"[ok] ttft:       slope = {t['slope_ms_per_char']:.4f} ms/char, "
                  f"intercept = {t['intercept_ms']:.2f} ms  "
                  f"(over {len(t['per_image'])} images)")
        else:
            print(f"[ok] ttft:       (aggregate fit unavailable)")

    if not merged:
        print("Error: no input files specified. "
              "Use --acc, --throughput, --energy, --ttft.")
        sys.exit(1)

    try:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\nMerged submission written to: {args.output}")
    except IOError as e:
        print(f"\nError: failed to write merged submission: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
