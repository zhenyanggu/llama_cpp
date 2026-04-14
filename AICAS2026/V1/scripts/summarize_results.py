#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Summarize the official V1 throughput and accuracy results.")
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--acc-json", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=False)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    throughput = load_json(args.throughput)
    acc_items = load_json(args.acc_json)
    run_meta = load_json(args.run_meta) if args.run_meta else {}

    acc_total = len(acc_items)
    acc_correct = sum(1 for item in acc_items if int(item.get("result", 0)) == 1)
    acc_ratio = (acc_correct / acc_total) if acc_total else 0.0

    summary = {
        "model": run_meta.get("model"),
        "mmproj": run_meta.get("mmproj"),
        "prompt_tokens": int(throughput.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(throughput.get("completion_tokens", 0) or 0),
        "total_tokens": int(throughput.get("total_tokens", 0) or 0),
        "prompt_ms": float(throughput.get("prompt_ms", 0.0) or 0.0),
        "decode_ms": float(throughput.get("decode_ms", 0.0) or 0.0),
        "total_ms": float(throughput.get("total_ms", 0.0) or 0.0),
        "prefill_tps": float(throughput.get("prefill_speed_tps", 0.0) or 0.0),
        "decode_tps": float(throughput.get("decode_speed_tps", 0.0) or 0.0),
        "acc_correct": acc_correct,
        "acc_total": acc_total,
        "acc_ratio": acc_ratio,
        "run_id": run_meta.get("run_id"),
        "remote_run_dir": run_meta.get("remote_run_dir"),
    }

    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Official Results",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model: `{summary['model']}`",
        f"- mmproj: `{summary['mmproj']}`",
        "",
        "## Throughput",
        "",
        f"- Prompt Tokens: `{summary['prompt_tokens']}`",
        f"- Completion Tokens: `{summary['completion_tokens']}`",
        f"- Total Tokens: `{summary['total_tokens']}`",
        f"- Prompt Time: `{summary['prompt_ms']:.3f} ms`",
        f"- Decode Time: `{summary['decode_ms']:.3f} ms`",
        f"- Total Time: `{summary['total_ms']:.3f} ms`",
        f"- Prefill Speed: `{summary['prefill_tps']:.6f} t/s`",
        f"- Decode Speed: `{summary['decode_tps']:.6f} t/s`",
        "",
        "## Accuracy",
        "",
        f"- Correct: `{summary['acc_correct']}`",
        f"- Total: `{summary['acc_total']}`",
        f"- Ratio: `{summary['acc_ratio']:.6f}`",
    ]

    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
