#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def classify(stats, spin_p95_us, irq_p50_us, irq_min_count):
    if stats["p95_us"] < spin_p95_us:
        return "spin"
    if stats["count"] >= irq_min_count and stats["p50_us"] >= irq_p50_us:
        return "irq"
    return "hybrid"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an adaptive NPU wait policy from NPU_WAIT_TRACE_JSONL output."
    )
    parser.add_argument("trace", help="Input npu_wait_trace.jsonl")
    parser.add_argument("-o", "--output", required=True, help="Output wait_policy.json")
    parser.add_argument(
        "--summary-output",
        help="Optional detailed grouped summary JSON. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--spin-p95-us", type=float, default=50.0)
    parser.add_argument("--irq-p50-us", type=float, default=300.0)
    parser.add_argument("--irq-min-count", type=int, default=3)
    parser.add_argument(
        "--default-strategy",
        choices=["hybrid", "spin", "irq"],
        default="hybrid",
        help="Strategy used by runtime when a shape is not listed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    groups = defaultdict(list)
    examples = {}

    with open(args.trace, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            op = str(record.get("op", "none"))
            shape_key = str(record.get("shape_key", ""))
            wait_us = float(record.get("wait_us", 0.0) or 0.0)
            key = (op, shape_key)
            groups[key].append(wait_us)
            examples.setdefault(key, record)

    summary = []
    for key, waits in sorted(groups.items()):
        op, shape_key = key
        stats = {
            "op": op,
            "shape_key": shape_key,
            "count": len(waits),
            "min_us": min(waits),
            "p50_us": percentile(waits, 0.50),
            "p90_us": percentile(waits, 0.90),
            "p95_us": percentile(waits, 0.95),
            "max_us": max(waits),
        }
        stats["strategy"] = classify(
            stats,
            spin_p95_us=args.spin_p95_us,
            irq_p50_us=args.irq_p50_us,
            irq_min_count=args.irq_min_count,
        )
        ex = examples[key]
        stats["bytes"] = int(ex.get("bytes", 0) or 0)
        summary.append(stats)

    policy_entries = [
        {
            "op": item["op"],
            "shape_key": item["shape_key"],
            "strategy": item["strategy"],
            "count": item["count"],
            "p50_us": item["p50_us"],
            "p95_us": item["p95_us"],
        }
        for item in summary
        if item["strategy"] != args.default_strategy
    ]

    policy = {
        "profile_kind": "npu_wait_policy",
        "default_strategy": args.default_strategy,
        "thresholds": {
            "spin_p95_us": args.spin_p95_us,
            "irq_p50_us": args.irq_p50_us,
            "irq_min_count": args.irq_min_count,
        },
        "entries": policy_entries,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary_output = args.summary_output or f"{args.output}.summary.json"
    with open(summary_output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "profile_kind": "npu_wait_trace_summary",
                "source": args.trace,
                "group_count": len(summary),
                "records": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print(f"groups={len(summary)} policy_entries={len(policy_entries)}")
    print(f"policy={args.output}")
    print(f"summary={summary_output}")


if __name__ == "__main__":
    main()
