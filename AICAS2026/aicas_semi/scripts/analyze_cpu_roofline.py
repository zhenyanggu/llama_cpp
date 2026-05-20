#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


TYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "q8_0": 1,
    "q4_0": 0.5,
    "q4_k": 0.5,
    "q5_k": 0.625,
    "q6_k": 0.75,
    "q8_k": 1,
}

OP_FLOPS_PER_ELEMENT = {
    "ADD": 1,
    "MUL": 1,
    "SUB": 1,
    "DIV": 1,
    "SCALE": 1,
    "RMS_NORM": 5,
    "NORM": 5,
    "ROPE": 6,
    "SWIGLU": 6,
    "GELU": 10,
    "SOFT_MAX": 5,
    "FLASH_ATTN_EXT": 8,
    "CONT": 0,
    "CPY": 0,
    "RESHAPE": 0,
    "VIEW": 0,
    "PERMUTE": 0,
    "TRANSPOSE": 0,
    "SET_ROWS": 0,
    "IM2COL": 0,
}


def product(values):
    out = 1
    for value in values:
        out *= int(value)
    return out


def tensor_nbytes(info):
    type_name = str(info.get("type", "")).lower()
    elem_bytes = TYPE_BYTES.get(type_name, 4)
    return product(info.get("shape", [1, 1, 1, 1])) * elem_bytes


def matmul_flops(sig):
    src0 = sig["src0"]["shape"]
    src1 = sig["src1"]["shape"]
    dst = sig["dst"]["shape"]
    k = int(src0[0])
    n = int(src0[1])
    m = int(src1[1])
    batch = max(int(src0[2]), int(src1[2]), int(dst[2]), 1) * max(int(src0[3]), int(src1[3]), int(dst[3]), 1)
    return 2.0 * k * n * m * batch


def matmul_bytes(sig):
    return tensor_nbytes(sig["src0"]) + tensor_nbytes(sig["src1"]) + tensor_nbytes(sig["dst"])


def signature_label(sig):
    src0_name = sig["src0"].get("name", "")
    dst_name = sig["dst"].get("name", "")
    if src0_name.strip():
        return src0_name
    return dst_name or "MUL_MAT"


def add_matmul_rows(rows, phase, profile):
    for sig in profile.get("mul_mat_signatures", []):
        count = int(sig.get("node_count", 0))
        duration_us = float(sig.get("duration_us", 0))
        if count <= 0 or duration_us <= 0:
            continue
        ops_per_node = matmul_flops(sig)
        bytes_per_node = matmul_bytes(sig)
        total_ops = ops_per_node * count
        total_bytes = bytes_per_node * count
        rows.append({
            "phase": phase,
            "operator": "MUL_MAT",
            "label": signature_label(sig),
            "node_count": count,
            "duration_us": duration_us,
            "ops": total_ops,
            "bytes": total_bytes,
            "arithmetic_intensity_ops_per_byte": total_ops / total_bytes if total_bytes else 0.0,
            "gops": total_ops / duration_us / 1000.0,
            "bandwidth_gbs": total_bytes / duration_us / 1000.0,
            "flop_source": "exact_matmul_shape",
            "src0_shape": "x".join(str(x) for x in sig["src0"].get("shape", [])),
            "src1_shape": "x".join(str(x) for x in sig["src1"].get("shape", [])),
            "dst_shape": "x".join(str(x) for x in sig["dst"].get("shape", [])),
        })


def op_estimated_ops(op):
    name = op["name"]
    return float(op.get("elements", 0)) * OP_FLOPS_PER_ELEMENT.get(name, 1)


def add_operator_rows(rows, phase, profile):
    for op in profile.get("operators", []):
        name = op["name"]
        if name == "MUL_MAT":
            continue
        duration_us = float(op.get("duration_us", 0))
        if duration_us <= 0:
            continue
        ops = op_estimated_ops(op)
        bytes_ = float(op.get("bytes", 0))
        rows.append({
            "phase": phase,
            "operator": name,
            "label": name,
            "node_count": int(op.get("node_count", 0)),
            "duration_us": duration_us,
            "ops": ops,
            "bytes": bytes_,
            "arithmetic_intensity_ops_per_byte": ops / bytes_ if bytes_ else 0.0,
            "gops": ops / duration_us / 1000.0 if ops else 0.0,
            "bandwidth_gbs": bytes_ / duration_us / 1000.0 if bytes_ else 0.0,
            "flop_source": "estimated_from_elements",
            "src0_shape": "",
            "src1_shape": "",
            "dst_shape": "",
        })


def combine_rows(rows):
    combined = {}
    for row in rows:
        key = (
            row["phase"],
            row["operator"],
            row["label"],
            row["flop_source"],
            row["src0_shape"],
            row["src1_shape"],
            row["dst_shape"],
        )
        if key not in combined:
            combined[key] = dict(row)
            continue
        dst = combined[key]
        dst["node_count"] += row["node_count"]
        dst["duration_us"] += row["duration_us"]
        dst["ops"] += row["ops"]
        dst["bytes"] += row["bytes"]

    out = []
    for row in combined.values():
        row["arithmetic_intensity_ops_per_byte"] = row["ops"] / row["bytes"] if row["bytes"] else 0.0
        row["gops"] = row["ops"] / row["duration_us"] / 1000.0 if row["duration_us"] and row["ops"] else 0.0
        row["bandwidth_gbs"] = row["bytes"] / row["duration_us"] / 1000.0 if row["duration_us"] and row["bytes"] else 0.0
        out.append(row)
    return out


def load_rows(result_dir):
    result_dir = Path(result_dir)
    with open(result_dir / "mtmd_prefill_summary.json", "r", encoding="utf-8") as handle:
        mtmd = json.load(handle)

    rows = []
    mmproj_profile = mtmd["mmproj"]["cpu_backend_profile_details"][0]
    add_matmul_rows(rows, "mmproj", mmproj_profile)
    add_operator_rows(rows, "mmproj", mmproj_profile)

    text_path = result_dir / "text_cpu_profile.jsonl"
    if text_path.exists():
        for line in text_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            phase = record.get("phase", "text")
            if phase == "merged_prefill":
                phase = "prefill"
            profile = record["cpu_backend_profile"]
            add_matmul_rows(rows, phase, profile)
            add_operator_rows(rows, phase, profile)

    return mtmd, combine_rows(rows)


def write_csv(path, rows):
    columns = [
        "phase", "operator", "label", "node_count", "duration_us", "ops", "bytes",
        "arithmetic_intensity_ops_per_byte", "gops", "bandwidth_gbs",
        "flop_source", "src0_shape", "src1_shape", "dst_shape",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def plot_roofline(path, rows, summary):
    exact = [r for r in rows if r["flop_source"] == "exact_matmul_shape" and r["gops"] > 0 and r["bytes"] > 0]
    estimated = [r for r in rows if r["flop_source"] != "exact_matmul_shape" and r["gops"] > 0 and r["bytes"] > 0]
    plotted = exact + estimated
    if not plotted:
        return

    peak_gops = summary["observed_peak_gops"]
    peak_bw = summary["observed_peak_bandwidth_gbs"]
    min_ai = min(r["arithmetic_intensity_ops_per_byte"] for r in plotted if r["arithmetic_intensity_ops_per_byte"] > 0)
    max_ai = max(r["arithmetic_intensity_ops_per_byte"] for r in plotted if r["arithmetic_intensity_ops_per_byte"] > 0)
    xs = [10 ** x for x in [math.log10(min_ai / 2.0) + i * (math.log10(max_ai * 2.0) - math.log10(min_ai / 2.0)) / 120 for i in range(121)]]
    ys = [min(peak_gops, peak_bw * x) for x in xs]

    phase_colors = {
        "mmproj": "#1f77b4",
        "prefill": "#d62728",
        "decode": "#2ca02c",
    }

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax0.plot(xs, ys, color="black", linewidth=1.5, label=f"observed roofline: {peak_gops:.2f} GOPS, {peak_bw:.2f} GB/s")
    for row in exact:
        ax0.scatter(
            row["arithmetic_intensity_ops_per_byte"],
            row["gops"],
            marker="o",
            s=max(35, min(220, row["duration_us"] / 35000.0)),
            color=phase_colors.get(row["phase"], "#7f7f7f"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
    for row in estimated:
        ax0.scatter(
            row["arithmetic_intensity_ops_per_byte"],
            row["gops"],
            marker="x",
            s=45,
            color=phase_colors.get(row["phase"], "#7f7f7f"),
            alpha=0.75,
        )

    top_labels = sorted(exact, key=lambda r: r["duration_us"], reverse=True)[:6]
    peak_label = max(exact, key=lambda r: r["gops"], default=None)
    if peak_label is not None and peak_label not in top_labels:
        top_labels.append(peak_label)
    for row in top_labels:
        label = f"{row['phase']}:{row['label'].split('|')[0]}"
        ax0.annotate(label[:34], (row["arithmetic_intensity_ops_per_byte"], row["gops"]), fontsize=7, xytext=(4, 3), textcoords="offset points")

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("Arithmetic intensity (ops/byte)")
    ax0.set_ylabel("Throughput (GOPS)")
    ax0.set_title("CPU baseline roofline")
    ax0.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax0.legend(fontsize=8, loc="lower right")

    phase_op = {}
    for row in rows:
        key = (row["phase"], row["operator"])
        phase_op[key] = phase_op.get(key, 0.0) + row["duration_us"] / 1000.0
    top = sorted(phase_op.items(), key=lambda kv: kv[1], reverse=True)[:14]
    labels = [f"{phase}\\n{op}" for (phase, op), _ in top]
    values = [value for _, value in top]
    colors = [phase_colors.get(phase, "#7f7f7f") for (phase, _), _ in top]
    ax1.barh(range(len(values)), values, color=colors)
    ax1.set_yticks(range(len(values)), labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Duration (ms)")
    ax1.set_title("Top operator buckets by time")
    ax1.grid(True, axis="x", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mtmd, rows = load_rows(result_dir)
    rows = sorted(rows, key=lambda r: (r["phase"], -r["duration_us"], r["operator"], r["label"]))

    exact = [r for r in rows if r["flop_source"] == "exact_matmul_shape" and r["gops"] > 0]
    bandwidth_candidates = [r for r in rows if r["bandwidth_gbs"] > 0 and r["ops"] > 0]
    layout_bandwidth_candidates = [r for r in rows if r["bandwidth_gbs"] > 0 and r["ops"] == 0]
    summary = {
        "result_dir": str(result_dir),
        "prefill_total_us": mtmd.get("prefill_total_us"),
        "mmproj_encode_us": mtmd.get("mmproj", {}).get("encode_us"),
        "observed_peak_gops": max((r["gops"] for r in exact), default=0.0),
        "observed_peak_gops_row": max(exact, key=lambda r: r["gops"], default=None),
        "observed_peak_bandwidth_gbs": max((r["bandwidth_gbs"] for r in bandwidth_candidates), default=0.0),
        "observed_peak_bandwidth_row": max(bandwidth_candidates, key=lambda r: r["bandwidth_gbs"], default=None),
        "observed_peak_layout_rate_gbs": max((r["bandwidth_gbs"] for r in layout_bandwidth_candidates), default=0.0),
        "observed_peak_layout_rate_row": max(layout_bandwidth_candidates, key=lambda r: r["bandwidth_gbs"], default=None),
        "total_exact_matmul_gop": sum(r["ops"] for r in exact) / 1e9,
        "total_exact_matmul_duration_ms": sum(r["duration_us"] for r in exact) / 1000.0,
        "notes": [
            "MUL_MAT FLOPs are exact shape estimates using 2*M*N*K*batch.",
            "MUL_MAT bytes are estimated as src0 + src1 + dst tensor traffic.",
            "Non-MUL operator FLOPs are heuristic element-count estimates; their profiler bytes are output-buffer bytes and should be treated as lower-bound traffic.",
            "The plotted roofline uses observed peaks from this run, not a standalone synthetic CPU microbenchmark.",
        ],
    }

    write_csv(out_dir / "cpu_roofline_rows.csv", rows)
    write_csv(out_dir / "cpu_roofline_matmul_rows.csv", exact)
    with open(out_dir / "cpu_roofline_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plot_roofline(out_dir / "cpu_baseline_roofline.png", rows, summary)

    print(json.dumps({
        "rows": len(rows),
        "matmul_rows": len(exact),
        "summary": str(out_dir / "cpu_roofline_summary.json"),
        "csv": str(out_dir / "cpu_roofline_rows.csv"),
        "matmul_csv": str(out_dir / "cpu_roofline_matmul_rows.csv"),
        "plot": str(out_dir / "cpu_baseline_roofline.png"),
        "observed_peak_gops": summary["observed_peak_gops"],
        "observed_peak_bandwidth_gbs": summary["observed_peak_bandwidth_gbs"],
    }, indent=2))


if __name__ == "__main__":
    main()
