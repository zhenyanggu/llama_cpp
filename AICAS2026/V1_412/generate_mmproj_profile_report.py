#!/usr/bin/env python3

import argparse
import datetime as dt
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(numerator, denominator):
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def ms_from_us(value):
    return float(value) / 1000.0


def ms_from_ns(value):
    return float(value) / 1_000_000.0


def top_n(items, key, n):
    return sorted(items, key=key, reverse=True)[:n]


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def summarize_semantic_ops(layer_summary):
    merged = {}
    for layer in layer_summary:
        key = layer["semantic_op"]
        agg = merged.setdefault(
            key,
            {
                "semantic_op": key,
                "layer_count": 0,
                "exec_tile_count": 0,
                "total_node_us": 0.0,
                "runtime_total_ns": 0,
                "runtime_dma_in_ns": 0,
                "runtime_compute_ns": 0,
                "runtime_dma_out_ns": 0,
                "runtime_layout_ns": 0,
                "runtime_wait_irq_ns": 0,
                "host_overhead_us": 0.0,
                "gemm_us": 0.0,
                "dma_out_us": 0.0,
                "examples": [],
            },
        )
        agg["layer_count"] += 1
        agg["exec_tile_count"] += layer.get("exec_tile_count", 0)
        agg["total_node_us"] += layer.get("total_node_us", 0.0)
        agg["runtime_total_ns"] += layer.get("runtime_total_ns", 0)
        agg["runtime_dma_in_ns"] += layer.get("runtime_dma_in_ns", 0)
        agg["runtime_compute_ns"] += layer.get("runtime_compute_ns", 0)
        agg["runtime_dma_out_ns"] += layer.get("runtime_dma_out_ns", 0)
        agg["runtime_layout_ns"] += layer.get("runtime_layout_ns", 0)
        agg["runtime_wait_irq_ns"] += layer.get("runtime_wait_irq_ns", 0)
        agg["host_overhead_us"] += layer.get("host_overhead_us", 0.0)
        agg["gemm_us"] += layer.get("gemm_us_total", 0.0)
        agg["dma_out_us"] += layer.get("dma_out_us_total", 0.0)
        example = layer.get("root_name") or layer.get("weight_name")
        if example and example not in agg["examples"] and len(agg["examples"]) < 3:
            agg["examples"].append(example)

    out = []
    total_node_us = sum(item["total_node_us"] for item in merged.values())
    for item in merged.values():
        item["total_node_ms"] = ms_from_us(item["total_node_us"])
        item["runtime_total_ms"] = ms_from_ns(item["runtime_total_ns"])
        item["share_of_npu_node_trace_pct"] = pct(item["total_node_us"], total_node_us)
        out.append(item)
    return sorted(out, key=lambda x: x["total_node_us"], reverse=True)


def build_layer_summary(node_trace, runtime_profile):
    runtime_layers = {
        int(layer["layer_id"]): layer for layer in runtime_profile.get("layers", [])
    }
    node_layers = {
        int(layer["layer_id"]): layer for layer in node_trace.get("nodes", [])
    }

    layer_ids = sorted(set(runtime_layers) | set(node_layers))
    total_node_us = safe_get(node_trace, "summary", "total_node_us", default=0.0)
    out = []

    for layer_id in layer_ids:
        runtime = runtime_layers.get(layer_id, {})
        node = node_layers.get(layer_id, {})
        root_name = safe_get(node, "root_tensor", "name")
        weight_name = safe_get(node, "weight_tensor", "name")
        dst_name = safe_get(node, "dst_tensor", "name")
        semantic_op = node.get("semantic_op") or runtime.get("origin_op_type") or "unknown"
        host_overhead_us = (
            node.get("activation_pack_us_total", 0.0)
            + node.get("host_copy_activation_us_total", 0.0)
            + node.get("host_copy_weight_us_total", 0.0)
            + node.get("bias_prepare_us_total", 0.0)
            + node.get("postprocess_us_total", 0.0)
        )
        top_tiles = top_n(node.get("tiles", []), key=lambda x: x.get("total_us", 0.0), n=8)

        layer = {
            "layer_id": layer_id,
            "semantic_op": semantic_op,
            "root_name": root_name,
            "weight_name": weight_name,
            "dst_name": dst_name,
            "root_op_name": node.get("root_op_name"),
            "compute_op_name": node.get("compute_op_name"),
            "device": runtime.get("device", "npu" if node else "unknown"),
            "m": node.get("m"),
            "n": node.get("n"),
            "k": node.get("k"),
            "use_aicas_w8a8": node.get("use_aicas_w8a8"),
            "activation_scale": node.get("activation_scale"),
            "activation_zero_point": node.get("activation_zero_point"),
            "summary": node.get("summary"),
            "exec_tile_count": node.get("exec_tile_count", 0),
            "weight_pack_count": node.get("weight_pack_count", 0),
            "bias_pack_count": node.get("bias_pack_count", 0),
            "total_node_us": node.get("total_node_us", 0.0),
            "total_node_ms": ms_from_us(node.get("total_node_us", 0.0)),
            "share_of_npu_node_trace_pct": pct(node.get("total_node_us", 0.0), total_node_us),
            "runtime_total_ns": runtime.get("total_ns", 0),
            "runtime_total_ms": ms_from_ns(runtime.get("total_ns", 0)),
            "runtime_dma_in_ns": runtime.get("dma_in_ns", 0),
            "runtime_compute_ns": runtime.get("compute_ns", 0),
            "runtime_dma_out_ns": runtime.get("dma_out_ns", 0),
            "runtime_layout_ns": runtime.get("layout_ns", 0),
            "runtime_wait_irq_ns": runtime.get("wait_irq_ns", 0),
            "runtime_other_ns": runtime.get("other_ns", 0),
            "runtime_mvin_calls": runtime.get("mvin_calls", 0),
            "runtime_compute_calls": runtime.get("compute_calls", 0),
            "runtime_mvout_calls": runtime.get("mvout_calls", 0),
            "runtime_layout_calls": runtime.get("layout_calls", 0),
            "activation_pack_us_total": node.get("activation_pack_us_total", 0.0),
            "host_copy_activation_us_total": node.get("host_copy_activation_us_total", 0.0),
            "host_copy_weight_us_total": node.get("host_copy_weight_us_total", 0.0),
            "bias_prepare_us_total": node.get("bias_prepare_us_total", 0.0),
            "host_overhead_us": host_overhead_us,
            "dma_in_activation_us_total": node.get("dma_in_activation_us_total", 0.0),
            "dma_in_weight_us_total": node.get("dma_in_weight_us_total", 0.0),
            "dma_in_bias_us_total": node.get("dma_in_bias_us_total", 0.0),
            "dma_in_total_us": node.get("dma_in_total_us", 0.0),
            "gemm_us_total": node.get("gemm_us_total", 0.0),
            "dma_out_us_total": node.get("dma_out_us_total", 0.0),
            "postprocess_us_total": node.get("postprocess_us_total", 0.0),
            "top_tiles": top_tiles,
            "status": node.get("status"),
            "error": node.get("error"),
        }
        out.append(layer)

    return sorted(out, key=lambda x: x["total_node_us"], reverse=True)


def build_report(args, mmproj_run, mmproj_operator, runtime_profile, node_trace):
    layer_summary = build_layer_summary(node_trace, runtime_profile)
    semantic_op_summary = summarize_semantic_ops(layer_summary)

    npu_names = set()
    for layer in layer_summary:
        for name in (layer.get("root_name"), layer.get("weight_name"), layer.get("dst_name")):
            if name:
                npu_names.add(name)

    ggml_top_nodes = mmproj_operator.get("top_nodes", [])
    non_npu_hot_nodes = [
        node for node in ggml_top_nodes
        if safe_get(node, "output", "name") not in npu_names
    ][:10]

    mmproj_total_us = mmproj_operator.get("total_us", 0.0)
    npu_node_total_us = safe_get(node_trace, "summary", "total_node_us", default=0.0)
    runtime_total_ns = safe_get(runtime_profile, "summary", "npu_total_ns", default=0)

    stage_breakdown = {
        "runtime_dma_in_ms": ms_from_ns(sum(layer["runtime_dma_in_ns"] for layer in layer_summary)),
        "runtime_compute_ms": ms_from_ns(sum(layer["runtime_compute_ns"] for layer in layer_summary)),
        "runtime_dma_out_ms": ms_from_ns(sum(layer["runtime_dma_out_ns"] for layer in layer_summary)),
        "runtime_layout_ms": ms_from_ns(sum(layer["runtime_layout_ns"] for layer in layer_summary)),
        "runtime_wait_irq_ms": ms_from_ns(sum(layer["runtime_wait_irq_ns"] for layer in layer_summary)),
        "host_pack_copy_post_ms": ms_from_us(sum(layer["host_overhead_us"] for layer in layer_summary)),
        "ggml_npu_gemm_call_ms": ms_from_us(sum(layer["gemm_us_total"] for layer in layer_summary)),
    }

    report = {
        "profile_kind": "kv260_mmproj_npu_performance_report",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {
            "board": args.board,
            "model": mmproj_run.get("model"),
            "mmproj": mmproj_run.get("mmproj"),
            "images": mmproj_run.get("images"),
            "threads": mmproj_run.get("threads"),
            "mode": mmproj_run.get("mode"),
        },
        "raw_artifacts": {
            "mmproj_run": str(args.mmproj_run),
            "mmproj_operator": str(args.mmproj_operator),
            "npu_runtime_profile": str(args.npu_runtime_profile),
            "npu_node_trace": str(args.npu_node_trace),
        },
        "raw_summaries": {
            "mmproj_run": mmproj_run,
            "mmproj_operator_summary": {
                "node_event_count": mmproj_operator.get("node_event_count"),
                "total_us": mmproj_total_us,
                "mul_mat_total_us": mmproj_operator.get("mul_mat_total_us"),
                "operators": mmproj_operator.get("operators", []),
            },
            "npu_runtime_summary": runtime_profile.get("summary", {}),
            "npu_node_trace_summary": node_trace.get("summary", {}),
        },
        "cross_checks": {
            "mmproj_total_us": mmproj_total_us,
            "npu_node_trace_total_us": npu_node_total_us,
            "runtime_npu_total_us": runtime_total_ns / 1000.0,
            "npu_covered_share_of_mmproj_pct": pct(npu_node_total_us, mmproj_total_us),
            "runtime_vs_node_trace_ratio": (runtime_total_ns / 1000.0 / npu_node_total_us) if npu_node_total_us else 0.0,
            "note": "mmproj operator profile is full ggml image-encode replay time. NPU node trace and runtime profile cover only the offloaded NPU subset, so the totals are expected to differ.",
        },
        "stage_breakdown": stage_breakdown,
        "layer_summary": layer_summary,
        "semantic_op_summary": semantic_op_summary,
        "ggml_operator_summary": mmproj_operator.get("operators", []),
        "ggml_mul_mat_signatures": mmproj_operator.get("mul_mat_signatures", []),
        "ggml_top_nodes": ggml_top_nodes,
        "non_npu_hot_mmproj_nodes": non_npu_hot_nodes,
        "npu_hot_layers": layer_summary[:15],
    }
    return report


def markdown_table(headers, rows):
    if not rows:
        return "_No data._\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def build_markdown(report):
    input_info = report["input"]
    stage = report["stage_breakdown"]
    cross = report["cross_checks"]
    hot_layers = report["npu_hot_layers"]
    ggml_ops = report["ggml_operator_summary"][:12]
    ggml_nodes = report["ggml_top_nodes"][:12]
    non_npu_nodes = report["non_npu_hot_mmproj_nodes"][:10]
    semantic_ops = report["semantic_op_summary"][:12]

    hottest_layer = hot_layers[0] if hot_layers else {}
    hottest_ggml_op = ggml_ops[0] if ggml_ops else {}
    hottest_ggml_node = ggml_nodes[0] if ggml_nodes else {}

    lines = []
    lines.append("# KV260 mmproj Performance Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Board: `{input_info.get('board')}`")
    lines.append(f"- Model: `{input_info.get('model')}`")
    lines.append(f"- mmproj: `{input_info.get('mmproj')}`")
    lines.append(f"- Image: `{', '.join(input_info.get('images') or [])}`")
    lines.append(f"- Threads: `{input_info.get('threads')}`")
    lines.append(f"- mmproj total profiled time: `{ms_from_us(cross.get('mmproj_total_us', 0.0)):.3f} ms`")
    lines.append(f"- NPU-covered node trace time: `{ms_from_us(cross.get('npu_node_trace_total_us', 0.0)):.3f} ms` (`{cross.get('npu_covered_share_of_mmproj_pct', 0.0):.2f}%` of mmproj replay)")
    lines.append(f"- Runtime NPU total: `{cross.get('runtime_npu_total_us', 0.0) / 1000.0:.3f} ms`")
    if hottest_layer:
        lines.append(
            f"- Hottest NPU layer: `{hottest_layer.get('semantic_op')}` / `{hottest_layer.get('root_name') or hottest_layer.get('weight_name')}` "
            f"at `{hottest_layer.get('total_node_ms', 0.0):.3f} ms`"
        )
    if hottest_ggml_op:
        lines.append(
            f"- Hottest ggml operator: `{hottest_ggml_op.get('operator_name')}` at `{hottest_ggml_op.get('duration_ms', 0.0):.3f} ms`"
        )
    if hottest_ggml_node:
        lines.append(
            f"- Hottest ggml node: `{safe_get(hottest_ggml_node, 'output', 'name', default='')}` "
            f"at `{hottest_ggml_node.get('duration_ms', 0.0):.3f} ms`"
        )
    lines.append("")
    lines.append("## NPU Stage Breakdown")
    lines.append("")
    lines.append(markdown_table(
        ["Stage", "Time (ms)"],
        [
            ["Runtime DMA In", f"{stage['runtime_dma_in_ms']:.3f}"],
            ["Runtime Compute", f"{stage['runtime_compute_ms']:.3f}"],
            ["Runtime DMA Out", f"{stage['runtime_dma_out_ms']:.3f}"],
            ["Runtime Layout", f"{stage['runtime_layout_ms']:.3f}"],
            ["Runtime Wait IRQ", f"{stage['runtime_wait_irq_ms']:.3f}"],
            ["Host Pack/Copy/Post", f"{stage['host_pack_copy_post_ms']:.3f}"],
            ["ggml-npu GEMM Call", f"{stage['ggml_npu_gemm_call_ms']:.3f}"],
        ],
    ))
    lines.append("## Hot NPU Layers")
    lines.append("")
    lines.append(markdown_table(
        ["Layer", "Semantic", "M/N/K", "Tiles", "Node ms", "Runtime ms", "DMA In ms", "Compute ms", "Wait IRQ ms"],
        [
            [
                layer["layer_id"],
                layer["semantic_op"],
                f"{layer.get('m')}/{layer.get('n')}/{layer.get('k')}",
                layer.get("exec_tile_count", 0),
                f"{layer.get('total_node_ms', 0.0):.3f}",
                f"{layer.get('runtime_total_ms', 0.0):.3f}",
                f"{ms_from_ns(layer.get('runtime_dma_in_ns', 0)):.3f}",
                f"{ms_from_ns(layer.get('runtime_compute_ns', 0)):.3f}",
                f"{ms_from_ns(layer.get('runtime_wait_irq_ns', 0)):.3f}",
            ]
            for layer in hot_layers[:15]
        ],
    ))
    lines.append("## Semantic Op Summary")
    lines.append("")
    lines.append(markdown_table(
        ["Semantic Op", "Layers", "Node ms", "Runtime ms", "Runtime Compute ms", "Examples"],
        [
            [
                item["semantic_op"],
                item["layer_count"],
                f"{item['total_node_ms']:.3f}",
                f"{item['runtime_total_ms']:.3f}",
                f"{ms_from_ns(item['runtime_compute_ns']):.3f}",
                ", ".join(item.get("examples", [])),
            ]
            for item in semantic_ops
        ],
    ))
    lines.append("## ggml Operator Summary")
    lines.append("")
    lines.append(markdown_table(
        ["Operator", "Time ms", "Share %", "Events"],
        [
            [
                op.get("operator_name"),
                f"{op.get('duration_ms', 0.0):.3f}",
                f"{op.get('share_of_total_time_pct', 0.0):.2f}",
                op.get("node_event_count", 0),
            ]
            for op in ggml_ops
        ],
    ))
    lines.append("## ggml Hot Nodes")
    lines.append("")
    lines.append(markdown_table(
        ["Node", "Operator", "Time ms", "Share %"],
        [
            [
                safe_get(node, "output", "name", default=""),
                node.get("operator_name", ""),
                f"{node.get('duration_ms', 0.0):.3f}",
                f"{node.get('share_of_total_time_pct', 0.0):.2f}",
            ]
            for node in ggml_nodes
        ],
    ))
    lines.append("## Non-NPU Hot Nodes")
    lines.append("")
    lines.append(markdown_table(
        ["Node", "Operator", "Time ms", "Share %"],
        [
            [
                safe_get(node, "output", "name", default=""),
                node.get("operator_name", ""),
                f"{node.get('duration_ms', 0.0):.3f}",
                f"{node.get('share_of_total_time_pct', 0.0):.2f}",
            ]
            for node in non_npu_nodes
        ],
    ))
    lines.append("## Notes")
    lines.append("")
    lines.append("- `mmproj_operator` totals come from ggml eval-callback replay of the full image-encode graph.")
    lines.append("- `npu_node_trace` and `npu_runtime_profile` only cover the NPU-offloaded subset, so totals are expected to be smaller.")
    lines.append("- Runtime `wait_irq` is counted separately and is useful for understanding hardware-side stall/latency not visible from the ggml wall-clock split.")
    lines.append("- Host `pack/copy/postprocess` comes from `ggml-npu` user-space instrumentation, not from the runtime JSON.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge KV260 mmproj raw profiling artifacts into a final JSON + markdown report.")
    parser.add_argument("--mmproj-run", type=Path, required=True)
    parser.add_argument("--mmproj-operator", type=Path, required=True)
    parser.add_argument("--npu-runtime-profile", type=Path, required=True)
    parser.add_argument("--npu-node-trace", type=Path, required=True)
    parser.add_argument("--board", default="ubuntu@192.168.0.10")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    mmproj_run = load_json(args.mmproj_run)
    mmproj_operator = load_json(args.mmproj_operator)
    runtime_profile = load_json(args.npu_runtime_profile)
    node_trace = load_json(args.npu_node_trace)

    report = build_report(args, mmproj_run, mmproj_operator, runtime_profile, node_trace)
    markdown = build_markdown(report)

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
