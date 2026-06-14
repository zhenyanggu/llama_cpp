#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_value(raw):
    if raw == "":
        return raw
    try:
        if any(ch in raw for ch in ".eE"):
            value = float(raw)
            if math.isfinite(value):
                return value
            return raw
        return int(raw, 0)
    except ValueError:
        return raw


def parse_kv_line(line):
    fields = line.strip().replace(",", " ").split()
    if not fields:
        return None, {}
    tag = fields[0]
    data = {}
    for field in fields[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        data[key] = parse_value(value)
    return tag, data


def load_records(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        tag, data = parse_kv_line(line)
        if tag in {
            "overlap_prefill_profile",
            "overlap_prefill_ablation",
            "overlap_output_check",
            "overlap_profile_counters",
            "overlap_exposed_wait",
            "overlap_cpu_prepare",
        }:
            data["_tag"] = tag
            records.append(data)
    return records


def require_record(records, tag, case=None):
    for record in records:
        if record.get("_tag") != tag:
            continue
        if case is not None and record.get("case") != case:
            continue
        return record
    suffix = f" case={case}" if case else ""
    raise SystemExit(f"missing record: {tag}{suffix}")


def f(record, key, default=0.0):
    value = record.get(key, default)
    return float(value) if value is not None else float(default)


def build_summary(args):
    records = load_records(args.log)
    serial = require_record(records, "overlap_prefill_profile", "serial_estimated")
    pingpong = require_record(records, "overlap_prefill_profile", "gemm_pingpong_only")
    full = require_record(records, "overlap_prefill_profile", "full_pipeline")
    ablation = require_record(records, "overlap_prefill_ablation")
    output_check = require_record(records, "overlap_output_check")
    pingpong_counters = require_record(records, "overlap_profile_counters", "gemm_pingpong_only")
    full_counters = require_record(records, "overlap_profile_counters", "full_pipeline")
    pingpong_wait = require_record(records, "overlap_exposed_wait", "gemm_pingpong_only")
    cpu_prepare = require_record(records, "overlap_cpu_prepare", "host_prepare_overlap")
    mvin_total_ms = f(pingpong_wait, "mvin_preload_total_ms")
    mvin_exposed_ms = f(pingpong_wait, "exposed_mvin_wait_ms")
    mvin_startup_ms = f(pingpong_wait, "startup_mvin_wait_ms")
    mvin_hidden_steady_ms = max(0.0, mvin_total_ms - mvin_exposed_ms - mvin_startup_ms)
    mvin_steady_total_ms = max(0.0, mvin_total_ms - mvin_startup_ms)

    serial_ms = f(ablation, "serial_ms")
    pingpong_ms = f(ablation, "pingpong_only_ms")
    full_ms = f(ablation, "full_overlap_ms")
    transfer_ms = f(serial, "mvin_a_ms") + f(serial, "mvin_w_ms") + f(serial, "mvout_ms")
    compute_ms = f(serial, "gemm_ms")
    host_ms = f(serial, "host_prepare_ms")
    output_copy_ms = f(serial, "output_copy_ms")
    other_ms = f(serial, "other_ms")

    configs = [
        {
            "config": "serial_estimated",
            "latency_ms": serial_ms,
            "measured": True,
            "description": "Blocking host prepare, CMA mvin, GEMM, mvout, and output copy.",
        },
        {
            "config": "gemm_pingpong_only",
            "latency_ms": pingpong_ms,
            "measured": False,
            "description": "Measured device ping-pong wall time plus serial host prepare/output-copy time.",
        },
        {
            "config": "full_pipeline",
            "latency_ms": full_ms,
            "measured": True,
            "description": "Runtime pipeline with host prepare/CMA transfer overlapped with GEMM where possible.",
        },
    ]

    result = {
        "schema": "aicas_prefill_overlap_ablation.v1",
        "run_id": args.run_id,
        "overlay": args.overlay,
        "phase": "prefill",
        "operator": "gemm",
        "shape": {
            "name": ablation.get("shape"),
            "m": ablation.get("M"),
            "k": ablation.get("K"),
            "n": ablation.get("N"),
            "output_tiles": serial.get("output_tiles"),
            "gemm_chunks": serial.get("gemm_chunks"),
        },
        "plot_recommendation": {
            "title": "Two-Level Overlap Ablation in Prefill GEMM",
            "chart": "horizontal bars",
            "x_axis": "Latency (ms)",
            "bars": ["serial_estimated", "gemm_pingpong_only", "full_pipeline"],
            "annotations": [
                "pingpong_gain = serial_estimated / gemm_pingpong_only",
                "host_pipeline_gain = gemm_pingpong_only / full_pipeline",
                "full_gain = serial_estimated / full_pipeline",
            ],
        },
        "latency_breakdown_ms": {
            "serial_total": serial_ms,
            "serial_host_prepare": host_ms,
            "serial_cma_transfer": transfer_ms,
            "serial_mvin_a": f(serial, "mvin_a_ms"),
            "serial_mvin_w": f(serial, "mvin_w_ms"),
            "serial_gemm_compute": compute_ms,
            "serial_mvout": f(serial, "mvout_ms"),
            "serial_output_copy": output_copy_ms,
            "serial_other": other_ms,
            "pingpong_device_wall": f(pingpong, "device_wall_ms"),
            "full_pipeline_wall": full_ms,
        },
        "configs": configs,
        "metrics": {
            "pingpong_gain": f(ablation, "pingpong_gain"),
            "host_pipeline_gain": f(ablation, "host_pipeline_gain"),
            "full_gain": f(ablation, "full_gain"),
            "pingpong_hidden_ms": f(ablation, "pingpong_hidden_ms"),
            "full_hidden_ms": f(ablation, "full_hidden_ms"),
            "transfer_hidden_ratio_pingpong": f(ablation, "transfer_hidden_ratio_pingpong"),
            "full_pipeline_reduction_ratio": f(ablation, "transfer_hidden_ratio_full"),
        },
        "matrix_transfer_overlap": {
            "target": "GEMM next-tile matrix preload",
            "mvin_total_ms": mvin_total_ms,
            "residual_exposed_wait_ms": mvin_exposed_ms,
            "startup_wait_ms": mvin_startup_ms,
            "critical_path_wait_ms": mvin_exposed_ms + mvin_startup_ms,
            "hidden_mvin_ms": mvin_hidden_steady_ms,
            "overall_hidden_ratio_including_startup": (
                mvin_hidden_steady_ms / mvin_total_ms if mvin_total_ms > 0.0 else 0.0
            ),
            "steady_state_hidden_ratio_excluding_startup": (
                mvin_hidden_steady_ms / mvin_steady_total_ms
                if mvin_steady_total_ms > 0.0 else 0.0
            ),
            "steady_state_residual_exposed_ratio": (
                mvin_exposed_ms / mvin_steady_total_ms
                if mvin_steady_total_ms > 0.0 else 0.0
            ),
            "preload_count": pingpong_wait.get("preload_count"),
            "exposed_wait_count": pingpong_wait.get("exposed_wait_count"),
            "definition": (
                "residual_exposed_wait_ms is the wait for next matrix preload "
                "after the current GEMM finishes. startup_wait_ms is first-tile "
                "warmup and is not counted as hidden. hidden_mvin_ms is measured "
                "preload time hidden in steady-state overlap."
            ),
        },
        "cpu_prepare_overlap": {
            "target": "CPU host prepare for GEMM tile packing/copy",
            "cpu_prepare_total_ms": f(cpu_prepare, "cpu_prepare_total_ms"),
            "hidden_cpu_prepare_ms": f(cpu_prepare, "cpu_prepare_hidden_ms"),
            "exposed_cpu_prepare_ms": f(cpu_prepare, "cpu_prepare_exposed_ms"),
            "startup_cpu_prepare_ms": f(cpu_prepare, "cpu_prepare_startup_ms"),
            "hidden_ratio_including_startup": f(cpu_prepare, "hidden_ratio"),
            "steady_state_hidden_ratio_excluding_startup": f(cpu_prepare, "steady_hidden_ratio"),
            "prepare_count": cpu_prepare.get("prepare_count"),
            "definition": cpu_prepare.get(
                "definition",
                "cpu_pack_copy_interval_overlap_with_gemm_compute_interval",
            ),
        },
        "profile_counters": {
            "gemm_pingpong_only": {
                k: pingpong_counters[k]
                for k in pingpong_counters
                if k not in {"_tag", "case"}
            },
            "full_pipeline": {
                k: full_counters[k]
                for k in full_counters
                if k not in {"_tag", "case"}
            },
        },
        "validity": {
            "serial_vs_pingpong_mismatches": output_check.get("serial_vs_pingpong_mismatches"),
            "serial_vs_full_mismatches": output_check.get("serial_vs_full_mismatches"),
            "elements": output_check.get("elements"),
            "decode_excluded_reason": (
                "Decode overlay is a fixed streaming hardware pipeline, so a true "
                "non-streaming ablation is not available for the single main overlap figure."
            ),
        },
        "source": {
            "log": str(Path(args.log).resolve()),
            "test_binary": "versa_p_prefill_overlap_ablation_test",
        },
    }
    return result


def write_notes(path, result):
    metrics = result["metrics"]
    transfer = result["matrix_transfer_overlap"]
    cpu_prepare = result["cpu_prepare_overlap"]
    breakdown = result["latency_breakdown_ms"]
    rows = result["configs"]
    lines = [
        "# Prefill overlap ablation",
        "",
        f"- run_id: `{result['run_id']}`",
        f"- overlay: `{result['overlay']}`",
        f"- shape: `{result['shape']['name']}` M={result['shape']['m']} K={result['shape']['k']} N={result['shape']['n']}",
        "",
        "| config | latency ms | note |",
        "| --- | ---: | --- |",
    ]
    for row in rows:
        measured = "measured" if row["measured"] else "estimated from measured parts"
        lines.append(
            f"| {row['config']} | {row['latency_ms']:.6f} | {measured} |"
        )
    lines.extend(
        [
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| pingpong_gain | {metrics['pingpong_gain']:.6f} |",
            f"| host_pipeline_gain | {metrics['host_pipeline_gain']:.6f} |",
            f"| full_gain | {metrics['full_gain']:.6f} |",
            f"| pingpong_hidden_ms | {metrics['pingpong_hidden_ms']:.6f} |",
            f"| full_hidden_ms | {metrics['full_hidden_ms']:.6f} |",
            f"| transfer_hidden_ratio_pingpong | {metrics['transfer_hidden_ratio_pingpong']:.6f} |",
            f"| full_pipeline_reduction_ratio | {metrics['full_pipeline_reduction_ratio']:.6f} |",
            "",
            "| matrix transfer overlap metric | value |",
            "| --- | ---: |",
            f"| mvin_total_ms | {transfer['mvin_total_ms']:.6f} |",
            f"| residual_exposed_wait_ms | {transfer['residual_exposed_wait_ms']:.6f} |",
            f"| startup_wait_ms | {transfer['startup_wait_ms']:.6f} |",
            f"| critical_path_wait_ms | {transfer['critical_path_wait_ms']:.6f} |",
            f"| hidden_mvin_ms | {transfer['hidden_mvin_ms']:.6f} |",
            f"| overall_hidden_ratio_including_startup | {transfer['overall_hidden_ratio_including_startup']:.6f} |",
            f"| steady_state_hidden_ratio_excluding_startup | {transfer['steady_state_hidden_ratio_excluding_startup']:.6f} |",
            f"| steady_state_residual_exposed_ratio | {transfer['steady_state_residual_exposed_ratio']:.6f} |",
            "",
            "| CPU prepare overlap metric | value |",
            "| --- | ---: |",
            f"| cpu_prepare_total_ms | {cpu_prepare['cpu_prepare_total_ms']:.6f} |",
            f"| hidden_cpu_prepare_ms | {cpu_prepare['hidden_cpu_prepare_ms']:.6f} |",
            f"| exposed_cpu_prepare_ms | {cpu_prepare['exposed_cpu_prepare_ms']:.6f} |",
            f"| startup_cpu_prepare_ms | {cpu_prepare['startup_cpu_prepare_ms']:.6f} |",
            f"| hidden_ratio_including_startup | {cpu_prepare['hidden_ratio_including_startup']:.6f} |",
            f"| steady_state_hidden_ratio_excluding_startup | {cpu_prepare['steady_state_hidden_ratio_excluding_startup']:.6f} |",
            "",
            "Serial breakdown for stacked/hatching plot:",
            "",
            f"- host_prepare_ms: {breakdown['serial_host_prepare']:.6f}",
            f"- cma_transfer_ms: {breakdown['serial_cma_transfer']:.6f}",
            f"- gemm_compute_ms: {breakdown['serial_gemm_compute']:.6f}",
            f"- output_copy_ms: {breakdown['serial_output_copy']:.6f}",
            "",
            "Decode is intentionally not included in the main figure because the decode overlay is already a fixed streaming pipeline and has no measured non-streaming ablation baseline.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()

    result = build_summary(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_notes(args.notes, result)


if __name__ == "__main__":
    main()
