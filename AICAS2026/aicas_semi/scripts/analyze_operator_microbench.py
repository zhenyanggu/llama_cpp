#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from statistics import median
from typing import Any


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^,\s]+)")


def parse_kv_line(line: str) -> tuple[str, dict[str, str]] | None:
    head = line.split(",", 1)[0].strip()
    tag = head.split(None, 1)[0].strip()
    if not tag:
        return None
    values = {key: value for key, value in KV_RE.findall(line)}
    return tag, values


def read_lines(path: Path) -> list[str]:
    if not path or not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_cpu_log(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in read_lines(path):
        parsed = parse_kv_line(line)
        if not parsed:
            continue
        tag, values = parsed
        if tag != "cpu_operator_perf":
            continue
        op = values.get("operator")
        if not op:
            continue
        out[op] = {
            "operator": op,
            "shape": values.get("shape", ""),
            "median_ms": as_float(values.get("median_ms")),
            "effective_gops": as_float(values.get("effective_gops")),
            "repeats": as_int(values.get("repeats")),
            "raw_line": line,
        }
    return out


def parse_prefill_gemm_log(path: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for line in read_lines(path):
        parsed = parse_kv_line(line)
        if not parsed:
            continue
        tag, values = parsed
        if tag != "phase_profile":
            continue
        layer = values.get("layer", "")
        if layer != "m5_ffn_up_768x768x2048":
            continue
        candidates.append({
            "operator": "gemm_prefill",
            "shape": "M768_K768_N2048",
            "npu_total_ms": as_float(values.get("total_ms")),
            "npu_compute_ms": as_float(values.get("gemm_ms")),
            "npu_hw_ms": as_float(values.get("hw_ms")),
            "source_test": "versa_p_layer_phase_profile_test",
            "raw_line": line,
        })
    return candidates[-1] if candidates else None


def parse_prefill_attention_log(path: Path) -> dict[str, Any] | None:
    for line in read_lines(path):
        parsed = parse_kv_line(line)
        if not parsed:
            continue
        tag, values = parsed
        if tag != "text_attention_qk_pv_log8_pingpong_summary":
            continue
        if values.get("T") != "1024":
            continue
        return {
            "operator": "attention_prefill",
            "shape": "T1024_QH15_KVH5_D64_causal",
            "npu_total_ms": as_float(values.get("layer_wall_ms")),
            "npu_compute_ms": cycles_to_ms(values, "gemm_busy_cycles", "global_cycles", "layer_wall_ms"),
            "source_test": "versa_p_attention_qk_pv_log8_test",
            "raw_line": line,
        }
    return None


def cycles_to_ms(values: dict[str, str], cycles_key: str, global_key: str, wall_key: str) -> float:
    cycles = as_float(values.get(cycles_key))
    global_cycles = as_float(values.get(global_key))
    wall_ms = as_float(values.get(wall_key))
    if cycles <= 0.0 or global_cycles <= 0.0 or wall_ms <= 0.0:
        return 0.0
    return wall_ms * cycles / global_cycles


def parse_decode_log(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    qkv_o = {"attn_q_proj", "attn_k_proj", "attn_v_proj", "attn_o_proj"}
    qk_total = 0.0
    pv_total = 0.0
    projection_total = 0.0
    gemv_record: dict[str, Any] | None = None

    for line in read_lines(path):
        parsed = parse_kv_line(line)
        if not parsed:
            continue
        tag, values = parsed
        if tag == "decode_full_perf":
            label = line.split(",", 2)[1] if line.count(",") >= 2 else ""
            if label == "attn_q_proj":
                gemv_record = {
                    "operator": "gemv_decode",
                    "shape": "M960_K960_N1",
                    "npu_total_ms": as_float(values.get("avg_ms")),
                    "npu_compute_ms": as_float(values.get("avg_ms")),
                    "effective_gops": as_float(values.get("effective_gops")),
                    "source_test": "kv260_decode_full_stream_perf_test",
                    "raw_line": line,
                }
            if label in qkv_o:
                projection_total += as_float(values.get("avg_ms"))
            if label.startswith("attention_qk_group_kvhead"):
                qk_total += as_float(values.get("total_ms"))
            if label.startswith("attention_pv_kvhead"):
                pv_total += as_float(values.get("avg_ms"))
        elif tag == "decode_full_breakdown":
            projection_total = (
                as_float(values.get("q_proj")) +
                as_float(values.get("k_proj")) +
                as_float(values.get("v_proj")) +
                as_float(values.get("o_proj"))
            )
            qk_total = as_float(values.get("qk_total"))
            pv_total = as_float(values.get("pv_total"))

    if gemv_record:
        records["gemv_decode"] = gemv_record
    if projection_total > 0.0 or qk_total > 0.0 or pv_total > 0.0:
        total = projection_total + qk_total + pv_total
        records["attention_decode"] = {
            "operator": "attention_decode",
            "shape": "T992_H960_QH15_KVH5_D64",
            "npu_total_ms": total,
            "npu_compute_ms": qk_total + pv_total,
            "source_test": "kv260_decode_full_stream_perf_test",
            "breakdown_ms": {
                "projections_qkvo": projection_total,
                "qk": qk_total,
                "pv": pv_total,
            },
        }
    return records


def make_measurement(
    op: str,
    cpu: dict[str, Any] | None,
    npu: dict[str, Any] | None,
    overlay: str,
    raw_logs: dict[str, str],
) -> dict[str, Any]:
    cpu_ms = as_float(cpu.get("median_ms") if cpu else 0.0)
    npu_total_ms = as_float(npu.get("npu_total_ms") if npu else 0.0)
    npu_compute_ms = as_float(npu.get("npu_compute_ms") if npu else 0.0)
    status = "ok" if cpu_ms > 0.0 and npu_total_ms > 0.0 else "missing"
    return {
        "operator": op,
        "status": status,
        "shape": (npu or cpu or {}).get("shape", ""),
        "overlay": overlay,
        "cpu_fp16_ms": cpu_ms,
        "npu_total_ms": npu_total_ms,
        "npu_compute_ms": npu_compute_ms,
        "speedup_total": cpu_ms / npu_total_ms if cpu_ms > 0.0 and npu_total_ms > 0.0 else None,
        "speedup_compute_only": cpu_ms / npu_compute_ms if cpu_ms > 0.0 and npu_compute_ms > 0.0 else None,
        "cpu_effective_gops": as_float(cpu.get("effective_gops") if cpu else 0.0),
        "npu_effective_gops": as_float(npu.get("effective_gops") if npu else 0.0),
        "timing_boundary": "npu_total_ms includes necessary low-level data movement/packing where the underlying microbench measures it",
        "source_test": (npu or {}).get("source_test", ""),
        "raw_logs": raw_logs,
        "raw_cpu_line": (cpu or {}).get("raw_line", ""),
        "raw_npu_line": (npu or {}).get("raw_line", ""),
        "breakdown_ms": (npu or {}).get("breakdown_ms", {}),
    }


def write_notes(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["measurements"]
    lines = [
        "# KV260 Operator Microbench Summary",
        "",
        "This run compares isolated low-level operators, not full inference phases.",
        "CPU baseline is the FP16 reference microbench for the same recorded shape.",
        "NPU total time includes required low-level data movement/packing when present in the source runtime test.",
        "",
        "## Measurements",
        "",
        "| operator | shape | CPU FP16 ms | NPU total ms | speedup | source |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        speedup = row["speedup_total"]
        lines.append(
            f"| {row['operator']} | {row['shape']} | {row['cpu_fp16_ms']:.6f} | "
            f"{row['npu_total_ms']:.6f} | {speedup:.3f} | {row['source_test']} |"
            if speedup is not None else
            f"| {row['operator']} | {row['shape']} | {row['cpu_fp16_ms']:.6f} | "
            f"{row['npu_total_ms']:.6f} | missing | {row['source_test']} |"
        )
    lines.extend([
        "",
        "## Raw Artifacts",
        "",
    ])
    for key, value in payload["raw_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build operator-level KV260 NPU/CPU speedup JSON.")
    parser.add_argument("--cpu-log", required=True, type=Path)
    parser.add_argument("--prefill-gemm-log", required=True, type=Path)
    parser.add_argument("--prefill-attention-log", required=True, type=Path)
    parser.add_argument("--decode-log", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--prefill-overlay", default="prefill_190m_qkpipe_20260608_0206_app")
    parser.add_argument("--decode-overlay", default="dec_200m_latest_0609a_app")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu = parse_cpu_log(args.cpu_log)
    prefill_gemm = parse_prefill_gemm_log(args.prefill_gemm_log)
    prefill_attn = parse_prefill_attention_log(args.prefill_attention_log)
    decode = parse_decode_log(args.decode_log)

    raw_artifacts = {
        "cpu_log": str(args.cpu_log),
        "prefill_gemm_log": str(args.prefill_gemm_log),
        "prefill_attention_log": str(args.prefill_attention_log),
        "decode_log": str(args.decode_log),
    }

    measurements = [
        make_measurement("gemm_prefill", cpu.get("gemm_prefill"), prefill_gemm, args.prefill_overlay, raw_artifacts),
        make_measurement("attention_prefill", cpu.get("attention_prefill"), prefill_attn, args.prefill_overlay, raw_artifacts),
        make_measurement("gemv_decode", cpu.get("gemv_decode"), decode.get("gemv_decode"), args.decode_overlay, raw_artifacts),
        make_measurement("attention_decode", cpu.get("attention_decode"), decode.get("attention_decode"), args.decode_overlay, raw_artifacts),
    ]

    payload = {
        "schema": "aicas_operator_microbench_speedup.v1",
        "run_id": args.run_id,
        "timing_unit": "ms",
        "metadata": {
            "scope": "isolated operator microbench, not end-to-end inference",
            "cpu_baseline": "FP16 reference microbench for matching shapes",
            "npu_timing_boundary": "includes necessary low-level movement/packing where measured by runtime tests",
            "prefill_overlay": args.prefill_overlay,
            "decode_overlay": args.decode_overlay,
        },
        "measurements": measurements,
        "raw_artifacts": raw_artifacts,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_notes(args.notes, payload)
    print(json.dumps({"out_json": str(args.out_json), "notes": str(args.notes), "measurements": len(measurements)}, indent=2))


if __name__ == "__main__":
    main()
