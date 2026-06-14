#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Any


TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
    "q4_k": 0.5,
    "q5_k": 0.625,
    "q6_k": 0.75,
    "q8_k": 1.0,
}


def product(values: list[Any]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def tensor_nbytes(info: dict[str, Any]) -> float:
    return product(info.get("shape", [1, 1, 1, 1])) * TYPE_BYTES.get(
        str(info.get("type", "f32")).lower(),
        4.0,
    )


def matmul_ops(sig: dict[str, Any]) -> float:
    src0 = sig["src0"]["shape"]
    src1 = sig["src1"]["shape"]
    dst = sig["dst"]["shape"]
    k = int(src0[0])
    n = int(src0[1])
    m = int(src1[1])
    batch = max(int(src0[2]), int(src1[2]), int(dst[2]), 1) * max(
        int(src0[3]), int(src1[3]), int(dst[3]), 1
    )
    return 2.0 * k * n * m * batch


def matmul_bytes(sig: dict[str, Any]) -> float:
    return tensor_nbytes(sig["src0"]) + tensor_nbytes(sig["src1"]) + tensor_nbytes(sig["dst"])


def add_cpu_profile(total: dict[str, float], profile: dict[str, Any], count_scale: float = 1.0) -> None:
    for sig in profile.get("mul_mat_signatures", []):
        count = float(sig.get("node_count", 0)) * count_scale
        if count <= 0:
            continue
        total["ops"] += matmul_ops(sig) * count
        total["bytes"] += matmul_bytes(sig) * count


def cpu_profile_totals(result_dir: Path) -> dict[str, dict[str, float]]:
    out = {
        "image_mmproj": {"ops": 0.0, "bytes": 0.0},
        "text_prefill_512": {"ops": 0.0, "bytes": 0.0},
        "text_decode_128avg": {"ops": 0.0, "bytes": 0.0},
    }
    mtmd_path = result_dir / "mtmd_prefill_summary.json"
    if mtmd_path.exists():
        mtmd = load_json(mtmd_path)
        for profile in mtmd.get("mmproj", {}).get("cpu_backend_profile_details", []):
            add_cpu_profile(out["image_mmproj"], profile)

    text_path = result_dir / "text_cpu_profile.jsonl"
    if text_path.exists():
        decode_records = 0
        decode_total = {"ops": 0.0, "bytes": 0.0}
        for record in load_jsonl(text_path):
            profile = record.get("cpu_backend_profile", {})
            phase = record.get("phase")
            if phase in ("prefill", "merged_prefill"):
                add_cpu_profile(out["text_prefill_512"], profile)
            elif phase == "decode":
                decode_records += 1
                add_cpu_profile(decode_total, profile)
        if decode_records > 0:
            out["text_decode_128avg"]["ops"] = decode_total["ops"] / decode_records
            out["text_decode_128avg"]["bytes"] = decode_total["bytes"] / decode_records
    return out


def infer_workload_from_path(path: Path) -> str | None:
    name = path.name.lower()
    for workload in ("image_mmproj", "text_prefill_512", "text_decode_128avg"):
        if workload in name:
            return workload
    return None


def load_latency_samples(result_dir: Path) -> dict[str, dict[str, float]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    if not result_dir.exists():
        return {}
    for path in sorted(result_dir.rglob("*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        workload = data.get("workload")
        if not isinstance(workload, str):
            workload = infer_workload_from_path(path)
        if workload not in ("image_mmproj", "text_prefill_512", "text_decode_128avg"):
            continue
        samples.setdefault(workload, []).append(data)

    out: dict[str, dict[str, float]] = {}
    for workload, rows in samples.items():
        prompt_ms = [float(r.get("prompt_ms", 0.0) or 0.0) for r in rows]
        decode_ms = [float(r.get("decode_ms", 0.0) or 0.0) for r in rows]
        completion = [float(r.get("completion_tokens", 0) or 0) for r in rows]
        prompt_tokens = [float(r.get("prompt_tokens", 0) or 0) for r in rows]
        if workload == "text_decode_128avg":
            per_tok = [
                d / c
                for d, c in zip(decode_ms, completion)
                if d > 0.0 and c > 0.0
            ]
            latency_ms = median(per_tok) if per_tok else 0.0
        else:
            latency_ms = median([v for v in prompt_ms if v > 0.0]) if prompt_ms else 0.0
        out[workload] = {
            "latency_ms": latency_ms,
            "prompt_tokens": median(prompt_tokens) if prompt_tokens else 0.0,
            "completion_tokens": median(completion) if completion else 0.0,
            "samples": float(len(rows)),
        }
    return out


def mtmd_workload_latency(result_dir: Path, workload: str) -> dict[str, float]:
    path = result_dir / "mtmd_prefill_summary.json"
    if not path.exists():
        return {}
    mtmd = load_json(path)
    if workload == "image_mmproj":
        encode_us = float(mtmd.get("mmproj", {}).get("encode_us", 0.0) or 0.0)
        if encode_us > 0.0:
            return {
                "latency_ms": encode_us / 1000.0,
                "prompt_tokens": float(mtmd.get("prompt_tokens_processed", 0) or 0),
                "completion_tokens": 0.0,
                "samples": 1.0,
            }
    return {}


def throughput_metrics_decode_latency(result_dir: Path) -> dict[str, float]:
    path = result_dir / "throughput_metrics_profile.json"
    if not path.exists():
        path = result_dir / "throughput_metrics.json"
    if not path.exists():
        return {}
    data = load_json(path)
    decode_ms = float(data.get("decode_ms", 0.0) or 0.0)
    completion_tokens = float(data.get("completion_tokens", 0.0) or 0.0)
    prompt_tokens = float(data.get("prompt_tokens", 0.0) or 0.0)
    if decode_ms <= 0.0 or completion_tokens <= 0.0:
        return {}
    return {
        "latency_ms": decode_ms / completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "samples": 1.0,
    }


def decode_profile_totals(profile_dir: Path) -> dict[str, float]:
    path = profile_dir / "ggml_npu_decode_profile.jsonl"
    records = load_jsonl(path)
    if not records:
        return {"ops": 0.0, "bytes": 0.0, "profiled_tokens": 0.0}
    ops = 0.0
    bytes_ = 0.0
    lm_head_records = 0
    for record in records:
        dims = record.get("dims", {})
        m = float(dims.get("m", 0) or 0)
        k = float(dims.get("k", 0) or 0)
        n_cols = float(dims.get("n_cols", 1) or 1)
        ops += 2.0 * m * k * n_cols
        bytes_ += sum(float(v or 0) for v in record.get("bytes", {}).values())
        if record.get("op_role") == "lm_head":
            lm_head_records += 1
    profiled_tokens = float(max(lm_head_records, 1))
    return {
        "ops": ops / profiled_tokens,
        "bytes": bytes_ / profiled_tokens,
        "profiled_tokens": profiled_tokens,
    }


def classify_npu_node(node: dict[str, Any]) -> str:
    weight_name = str(node.get("weight_name", ""))
    root_name = str(node.get("root_name", ""))
    if weight_name.startswith("v.") or root_name.startswith("node_"):
        return "image_mmproj"
    return "text_prefill_512"


def npu_prefill_profile_totals(profile_dir: Path) -> dict[str, dict[str, Any]]:
    path = profile_dir / "ggml_npu_profile.json"
    out = {
        "image_mmproj": {"ops": 0.0, "bytes": 0.0, "ops_source": "npu_profile_nodes"},
        "text_prefill_512": {"ops": 0.0, "bytes": 0.0, "ops_source": "npu_profile_nodes"},
    }
    if not path.exists():
        return out
    profile = load_json(path)

    nodes = profile.get("nodes") or []
    if nodes:
        for node in nodes:
            workload = classify_npu_node(node)
            m = float(node.get("m", 0) or 0)
            n = float(node.get("n", 0) or 0)
            k = float(node.get("k", 0) or 0)
            out[workload]["ops"] += 2.0 * m * n * k
            out[workload]["bytes"] += sum(
                float(node.get(key, 0) or 0)
                for key in (
                    "dma_in_activation_bytes_total",
                    "dma_in_weight_bytes_total",
                    "acc_readback_bytes_total",
                )
            )
        return out

    for row in profile.get("by_domain", []):
        domain = row.get("domain")
        workload = "image_mmproj" if domain == "mmproj" else "text_prefill_512"
        out[workload]["bytes"] += sum(
            float(row.get(key, 0) or 0)
            for key in (
                "dma_in_activation_bytes_total",
                "dma_in_weight_bytes_total",
                "acc_readback_bytes_total",
            )
        )
        out[workload]["ops_source"] = "model_estimate_from_aggregate_profile"

    mtmd_path = profile_dir / "mtmd_prefill_summary.json"
    prompt_tokens = 512
    media_tokens = 64
    if mtmd_path.exists():
        mtmd = load_json(mtmd_path)
        prompt_tokens = int(mtmd.get("prompt_tokens_processed") or prompt_tokens)
        media_tokens = int(mtmd.get("mmproj", {}).get("media_tokens") or media_tokens)
    out["text_prefill_512"]["ops"] = estimate_text_prefill_linear_ops(prompt_tokens)
    out["image_mmproj"]["ops"] = estimate_mmproj_linear_ops(media_tokens)
    return out


def estimate_text_prefill_linear_ops(tokens: int) -> float:
    hidden = 960
    ffn = 2560
    layers = 32
    per_layer = 2.0 * tokens * (
        hidden * hidden
        + hidden * 320
        + hidden * 320
        + hidden * hidden
        + hidden * ffn
        + hidden * ffn
        + ffn * hidden
    )
    return per_layer * layers


def estimate_mmproj_linear_ops(media_tokens: int) -> float:
    # Vision projector linear-GEMM estimate for SmolVLM2. Exact profile nodes
    # override this when diagnostic profile data is available.
    return 2.0 * media_tokens * (
        768 * 1024
        + 12 * (768 * 768 * 4 + 768 * 3072 * 3)
        + 3072 * 960
    )


def parse_microbench_logs(logs: list[Path]) -> dict[str, dict[str, float | str]]:
    roofs: dict[str, dict[str, float | str]] = {}
    for path in logs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        gops_values = [float(x) for x in re.findall(r"effective_gops(?:_gemm|_total)?=([0-9.]+)", text)]
        bandwidth_values = [float(x) for x in re.findall(r"(?:payload_gbps|bw_mib_s)=([0-9.]+)", text)]
        if "stream_" in text or "decode_full_" in text:
            key = "NPU_decode_overlay"
        elif "network_gemm" in text or "layer_timing" in text:
            key = "NPU_prefill_overlay"
        else:
            key = path.stem
        peak_tops = max(gops_values) / 1000.0 if gops_values else 0.0
        # payload_gbps uses decimal GB/s in current tests; bw_mib_s is converted.
        if "bw_mib_s" in text and "payload_gbps" not in text:
            bandwidth_gbs = max(bandwidth_values) * 1024.0 * 1024.0 / 1.0e9 if bandwidth_values else 0.0
        else:
            bandwidth_gbs = max(bandwidth_values) if bandwidth_values else 0.0
        old = roofs.get(key)
        roofs[key] = {
            "peak_tops": max(float(old.get("peak_tops", 0.0)) if old else 0.0, peak_tops),
            "memory_bandwidth_gbs": max(
                float(old.get("memory_bandwidth_gbs", 0.0)) if old else 0.0,
                bandwidth_gbs,
            ),
            "source": str(path),
        }
    return roofs


def make_point(
    workload: str,
    hardware: str,
    totals: dict[str, Any],
    latency: dict[str, float],
    latency_source: str,
) -> dict[str, Any]:
    ops = float(totals.get("ops", 0.0) or 0.0)
    bytes_ = float(totals.get("bytes", 0.0) or 0.0)
    latency_ms = float(latency.get("latency_ms", 0.0) or 0.0)
    return {
        "workload": workload,
        "hardware": hardware,
        "ops": ops,
        "bytes": bytes_,
        "latency_ms": latency_ms,
        "operational_intensity_ops_per_byte": ops / bytes_ if bytes_ > 0 else 0.0,
        "achieved_tops": ops / (latency_ms / 1000.0) / 1.0e12 if latency_ms > 0 else 0.0,
        "ops_source": totals.get("ops_source", "profile_exact_matmul_mac2"),
        "bytes_source": totals.get("bytes_source", "profile_effective_bytes"),
        "latency_source": latency_source,
        "prompt_tokens": latency.get("prompt_tokens", ""),
        "completion_tokens": latency.get("completion_tokens", ""),
        "samples": latency.get("samples", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate KV260 roofline CSV/JSON data.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cpu-profile-dir", default="")
    parser.add_argument("--cpu-latency-dir", default="")
    parser.add_argument("--npu-prefill-profile-dir", default="")
    parser.add_argument("--npu-prefill-latency-profile-dir", default="")
    parser.add_argument("--npu-decode-profile-dir", default="")
    parser.add_argument("--npu-latency-dir", default="")
    parser.add_argument("--microbench-log", action="append", default=[])
    parser.add_argument("--cpu-peak-tops", type=float, default=None)
    parser.add_argument("--cpu-memory-bandwidth-gbs", type=float, default=None)
    parser.add_argument("--npu-prefill-peak-tops", type=float, default=None)
    parser.add_argument("--npu-prefill-memory-bandwidth-gbs", type=float, default=None)
    parser.add_argument("--npu-decode-peak-tops", type=float, default=None)
    parser.add_argument("--npu-decode-memory-bandwidth-gbs", type=float, default=None)
    return parser.parse_args()


def make_hardware_row(
    hardware: str,
    peak_tops: float,
    memory_bandwidth_gbs: float,
    peak_source: str,
    bandwidth_source: str,
    config: str,
) -> dict[str, Any]:
    return {
        "hardware": hardware,
        "peak_tops": peak_tops,
        "memory_bandwidth_gbs": memory_bandwidth_gbs,
        "peak_source": peak_source,
        "bandwidth_source": bandwidth_source,
        "overlay_app": "",
        "config": config,
    }


def append_roof_row(
    rows: list[dict[str, Any]],
    hardware: str,
    parsed_roofs: dict[str, dict[str, float | str]],
    explicit_peak_tops: float | None,
    explicit_memory_bandwidth_gbs: float | None,
    config: str,
) -> None:
    parsed = parsed_roofs.get(hardware, {})
    parsed_peak = float(parsed.get("peak_tops", 0.0) or 0.0)
    parsed_bw = float(parsed.get("memory_bandwidth_gbs", 0.0) or 0.0)
    peak_tops = explicit_peak_tops if explicit_peak_tops is not None else parsed_peak
    memory_bandwidth_gbs = (
        explicit_memory_bandwidth_gbs
        if explicit_memory_bandwidth_gbs is not None
        else parsed_bw
    )
    if peak_tops <= 0.0 and memory_bandwidth_gbs <= 0.0 and not parsed:
        return
    parsed_source = str(parsed.get("source", ""))
    rows.append(make_hardware_row(
        hardware=hardware,
        peak_tops=peak_tops,
        memory_bandwidth_gbs=memory_bandwidth_gbs,
        peak_source="explicit_cli" if explicit_peak_tops is not None else parsed_source,
        bandwidth_source=(
            "explicit_cli"
            if explicit_memory_bandwidth_gbs is not None
            else parsed_source
        ),
        config=config,
    ))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    points: list[dict[str, Any]] = []

    if args.cpu_profile_dir:
        cpu_profile_dir = Path(args.cpu_profile_dir)
        cpu_totals = cpu_profile_totals(cpu_profile_dir)
        cpu_latency = load_latency_samples(Path(args.cpu_latency_dir)) if args.cpu_latency_dir else {}
        for workload, totals in cpu_totals.items():
            latency = cpu_latency.get(workload, {})
            latency_source = args.cpu_latency_dir or args.cpu_profile_dir
            if workload == "image_mmproj":
                mtmd_latency = mtmd_workload_latency(cpu_profile_dir, workload)
                if mtmd_latency:
                    latency = mtmd_latency
                    latency_source = str(cpu_profile_dir / "mtmd_prefill_summary.json")
            points.append(
                make_point(
                    workload,
                    "CPU",
                    {**totals, "bytes_source": "cpu_profiler_src_dst_tensor_bytes"},
                    latency,
                    latency_source,
                )
            )

    if args.npu_prefill_profile_dir:
        npu_prefill_profile_dir = Path(args.npu_prefill_profile_dir)
        npu_prefill_latency_profile_dir = (
            Path(args.npu_prefill_latency_profile_dir)
            if args.npu_prefill_latency_profile_dir
            else npu_prefill_profile_dir
        )
        npu_totals = npu_prefill_profile_totals(npu_prefill_profile_dir)
        npu_latency = load_latency_samples(Path(args.npu_latency_dir)) if args.npu_latency_dir else {}
        for workload in ("image_mmproj", "text_prefill_512"):
            latency = npu_latency.get(workload, {})
            latency_source = args.npu_latency_dir or args.npu_prefill_profile_dir
            if workload == "image_mmproj":
                mtmd_latency = mtmd_workload_latency(npu_prefill_latency_profile_dir, workload)
                if mtmd_latency:
                    latency = mtmd_latency
                    latency_source = str(npu_prefill_latency_profile_dir / "mtmd_prefill_summary.json")
            points.append(
                make_point(
                    workload,
                    "NPU_prefill_overlay",
                    {**npu_totals[workload], "bytes_source": "npu_dma_effective_bytes"},
                    latency,
                    latency_source,
                )
            )

    if args.npu_decode_profile_dir:
        npu_decode_profile_dir = Path(args.npu_decode_profile_dir)
        decode_totals = decode_profile_totals(npu_decode_profile_dir)
        npu_latency = load_latency_samples(Path(args.npu_latency_dir)) if args.npu_latency_dir else {}
        decode_latency = npu_latency.get("text_decode_128avg", {})
        latency_source = args.npu_latency_dir or args.npu_decode_profile_dir
        if not decode_latency:
            decode_latency = throughput_metrics_decode_latency(npu_decode_profile_dir)
            if decode_latency:
                latency_source = str(
                    npu_decode_profile_dir / "throughput_metrics_profile.json"
                )
        points.append(
            make_point(
                "text_decode_128avg",
                "NPU_decode_overlay",
                {
                    **decode_totals,
                    "ops_source": "decode_profile_dims_mac2_per_token",
                    "bytes_source": "decode_profile_payload_bytes_per_token",
                },
                decode_latency,
                latency_source,
            )
        )

    hardware_rows: list[dict[str, Any]] = []
    roofs = parse_microbench_logs([Path(p) for p in args.microbench_log])
    if args.cpu_profile_dir or args.cpu_peak_tops is not None or args.cpu_memory_bandwidth_gbs is not None:
        cpu_totals = cpu_profile_totals(Path(args.cpu_profile_dir)) if args.cpu_profile_dir else {}
        cpu_peak = 0.0
        cpu_bw = 0.0
        cpu_latency = load_latency_samples(Path(args.cpu_latency_dir)) if args.cpu_latency_dir else {}
        for workload, totals in cpu_totals.items():
            latency = cpu_latency.get(workload, {})
            if workload == "image_mmproj" and args.cpu_profile_dir:
                latency = mtmd_workload_latency(Path(args.cpu_profile_dir), workload) or latency
            latency_ms = float(latency.get("latency_ms", 0.0) or 0.0)
            if latency_ms > 0:
                cpu_peak = max(cpu_peak, totals["ops"] / (latency_ms / 1000.0) / 1.0e12)
                cpu_bw = max(cpu_bw, totals["bytes"] / (latency_ms / 1000.0) / 1.0e9)
        hardware_rows.append(make_hardware_row(
            hardware="CPU",
            peak_tops=args.cpu_peak_tops if args.cpu_peak_tops is not None else cpu_peak,
            memory_bandwidth_gbs=(
                args.cpu_memory_bandwidth_gbs
                if args.cpu_memory_bandwidth_gbs is not None
                else cpu_bw
            ),
            peak_source=(
                "explicit_cli"
                if args.cpu_peak_tops is not None
                else "observed_workload_peak"
            ),
            bandwidth_source=(
                "explicit_cli"
                if args.cpu_memory_bandwidth_gbs is not None
                else "observed_workload_tensor_bytes"
            ),
            config="official_fp16_baseline",
        ))
    append_roof_row(
        hardware_rows,
        "NPU_prefill_overlay",
        roofs,
        args.npu_prefill_peak_tops,
        args.npu_prefill_memory_bandwidth_gbs,
        "current_default",
    )
    append_roof_row(
        hardware_rows,
        "NPU_decode_overlay",
        roofs,
        args.npu_decode_peak_tops,
        args.npu_decode_memory_bandwidth_gbs,
        "current_default",
    )
    for hardware, data in sorted(roofs.items()):
        if hardware in {"NPU_prefill_overlay", "NPU_decode_overlay"}:
            continue
        hardware_rows.append(make_hardware_row(
            hardware=hardware,
            peak_tops=float(data.get("peak_tops", 0.0) or 0.0),
            memory_bandwidth_gbs=float(data.get("memory_bandwidth_gbs", 0.0) or 0.0),
            peak_source=str(data.get("source", "")),
            bandwidth_source=str(data.get("source", "")),
            config="current_default",
        ))

    write_csv(out_dir / "roofline_points.csv", points)
    write_csv(out_dir / "hardware_roofs.csv", hardware_rows)
    summary = {
        "schema": "aicas_roofline_summary.v1",
        "points": points,
        "hardware_roofs": hardware_rows,
        "notes": [
            "All operation counts use MAC = 2 OPs.",
            "CPU bytes are profiler tensor traffic estimates.",
            "NPU bytes are effective profiled payload/DMA bytes when available.",
            "Profile runs are intended for ops/bytes only; latency should come from non-profile runs.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roofline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "points": len(points),
        "hardware_roofs": len(hardware_rows),
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
