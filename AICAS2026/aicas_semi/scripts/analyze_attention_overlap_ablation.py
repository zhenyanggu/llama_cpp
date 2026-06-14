#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_value(value):
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value, 0)
    except ValueError:
        return value


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


def find_rows(log_path):
    rows = {}
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        tag, data = parse_kv_line(line)
        if tag == "attention_p_overlap_ablation":
            rows["p_overlap"] = data
        elif tag == "attention_qk_logp_mvout_overlap":
            rows["qk_logp_mvout_overlap"] = data
    if "p_overlap" not in rows:
        raise SystemExit("missing attention_p_overlap_ablation line")
    return rows


def build_summary(args):
    rows = find_rows(args.log)
    row = rows["p_overlap"]
    serial_ms = float(row["serial_pv_ms"])
    overlap_ms = float(row["overlap_pv_ms"])
    mvin_total_ms = float(row["overlap_mvin_p_total_ms"])
    residual_exposed_wait_ms = float(row["exposed_mvin_p_wait_ms"])
    startup_wait_ms = float(row["startup_mvin_p_wait_ms"])
    hidden_mvin_ms = max(0.0, mvin_total_ms - residual_exposed_wait_ms - startup_wait_ms)
    steady_total_ms = max(0.0, mvin_total_ms - startup_wait_ms)
    overall_hidden_ratio = hidden_mvin_ms / mvin_total_ms if mvin_total_ms > 0.0 else 0.0
    steady_hidden_ratio = hidden_mvin_ms / steady_total_ms if steady_total_ms > 0.0 else 0.0
    residual_exposed_ratio = (
        residual_exposed_wait_ms / steady_total_ms if steady_total_ms > 0.0 else 0.0
    )
    result = {
        "schema": "aicas_prefill_attention_p_overlap_ablation.v1",
        "run_id": args.run_id,
        "overlay": args.overlay,
        "phase": "prefill_attention",
        "operator": "qk_pv_attention",
        "focus": "P matrix mvin overlap in PV phase",
        "shape": {
            "tokens": row["T"],
            "q_heads_measured": row["q_heads"],
            "kv_heads_measured": row["kv_heads"],
            "head_dim": row["head_dim"],
            "chunks": row["chunks"],
            "chunk_rows": row["chunk_rows"],
            "p_bytes": row["p_bytes"],
            "pv_out_bytes": row["pv_out_bytes"],
        },
        "plot_recommendation": {
            "title": "P-Matrix Transfer Hiding in Prefill Attention",
            "chart": "horizontal bars",
            "x_axis": "PV phase latency (ms)",
            "bars": ["serial_pv", "overlapped_pv"],
            "annotation": "The bar is split into hidden mvin(P) and exposed wait after overlap.",
        },
        "configs": [
            {
                "config": "serial_pv",
                "latency_ms": serial_ms,
                "measured": True,
                "description": "For each P chunk: blocking mvin_p, PV compute, and PV mvout.",
            },
            {
                "config": "overlapped_pv",
                "latency_ms": overlap_ms,
                "measured": True,
                "description": "Ping-pong PV pipeline: next mvin_p overlaps current PV compute and mvout overlaps adjacent work.",
            },
        ],
        "latency_breakdown_ms": {
            "qk_setup": float(row["qk_setup_ms"]),
            "serial_pv_total": serial_ms,
            "serial_mvin_p": float(row["serial_mvin_p_ms"]),
            "serial_pv_compute": float(row["serial_pv_compute_ms"]),
            "serial_pv_mvout": float(row["serial_pv_mvout_ms"]),
            "overlap_pv_total": overlap_ms,
            "hidden_mvin_p": hidden_mvin_ms,
            "residual_exposed_mvin_p_wait": residual_exposed_wait_ms,
            "startup_mvin_p_wait": startup_wait_ms,
        },
        "matrix_transfer_overlap": {
            "target": "P matrix mvin in PV phase",
            "mvin_total_ms": mvin_total_ms,
            "residual_exposed_wait_ms": residual_exposed_wait_ms,
            "startup_wait_ms": startup_wait_ms,
            "critical_path_wait_ms": residual_exposed_wait_ms + startup_wait_ms,
            "hidden_mvin_ms": hidden_mvin_ms,
            "overall_hidden_ratio_including_startup": overall_hidden_ratio,
            "steady_state_hidden_ratio_excluding_startup": steady_hidden_ratio,
            "steady_state_residual_exposed_ratio": residual_exposed_ratio,
            "mvin_count": row["mvin_p_count"],
            "exposed_wait_count": row["exposed_wait_count"],
            "head_boundary_exposed_wait_ms": float(row.get("head_boundary_exposed_wait_ms", 0.0)),
            "intra_head_exposed_wait_ms": float(row.get("intra_head_exposed_wait_ms", residual_exposed_wait_ms)),
            "head_boundary_wait_count": row.get("head_boundary_wait_count", 0),
            "intra_head_wait_count": row.get("intra_head_wait_count", row["exposed_wait_count"]),
            "definition": (
                "residual_exposed_wait_ms is the wait for next P chunk after "
                "the current PV compute finishes. startup_wait_ms is first-tile "
                "warmup and is not counted as hidden. hidden_mvin_ms is measured "
                "P transfer time hidden in steady-state overlap."
            ),
        },
        "metrics": {
            "speedup": float(row["speedup"]),
            "hidden_mvin_p_ms": hidden_mvin_ms,
            "residual_exposed_mvin_p_wait_ms": residual_exposed_wait_ms,
            "startup_mvin_p_wait_ms": startup_wait_ms,
            "overall_hidden_mvin_p_ratio_including_startup": overall_hidden_ratio,
            "steady_state_hidden_mvin_p_ratio_excluding_startup": steady_hidden_ratio,
            "steady_state_residual_exposed_ratio": residual_exposed_ratio,
            "head_boundary_exposed_wait_ms": float(row.get("head_boundary_exposed_wait_ms", 0.0)),
            "intra_head_exposed_wait_ms": float(row.get("intra_head_exposed_wait_ms", residual_exposed_wait_ms)),
        },
        "validity": {
            "correctness": "serial and overlapped PV outputs are both checked against the same software reference in the board test",
            "note": row.get("note", ""),
        },
        "source": {
            "log": str(Path(args.log).resolve()),
            "test_binary": (
                "versa_p_attention_qk_pv_log8_test --p-overlap-only"
                if int(row["q_heads"]) == 1
                else f"versa_p_attention_qk_pv_log8_test --p-overlap-heads {row['q_heads']}"
            ),
        },
    }
    qk = rows.get("qk_logp_mvout_overlap")
    if qk:
        logp_mvout_total_ms = float(qk["logp_mvout_total_ms"])
        residual_exposed_ms = float(qk["residual_exposed_logp_mvout_wait_ms"])
        tail_ms = float(qk["tail_logp_mvout_wait_ms"])
        hidden_ms = max(0.0, logp_mvout_total_ms - residual_exposed_ms - tail_ms)
        steady_total_ms = max(0.0, logp_mvout_total_ms - tail_ms)
        result["qk_logp_mvout_overlap"] = {
            "target": "P/logP matrix mvout after QK phase",
            "qk_profile_wall_ms": float(qk["qk_profile_wall_ms"]),
            "logp_mvout_total_ms": logp_mvout_total_ms,
            "residual_exposed_wait_ms": residual_exposed_ms,
            "tail_wait_ms": tail_ms,
            "critical_path_wait_ms": residual_exposed_ms + tail_ms,
            "hidden_logp_mvout_ms": hidden_ms,
            "overall_hidden_ratio_including_tail": (
                hidden_ms / logp_mvout_total_ms if logp_mvout_total_ms > 0.0 else 0.0
            ),
            "steady_state_hidden_ratio_excluding_tail": (
                hidden_ms / steady_total_ms if steady_total_ms > 0.0 else 0.0
            ),
            "steady_state_residual_exposed_ratio": (
                residual_exposed_ms / steady_total_ms if steady_total_ms > 0.0 else 0.0
            ),
            "logp_mvout_count": qk["logp_mvout_count"],
            "exposed_wait_count": qk["exposed_wait_count"],
            "tail_wait_count": qk["tail_wait_count"],
            "head_boundary_exposed_wait_ms": float(qk.get("head_boundary_exposed_wait_ms", 0.0)),
            "intra_head_exposed_wait_ms": float(qk.get("intra_head_exposed_wait_ms", residual_exposed_ms)),
            "head_boundary_wait_count": qk.get("head_boundary_wait_count", 0),
            "intra_head_wait_count": qk.get("intra_head_wait_count", qk["exposed_wait_count"]),
            "definition": (
                "residual_exposed_wait_ms is the wait for a P/logP mvout after "
                "the next QK chunk has already finished. tail_wait_ms is the final "
                "P/logP mvout with no following QK chunk to hide behind. hidden_logp_mvout_ms "
                "is measured P/logP output transfer time hidden by QK compute."
            ),
        }
    return result


def write_notes(path, result):
    shape = result["shape"]
    metrics = result["metrics"]
    transfer = result["matrix_transfer_overlap"]
    bd = result["latency_breakdown_ms"]
    lines = [
        "# Prefill attention P-overlap ablation",
        "",
        f"- run_id: `{result['run_id']}`",
        f"- overlay: `{result['overlay']}`",
        f"- shape: T={shape['tokens']}, head_dim={shape['head_dim']}, chunks={shape['chunks']}, chunk_rows={shape['chunk_rows']}",
        f"- P matrix bytes measured: {shape['p_bytes']}",
        "",
        "| config | latency ms | note |",
        "| --- | ---: | --- |",
        f"| serial_pv | {bd['serial_pv_total']:.6f} | blocking mvin(P) + PV + mvout |",
        f"| overlapped_pv | {bd['overlap_pv_total']:.6f} | ping-pong mvin(P)/PV/mvout pipeline |",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| speedup | {metrics['speedup']:.6f} |",
        f"| hidden_mvin_p_ms | {metrics['hidden_mvin_p_ms']:.6f} |",
        f"| residual_exposed_mvin_p_wait_ms | {metrics['residual_exposed_mvin_p_wait_ms']:.6f} |",
        f"| startup_mvin_p_wait_ms | {metrics['startup_mvin_p_wait_ms']:.6f} |",
        f"| overall_hidden_mvin_p_ratio_including_startup | {metrics['overall_hidden_mvin_p_ratio_including_startup']:.6f} |",
        f"| steady_state_hidden_mvin_p_ratio_excluding_startup | {metrics['steady_state_hidden_mvin_p_ratio_excluding_startup']:.6f} |",
        f"| steady_state_residual_exposed_ratio | {metrics['steady_state_residual_exposed_ratio']:.6f} |",
        f"| head_boundary_exposed_wait_ms | {metrics['head_boundary_exposed_wait_ms']:.6f} |",
        f"| intra_head_exposed_wait_ms | {metrics['intra_head_exposed_wait_ms']:.6f} |",
        "",
        "| P transfer overlap metric | value |",
        "| --- | ---: |",
        f"| mvin_total_ms | {transfer['mvin_total_ms']:.6f} |",
        f"| residual_exposed_wait_ms | {transfer['residual_exposed_wait_ms']:.6f} |",
        f"| startup_wait_ms | {transfer['startup_wait_ms']:.6f} |",
        f"| critical_path_wait_ms | {transfer['critical_path_wait_ms']:.6f} |",
        f"| hidden_mvin_ms | {transfer['hidden_mvin_ms']:.6f} |",
        "",
        "Serial PV breakdown:",
        "",
        f"- mvin_p_ms: {bd['serial_mvin_p']:.6f}",
        f"- pv_compute_ms: {bd['serial_pv_compute']:.6f}",
        f"- pv_mvout_ms: {bd['serial_pv_mvout']:.6f}",
        "",
        "The QK/logP setup is measured separately and excluded from the PV overlap bars.",
    ]
    qk = result.get("qk_logp_mvout_overlap")
    if qk:
        lines.extend([
            "",
            "QK -> P/logP mvout overlap:",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| qk_profile_wall_ms | {qk['qk_profile_wall_ms']:.6f} |",
            f"| logp_mvout_total_ms | {qk['logp_mvout_total_ms']:.6f} |",
            f"| hidden_logp_mvout_ms | {qk['hidden_logp_mvout_ms']:.6f} |",
            f"| residual_exposed_wait_ms | {qk['residual_exposed_wait_ms']:.6f} |",
            f"| tail_wait_ms | {qk['tail_wait_ms']:.6f} |",
            f"| steady_state_hidden_ratio_excluding_tail | {qk['steady_state_hidden_ratio_excluding_tail']:.6f} |",
            f"| steady_state_residual_exposed_ratio | {qk['steady_state_residual_exposed_ratio']:.6f} |",
            f"| head_boundary_exposed_wait_ms | {qk['head_boundary_exposed_wait_ms']:.6f} |",
            f"| intra_head_exposed_wait_ms | {qk['intra_head_exposed_wait_ms']:.6f} |",
        ])
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
