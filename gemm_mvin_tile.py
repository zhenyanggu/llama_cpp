#!/usr/bin/env python3
"""Profile-driven GEMM MVIN tiling explorer for the current ggml-npu path."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SA_ROWS = 32
SA_COLS = 32
DEFAULT_SPM_BYTES = 512 * 1024
DEFAULT_ACC_BYTES = 512 * 1024
DEFAULT_GUARD_BYTES = 4 * 1024
DEFAULT_STAGE2_K = 2048
DEFAULT_DMA_SETUP_US = 1.54
DEFAULT_DMA_BW_BYTES_PER_US = 1300.0


@dataclass(frozen=True)
class LayerProfile:
    layer_id: int
    semantic_op: str
    root_op_name: str
    m: int
    n: int
    k: int
    spm_bytes: int
    acc_bytes: int
    stage2_k_block: int
    first_stage_tm: int
    first_stage_tn: int
    first_stage_tk: int
    exec_tile_count: int
    activation_bytes: int
    weight_bytes: int
    output_bytes: int
    activation_calls: int
    weight_calls: int
    activation_us_total: Optional[float] = None
    weight_us_total: Optional[float] = None
    raw_node: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class LayerGroup:
    group_key: str
    semantic_op: str
    m: int
    n: int
    k: int
    spm_bytes: int
    acc_bytes: int
    stage2_k_block: int
    count: int
    members: Sequence[LayerProfile]


@dataclass(frozen=True)
class CandidateResult:
    tm: int
    tn: int
    tk: int
    schedule: str
    spm_bytes: int
    acc_bytes: int
    spm_util: float
    acc_util: float
    activation_loads: int
    weight_loads: int
    activation_bytes: int
    weight_bytes: int
    both_change_events: int
    activation_only_events: int
    weight_only_events: int
    est_mvin_us: float

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_down_multiple(value: int, multiple: int) -> int:
    if value < multiple:
        return value
    return (value // multiple) * multiple


def _candidate_dims(dim: int) -> List[int]:
    if dim <= 0:
        return []

    candidates = {dim}
    aligned = _round_down_multiple(dim, SA_ROWS)
    if aligned >= SA_ROWS:
        for value in range(SA_ROWS, aligned + 1, SA_ROWS):
            candidates.add(value)
    else:
        candidates.add(dim)

    if dim > SA_ROWS:
        candidates.add(SA_ROWS)

    return sorted(value for value in candidates if 0 < value <= dim)


def _split_blocks(dim: int, block: int) -> List[int]:
    if dim <= 0 or block <= 0:
        return []
    values: List[int] = []
    offset = 0
    while offset < dim:
        size = min(block, dim - offset)
        values.append(size)
        offset += size
    return values


def _macro_micro_blocks(dim: int, macro: int, micro: int) -> List[int]:
    blocks: List[int] = []
    for macro_size in _split_blocks(dim, macro):
        blocks.extend(_split_blocks(macro_size, micro))
    return blocks


def _k_subtiles(total_k: int, tk: int, stage2_k: int) -> List[int]:
    blocks: List[int] = []
    for macro_k in _split_blocks(total_k, tk):
        blocks.extend(_split_blocks(macro_k, stage2_k))
    return blocks


def parse_profile(
    profile_json: str,
    semantic_op: Optional[str] = None,
    layer_id: Optional[int] = None,
) -> List[LayerProfile]:
    with open(profile_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result: List[LayerProfile] = []
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue

        current_layer_id = _safe_int(node.get("layer_id"), -1)
        current_semantic = str(node.get("semantic_op") or "")
        if semantic_op is not None and current_semantic != semantic_op:
            continue
        if layer_id is not None and current_layer_id != layer_id:
            continue

        m = _safe_int(node.get("m"))
        n = _safe_int(node.get("n"))
        k = _safe_int(node.get("k"))
        if min(m, n, k) <= 0:
            continue

        activation_bytes = 0
        weight_bytes = 0
        output_bytes = 0
        for tile in node.get("tiles", []):
            if not isinstance(tile, dict):
                continue
            activation_bytes += _safe_int(tile.get("activation_bytes"))
            weight_bytes += _safe_int(tile.get("weight_bytes"))
            output_bytes += _safe_int(tile.get("output_write_bytes"))

        result.append(
            LayerProfile(
                layer_id=current_layer_id,
                semantic_op=current_semantic,
                root_op_name=str(node.get("root_op_name") or ""),
                m=m,
                n=n,
                k=k,
                spm_bytes=_safe_int(node.get("spm_bytes"), DEFAULT_SPM_BYTES),
                acc_bytes=_safe_int(node.get("acc_bytes"), DEFAULT_ACC_BYTES),
                stage2_k_block=_safe_int(node.get("stage2_k_block"), DEFAULT_STAGE2_K),
                first_stage_tm=_safe_int(node.get("first_stage_tm")),
                first_stage_tn=_safe_int(node.get("first_stage_tn")),
                first_stage_tk=_safe_int(node.get("first_stage_tk")),
                exec_tile_count=_safe_int(node.get("exec_tile_count")),
                activation_bytes=activation_bytes,
                weight_bytes=weight_bytes,
                output_bytes=output_bytes,
                activation_calls=_safe_int(node.get("dma_in_activation_calls")),
                weight_calls=_safe_int(node.get("dma_in_weight_calls")),
                activation_us_total=_safe_float(node.get("dma_in_activation_us_total")),
                weight_us_total=_safe_float(node.get("dma_in_weight_us_total")),
                raw_node=node,
            )
        )

    return result


def build_groups(layers: Sequence[LayerProfile], group_by: str) -> List[LayerGroup]:
    buckets: Dict[str, List[LayerProfile]] = {}
    for layer in layers:
        if group_by == "node":
            key = f"layer:{layer.layer_id}"
        else:
            key = (
                f"shape:{layer.semantic_op}:{layer.m}x{layer.n}x{layer.k}:"
                f"spm={layer.spm_bytes}:acc={layer.acc_bytes}:stage2={layer.stage2_k_block}"
            )
        buckets.setdefault(key, []).append(layer)

    groups: List[LayerGroup] = []
    for key, members in buckets.items():
        first = members[0]
        groups.append(
            LayerGroup(
                group_key=key,
                semantic_op=first.semantic_op,
                m=first.m,
                n=first.n,
                k=first.k,
                spm_bytes=first.spm_bytes,
                acc_bytes=first.acc_bytes,
                stage2_k_block=first.stage2_k_block,
                count=len(members),
                members=members,
            )
        )
    groups.sort(key=lambda item: (item.semantic_op, item.m, item.n, item.k, item.group_key))
    return groups


def estimate_dma_us(bytes_count: int, dma_setup_us: float, dma_bw_bytes_per_us: float) -> float:
    if bytes_count <= 0:
        return 0.0
    return dma_setup_us + (bytes_count / dma_bw_bytes_per_us)


def estimate_split_dma_us(bytes_count: int, dma_setup_us: float, dma_bw_bytes_per_us: float) -> float:
    if bytes_count <= 0:
        return 0.0
    lhs = (bytes_count + 1) // 2
    rhs = bytes_count - lhs
    return max(
        estimate_dma_us(lhs, dma_setup_us, dma_bw_bytes_per_us),
        estimate_dma_us(rhs, dma_setup_us, dma_bw_bytes_per_us),
    )


def estimate_events_us(
    events: Iterable[Tuple[int, int]],
    dma_setup_us: float,
    dma_bw_bytes_per_us: float,
    split_same_matrix: bool,
) -> float:
    total = 0.0
    for activation_bytes, weight_bytes in events:
        if activation_bytes > 0 and weight_bytes > 0:
            total += max(
                estimate_dma_us(activation_bytes, dma_setup_us, dma_bw_bytes_per_us),
                estimate_dma_us(weight_bytes, dma_setup_us, dma_bw_bytes_per_us),
            )
        elif activation_bytes > 0:
            total += (
                estimate_split_dma_us(activation_bytes, dma_setup_us, dma_bw_bytes_per_us)
                if split_same_matrix
                else estimate_dma_us(activation_bytes, dma_setup_us, dma_bw_bytes_per_us)
            )
        elif weight_bytes > 0:
            total += (
                estimate_split_dma_us(weight_bytes, dma_setup_us, dma_bw_bytes_per_us)
                if split_same_matrix
                else estimate_dma_us(weight_bytes, dma_setup_us, dma_bw_bytes_per_us)
            )
    return total


def _estimate_single_matrix_sum_us(
    block_sizes: Sequence[int],
    k_blocks: Sequence[int],
    repeat: int,
    dma_setup_us: float,
    dma_bw_bytes_per_us: float,
    split_same_matrix: bool,
) -> float:
    total = 0.0
    for block_size in block_sizes:
        for sub_k in k_blocks:
            bytes_count = block_size * sub_k
            if split_same_matrix:
                total += estimate_split_dma_us(bytes_count, dma_setup_us, dma_bw_bytes_per_us)
            else:
                total += estimate_dma_us(bytes_count, dma_setup_us, dma_bw_bytes_per_us)
    return total * repeat


def summarize_candidate(
    m: int,
    n: int,
    k: int,
    tm: int,
    tn: int,
    tk: int,
    stage2_k_block: int,
    schedule: str,
    dma_setup_us: float,
    dma_bw_bytes_per_us: float,
    split_same_matrix: bool,
) -> Dict[str, float]:
    n_blocks = _macro_micro_blocks(n, tn, SA_ROWS)
    m_blocks = _macro_micro_blocks(m, tm, SA_COLS)
    k_blocks = _k_subtiles(k, tk, stage2_k_block)

    if schedule == "reuse_weight":
        activation_loads = len(k_blocks) * len(m_blocks) * len(n_blocks)
        weight_loads = len(k_blocks) * len(m_blocks)
        activation_bytes = sum(n_blocks) * sum(k_blocks) * len(m_blocks)
        weight_bytes = sum(m_blocks) * sum(k_blocks)
        activation_only_events = activation_loads
        weight_only_events = weight_loads
        est_mvin_us = _estimate_single_matrix_sum_us(
            block_sizes=n_blocks,
            k_blocks=k_blocks,
            repeat=len(m_blocks),
            dma_setup_us=dma_setup_us,
            dma_bw_bytes_per_us=dma_bw_bytes_per_us,
            split_same_matrix=split_same_matrix,
        ) + _estimate_single_matrix_sum_us(
            block_sizes=m_blocks,
            k_blocks=k_blocks,
            repeat=1,
            dma_setup_us=dma_setup_us,
            dma_bw_bytes_per_us=dma_bw_bytes_per_us,
            split_same_matrix=split_same_matrix,
        )
    elif schedule == "reuse_activation":
        activation_loads = len(k_blocks) * len(n_blocks)
        weight_loads = len(k_blocks) * len(n_blocks) * len(m_blocks)
        activation_bytes = sum(n_blocks) * sum(k_blocks)
        weight_bytes = sum(m_blocks) * sum(k_blocks) * len(n_blocks)
        activation_only_events = activation_loads
        weight_only_events = weight_loads
        est_mvin_us = _estimate_single_matrix_sum_us(
            block_sizes=n_blocks,
            k_blocks=k_blocks,
            repeat=1,
            dma_setup_us=dma_setup_us,
            dma_bw_bytes_per_us=dma_bw_bytes_per_us,
            split_same_matrix=split_same_matrix,
        ) + _estimate_single_matrix_sum_us(
            block_sizes=m_blocks,
            k_blocks=k_blocks,
            repeat=len(n_blocks),
            dma_setup_us=dma_setup_us,
            dma_bw_bytes_per_us=dma_bw_bytes_per_us,
            split_same_matrix=split_same_matrix,
        )
    else:
        raise ValueError(f"unsupported schedule: {schedule}")

    return {
        "activation_loads": activation_loads,
        "weight_loads": weight_loads,
        "activation_bytes": activation_bytes,
        "weight_bytes": weight_bytes,
        "both_change_events": 0,
        "activation_only_events": activation_only_events,
        "weight_only_events": weight_only_events,
        "est_mvin_us": est_mvin_us,
    }


def evaluate_candidate(
    group: LayerGroup,
    tm: int,
    tn: int,
    tk: int,
    schedule: str,
    split_same_matrix: bool,
    dma_setup_us: float,
    dma_bw_bytes_per_us: float,
    guard_bytes: int,
) -> Optional[CandidateResult]:
    if min(tm, tn, tk) <= 0:
        return None

    micro_n = min(tn, SA_ROWS)
    micro_m = min(tm, SA_COLS)
    micro_k = min(tk, group.stage2_k_block)
    spm_needed = micro_n * micro_k + micro_m * micro_k
    acc_needed = micro_n * micro_m * 4
    spm_cap = max(0, group.spm_bytes - guard_bytes)
    if spm_needed > spm_cap or acc_needed > group.acc_bytes:
        return None

    summary = summarize_candidate(
        m=group.m,
        n=group.n,
        k=group.k,
        tm=tm,
        tn=tn,
        tk=tk,
        stage2_k_block=group.stage2_k_block,
        schedule=schedule,
        dma_setup_us=dma_setup_us,
        dma_bw_bytes_per_us=dma_bw_bytes_per_us,
        split_same_matrix=split_same_matrix,
    )

    return CandidateResult(
        tm=tm,
        tn=tn,
        tk=tk,
        schedule=schedule,
        spm_bytes=spm_needed,
        acc_bytes=acc_needed,
        spm_util=(spm_needed / spm_cap) if spm_cap > 0 else float("inf"),
        acc_util=(acc_needed / group.acc_bytes) if group.acc_bytes > 0 else float("inf"),
        activation_loads=int(summary["activation_loads"]),
        weight_loads=int(summary["weight_loads"]),
        activation_bytes=int(summary["activation_bytes"]),
        weight_bytes=int(summary["weight_bytes"]),
        both_change_events=int(summary["both_change_events"]),
        activation_only_events=int(summary["activation_only_events"]),
        weight_only_events=int(summary["weight_only_events"]),
        est_mvin_us=summary["est_mvin_us"],
    )


def _candidate_sort_key(candidate: CandidateResult) -> Tuple[float, int, int]:
    return (
        candidate.est_mvin_us,
        candidate.total_bytes,
        abs(candidate.tm - candidate.tn),
    )


def search_group(
    group: LayerGroup,
    dma_setup_us: float,
    dma_bw_bytes_per_us: float,
    guard_bytes: int,
    topk: int,
) -> Dict[str, object]:
    supported_candidates: List[CandidateResult] = []
    split_candidates: List[CandidateResult] = []

    tm_candidates = _candidate_dims(group.m)
    tn_candidates = _candidate_dims(group.n)
    tk_candidates = _candidate_dims(group.k)

    for tm in tm_candidates:
        for tn in tn_candidates:
            for tk in tk_candidates:
                for schedule in ("reuse_weight", "reuse_activation"):
                    supported = evaluate_candidate(
                        group=group,
                        tm=tm,
                        tn=tn,
                        tk=tk,
                        schedule=schedule,
                        split_same_matrix=False,
                        dma_setup_us=dma_setup_us,
                        dma_bw_bytes_per_us=dma_bw_bytes_per_us,
                        guard_bytes=guard_bytes,
                    )
                    if supported is not None:
                        supported_candidates.append(supported)

                    split_mode = evaluate_candidate(
                        group=group,
                        tm=tm,
                        tn=tn,
                        tk=tk,
                        schedule=schedule,
                        split_same_matrix=True,
                        dma_setup_us=dma_setup_us,
                        dma_bw_bytes_per_us=dma_bw_bytes_per_us,
                        guard_bytes=guard_bytes,
                    )
                    if split_mode is not None:
                        split_candidates.append(split_mode)

    supported_candidates.sort(key=_candidate_sort_key)
    split_candidates.sort(key=_candidate_sort_key)

    baseline_activation_bytes = sum(member.activation_bytes for member in group.members)
    baseline_weight_bytes = sum(member.weight_bytes for member in group.members)
    baseline_activation_calls = sum(member.activation_calls for member in group.members)
    baseline_weight_calls = sum(member.weight_calls for member in group.members)
    baseline_activation_us = sum(member.activation_us_total or 0.0 for member in group.members)
    baseline_weight_us = sum(member.weight_us_total or 0.0 for member in group.members)

    best_supported = supported_candidates[0] if supported_candidates else None
    best_split = split_candidates[0] if split_candidates else None

    def scale_candidate(candidate: Optional[CandidateResult]) -> Optional[dict]:
        if candidate is None:
            return None
        scaled = asdict(candidate)
        scaled["count"] = group.count
        scaled["activation_loads"] *= group.count
        scaled["weight_loads"] *= group.count
        scaled["activation_bytes"] *= group.count
        scaled["weight_bytes"] *= group.count
        scaled["both_change_events"] *= group.count
        scaled["activation_only_events"] *= group.count
        scaled["weight_only_events"] *= group.count
        scaled["total_bytes"] = scaled["activation_bytes"] + scaled["weight_bytes"]
        scaled["est_mvin_us"] *= group.count
        return scaled

    baseline_total_est_mvin_us = baseline_activation_us + baseline_weight_us
    if baseline_total_est_mvin_us <= 0.0:
        # fallback when profile has no per-stage totals
        baseline_events = [
            (member.activation_bytes, member.weight_bytes)
            for member in group.members
            if member.activation_bytes > 0 or member.weight_bytes > 0
        ]
        baseline_total_est_mvin_us = estimate_events_us(
            events=baseline_events,
            dma_setup_us=dma_setup_us,
            dma_bw_bytes_per_us=dma_bw_bytes_per_us,
            split_same_matrix=False,
        )

    return {
        "group_key": group.group_key,
        "semantic_op": group.semantic_op,
        "shape": {
            "m": group.m,
            "n": group.n,
            "k": group.k,
        },
        "hardware": {
            "spm_bytes": group.spm_bytes,
            "acc_bytes": group.acc_bytes,
            "stage2_k_block": group.stage2_k_block,
        },
        "count": group.count,
        "baseline": {
            "activation_loads": baseline_activation_calls,
            "weight_loads": baseline_weight_calls,
            "activation_bytes": baseline_activation_bytes,
            "weight_bytes": baseline_weight_bytes,
            "total_bytes": baseline_activation_bytes + baseline_weight_bytes,
            "est_mvin_us": baseline_total_est_mvin_us,
            "first_stage_tm": group.members[0].first_stage_tm,
            "first_stage_tn": group.members[0].first_stage_tn,
            "first_stage_tk": group.members[0].first_stage_tk,
        },
        "best_supported": scale_candidate(best_supported),
        "best_split_same_matrix": scale_candidate(best_split),
        "top_candidates_supported": [scale_candidate(item) for item in supported_candidates[:topk]],
        "top_candidates_split_same_matrix": [scale_candidate(item) for item in split_candidates[:topk]],
    }


def _search_group_worker(task: Tuple[LayerGroup, float, float, int, int]) -> Dict[str, object]:
    group, dma_setup_us, dma_bw_bytes_per_us, guard_bytes, topk = task
    return search_group(
        group=group,
        dma_setup_us=dma_setup_us,
        dma_bw_bytes_per_us=dma_bw_bytes_per_us,
        guard_bytes=guard_bytes,
        topk=topk,
    )


def _improvement_ratio(baseline_us: float, candidate_us: Optional[float]) -> Optional[float]:
    if candidate_us is None or baseline_us <= 0:
        return None
    return baseline_us / candidate_us


def run_search(args: argparse.Namespace) -> dict:
    layers = parse_profile(
        profile_json=args.profile_json,
        semantic_op=args.semantic_op,
        layer_id=args.layer_id,
    )
    groups = build_groups(layers, args.group_by)
    worker_count = max(1, min(args.jobs, len(groups))) if groups else 1
    tasks = [
        (
            group,
            args.dma_setup_us,
            args.dma_bw_bytes_us,
            args.guard_kb * 1024,
            args.topk,
        )
        for group in groups
    ]

    if worker_count == 1:
        group_results = [_search_group_worker(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            group_results = list(executor.map(_search_group_worker, tasks))

    group_results.sort(key=lambda item: item["group_key"])

    baseline_total_bytes = sum(item["baseline"]["total_bytes"] for item in group_results)
    baseline_total_est_mvin_us = sum(item["baseline"]["est_mvin_us"] for item in group_results)
    best_supported_total_est_mvin_us = sum(
        item["best_supported"]["est_mvin_us"] for item in group_results if item["best_supported"] is not None
    )
    best_split_total_est_mvin_us = sum(
        item["best_split_same_matrix"]["est_mvin_us"]
        for item in group_results
        if item["best_split_same_matrix"] is not None
    )

    return {
        "input": {
            "profile_json": str(Path(args.profile_json)),
            "group_by": args.group_by,
            "semantic_op": args.semantic_op,
            "layer_id": args.layer_id,
            "jobs": worker_count,
        },
        "hardware": {
            "dma_setup_us": args.dma_setup_us,
            "dma_bw_bytes_per_us": args.dma_bw_bytes_us,
            "guard_bytes": args.guard_kb * 1024,
        },
        "model": {
            "group_count": len(group_results),
            "layer_count": len(layers),
        },
        "summary": {
            "baseline_total_bytes": baseline_total_bytes,
            "baseline_total_est_mvin_us": baseline_total_est_mvin_us,
            "best_supported_total_est_mvin_us": best_supported_total_est_mvin_us,
            "best_supported_improvement": _improvement_ratio(
                baseline_total_est_mvin_us,
                best_supported_total_est_mvin_us if best_supported_total_est_mvin_us > 0 else None,
            ),
            "best_split_same_matrix_total_est_mvin_us": best_split_total_est_mvin_us,
            "best_split_same_matrix_improvement": _improvement_ratio(
                baseline_total_est_mvin_us,
                best_split_total_est_mvin_us if best_split_total_est_mvin_us > 0 else None,
            ),
        },
        "groups": group_results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search GEMM MVIN-friendly tiling from ggml_npu_profile.json")
    parser.add_argument("--profile-json", required=True, help="Path to ggml_npu_profile.json")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("--semantic-op", default=None, help="Only analyze one semantic_op")
    parser.add_argument("--layer-id", type=int, default=None, help="Only analyze one layer_id")
    parser.add_argument("--group-by", choices=["shape", "node"], default="shape", help="Aggregate by repeated shape or per-node")
    parser.add_argument("--topk", type=int, default=10, help="Number of top candidates per mode")
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker count across profile groups/shapes",
    )
    parser.add_argument("--dma-setup-us", type=float, default=DEFAULT_DMA_SETUP_US, help="DMA fixed setup overhead per transfer")
    parser.add_argument(
        "--dma-bw-bytes-us",
        type=float,
        default=DEFAULT_DMA_BW_BYTES_PER_US,
        help="Effective DMA bandwidth in bytes/us",
    )
    parser.add_argument("--guard-kb", type=int, default=DEFAULT_GUARD_BYTES // 1024, help="Reserved SPM guard size")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_search(args)

    output_path = args.output
    if output_path is None:
        profile_path = Path(args.profile_json)
        output_path = str(profile_path.with_name(f"{profile_path.stem}_gemm_mvin_search.json"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
