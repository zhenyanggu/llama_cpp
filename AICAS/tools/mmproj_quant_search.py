#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
AICAS_DIR = ROOT_DIR / "AICAS"
DEFAULT_GROUP_CATALOG = AICAS_DIR / "artifacts" / "mmproj_group_catalog.json"
DEFAULT_LAYER_MANIFEST = AICAS_DIR / "artifacts" / "layer_manifest.json"
DEFAULT_INVENTORY = AICAS_DIR / "artifacts" / "mmproj_mulmat_inventory.json"
DEFAULT_BASE_POLICY = AICAS_DIR / "artifacts" / "quant_params.json"
DEFAULT_OUTPUT_ROOT = AICAS_DIR / "output" / "mmproj-search"
DEFAULT_FP16_TEXT_MODEL = AICAS_DIR / "gguf" / "SmolVLM2-500M-Video-Instruct-f16.gguf"
DEFAULT_FP16_MMPROJ = AICAS_DIR / "gguf" / "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
DEFAULT_Q8_TEXT_MODEL = AICAS_DIR / "gguf" / "SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
DEFAULT_Q8_MMPROJ = AICAS_DIR / "gguf" / "mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
DEFAULT_EVAL_JSON = AICAS_DIR / "sampled.json"
DEFAULT_IMAGE_DIR = AICAS_DIR / "data"
DEFAULT_REGISTRY_JSONL = DEFAULT_OUTPUT_ROOT / "registry.jsonl"
DEFAULT_REGISTRY_CSV = DEFAULT_OUTPUT_ROOT / "registry.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_ROOT / "final_summary.json"
DEFAULT_PILOT_JSON = AICAS_DIR / "artifacts" / "eval_subsets" / "pilot24.json"
TRACKED_CODE_PATHS = [
    AICAS_DIR / "acc_eval.py",
    AICAS_DIR / "scripts" / "run_local_acc_eval.sh",
    AICAS_DIR / "tools" / "mmproj_pack_gguf.py",
    AICAS_DIR / "tools" / "mmproj_calibrate.py",
    ROOT_DIR / "tools" / "mtmd" / "clip.cpp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mmproj 量化搜索与实验缓存入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="生成 mmproj 分组目录")
    catalog.add_argument("--layer-manifest", default=str(DEFAULT_LAYER_MANIFEST))
    catalog.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    catalog.add_argument("--output", default=str(DEFAULT_GROUP_CATALOG))

    run = subparsers.add_parser("run", help="执行一个配置的打包与 acc 评测")
    add_shared_run_args(run)
    run.add_argument("--experiment-id", default="", help="实验 ID，留空时自动生成")
    run.add_argument("--phase", default="manual")
    run.add_argument("--enable-group", action="append", default=[], help="启用的 group_id，可重复传入")
    run.add_argument("--disable-group", action="append", default=[], help="从启用列表中剔除的 group_id，可重复传入")
    run.add_argument("--notes", default="")
    run.add_argument("--fp16-baseline-score", type=int, default=-1)
    run.add_argument("--threshold", type=int, default=-1)
    run.add_argument("--print-only", action="store_true", help="只打印将执行的配置，不打包不评测")
    run.add_argument("--server-bin", default="", help="传给 run_local_acc_eval.sh 的 --server-bin")
    run.add_argument("--port", type=int, default=18080)
    run.add_argument("--threads", type=int, default=0)
    run.add_argument("--python", default="python3")
    run.add_argument("--skip-venv", action="store_true")
    run.add_argument("--request-timeout", type=float, default=60.0)
    run.add_argument("--request-retries", type=int, default=0)
    run.add_argument("--retry-delay", type=float, default=0.5)

    search = subparsers.add_parser("search", help="按计划执行基线、粗筛与贪心搜索")
    add_shared_run_args(search)
    search.add_argument("--notes", default="")
    search.add_argument("--pilot-size", type=int, default=24)
    search.add_argument("--max-refine-kinds", type=int, default=2)
    search.add_argument("--top-block-candidates", type=int, default=4)
    search.add_argument("--min-remaining-param-stop", type=int, default=7077888)
    search.add_argument("--server-bin", default="", help="传给 run_local_acc_eval.sh 的 --server-bin")
    search.add_argument("--port-base", type=int, default=18080)
    search.add_argument("--threads", type=int, default=0)
    search.add_argument("--python", default="python3")
    search.add_argument("--skip-venv", action="store_true")
    search.add_argument("--request-timeout", type=float, default=60.0)
    search.add_argument("--request-retries", type=int, default=0)
    search.add_argument("--retry-delay", type=float, default=0.5)

    return parser.parse_args()


def add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--group-catalog", default=str(DEFAULT_GROUP_CATALOG))
    parser.add_argument("--base-policy", default=str(DEFAULT_BASE_POLICY))
    parser.add_argument("--base-mmproj", default=str(DEFAULT_FP16_MMPROJ))
    parser.add_argument("--text-model", default=str(DEFAULT_FP16_TEXT_MODEL))
    parser.add_argument("--image-folder", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL_JSON))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--registry-jsonl", default=str(DEFAULT_REGISTRY_JSONL))
    parser.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY_CSV))
    parser.add_argument("--weight-granularity", default="per_channel", choices=["per_channel", "per_tensor"])
    parser.add_argument("--act-quant-mode", default="asymmetric_u8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_registry_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                elif value is None:
                    normalized[key] = ""
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def append_registry(jsonl_path: Path, csv_path: Path, row: dict) -> None:
    rows = load_registry(jsonl_path)
    rows.append(row)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    save_registry_csv(csv_path, rows)


def build_group_catalog(layer_manifest: Path, inventory: Path, output: Path) -> dict:
    cmd = [
        sys.executable,
        str(AICAS_DIR / "tools" / "mmproj_group_catalog.py"),
        "--layer-manifest",
        str(layer_manifest),
        "--inventory",
        str(inventory),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    return read_json(output)


def ensure_group_catalog(path: Path) -> dict:
    if path.exists():
        return read_json(path)
    return build_group_catalog(DEFAULT_LAYER_MANIFEST, DEFAULT_INVENTORY, path)


def build_group_maps(catalog: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    group_map = {group["group_id"]: group for group in catalog["groups"]}
    layer_map = {layer["tensor_name"]: layer for layer in catalog["layers"]}
    return group_map, layer_map


def resolve_enabled_tensors(group_map: dict[str, dict], enabled_groups: list[str], disabled_groups: list[str]) -> tuple[list[str], list[str], list[str]]:
    missing = [group_id for group_id in enabled_groups + disabled_groups if group_id not in group_map]
    if missing:
        raise SystemExit(f"未知 group_id: {missing}")

    enabled_set: set[str] = set()
    for group_id in enabled_groups:
        enabled_set.update(group_map[group_id]["tensor_names"])
    for group_id in disabled_groups:
        enabled_set.difference_update(group_map[group_id]["tensor_names"])

    group_ids_enabled = sorted(set(enabled_groups))
    group_ids_disabled = sorted(set(disabled_groups))
    tensor_names_enabled = sorted(enabled_set)
    return group_ids_enabled, group_ids_disabled, tensor_names_enabled


def build_policy_doc(base_policy: dict, tensor_names_enabled: list[str]) -> dict:
    enabled = set(tensor_names_enabled)
    policy_doc = json.loads(json.dumps(base_policy))
    for layer in policy_doc.get("layers", []):
        if layer["tensor_name"] in enabled:
            layer["enabled"] = True
            layer["policy"] = "W8A8"
        else:
            layer["enabled"] = False
            layer["policy"] = "F16_FALLBACK"
    return policy_doc


def compute_quantized_param_count(tensor_names_enabled: list[str], layer_map: dict[str, dict]) -> int:
    return sum(int(layer_map[name]["param_count"]) for name in tensor_names_enabled)


def score_eval_result(result_json: Path) -> tuple[int, int]:
    payload = read_json(result_json)
    if not isinstance(payload, list):
        raise SystemExit(f"结果文件不是样本列表: {result_json}")
    score = 0
    errors = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        if row.get("result") == 1:
            score += 1
        predict = row.get("predict")
        if isinstance(predict, str) and predict.startswith(("API_ERROR:", "ERROR:")):
            errors += 1
    return score, errors


def make_code_fingerprint() -> dict:
    return {str(path.relative_to(ROOT_DIR)): file_sha256(path) for path in TRACKED_CODE_PATHS}


def make_cache_key(
    *,
    eval_json: Path,
    base_policy: Path,
    group_catalog: Path,
    text_model: Path,
    base_mmproj: Path,
    weight_granularity: str,
    act_quant_mode: str,
    tensor_names_enabled: list[str],
) -> tuple[str, dict]:
    payload = {
        "eval_json": str(eval_json.resolve()),
        "eval_json_sha256": file_sha256(eval_json),
        "base_policy": str(base_policy.resolve()),
        "base_policy_sha256": file_sha256(base_policy),
        "group_catalog": str(group_catalog.resolve()),
        "group_catalog_sha256": file_sha256(group_catalog),
        "text_model": str(text_model.resolve()),
        "text_model_sha256": file_sha256(text_model),
        "base_mmproj": str(base_mmproj.resolve()),
        "base_mmproj_sha256": file_sha256(base_mmproj),
        "weight_granularity": weight_granularity,
        "act_quant_mode": act_quant_mode,
        "tensor_names_enabled": tensor_names_enabled,
        "code_fingerprint": make_code_fingerprint(),
    }
    return stable_hash(payload), payload["code_fingerprint"]


def find_cached_row(rows: list[dict], cache_key: str) -> dict | None:
    for row in reversed(rows):
        if row.get("cache_key") == cache_key:
            result_json = row.get("result_json", "")
            if result_json and Path(result_json).exists():
                return row
    return None


def run_subprocess(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def choose_experiment_id(prefix: str, enabled_groups: list[str], tensor_names_enabled: list[str]) -> str:
    if prefix:
        return prefix
    token = stable_hash({"groups": enabled_groups, "tensors": tensor_names_enabled})[:12]
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{token}"


def execute_run(args: argparse.Namespace, *, port: int | None = None) -> dict:
    group_catalog_path = Path(args.group_catalog).resolve()
    base_policy_path = Path(args.base_policy).resolve()
    base_mmproj_path = Path(args.base_mmproj).resolve()
    text_model_path = Path(args.text_model).resolve()
    eval_json_path = Path(args.eval_json).resolve()
    image_folder_path = Path(args.image_folder).resolve()
    output_root = Path(args.output_root).resolve()
    registry_jsonl = Path(args.registry_jsonl).resolve()
    registry_csv = Path(args.registry_csv).resolve()

    catalog = ensure_group_catalog(group_catalog_path)
    group_map, layer_map = build_group_maps(catalog)
    base_policy = read_json(base_policy_path)
    declared_act_quant_mode = base_policy.get("act_quant_mode")
    if declared_act_quant_mode and declared_act_quant_mode != args.act_quant_mode:
        raise SystemExit(
            f"base policy act_quant_mode={declared_act_quant_mode}，与请求的 {args.act_quant_mode} 不一致"
        )
    group_ids_enabled, group_ids_disabled, tensor_names_enabled = resolve_enabled_tensors(
        group_map, args.enable_group, args.disable_group
    )
    quantized_param_count = compute_quantized_param_count(tensor_names_enabled, layer_map)
    quantized_layer_count = len(tensor_names_enabled)
    cache_key, code_fingerprint = make_cache_key(
        eval_json=eval_json_path,
        base_policy=base_policy_path,
        group_catalog=group_catalog_path,
        text_model=text_model_path,
        base_mmproj=base_mmproj_path,
        weight_granularity=args.weight_granularity,
        act_quant_mode=args.act_quant_mode,
        tensor_names_enabled=tensor_names_enabled,
    )

    existing_rows = load_registry(registry_jsonl)
    cached_row = find_cached_row(existing_rows, cache_key)
    if cached_row is not None:
        replay = dict(cached_row)
        replay["cached"] = True
        return replay

    experiment_id = choose_experiment_id(args.experiment_id, group_ids_enabled, tensor_names_enabled)
    experiment_dir = output_root / experiment_id
    policy_path = experiment_dir / "policy.json"
    packed_mmproj_path = experiment_dir / "mmproj.gguf"
    pack_summary_path = experiment_dir / "pack_summary.json"
    result_json = experiment_dir / "result.json"

    row = {
        "timestamp_utc": timestamp_utc(),
        "experiment_id": experiment_id,
        "phase": args.phase,
        "notes": args.notes,
        "text_model_path": str(text_model_path),
        "mmproj_base_path": str(base_mmproj_path),
        "mmproj_packed_path": str(packed_mmproj_path),
        "eval_json_path": str(eval_json_path),
        "policy_json": str(policy_path),
        "pack_summary_json": str(pack_summary_path),
        "result_json": str(result_json),
        "group_ids_enabled": group_ids_enabled,
        "group_ids_disabled": group_ids_disabled,
        "tensor_names_enabled": tensor_names_enabled,
        "only_mulmat": True,
        "skip_decode_gemv": True,
        "act_quant_mode": args.act_quant_mode,
        "weight_granularity": args.weight_granularity,
        "quantized_layer_count": quantized_layer_count,
        "quantized_param_count": quantized_param_count,
        "fp16_baseline_score": None if args.fp16_baseline_score < 0 else int(args.fp16_baseline_score),
        "threshold": None if args.threshold < 0 else int(args.threshold),
        "score": None,
        "delta_vs_fp16": None,
        "passes_threshold": None,
        "errors": None,
        "eval_failed": False,
        "eval_failure_kind": "",
        "cache_key": cache_key,
        "code_fingerprint": code_fingerprint,
        "cached": False,
    }

    if args.print_only:
        return row

    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(policy_path, build_policy_doc(base_policy, tensor_names_enabled))

    pack_cmd = [
        sys.executable,
        str(AICAS_DIR / "tools" / "mmproj_pack_gguf.py"),
        "--input-gguf",
        str(base_mmproj_path),
        "--layer-manifest",
        str(DEFAULT_LAYER_MANIFEST.resolve()),
        "--quant-params",
        str(policy_path),
        "--output-gguf",
        str(packed_mmproj_path),
        "--output-quant-params",
        str(pack_summary_path),
        "--mode",
        "quantize",
        "--weight-granularity",
        args.weight_granularity,
    ]
    run_subprocess(pack_cmd, ROOT_DIR)

    eval_cmd = [
        "bash",
        str(AICAS_DIR / "scripts" / "run_local_acc_eval.sh"),
        "--model",
        str(text_model_path),
        "--mmproj",
        str(packed_mmproj_path),
        "--image-folder",
        str(image_folder_path),
        "--ocrbench-file",
        str(eval_json_path),
        "--output-folder",
        str(experiment_dir),
        "--save-name",
        "result",
    ]
    if getattr(args, "server_bin", ""):
        eval_cmd.extend(["--server-bin", args.server_bin])
    run_port = port if port is not None else int(getattr(args, "port", 18080))
    eval_cmd.extend(["--port", str(run_port)])
    if getattr(args, "threads", 0):
        eval_cmd.extend(["--threads", str(args.threads)])
    if getattr(args, "python", ""):
        eval_cmd.extend(["--python", args.python])
    if getattr(args, "skip_venv", False):
        eval_cmd.append("--skip-venv")
    eval_cmd.extend(["--request-timeout", str(getattr(args, "request_timeout", 60.0))])
    eval_cmd.extend(["--request-retries", str(getattr(args, "request_retries", 0))])
    eval_cmd.extend(["--retry-delay", str(getattr(args, "retry_delay", 0.5))])

    try:
        run_subprocess(eval_cmd, ROOT_DIR)
    except subprocess.CalledProcessError as exc:
        row["eval_failed"] = True
        row["eval_failure_kind"] = f"subprocess_exit_{exc.returncode}"
        if not result_json.exists():
            row["score"] = -1
            row["errors"] = -1
            if row["fp16_baseline_score"] is not None:
                row["delta_vs_fp16"] = int(row["fp16_baseline_score"]) - row["score"]
            if row["threshold"] is not None:
                row["passes_threshold"] = False
            append_registry(registry_jsonl, registry_csv, row)
            return row

    score, errors = score_eval_result(result_json)
    row["score"] = score
    row["errors"] = errors
    if row["fp16_baseline_score"] is not None:
        row["delta_vs_fp16"] = int(row["fp16_baseline_score"]) - score
    if row["threshold"] is not None:
        row["passes_threshold"] = score >= int(row["threshold"])

    append_registry(registry_jsonl, registry_csv, row)
    return row


def summarize_run(row: dict) -> str:
    return (
        f"experiment={row['experiment_id']} score={row.get('score')} "
        f"threshold={row.get('threshold')} pass={row.get('passes_threshold')} "
        f"layers={row.get('quantized_layer_count')} params={row.get('quantized_param_count')} "
        f"cached={row.get('cached')}"
    )


def build_pilot_subset(source_json: Path, output_json: Path, pilot_size: int) -> list[dict]:
    data = read_json(source_json)
    if not isinstance(data, list):
        raise SystemExit(f"评测集不是列表: {source_json}")
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in data:
        key = (str(item.get("type", "")), str(item.get("dataset_name", "")))
        buckets[key].append(item)

    selected = []
    max_bucket_len = max((len(bucket) for bucket in buckets.values()), default=0)
    ordered_keys = sorted(buckets.keys())
    for index in range(max_bucket_len):
        for key in ordered_keys:
            bucket = buckets[key]
            if index < len(bucket):
                selected.append(bucket[index])
                if len(selected) >= pilot_size:
                    write_json(output_json, selected)
                    return selected

    write_json(output_json, selected)
    return selected


def choose_sort_key(row: dict, group_map: dict[str, dict]) -> tuple[int, int, str]:
    enabled_groups = row.get("group_ids_enabled", [])
    group_id = enabled_groups[0] if enabled_groups else ""
    param_count = int(group_map.get(group_id, {}).get("param_count", 0))
    delta = 1 << 30 if row.get("delta_vs_fp16") is None else int(row["delta_vs_fp16"])
    return (delta, -param_count, group_id)


def execute_search(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "final_summary.json"
    catalog = ensure_group_catalog(Path(args.group_catalog).resolve())
    group_map, layer_map = build_group_maps(catalog)

    pilot_subset = build_pilot_subset(Path(args.eval_json).resolve(), DEFAULT_PILOT_JSON.resolve(), args.pilot_size)

    class RunArgs:
        pass

    def make_run_args(**overrides: Any) -> argparse.Namespace:
        run_args = RunArgs()
        for key, value in vars(args).items():
            setattr(run_args, key, value)
        run_args.command = "run"
        run_args.print_only = False
        run_args.experiment_id = ""
        run_args.phase = overrides.get("phase", "search")
        run_args.enable_group = overrides.get("enable_group", [])
        run_args.disable_group = overrides.get("disable_group", [])
        run_args.notes = overrides.get("notes", "")
        run_args.fp16_baseline_score = overrides.get("fp16_baseline_score", -1)
        run_args.threshold = overrides.get("threshold", -1)
        run_args.text_model = overrides.get("text_model", args.text_model)
        run_args.base_mmproj = overrides.get("base_mmproj", args.base_mmproj)
        run_args.eval_json = overrides.get("eval_json", args.eval_json)
        run_args.experiment_id = overrides.get("experiment_id", "")
        return run_args

    port = args.port_base

    fp16_row = execute_run(
        make_run_args(
            phase="baseline_fp16",
            notes="fp16 文本模型 + fp16 mmproj 基线",
            enable_group=[],
            threshold=-1,
            fp16_baseline_score=-1,
            experiment_id="baseline-fp16-mmproj-f16",
        ),
        port=port,
    )
    port += 1
    fp16_acc = int(fp16_row["score"])
    threshold = fp16_acc - 5

    q8_row = execute_run(
        make_run_args(
            phase="baseline_q8_ref",
            notes="官方 Q8_0 文本模型 + 官方 Q8_0 mmproj 参考",
            enable_group=[],
            threshold=threshold,
            fp16_baseline_score=fp16_acc,
            text_model=str(DEFAULT_Q8_TEXT_MODEL),
            base_mmproj=str(DEFAULT_Q8_MMPROJ),
            experiment_id="baseline-q8-mmproj-q8",
        ),
        port=port,
    )
    port += 1

    full_group_ids = [group["group_id"] for group in catalog["groups"] if group["level"] == "kind"]
    full_w8a8_row = execute_run(
        make_run_args(
            phase="baseline_full_w8a8",
            notes="73 层 mmproj 全开 W8A8 下界参考",
            enable_group=full_group_ids,
            threshold=threshold,
            fp16_baseline_score=fp16_acc,
            experiment_id="baseline-full-w8a8",
        ),
        port=port,
    )
    port += 1

    coarse_group_ids = [
        "kind:projector_all",
        "kind:attn_q_all",
        "kind:attn_k_all",
        "kind:attn_v_all",
        "kind:attn_out_all",
        "kind:ffn_up_all",
        "kind:ffn_down_all",
    ]
    single_group_rows = []
    for group_id in coarse_group_ids:
        row = execute_run(
            make_run_args(
                phase="single_group_screen",
                notes=f"单组启用粗筛: {group_id}",
                enable_group=[group_id],
                threshold=threshold,
                fp16_baseline_score=fp16_acc,
                experiment_id=f"single-{group_id.replace(':', '_')}",
            ),
            port=port,
        )
        single_group_rows.append(row)
        port += 1

    single_group_rows = sorted(single_group_rows, key=lambda row: choose_sort_key(row, group_map))
    coarse_order = [row["group_ids_enabled"][0] for row in single_group_rows]

    accepted_coarse: list[str] = []
    rejected_coarse: list[dict] = []
    best_row = fp16_row

    for group_id in coarse_order:
        candidate_groups = sorted(set(accepted_coarse + [group_id]))
        row = execute_run(
            make_run_args(
                phase="coarse_greedy",
                notes=f"粗粒度贪心加入 {group_id}",
                enable_group=candidate_groups,
                threshold=threshold,
                fp16_baseline_score=fp16_acc,
                experiment_id=f"greedy-{'-'.join(group.replace(':', '_') for group in candidate_groups)}",
            ),
            port=port,
        )
        port += 1
        if row.get("passes_threshold"):
            accepted_coarse = candidate_groups
            best_row = row
        else:
            rejected_coarse.append(row)

    remaining_param = sum(group_map[group_id]["param_count"] for group_id in coarse_group_ids if group_id not in accepted_coarse)
    refine_rows = []
    refine_candidates = []
    for row in rejected_coarse:
        enabled_groups = row.get("group_ids_enabled", [])
        group_id = enabled_groups[-1] if enabled_groups else ""
        if group_id.startswith("kind:projector"):
            continue
        refine_candidates.append(row)

    refine_candidates.sort(
        key=lambda row: (
            1 << 30 if row.get("score") is None else abs(int(row["score"]) - threshold),
            -int(group_map[row["group_ids_enabled"][-1]]["param_count"]),
        )
    )
    refine_candidates = refine_candidates[: args.max_refine_kinds]

    if remaining_param >= args.min_remaining_param_stop:
        for coarse_row in refine_candidates:
            coarse_group_id = coarse_row["group_ids_enabled"][-1]
            kind = group_map[coarse_group_id]["kind"]
            pilot_rows = []
            for block_idx in range(12):
                group_id = f"block_kind:blk_{block_idx:02d}.{kind}"
                row = execute_run(
                    make_run_args(
                        phase="pilot_block_screen",
                        notes=f"{coarse_group_id} 的 block 细化 pilot: {group_id}",
                        enable_group=sorted(set(accepted_coarse + [group_id])),
                        threshold=threshold,
                        fp16_baseline_score=fp16_acc,
                        eval_json=str(DEFAULT_PILOT_JSON),
                        experiment_id=f"pilot-{kind}-blk-{block_idx:02d}",
                    ),
                    port=port,
                )
                pilot_rows.append(row)
                port += 1

            pilot_rows.sort(key=lambda row: choose_sort_key(row, group_map))
            top_group_ids = [row["group_ids_enabled"][-1] for row in pilot_rows[: args.top_block_candidates]]

            for group_id in top_group_ids:
                row = execute_run(
                    make_run_args(
                        phase="block_refine",
                        notes=f"从 pilot 候选中尝试加入 {group_id}",
                        enable_group=sorted(set(best_row["group_ids_enabled"] + [group_id])),
                        threshold=threshold,
                        fp16_baseline_score=fp16_acc,
                        experiment_id=f"refine-{group_id.replace(':', '_')}",
                    ),
                    port=port,
                )
                refine_rows.append(row)
                port += 1
                if row.get("passes_threshold"):
                    best_row = row

    all_candidate_rows = [best_row] + refine_rows
    passing_rows = [row for row in all_candidate_rows if row.get("passes_threshold")]
    if passing_rows:
        best_row = sorted(
            passing_rows,
            key=lambda row: (
                -int(row["quantized_param_count"]),
                -(int(row["score"]) if row.get("score") is not None else -1),
                len(row.get("group_ids_disabled", [])),
            ),
        )[0]

    sensitivity_rows = []
    for row in single_group_rows:
        group_id = row["group_ids_enabled"][0]
        group = group_map[group_id]
        sensitivity_rows.append(
            {
                "group_id": group_id,
                "kind": group["kind"],
                "param_count": group["param_count"],
                "score": row["score"],
                "delta_vs_fp16": row["delta_vs_fp16"],
                "passes_threshold": row["passes_threshold"],
            }
        )

    summary = {
        "schema": "aicas.mmproj.quant_search_summary.v1",
        "timestamp_utc": timestamp_utc(),
        "fp16_acc": fp16_acc,
        "q8_ref_acc": q8_row["score"],
        "threshold": threshold,
        "pilot_subset_path": str(DEFAULT_PILOT_JSON.resolve()),
        "pilot_subset_size": len(pilot_subset),
        "coarse_order": coarse_order,
        "accepted_coarse_groups": accepted_coarse,
        "rejected_coarse_groups": [row["group_ids_enabled"][-1] for row in rejected_coarse],
        "single_group_sensitivity": sensitivity_rows,
        "best_known_config": {
            "experiment_id": best_row["experiment_id"],
            "group_ids_enabled": best_row["group_ids_enabled"],
            "tensor_names_enabled": best_row["tensor_names_enabled"],
            "quantized_layer_count": best_row["quantized_layer_count"],
            "quantized_param_count": best_row["quantized_param_count"],
            "score": best_row["score"],
            "delta_vs_fp16": best_row["delta_vs_fp16"],
            "passes_threshold": best_row["passes_threshold"],
            "result_json": best_row["result_json"],
        },
        "baseline_rows": {
            "fp16": fp16_row,
            "q8_ref": q8_row,
            "full_w8a8": full_w8a8_row,
        },
        "next_try_directions": [row["group_ids_enabled"][-1] for row in refine_candidates],
    }
    write_json(summary_json.resolve(), summary)
    return summary


def main() -> int:
    args = parse_args()

    if args.command == "catalog":
        build_group_catalog(
            Path(args.layer_manifest).resolve(),
            Path(args.inventory).resolve(),
            Path(args.output).resolve(),
        )
        return 0

    if args.command == "run":
        row = execute_run(args)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print(summarize_run(row))
        return 0

    if args.command == "search":
        summary = execute_search(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
