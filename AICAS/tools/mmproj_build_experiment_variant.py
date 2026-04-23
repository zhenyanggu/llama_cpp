#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AICAS = ROOT / "AICAS"
DEFAULT_LAYER_MANIFEST = AICAS / "artifacts" / "layer_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a consistent mmproj experiment variant from SmoothQuant candidates."
    )
    parser.add_argument("--candidates", required=True, help="SmoothQuant candidate JSON")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value to materialize")
    parser.add_argument("--clip-mode", choices=["minmax", "percentile"], default="minmax")
    parser.add_argument("--variant-name", required=True, help="Variant name used for output filenames")
    parser.add_argument("--output-dir", required=True, help="Output directory for policy/GGUF/summary")
    parser.add_argument("--input-gguf", required=True, help="Base GGUF used for metadata and F16 fallback tensors")
    parser.add_argument(
        "--source-weights-gguf",
        default="",
        help="Optional float GGUF used as quantization source. Defaults to --input-gguf",
    )
    parser.add_argument("--layer-manifest", default=str(DEFAULT_LAYER_MANIFEST))
    parser.add_argument("--weight-granularity", choices=["per_channel", "per_tensor"], default="per_tensor")
    parser.add_argument("--schema", default="smolvlm2_idefics3_static_w8a8_v1")
    parser.add_argument(
        "--mode",
        choices=["quantize", "metadata-only"],
        default="quantize",
        help="Pass-through mode for mmproj_pack_gguf.py",
    )
    parser.add_argument(
        "--reference-active-gguf",
        action="append",
        default=[],
        help="Seed the enabled set from active W8A8 tensors in an existing GGUF. May be repeated.",
    )
    parser.add_argument("--disable-all-w8a8", action="store_true", help="Start with all tensors disabled")
    parser.add_argument("--enable-kind", action="append", default=[])
    parser.add_argument("--disable-kind", action="append", default=[])
    parser.add_argument("--enable-block", action="append", type=int, default=[])
    parser.add_argument("--disable-block", action="append", type=int, default=[])
    parser.add_argument("--enable-block-range", action="append", default=[])
    parser.add_argument("--disable-block-range", action="append", default=[])
    parser.add_argument("--enable-tensor", action="append", default=[])
    parser.add_argument("--disable-tensor", action="append", default=[])
    parser.add_argument("--enable-regex", action="append", default=[])
    parser.add_argument("--disable-regex", action="append", default=[])
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke mmproj_pack_gguf.py",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_range(text: str) -> tuple[int, int]:
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid block range '{text}', expected START:END")
    start = int(parts[0])
    end = int(parts[1])
    if end < start:
        raise ValueError(f"invalid block range '{text}', END must be >= START")
    return start, end


def active_tensors_from_gguf(path: Path) -> set[str]:
    sys.path.insert(0, str(ROOT / "gguf-py"))
    import gguf  # type: ignore

    reader = gguf.GGUFReader(str(path), "r")
    field = reader.get_field("aicas.w8a8.tensor_count")
    if field is None:
        return set()

    active: set[str] = set()
    tensor_count = int(field.contents())
    for idx in range(tensor_count):
        prefix = f"aicas.w8a8.tensor.{idx}."
        name = str(reader.get_field(prefix + "name").contents())
        enabled_field = reader.get_field(prefix + "enabled")
        policy_field = reader.get_field(prefix + "policy")
        enabled = bool(enabled_field.contents()) if enabled_field is not None else False
        policy = str(policy_field.contents()) if policy_field is not None else ""
        if enabled and policy == "W8A8":
            active.add(name)
    return active


def resolve_alpha_entry(doc: dict[str, Any], alpha: float) -> dict[str, Any]:
    for entry in doc.get("alpha_results", []):
        if abs(float(entry["alpha"]) - float(alpha)) < 1e-8:
            return entry
    raise ValueError(f"alpha {alpha} not found in candidates document")


def build_initial_enabled_set(args: argparse.Namespace, layers: list[dict[str, Any]]) -> set[str]:
    if args.disable_all_w8a8:
        enabled: set[str] = set()
    else:
        enabled = {layer["tensor_name"] for layer in layers}

    for ref in args.reference_active_gguf:
        enabled = active_tensors_from_gguf(Path(ref).resolve())

    return enabled


def apply_layer_selection(args: argparse.Namespace, layers: list[dict[str, Any]], enabled: set[str]) -> list[dict[str, Any]]:
    by_name = {layer["tensor_name"]: layer for layer in layers}
    compiled_enable = [re.compile(pattern) for pattern in args.enable_regex]
    compiled_disable = [re.compile(pattern) for pattern in args.disable_regex]
    enable_kinds = set(args.enable_kind)
    disable_kinds = set(args.disable_kind)
    enable_blocks = set(args.enable_block)
    disable_blocks = set(args.disable_block)
    enable_ranges = [parse_range(text) for text in args.enable_block_range]
    disable_ranges = [parse_range(text) for text in args.disable_block_range]

    decisions: list[dict[str, Any]] = []
    for layer in layers:
        name = layer["tensor_name"]
        reason = "default_enabled" if name in enabled else "default_disabled"
        is_enabled = name in enabled
        layer_index = int(layer.get("layer_index", -1))
        kind = str(layer.get("kind", ""))

        if kind in enable_kinds:
            is_enabled = True
            reason = f"enable_kind:{kind}"
        if layer_index in enable_blocks:
            is_enabled = True
            reason = f"enable_block:{layer_index}"
        for start, end in enable_ranges:
            if start <= layer_index <= end:
                is_enabled = True
                reason = f"enable_block_range:{start}:{end}"
        if name in args.enable_tensor:
            is_enabled = True
            reason = f"enable_tensor:{name}"
        for regex in compiled_enable:
            if regex.search(name):
                is_enabled = True
                reason = f"enable_regex:{regex.pattern}"

        if kind in disable_kinds:
            is_enabled = False
            reason = f"disable_kind:{kind}"
        if layer_index in disable_blocks:
            is_enabled = False
            reason = f"disable_block:{layer_index}"
        for start, end in disable_ranges:
            if start <= layer_index <= end:
                is_enabled = False
                reason = f"disable_block_range:{start}:{end}"
        if name in args.disable_tensor:
            is_enabled = False
            reason = f"disable_tensor:{name}"
        for regex in compiled_disable:
            if regex.search(name):
                is_enabled = False
                reason = f"disable_regex:{regex.pattern}"

        decisions.append(
            {
                "tensor_name": name,
                "enabled": is_enabled,
                "reason": reason,
                "kind": kind,
                "layer_index": layer_index,
            }
        )

    missing = [name for name in args.enable_tensor + args.disable_tensor if name not in by_name]
    if missing:
        raise ValueError(f"requested tensor names not found in candidates: {missing}")
    return decisions


def materialize_policy(doc: dict[str, Any], alpha_entry: dict[str, Any], clip_mode: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    decision_map = {item["tensor_name"]: item for item in decisions}
    layers_out = []
    for layer in alpha_entry.get("layers", []):
        selected = layer["smoothed_activation"][clip_mode]
        decision = decision_map[layer["tensor_name"]]
        enabled = bool(decision["enabled"])
        layers_out.append(
            {
                "id": layer["id"],
                "tensor_name": layer["tensor_name"],
                "kind": layer["kind"],
                "layer_index": layer["layer_index"],
                "in_features": layer["in_features"],
                "out_features": layer["out_features"],
                "enabled": enabled,
                "policy": "W8A8" if enabled else "F16_FALLBACK",
                "reason": decision["reason"],
                "scheme": layer["scheme"],
                "act_quant_mode": doc.get("act_quant_mode", layer.get("act_quant_mode", "asymmetric_u8")),
                "alpha": layer["alpha"],
                "eps": layer["eps"],
                "smooth_enabled": layer["smooth_enabled"],
                "smooth_scale": layer["smooth_scale"],
                "act_scale": selected["act_scale"],
                "act_zero_point": selected["act_zero_point"],
                "clip_mode": clip_mode,
                "clip_min": selected["clip_min"],
                "clip_max": selected["clip_max"],
                "clipped_ratio": selected["clipped_ratio"],
            }
        )

    return {
        "schema": "aicas.mmproj.smoothquant_policy.v1",
        "candidates": doc.get("candidates") or "",
        "alpha": float(alpha_entry["alpha"]),
        "clip_mode": clip_mode,
        "act_quant_mode": doc.get("act_quant_mode", "asymmetric_u8"),
        "layers": layers_out,
    }


def run_pack(args: argparse.Namespace, policy_path: Path, output_gguf: Path, pack_summary_path: Path) -> None:
    cmd = [
        args.python,
        str(AICAS / "tools" / "mmproj_pack_gguf.py"),
        "--input-gguf",
        str(Path(args.input_gguf).resolve()),
        "--layer-manifest",
        str(Path(args.layer_manifest).resolve()),
        "--quant-params",
        str(policy_path),
        "--output-gguf",
        str(output_gguf),
        "--output-quant-params",
        str(pack_summary_path),
        "--mode",
        args.mode,
        "--weight-granularity",
        args.weight_granularity,
        "--schema",
        args.schema,
    ]
    if args.source_weights_gguf:
        cmd.extend(["--source-weights-gguf", str(Path(args.source_weights_gguf).resolve())])
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_summary(policy: dict[str, Any], args: argparse.Namespace, output_gguf: Path, policy_path: Path, pack_summary_path: Path) -> dict[str, Any]:
    enabled_layers = [layer for layer in policy["layers"] if layer["enabled"]]
    by_kind: dict[str, int] = {}
    for layer in enabled_layers:
        kind = str(layer["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "variant_name": args.variant_name,
        "clip_mode": args.clip_mode,
        "alpha": args.alpha,
        "weight_granularity": args.weight_granularity,
        "input_gguf": str(Path(args.input_gguf).resolve()),
        "source_weights_gguf": str(Path(args.source_weights_gguf or args.input_gguf).resolve()),
        "output_gguf": str(output_gguf.resolve()),
        "policy_json": str(policy_path.resolve()),
        "pack_summary_json": str(pack_summary_path.resolve()),
        "enabled_count": len(enabled_layers),
        "enabled_by_kind": by_kind,
        "enabled_tensors": [layer["tensor_name"] for layer in enabled_layers],
    }


def main() -> int:
    args = parse_args()
    candidates_path = Path(args.candidates).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = load_json(candidates_path)
    alpha_entry = resolve_alpha_entry(doc, args.alpha)
    layers = list(alpha_entry.get("layers", []))
    enabled = build_initial_enabled_set(args, layers)
    decisions = apply_layer_selection(args, layers, enabled)
    policy = materialize_policy(doc, alpha_entry, args.clip_mode, decisions)

    policy_path = output_dir / f"{args.variant_name}.policy.json"
    output_gguf = output_dir / f"{args.variant_name}.gguf"
    pack_summary_path = output_dir / f"{args.variant_name}.pack_summary.json"
    summary_path = output_dir / f"{args.variant_name}.summary.json"

    write_json(policy_path, policy)
    run_pack(args, policy_path, output_gguf, pack_summary_path)
    write_json(summary_path, build_summary(policy, args, output_gguf, policy_path, pack_summary_path))

    print(f"Wrote policy: {policy_path}")
    print(f"Wrote GGUF: {output_gguf}")
    print(f"Wrote pack summary: {pack_summary_path}")
    print(f"Wrote variant summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
