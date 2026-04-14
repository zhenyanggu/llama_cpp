#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "gguf-py")))
import gguf  # type: ignore


@dataclass
class LayerPolicy:
    tensor_name: str
    enabled: bool
    policy: str
    act_scale: float
    act_zero_point: int
    act_quant_mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack mmproj GGUF with AICAS W8A8 metadata and optional I8 tensors.")
    p.add_argument("--input-gguf", required=True, help="Input mmproj GGUF")
    p.add_argument("--layer-manifest", required=True, help="Layer manifest JSON")
    p.add_argument("--quant-params", default="", help="Optional calibration params JSON that overrides manifest act params/policy")
    p.add_argument("--output-gguf", required=True, help="Output GGUF")
    p.add_argument("--output-quant-params", default="AICAS/artifacts/quant_params.json")
    p.add_argument("--schema", default="smolvlm2_idefics3_static_w8a8_v1")
    p.add_argument("--mode", choices=["quantize", "metadata-only"], default="quantize")
    p.add_argument(
        "--weight-granularity",
        choices=["per_channel", "per_tensor"],
        default="per_channel",
        help="Weight quantization granularity for emitted I8 tensors",
    )
    p.add_argument("--default-act-scale", type=float, default=0.02)
    p.add_argument("--default-act-zero-point", type=int, default=128)
    return p.parse_args()


def load_manifest(path: str, default_scale: float, default_zp: int) -> list[LayerPolicy]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layers = data.get("layers", [])
    out: list[LayerPolicy] = []
    for layer in layers:
        out.append(
            LayerPolicy(
                tensor_name=layer["tensor_name"],
                enabled=bool(layer.get("enabled", True)),
                policy=str(layer.get("policy", "W8A8")),
                act_scale=float(layer.get("act_scale", default_scale)),
                act_zero_point=int(layer.get("act_zero_point", default_zp)),
                act_quant_mode=str(layer.get("act_quant_mode", "asymmetric_u8")),
            )
        )
    return out


def overlay_quant_params(layers: list[LayerPolicy], quant_params_path: str) -> list[LayerPolicy]:
    if not quant_params_path:
        return layers

    with open(quant_params_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    default_act_quant_mode = str(data.get("act_quant_mode", "asymmetric_u8"))
    overrides = {item["tensor_name"]: item for item in data.get("layers", [])}
    out: list[LayerPolicy] = []
    for layer in layers:
        item = overrides.get(layer.tensor_name)
        if item is None:
            out.append(layer)
            continue

        out.append(
            LayerPolicy(
                tensor_name=layer.tensor_name,
                enabled=bool(item.get("enabled", layer.enabled)),
                policy=str(item.get("policy", layer.policy)),
                act_scale=float(item.get("act_scale", layer.act_scale)),
                act_zero_point=int(item.get("act_zero_point", layer.act_zero_point)),
                act_quant_mode=str(item.get("act_quant_mode", default_act_quant_mode)),
            )
        )

    return out


def quantize_per_channel_i8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weight.ndim != 2:
        raise ValueError(f"Only 2D weight tensors are supported, got ndim={weight.ndim}")

    # GGUFReader exposes tensor.data in backend storage order. For these mmproj
    # matmul weights that means [OUT, K], where each row is one output channel.
    w = np.asarray(weight, dtype=np.float32)
    max_abs = np.max(np.abs(w), axis=1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    return q, scales


def quantize_per_tensor_i8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weight.ndim != 2:
        raise ValueError(f"Only 2D weight tensors are supported, got ndim={weight.ndim}")

    w = np.asarray(weight, dtype=np.float32)
    max_abs = float(np.max(np.abs(w)))
    scale = np.float32(max_abs / 127.0) if max_abs > 0.0 else np.float32(1.0)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, np.asarray([scale], dtype=np.float32)


def maybe_build_sum_w(q: np.ndarray, act_quant_mode: str, act_zero_point: int) -> np.ndarray | None:
    if act_quant_mode == "symmetric_u8":
        if act_zero_point != 128:
            raise ValueError("symmetric_u8 requires act_zero_point=128 when omitting sum_w")
        return None

    return q.astype(np.int32).sum(axis=1).astype(np.int32)


def float_to_q8_24(scale: float) -> int:
    scaled = int(np.rint(np.float64(scale) * np.float64(1 << 24)))
    i32 = np.iinfo(np.int32)
    return int(np.clip(scaled, i32.min, i32.max))


def array_to_q8_24(scales: np.ndarray) -> np.ndarray:
    scaled = np.rint(scales.astype(np.float64) * np.float64(1 << 24)).astype(np.int64)
    i32 = np.iinfo(np.int32)
    clipped = np.clip(scaled, i32.min, i32.max)
    return clipped.astype(np.int32)


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        if field.name.startswith("aicas.w8a8."):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)


def main() -> int:
    args = parse_args()
    quant_params_input = os.path.abspath(args.quant_params) if args.quant_params else ""
    quant_params_output = os.path.abspath(args.output_quant_params)
    if quant_params_input and quant_params_input == quant_params_output:
        raise SystemExit(
            "--quant-params and --output-quant-params must be different paths; "
            "otherwise the calibration file gets overwritten by the pack summary"
        )

    layers = load_manifest(args.layer_manifest, args.default_act_scale, args.default_act_zero_point)
    layers = overlay_quant_params(layers, args.quant_params)
    layer_map = {x.tensor_name: x for x in layers}

    reader = gguf.GGUFReader(args.input_gguf, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output_gguf, arch=arch, endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    copy_metadata(reader, writer)

    # Apply AICAS metadata.
    writer.add_string("aicas.w8a8.schema", args.schema)
    writer.add_int32("aicas.w8a8.tensor_count", len(layers))

    quant_params: dict[str, Any] = {
        "schema": args.schema,
        "source_gguf": os.path.abspath(args.input_gguf),
        "layer_manifest": os.path.abspath(args.layer_manifest),
        "quant_params_input": quant_params_input,
        "mode": args.mode,
        "weight_granularity": args.weight_granularity,
        "act_quant_modes": sorted({layer.act_quant_mode for layer in layers}),
        "layers": [],
    }

    # First pass: register tensor infos and stage tensor payloads.
    staged: list[np.ndarray] = []
    replaced_tensors = 0
    for tensor in reader.tensors:
        name = tensor.name
        np_data = np.asarray(tensor.data)
        layer = layer_map.get(name)

        resolved_policy = "F16_FALLBACK"
        scale = None
        sum_w = None
        out_data = np_data
        out_raw_dtype = tensor.tensor_type

        if layer is not None and layer.enabled and layer.policy == "W8A8":
            can_quantize = (
                args.mode == "quantize"
                and tensor.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
                and np_data.ndim == 2
            )
            if can_quantize:
                if layer.act_quant_mode == "symmetric_u8" and layer.act_zero_point != 128:
                    raise ValueError(
                        f"{name}: symmetric_u8 requires act_zero_point=128, got {layer.act_zero_point}"
                    )
                if args.weight_granularity == "per_tensor":
                    q, scale = quantize_per_tensor_i8(np_data)
                else:
                    q, scale = quantize_per_channel_i8(np_data)
                sum_w = maybe_build_sum_w(q, layer.act_quant_mode, layer.act_zero_point)
                out_data = np.ascontiguousarray(q)
                out_raw_dtype = None
                resolved_policy = "W8A8"
                replaced_tensors += 1
            else:
                resolved_policy = "F16_FALLBACK"

        out_data = np.ascontiguousarray(out_data)
        writer.add_tensor_info(name, out_data.shape, out_data.dtype, out_data.nbytes, raw_dtype=out_raw_dtype)
        staged.append(out_data)

        if layer is not None:
            i = len(quant_params["layers"])
            prefix = f"aicas.w8a8.tensor.{i}."
            writer.add_string(prefix + "name", layer.tensor_name)
            writer.add_bool(prefix + "enabled", layer.enabled)
            writer.add_string(prefix + "policy", resolved_policy)
            if resolved_policy == "W8A8":
                act_scale_q8_24 = float_to_q8_24(layer.act_scale)
                dequant_scale_q8_24 = array_to_q8_24(scale * np.float32(layer.act_scale))
                writer.add_float32(prefix + "act_scale", layer.act_scale)
                writer.add_int32(prefix + "act_scale_q8_24", act_scale_q8_24)
                writer.add_int32(prefix + "act_zero_point", layer.act_zero_point)
                writer.add_string(prefix + "act_quant_mode", layer.act_quant_mode)
                writer.add_string(prefix + "weight_scale_mode", args.weight_granularity)
                writer.add_array(prefix + "weight_scale", scale.tolist())
                writer.add_array(prefix + "dequant_scale_q8_24", dequant_scale_q8_24.tolist())
                if sum_w is not None:
                    writer.add_array(prefix + "sum_w", sum_w.tolist())

            quant_params["layers"].append(
                {
                    "tensor_name": layer.tensor_name,
                    "enabled": layer.enabled,
                    "policy": resolved_policy,
                    "act_scale": layer.act_scale,
                    "act_scale_q8_24": float_to_q8_24(layer.act_scale) if scale is not None else 0,
                    "act_zero_point": layer.act_zero_point,
                    "act_quant_mode": layer.act_quant_mode,
                    "weight_scale_mode": args.weight_granularity if scale is not None else "",
                    "weight_scale_len": int(len(scale)) if scale is not None else 0,
                    "dequant_scale_q8_24_len": int(len(scale)) if scale is not None else 0,
                    "sum_w_len": int(len(sum_w)) if sum_w is not None else 0,
                    "sum_w_mode": "omitted" if sum_w is None and scale is not None else "per_output_channel",
                }
            )

    # Write file.
    Path(args.output_gguf).parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in staged:
        writer.write_tensor_data(data)
    writer.close()

    Path(args.output_quant_params).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_quant_params, "w", encoding="utf-8") as f:
        json.dump(quant_params, f, indent=2, ensure_ascii=False)

    print(f"Wrote GGUF: {args.output_gguf}")
    print(f"Wrote quant params: {args.output_quant_params}")
    print(f"Replaced tensors with I8: {replaced_tensors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
