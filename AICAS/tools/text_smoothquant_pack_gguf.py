#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf  # type: ignore


@dataclass
class LayerPolicy:
    tensor_name: str
    enabled: bool
    policy: str
    reason: str
    alpha: float
    eps: float
    smooth_scale: np.ndarray
    act_scale: float
    act_zero_point: int
    act_quant_mode: str
    clip_mode: str
    quant_tensor_name: str
    stat_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack text-model SmoothQuant GGUF.")
    parser.add_argument("--input-gguf", required=True)
    parser.add_argument("--source-weights-gguf", default="")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-gguf", required=True)
    parser.add_argument("--output-summary", default="AICAS/artifacts/text_sq_pack_summary.json")
    parser.add_argument("--weight-granularity", choices=["per_channel", "per_tensor"], default="per_channel")
    return parser.parse_args()


def load_policy(path: str) -> list[LayerPolicy]:
    doc = json.load(open(path, "r", encoding="utf-8"))
    layers = []
    for item in doc["layers"]:
        layers.append(
            LayerPolicy(
                tensor_name=item["tensor_name"],
                enabled=bool(item["enabled"]),
                policy=str(item["policy"]),
                reason=str(item.get("reason", "")),
                alpha=float(item["alpha"]),
                eps=float(item["eps"]),
                smooth_scale=np.asarray(item["smooth_scale"], dtype=np.float32),
                act_scale=float(item["act_scale"]),
                act_zero_point=int(item["act_zero_point"]),
                act_quant_mode=str(item["act_quant_mode"]),
                clip_mode=str(item["clip_mode"]),
                quant_tensor_name=str(item.get("quant_tensor_name", item["tensor_name"] + ".aicas_i8")),
                stat_source=str(item.get("stat_source", "observed")),
            )
        )
    return layers


def quantize_per_channel(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(weight), axis=1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(weight / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    return q, scales


def quantize_per_tensor(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = float(np.max(np.abs(weight)))
    scale = np.float32(max_abs / 127.0) if max_abs > 0.0 else np.float32(1.0)
    q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    return q, np.asarray([scale], dtype=np.float32)


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        if field.name.startswith("aicas.text_sq."):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)


def main() -> int:
    args = parse_args()
    layers = load_policy(args.policy)
    reader = gguf.GGUFReader(args.input_gguf, "r")
    source_reader = gguf.GGUFReader(args.source_weights_gguf, "r") if args.source_weights_gguf else reader
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output_gguf, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    copy_metadata(reader, writer)

    writer.add_string("aicas.text_sq.schema", "llama_text_smoothquant_v1")
    writer.add_int32("aicas.text_sq.tensor_count", len(layers))
    writer.add_string("aicas.text_sq.weight_granularity", args.weight_granularity)

    policy_map = {layer.tensor_name: layer for layer in layers}
    source_tensor_map = {tensor.name: tensor for tensor in source_reader.tensors}
    staged = []
    summary = {
        "schema": "aicas.llama.text_sq_pack_summary.v1",
        "input_gguf": os.path.abspath(args.input_gguf),
        "source_weights_gguf": os.path.abspath(args.source_weights_gguf) if args.source_weights_gguf else os.path.abspath(args.input_gguf),
        "policy": os.path.abspath(args.policy),
        "weight_granularity": args.weight_granularity,
        "layers": [],
    }

    for tensor in reader.tensors:
        arr = np.ascontiguousarray(np.asarray(tensor.data))
        writer.add_tensor_info(tensor.name, arr.shape, arr.dtype, arr.nbytes, raw_dtype=tensor.tensor_type)
        staged.append(arr)

    for index, layer in enumerate(layers):
        prefix = f"aicas.text_sq.tensor.{index}."
        writer.add_string(prefix + "name", layer.tensor_name)
        writer.add_bool(prefix + "enabled", layer.enabled)
        writer.add_string(prefix + "policy", layer.policy)
        writer.add_string(prefix + "reason", layer.reason)
        writer.add_float32(prefix + "act_scale", layer.act_scale)
        writer.add_int32(prefix + "act_zero_point", layer.act_zero_point)
        writer.add_string(prefix + "act_quant_mode", layer.act_quant_mode)
        writer.add_string(prefix + "weight_scale_mode", args.weight_granularity)
        writer.add_float32(prefix + "smooth_alpha", layer.alpha)
        writer.add_float32(prefix + "smooth_eps", layer.eps)
        writer.add_array(prefix + "smooth_scale", layer.smooth_scale.astype(np.float32).tolist())
        writer.add_string(prefix + "quant_tensor_name", layer.quant_tensor_name)

        entry = {
            "tensor_name": layer.tensor_name,
            "enabled": layer.enabled,
            "policy": layer.policy,
            "reason": layer.reason,
            "clip_mode": layer.clip_mode,
            "stat_source": layer.stat_source,
            "weight_scale_len": 0,
            "sum_w_len": 0,
            "quant_tensor_name": layer.quant_tensor_name,
        }

        if layer.enabled and layer.policy == "W8A8":
            if layer.tensor_name not in source_tensor_map:
                raise RuntimeError(f"missing source tensor for {layer.tensor_name}")
            source = np.asarray(source_tensor_map[layer.tensor_name].data, dtype=np.float32)
            weight = np.asarray(source * layer.smooth_scale[np.newaxis, :], dtype=np.float32)
            if args.weight_granularity == "per_tensor":
                q, scales = quantize_per_tensor(weight)
            else:
                q, scales = quantize_per_channel(weight)
            sum_w = q.astype(np.int32).sum(axis=1).astype(np.int32)

            writer.add_array(prefix + "weight_scale", scales.tolist())
            writer.add_array(prefix + "sum_w", sum_w.tolist())
            writer.add_tensor_info(layer.quant_tensor_name, q.shape, q.dtype, q.nbytes, raw_dtype=None)
            staged.append(np.ascontiguousarray(q))

            entry["weight_scale_len"] = int(len(scales))
            entry["sum_w_len"] = int(len(sum_w))

        summary["layers"].append(entry)

    Path(args.output_gguf).parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in staged:
        writer.write_tensor_data(data)
    writer.close()

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(output_summary, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote text SmoothQuant GGUF: {args.output_gguf}")
    print(f"Wrote pack summary: {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
