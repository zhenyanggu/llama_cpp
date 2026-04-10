#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a mixed W8A8/F16 layer policy from calibrated quant params.")
    p.add_argument("--input-quant-params", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fallback-kind", action="append", default=[], help="Fallback all layers whose inferred kind matches this value, e.g. ffn_up, attn_out, attn_q")
    p.add_argument("--fallback-tensor", action="append", default=[], help="Fallback a specific tensor name")
    p.add_argument("--fallback-regex", action="append", default=[], help="Fallback tensors whose names match this regex")
    return p.parse_args()


def infer_kind(tensor_name: str) -> str:
    if tensor_name == "mm.model.fc.weight":
        return "projector"
    parts = tensor_name.split('.')
    if len(parts) >= 5:
        token = parts[3]
        if token == 'attn':
            return parts[4].replace('.weight', '')
        if token == 'ffn':
            return parts[4].replace('.weight', '')
    if '.attn_q.' in tensor_name:
        return 'attn_q'
    if '.attn_k.' in tensor_name:
        return 'attn_k'
    if '.attn_v.' in tensor_name:
        return 'attn_v'
    if '.attn_out.' in tensor_name:
        return 'attn_out'
    if '.ffn_up.' in tensor_name:
        return 'ffn_up'
    if '.ffn_down.' in tensor_name:
        return 'ffn_down'
    return tensor_name


def main() -> int:
    args = parse_args()
    with open(args.input_quant_params, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    regexes = [re.compile(x) for x in args.fallback_regex]
    tensor_set = set(args.fallback_tensor)
    kind_set = set(args.fallback_kind)

    changed = []
    for layer in doc.get('layers', []):
        name = layer['tensor_name']
        kind = infer_kind(name)
        hit = name in tensor_set or kind in kind_set or any(r.search(name) for r in regexes)
        if hit:
            layer['enabled'] = False
            layer['policy'] = 'F16_FALLBACK'
            changed.append((name, kind))
        else:
            layer['enabled'] = True
            layer['policy'] = 'W8A8'

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f'Wrote layer policy: {out}')
    print(f'Fallback layers: {len(changed)}')
    for name, kind in changed[:40]:
        print(f'{kind:<12} {name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
