#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def load_json(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--throughput', type=Path, required=True)
    parser.add_argument('--acc-json', type=Path, required=True)
    parser.add_argument('--run-meta', type=Path, required=False)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-zh', type=Path, required=True)
    parser.add_argument('--output-en', type=Path, required=True)
    args = parser.parse_args()

    throughput = load_json(args.throughput)
    acc_items = load_json(args.acc_json)
    run_meta = load_json(args.run_meta) if args.run_meta else {}
    acc_total = len(acc_items)
    acc_correct = sum(1 for item in acc_items if int(item.get('result', 0)) == 1)
    acc_ratio = (acc_correct / acc_total) if acc_total else 0.0
    summary = {
        'model': run_meta.get('model'),
        'mmproj': run_meta.get('mmproj'),
        'prompt_tokens': int(throughput.get('prompt_tokens', 0) or 0),
        'completion_tokens': int(throughput.get('completion_tokens', 0) or 0),
        'total_tokens': int(throughput.get('total_tokens', 0) or 0),
        'prompt_ms': float(throughput.get('prompt_ms', 0.0) or 0.0),
        'decode_ms': float(throughput.get('decode_ms', 0.0) or 0.0),
        'total_ms': float(throughput.get('total_ms', 0.0) or 0.0),
        'prefill_tps': float(throughput.get('prefill_speed_tps', 0.0) or 0.0),
        'decode_tps': float(throughput.get('decode_speed_tps', 0.0) or 0.0),
        'acc_correct': acc_correct,
        'acc_total': acc_total,
        'acc_ratio': acc_ratio,
        'run_id': run_meta.get('run_id'),
        'remote_run_dir': run_meta.get('remote_run_dir'),
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    zh = f'''# 官方结果\n\n- Run ID: `{summary['run_id']}`\n- Model: `{summary['model']}`\n- mmproj: `{summary['mmproj']}`\n\n## 吞吐\n\n- Prompt Tokens: `{summary['prompt_tokens']}`\n- Completion Tokens: `{summary['completion_tokens']}`\n- Total Tokens: `{summary['total_tokens']}`\n- Prompt Time: `{summary['prompt_ms']:.3f} ms`\n- Decode Time: `{summary['decode_ms']:.3f} ms`\n- Total Time: `{summary['total_ms']:.3f} ms`\n- Prefill Speed: `{summary['prefill_tps']:.6f} t/s`\n- Decode Speed: `{summary['decode_tps']:.6f} t/s`\n\n## 正确率\n\n- Correct: `{summary['acc_correct']}`\n- Total: `{summary['acc_total']}`\n- Ratio: `{summary['acc_ratio']:.6f}`\n'''
    en = f'''# Official Results\n\n- Run ID: `{summary['run_id']}`\n- Model: `{summary['model']}`\n- mmproj: `{summary['mmproj']}`\n\n## Throughput\n\n- Prompt Tokens: `{summary['prompt_tokens']}`\n- Completion Tokens: `{summary['completion_tokens']}`\n- Total Tokens: `{summary['total_tokens']}`\n- Prompt Time: `{summary['prompt_ms']:.3f} ms`\n- Decode Time: `{summary['decode_ms']:.3f} ms`\n- Total Time: `{summary['total_ms']:.3f} ms`\n- Prefill Speed: `{summary['prefill_tps']:.6f} t/s`\n- Decode Speed: `{summary['decode_tps']:.6f} t/s`\n\n## Accuracy\n\n- Correct: `{summary['acc_correct']}`\n- Total: `{summary['acc_total']}`\n- Ratio: `{summary['acc_ratio']:.6f}`\n'''
    args.output_zh.write_text(zh, encoding='utf-8')
    args.output_en.write_text(en, encoding='utf-8')

if __name__ == '__main__':
    main()
