import os
import sys
import time
import json
import argparse

from llama_server_client import (
    DEFAULT_BASE_URL,
    chat_completion,
    image_to_data_url,
)

try:
    from PIL import Image
except Exception:
    Image = None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_images(image_configs, workdir):
    for cfg in image_configs:
        source_path = os.path.join(workdir, cfg['source'])

        if 'resize' in cfg:
            if Image is None:
                raise RuntimeError("Pillow is required for resize cases in ttft_config.json")
            resize = cfg['resize']
            img = Image.open(source_path)
            img = img.resize((resize['width'], resize['height']), Image.LANCZOS)
            output_name = cfg.get('output', f"{cfg['name']}.jpg")
            output_path = os.path.join(workdir, output_name)
            img.save(output_path, 'JPEG')
            cfg['_path'] = output_path
            print(f"[Image] {cfg['name']}: resized to {resize['width']}x{resize['height']} -> {output_name}")
        else:
            cfg['_path'] = source_path
            print(f"[Image] {cfg['name']}: using original -> {cfg['source']}")

    return image_configs


def measure_ttft(base_url, model, messages_payload, request_timeout):
    t0 = time.perf_counter()
    ttft_ms = None
    try:
        stream = chat_completion(
            base_url=base_url,
            model=model,
            messages=messages_payload,
            max_tokens=1,
            temperature=0.0,
            stream=True,
            timeout=request_timeout,
        )
        for chunk in stream:
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                t1 = time.perf_counter()
                ttft_ms = (t1 - t0) * 1000.0
                break
    finally:
        pass

    return ttft_ms


def run_single_case(base_url, image_cfg, prompt_cfg, model, request_timeout):
    def build_payload():
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[Nonce: {time.time()}]\n"},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_cfg['_path'])}},
                    {"type": "text", "text": prompt_cfg['text']}
                ]
            }
        ]

    ttft_ms = measure_ttft(base_url, model, build_payload(), request_timeout)
    if ttft_ms is None:
        print(f"          no content received from server")
        return None

    result = {
        "image": image_cfg['name'],
        "prompt": prompt_cfg['name'],
        "prompt_chars": len(prompt_cfg['text']),
        "ttft_ms": ttft_ms
    }
    return result


def linear_fit(points):
    """Fit y = slope * x + intercept using ordinary least squares.

    Returns (slope, intercept) or (None, None) when the fit is undefined
    (fewer than 2 points or zero variance in x).
    """
    n = len(points)
    if n < 2:
        return None, None
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    mean_x = sum_x / n
    mean_y = sum_y / n
    cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    var = sum((p[0] - mean_x) ** 2 for p in points)
    if var == 0:
        return None, None
    slope = cov / var
    intercept = mean_y - slope * mean_x
    return slope, intercept


def aggregate_fits(cases):
    """Group cases by image, fit a line per image, then average across images.

    Returns (per_image_fits, aggregate_fit). Each per-image fit carries the
    raw points used so the committee can reproduce or audit the fit.
    """
    by_image = {}
    for case in cases:
        by_image.setdefault(case['image'], []).append(
            (case['prompt_chars'], case['ttft_ms'])
        )

    per_image = []
    valid_slopes = []
    valid_intercepts = []
    for img_name, points in by_image.items():
        # Sort by prompt length so the points list reads naturally.
        points = sorted(points, key=lambda p: p[0])
        slope, intercept = linear_fit(points)
        per_image.append({
            "image": img_name,
            "num_points": len(points),
            "slope_ms_per_char": slope,
            "intercept_ms": intercept,
            "points": [
                {"prompt_chars": p[0], "ttft_ms": p[1]} for p in points
            ]
        })
        if slope is not None and intercept is not None:
            valid_slopes.append(slope)
            valid_intercepts.append(intercept)

    aggregate = None
    if valid_slopes:
        aggregate = {
            "slope_ms_per_char": sum(valid_slopes) / len(valid_slopes),
            "intercept_ms": sum(valid_intercepts) / len(valid_intercepts),
            "num_images_aggregated": len(valid_slopes)
        }

    return per_image, aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Multi-image, multi-prompt TTFT evaluation."
    )
    parser.add_argument(
        '-c', '--config',
        default='ttft_config.json',
        help='Path to the JSON config file (default: ttft_config.json)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Override output path for results JSON'
    )
    parser.add_argument(
        '--request-timeout',
        type=float,
        default=3600.0,
        help='HTTP timeout in seconds for each streaming TTFT request'
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    config = load_json(args.config)
    workdir = os.path.dirname(os.path.abspath(args.config))

    server_url = config.get('server_url', DEFAULT_BASE_URL)
    model = config.get('model', 'local-model')
    output_path = args.output or config.get('output', 'ttft_eval_results.json')

    image_configs = config.get('images', [])
    prompt_configs = config.get('prompts', [])

    if not image_configs:
        print("Error: no images defined in config")
        sys.exit(1)
    if not prompt_configs:
        print("Error: no prompts defined in config")
        sys.exit(1)

    total_cases = len(image_configs) * len(prompt_configs)

    print("=" * 60)
    print("  TTFT Multi-Prompt Evaluation")
    print("=" * 60)
    print(f"  Config:    {args.config}")
    print(f"  Server:    {server_url}")
    print(f"  Model:     {model}")
    print(f"  Images:    {len(image_configs)}")
    print(f"  Prompts:   {len(prompt_configs)}")
    print(f"  Cases:     {total_cases}")
    print(f"  Output:    {output_path}")
    print("=" * 60)

    image_configs = prepare_images(image_configs, workdir)

    all_cases = []
    case_index = 0
    for img_cfg in image_configs:
        for prompt_cfg in prompt_configs:
            case_index += 1
            case_label = f"[{img_cfg['name']}] x [{prompt_cfg['name']}]"
            print(f"\n  Case {case_index}/{total_cases}: {case_label}")
            print(f"        Prompt chars: {len(prompt_cfg['text'])}")
            print("-" * 40)

            result = run_single_case(server_url, img_cfg, prompt_cfg, model, args.request_timeout)

            if result is None:
                print(f"    SKIPPED (no valid sample)")
                continue

            all_cases.append(result)

            print(f"    TTFT = {result['ttft_ms']:.2f} ms")

    if not all_cases:
        print("\nError: no valid cases collected")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for case in all_cases:
        print(f"  [{case['image']:>5}] x [{case['prompt']:>5}]  "
              f"chars={case['prompt_chars']:>5}  "
              f"ttft={case['ttft_ms']:>8.2f} ms")

    per_image_fits, aggregate_fit = aggregate_fits(all_cases)

    print("\n" + "=" * 60)
    print("  Linear Fit (TTFT = slope * prompt_chars + intercept)")
    print("=" * 60)
    for fit in per_image_fits:
        if fit['slope_ms_per_char'] is not None:
            print(f"  [{fit['image']:>5}]  num_points={fit['num_points']}  "
                  f"slope={fit['slope_ms_per_char']:.4f} ms/char  "
                  f"intercept={fit['intercept_ms']:.2f} ms")
        else:
            print(f"  [{fit['image']:>5}]  num_points={fit['num_points']}  fit undefined (<2 points or zero variance)")
    if aggregate_fit:
        print(f"  [aggregate]  slope={aggregate_fit['slope_ms_per_char']:.4f} ms/char  "
              f"intercept={aggregate_fit['intercept_ms']:.2f} ms  "
              f"(averaged over {aggregate_fit['num_images_aggregated']} images)")
    else:
        print("  [aggregate]  no valid per-image fit available")

    output_data = {
        "config_summary": {
            "server_url": server_url,
            "model": model,
            "num_images": len(image_configs),
            "num_prompts": len(prompt_configs),
            "total_cases": total_cases
        },
        "cases": all_cases,
        "linear_fit": {
            "per_image": per_image_fits,
            "aggregate": aggregate_fit
        }
    }

    output_abs = os.path.join(workdir, os.path.basename(output_path)) if not os.path.isabs(output_path) else output_path
    save_json(output_data, output_abs)
    print(f"\nResults saved to: {output_abs}")


if __name__ == "__main__":
    main()
