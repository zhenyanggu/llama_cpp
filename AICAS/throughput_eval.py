import os
import sys
import argparse
import json

from llama_server_client import (
    DEFAULT_BASE_URL,
    chat_completion,
    extract_text_content,
    image_to_data_url,
)

LONG_PROMPT = """
Please analyze this image in detail.
1.  First, please perform a full OCR (Optical Character Recognition), extract all visible text
    in the image, and list it in order from top-to-bottom, left-to-right.
2.  Second, please describe the main visual elements in the image, including but not limited to
    objects, people, scenery, and atmosphere.
3.  Finally, based on the extracted text and visual elements, summarize the theme
    and possible context of this image.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run a non-streaming test against a llama-server.")
    parser.add_argument(
        "-i", "--image",
        help="Path to the input image file.",
        default="image.png"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output metrics JSON file.",
        default="throughput_metrics.json"
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for llama-server's OpenAI-compatible API.",
        default=DEFAULT_BASE_URL
    )
    parser.add_argument(
        "--model",
        help="Model alias exposed by llama-server.",
        default="local-model"
    )
    parser.add_argument(
        "--request-timeout",
        help="HTTP timeout in seconds.",
        type=float,
        default=300.0
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path not found: {args.image}")
        sys.exit(1)

    try:
        messages_payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(args.image)
                        }
                    },
                    {
                        "type": "text",
                        "text": LONG_PROMPT
                    }
                ]
            }
        ]

        response = chat_completion(
            base_url=args.base_url,
            model=args.model,
            messages=messages_payload,
            max_tokens=4096,
            temperature=0.0,
            stream=False,
            timeout=args.request_timeout,
        )

        full_response = extract_text_content(response)
        print(full_response)
        print("\n" + "--- Generation Finished ---")

        print("\n--- Performance Metrics (from llama-server) ---")

        timings = response.get("timings", {})
        usage = response.get("usage", {})

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        decode_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
        prefill_speed = float(timings.get("prompt_per_second", 0.0) or 0.0)
        decode_speed = float(timings.get("predicted_per_second", 0.0) or 0.0)

        print(f"[Token Stats]")
        print(f"  Prompt Tokens:     {prompt_tokens} tokens")
        print(f"  Completion Tokens: {completion_tokens} tokens")
        print(f"  Total Tokens:      {total_tokens} tokens")

        print(f"\n[Server-Side Timing (ms)]")
        print(f"  Prefill Time: {prompt_ms:.2f} ms")
        print(f"  Decode Time:  {decode_ms:.2f} ms")
        print(f"  Total Time (Server): {(prompt_ms + decode_ms):.2f} ms")

        print(f"\n[Speed (Tokens/sec)]")
        print(f"  Prefill Speed:  {prefill_speed:.2f} t/s")
        print(f"  Decode Speed:   {decode_speed:.2f} t/s")

        metrics_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_ms": prompt_ms,
            "decode_ms": decode_ms,
            "total_ms": prompt_ms + decode_ms,
            "prefill_speed_tps": prefill_speed,
            "decode_speed_tps": decode_speed
        }

        output_json_path = args.output

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=4)
            print(f"\nSuccessfully saved metrics to: {output_json_path}")
        except IOError as e:
            print(f"\nError: Failed to write metrics file: {e}")

    except Exception as e:
        print(f"\n\n--- AN ERROR OCCURRED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    main()
