import os
import sys
import base64
import argparse
import json
from openai import OpenAI

LONG_PROMPT = """
Please analyze this image in detail.
1.  First, please perform a full OCR (Optical Character Recognition), extract all visible text
    in the image, and list it in order from top-to-bottom, left-to-right.
2.  Second, please describe the main visual elements in the image, including but not limited to
    objects, people, scenery, and atmosphere.
3.  Finally, based on the extracted text and visual elements, summarize the theme
    and possible context of this image.
"""

# llama-server API address
SERVER_URL = "http://127.0.0.1:8080/v1"

def parse_args():
    """Parse command-line arguments."""
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
    return parser.parse_args()


def image_to_base64(image_path):
    """Helper function: encode an image file as a Base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main():
    args = parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image path not found: {args.image}")
        sys.exit(1)

    client = OpenAI(
        base_url=SERVER_URL,
        api_key="NA"
    )

    try:
        img_b64 = image_to_base64(args.image)
        messages_payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": LONG_PROMPT
                    }
                ]
            }
        ]

        # Execute the blocking (non-streaming) call
        response = client.chat.completions.create(
            model="local-model",
            messages=messages_payload,
            max_tokens=4096,
            temperature=0.0,
            stream=False
        )
        
        # Print the full response content
        full_response = response.choices[0].message.content
        print(full_response)
        print("\n" + "--- Generation Finished ---")
        
        # Parse and print metrics from the response object
        print("\n--- Performance Metrics (from llama-server) ---")

        timings = response.timings
        usage = response.usage

        print(f"[Token Stats]")
        print(f"  Prompt Tokens:     {usage.prompt_tokens} tokens")
        print(f"  Completion Tokens: {usage.completion_tokens} tokens")
        print(f"  Total Tokens:      {usage.total_tokens} tokens")
        
        print(f"\n[Server-Side Timing (ms)]")
        print(f"  Prefill Time: {timings['prompt_ms']:.2f} ms")
        print(f"  Decode Time:  {timings['predicted_ms']:.2f} ms")
        print(f"  Total Time (Server): {(timings['prompt_ms'] + timings['predicted_ms']):.2f} ms")

        prefill_speed = timings['prompt_per_second']
        decode_speed = timings['predicted_per_second']

        print(f"\n[Speed (Tokens/sec)]")
        print(f"  Prefill Speed:  {prefill_speed:.2f} t/s")
        print(f"  Decode Speed:   {decode_speed:.2f} t/s")

        # --- Save metrics to JSON ---
        metrics_data = {
            "prefill_speed_tps": prefill_speed,
            "decode_speed_tps": decode_speed
        }
        
        # Use the output path from command-line arguments
        output_json_path = args.output

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=4)
            print(f"\nSuccessfully saved metrics to: {output_json_path}")
        except IOError as e:
            print(f"\nError: Failed to write metrics file: {e}")
        # --- End of JSON saving ---

    except Exception as e:
        print(f"\n\n--- AN ERROR OCCURRED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    main()