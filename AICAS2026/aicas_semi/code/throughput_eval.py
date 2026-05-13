import argparse
import os
import sys
import json

from llama_server_client import (
    DEFAULT_BASE_URL,
    chat_completion,
    extract_text_content,
    image_to_data_url,
)

LONG_PROMPT = """
Act as an interdisciplinary expert combining the skills of a master art historian, a rigorous forensic image analyst, and a computational aesthetician. I am presenting you with a landscape painting. To thoroughly evaluate the visual data, you must execute a comprehensive, multi-layered analysis. Do not hallucinate details, but extract every possible piece of data from the image provided. You must not skip any section.
Phase 1: Granular Visual Deconstruction
Spatial Mapping: Describe the foreground, midground, and background in exhaustive detail. Establish the exact spatial relationships and distances between the primary topographical features (e.g., mountains, bodies of water, vegetation, architecture). If there are any human, animal, or dynamic figures, pinpoint their exact relative locations and describe their scale relative to the environment.
Chromatic and Luminance Analysis: Identify the dominant color palette. Break down the specific hues, saturation levels, and contrast ratios. Analyze the primary and secondary light sources—are they diffuse, directional, natural, or artificial? Describe exactly how the lighting creates atmospheric perspective and volumetric shadows across the terrain.
Textural and Topographical Fidelity: Evaluate the simulated texture of the physical elements. How does the artist visually differentiate the texture of dense foliage from jagged rock, or flowing water from an overcast sky? Describe the structural brushwork or stylistic rendering techniques used to build these surfaces.
Phase 2: Semantic and Thematic Extractio
Meteorological and Temporal Inference: Based strictly on the lighting angles, cloud formations, atmospheric haze, and shadow lengths, deduce the specific time of day, the season, and the weather conditions at the moment depicted. Justify your deductions with direct visual evidence from the painting.
Compositional Architecture: Analyze the geometric structure of the painting. Identify any leading lines, framing devices, or adherence to the rule of thirds. How do these compositional choices guide the viewer's eye through the landscape?
Ensure your total response is expansive, rigorously detailed, logically sequenced, and leaves no visual element unexamined.
"""

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a non-streaming test against a llama-server.")
    parser.add_argument(
        "-i", "--image",
        help="Path to the input image file.",
        default="test2.jpg"
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
    parser.add_argument(
        "--max-tokens",
        help="Maximum generated tokens for this throughput request.",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text for this throughput request.",
        default=LONG_PROMPT
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if image exists
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
                        "text": args.prompt
                    }
                ]
            }
        ]

        # Execute the blocking (non-streaming) call
        response = chat_completion(
            base_url=args.base_url,
            model=args.model,
            messages=messages_payload,
            max_tokens=args.max_tokens,
            temperature=0.0,
            stream=False,
            timeout=args.request_timeout,
        )
        
        # Print the full response content
        full_response = extract_text_content(response)
        print(full_response)
        print("\n" + "--- Generation Finished ---")
        
        # Parse and print metrics from the response object
        print("\n--- Performance Metrics (from llama-server) ---")

        timings = response.get("timings", {})
        usage = response.get("usage", {})

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        print(f"[Token Stats]")
        print(f"  Prompt Tokens:     {prompt_tokens} tokens")
        print(f"  Completion Tokens: {completion_tokens} tokens")
        print(f"  Total Tokens:      {total_tokens} tokens")
        
        print(f"\n[Server-Side Timing (ms)]")
        prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        decode_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
        print(f"  Prefill Time: {prompt_ms:.2f} ms")
        print(f"  Decode Time:  {decode_ms:.2f} ms")
        print(f"  Total Time (Server): {(prompt_ms + decode_ms):.2f} ms")

        prefill_speed = float(timings.get("prompt_per_second", 0.0) or 0.0)
        decode_speed = float(timings.get("predicted_per_second", 0.0) or 0.0)

        print(f"\n[Speed (Tokens/sec)]")
        print(f"  Prefill Speed:  {prefill_speed:.2f} t/s")
        print(f"  Decode Speed:   {decode_speed:.2f} t/s")

        # --- Save metrics to JSON ---
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
        sys.exit(1)

if __name__ == "__main__":
    main()
