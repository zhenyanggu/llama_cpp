import os
import sys
import base64
import argparse
import json
from openai import OpenAI

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

# llama-server API address
SERVER_URL = "http://127.0.0.1:8080/v1"

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