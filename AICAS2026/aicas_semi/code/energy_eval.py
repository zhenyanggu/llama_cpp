import os
import sys
import time
import json
import base64
import argparse
import threading
from glob import glob
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

# Default sysfs hwmon power input file. KV260 PYNQ images expose on-board
# INA226 sensors through hwmon, where the power readings are typically
# reported in microwatts (uW). The exact hwmonN index can vary across images;
# override with --power_path if needed (use --list_hwmon to enumerate).
DEFAULT_POWER_PATH = "/sys/class/hwmon/hwmon2/power1_input"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Measure end-to-end energy efficiency (Tokens/Joule) on KV260.")
    parser.add_argument(
        "-i", "--image",
        help="Path to the input image file.",
        default="test2.jpg"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output metrics JSON file.",
        default="energy_metrics.json"
    )
    parser.add_argument(
        "--sample_hz",
        type=float,
        default=100.0,
        help="Power sampling frequency in Hz (default: 100)."
    )
    parser.add_argument(
        "--power_path",
        default=DEFAULT_POWER_PATH,
        help=f"Sysfs hwmon power input file (default: {DEFAULT_POWER_PATH})."
    )
    parser.add_argument(
        "--power_unit",
        choices=["uW", "mW", "W"],
        default="uW",
        help="Unit reported by the hwmon file (default: uW)."
    )
    parser.add_argument(
        "--list_hwmon",
        action="store_true",
        help="List available hwmon devices and exit."
    )
    return parser.parse_args()


def image_to_base64(image_path):
    """Helper function: encode an image file as a Base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


class PowerSampler(threading.Thread):
    """Background thread that samples instantaneous board power from sysfs hwmon."""

    def __init__(self, power_path, sample_hz, scale):
        super().__init__(daemon=True)
        self.power_path = power_path
        self.interval = 1.0 / sample_hz
        # Multiplier that converts the raw reading to Watts.
        self.scale = scale
        # List of (timestamp_s, power_w) tuples populated during the run.
        self.samples = []
        self._stop_event = threading.Event()

    def run(self):
        next_t = time.perf_counter()
        # Open the sysfs file once and reuse the descriptor for every
        # sample. Linux hwmon regenerates the reading on each read() from
        # offset 0, so seek(0) + read() returns the latest sensor value
        # without the per-iteration open/close syscall overhead.
        try:
            f = open(self.power_path, 'r')
        except Exception:
            # Cannot open the power path at all; the thread exits silently
            # and main() will see an empty samples list.
            return
        try:
            while not self._stop_event.is_set():
                try:
                    f.seek(0)
                    raw = float(f.read().strip())
                    power_w = raw * self.scale
                    t = time.perf_counter()
                    self.samples.append((t, power_w))
                except Exception as e:
                    # Skip the sample on any read error; the thread keeps running.
                    import sys
                    print(f"Sampling error: {e}", file=sys.stderr)
                    pass
                next_t += self.interval
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_t = time.perf_counter()
        finally:
            f.close()

    def stop(self):
        self._stop_event.set()


def trapezoid_integral(samples):
    """Trapezoidal integration of (timestamp_s, value) pairs."""
    if len(samples) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(samples)):
        t0, v0 = samples[i - 1]
        t1, v1 = samples[i]
        total += 0.5 * (v0 + v1) * (t1 - t0)
    return total


def list_hwmon_devices():
    """Print available hwmon devices and their power_*_input files."""
    print("Available hwmon devices:")
    for hwmon_dir in sorted(glob("/sys/class/hwmon/hwmon*/")):
        name_path = os.path.join(hwmon_dir, "name")
        try:
            with open(name_path, 'r') as f:
                name = f.read().strip()
        except Exception:
            name = "?"
        print(f"  {hwmon_dir}  (name: {name})")
        for power_file in sorted(glob(os.path.join(hwmon_dir, "power*_input"))):
            try:
                with open(power_file, 'r') as f:
                    val = f.read().strip()
            except Exception:
                val = "?"
            print(f"    {power_file}  -> {val}")


def main():
    args = parse_args()

    if args.list_hwmon:
        list_hwmon_devices()
        return

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image path not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.power_path):
        print(f"Error: Power input file not found: {args.power_path}")
        print("Use --list_hwmon to enumerate available hwmon devices.")
        sys.exit(1)

    unit_scale = {"uW": 1e-6, "mW": 1e-3, "W": 1.0}[args.power_unit]

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

        sampler = PowerSampler(args.power_path, args.sample_hz, unit_scale)
        sampler.start()
        # Allow the sampler thread to settle before triggering inference.
        time.sleep(0.5)

        print("Triggering inference...")
        t_start = time.perf_counter()
        response = client.chat.completions.create(
            model="local-model",
            messages=messages_payload,
            max_tokens=4096,
            temperature=0.0,
            stream=False
        )
        t_end = time.perf_counter()

        sampler.stop()
        sampler.join()

        # Trim recorded samples to the inference window.
        inference_samples = [(t, p) for (t, p) in sampler.samples if t_start <= t <= t_end]

        if len(inference_samples) < 2:
            print("Error: Insufficient power samples collected during the inference window.")
            print(f"  Inference window: {t_end - t_start:.3f}s, samples: {len(inference_samples)}")
            print("  Consider raising --sample_hz, or check that --power_path is readable.")
            sys.exit(1)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        inference_duration_s = t_end - t_start
        energy_j = trapezoid_integral(inference_samples)
        avg_power_w = energy_j / inference_duration_s if inference_duration_s > 0 else 0.0
        # Tokens/Joule based on completion (output) tokens, the standard
        # definition for end-side inference energy efficiency.
        tokens_per_joule = completion_tokens / energy_j if energy_j > 0 else 0.0

        print("\n--- Energy Efficiency Results ---")
        print(f"  Inference Duration:  {inference_duration_s:.3f} s")
        print(f"  Power Samples:       {len(inference_samples)}")
        print(f"  Average Power:       {avg_power_w:.3f} W")
        print(f"  Total Energy:        {energy_j:.3f} J")
        print(f"  Prompt Tokens:       {prompt_tokens}")
        print(f"  Completion Tokens:   {completion_tokens}")
        print(f"  Total Tokens:        {total_tokens}")
        print(f"  Energy Efficiency:   {tokens_per_joule:.4f} tokens/J")

        # --- Save metrics to JSON ---
        metrics_data = {
            "inference_duration_s": inference_duration_s,
            "average_power_w": avg_power_w,
            "total_energy_j": energy_j,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_joule": tokens_per_joule,
            "sample_hz": args.sample_hz,
            "num_samples": len(inference_samples),
            "power_path": args.power_path,
            "power_unit": args.power_unit,
            # Power trace stored as parallel arrays (relative to t_start) to
            # keep the JSON compact while remaining self-describing.
            "power_trace_t_s": [t - t_start for (t, _) in inference_samples],
            "power_trace_p_w": [p for (_, p) in inference_samples],
        }

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
