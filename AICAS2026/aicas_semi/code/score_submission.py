import argparse
import json
from pathlib import Path


BASELINE = {
    "throughput_p": 3.01,
    "throughput_d": 5.55,
    "energy": 0.97,
    "ttft_slope": 12.0,
    "ttft_intercept": 54000.0,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ratio_ttft(ttft: dict, baseline: dict) -> float | None:
    slope = ttft.get("slope_ms_per_char")
    intercept = ttft.get("intercept_ms")
    if slope in (None, 0) or intercept in (None, 0):
        return None
    return 0.5 * (
        baseline["ttft_slope"] / float(slope)
        + baseline["ttft_intercept"] / float(intercept)
    )


def main():
    parser = argparse.ArgumentParser(description="Score AICAS 2026 semi-final outputs.")
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--acc-ori", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("score_report.json"))
    args = parser.parse_args()

    merged = load_json(args.merged)
    acc = merged.get("accuracy", {})
    throughput = merged.get("throughput", {})
    energy = merged.get("energy", {})
    ttft = merged.get("ttft", {})

    acc_opt = acc.get("ratio_accuracy")
    tp = throughput.get("prefill_speed_tps")
    td = throughput.get("decode_speed_tps")
    eff = energy.get("tokens_per_joule")
    ttft_ratio = ratio_ttft(ttft, BASELINE) if isinstance(ttft, dict) else None

    ratios = {
        "throughput_p": (tp / BASELINE["throughput_p"]) if tp is not None else None,
        "throughput_d": (td / BASELINE["throughput_d"]) if td is not None else None,
        "energy": (eff / BASELINE["energy"]) if eff is not None else None,
        "ttft": ttft_ratio,
    }

    weighted = 0.0
    missing = []
    for key, weight in (("throughput_p", 20), ("throughput_d", 30), ("energy", 30), ("ttft", 20)):
        value = ratios[key]
        if value is None:
            missing.append(key)
            continue
        weighted += weight * float(value)

    gate = None
    if args.acc_ori is not None and acc_opt is not None:
        gate = acc_opt >= (args.acc_ori - 0.05)

    report = {
        "accuracy_ratio": acc_opt,
        "accuracy_gate": gate,
        "ratios": ratios,
        "weighted_score": weighted,
        "missing": missing,
        "baseline": BASELINE,
        "note": "weighted_score is the local provisional score; official final score still needs committee max-normalization",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
