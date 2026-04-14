#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "gguf-py")))
import gguf  # type: ignore


def f32_to_bits(value: float) -> int:
    return int(np.array([np.float32(value)], dtype=np.float32).view(np.uint32)[0])


def bits_to_f32(bits: int) -> float:
    return float(np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0])


def fp32_bits_to_q_fixed(fp_bits: int, frac_width: int) -> int:
    sign_bit = (fp_bits >> 31) != 0
    exp_bits = (fp_bits >> 23) & 0xFF
    frac_field = fp_bits & 0x7FFFFF
    is_nan = exp_bits == 0xFF and frac_field != 0
    is_inf = exp_bits == 0xFF and frac_field == 0

    if is_nan:
        return 0
    if is_inf:
        return -0x8000_0000 if sign_bit else 0x7FFF_FFFF
    if exp_bits == 0 and frac_field == 0:
        return 0

    mantissa = frac_field if exp_bits == 0 else ((1 << 23) | frac_field)
    exp_unbiased = -126 if exp_bits == 0 else (exp_bits - 127)
    shift_amount = exp_unbiased + (frac_width - 23)

    abs_value = 0
    if shift_amount >= 0:
        abs_value = mantissa << shift_amount if shift_amount < 40 else (1 << 63) - 1
    else:
        right_shift = -shift_amount
        if right_shift < 64:
            base_value = mantissa >> right_shift
            lower_mask = 0x1 if right_shift == 1 else ((1 << right_shift) - 1)
            half_ulp = 0x1 if right_shift == 1 else (1 << (right_shift - 1))
            remainder = mantissa & lower_mask
            abs_value = base_value
            if remainder > half_ulp or (remainder == half_ulp and (base_value & 1)):
                abs_value += 1

    if not sign_bit:
        return min(abs_value, 0x7FFF_FFFF)
    return -0x8000_0000 if abs_value >= 0x8000_0000 else -int(abs_value)


def fixed_mul_to_i32(int_value: int, scale_value: int, frac_width: int) -> tuple[int, bool]:
    product = int(int_value) * int(scale_value)
    product_is_neg = product < 0
    abs_product = -product if product_is_neg else product

    quotient = abs_product >> frac_width
    remainder_mask = (1 << frac_width) - 1
    remainder = abs_product & remainder_mask
    half_ulp = 1 << (frac_width - 1)
    if remainder > half_ulp or (remainder == half_ulp and (quotient & 1)):
        quotient += 1

    rounded_value = -quotient if product_is_neg else quotient
    if rounded_value > 0x7FFF_FFFF:
        return 0x7FFF_FFFF, True
    if rounded_value < -0x8000_0000:
        return -0x8000_0000, True
    return int(rounded_value), False


def i32_to_fp32_bits_exact(int_value: int) -> int:
    if int_value == 0:
        return 0

    sign_bit = int_value < 0
    abs_value = -int_value if sign_bit else int_value
    msb_idx = abs_value.bit_length() - 1
    exponent_bits = msb_idx + 127

    if msb_idx <= 23:
        mantissa_24 = abs_value << (23 - msb_idx)
    else:
        right_shift = msb_idx - 23
        mantissa_24 = abs_value >> right_shift
        remainder_mask = (1 << right_shift) - 1
        remainder_bits = abs_value & remainder_mask
        half_ulp = 1 << (right_shift - 1)
        if remainder_bits > half_ulp or (remainder_bits == half_ulp and (mantissa_24 & 1)):
            mantissa_24 += 1
        if mantissa_24 == 0x1000000:
            exponent_bits += 1
            mantissa_24 = 0x800000

    return ((0x80000000 if sign_bit else 0) | (exponent_bits << 23) | (mantissa_24 & 0x7FFFFF)) & 0xFFFFFFFF


def dequant_scale_to_scale_shift(scale: float) -> tuple[int, int]:
    if not math.isfinite(scale) or scale == 0.0:
        return 0, 0

    mantissa, exponent = math.frexp(abs(scale))
    scale_i64 = int(round(mantissa * (1 << 31)))
    if scale_i64 >= (1 << 31):
        scale_i64 >>= 1
        exponent += 1
    scale_i64 = min(scale_i64, 0x7FFF_FFFF)
    if math.copysign(1.0, scale) < 0:
        scale_i64 = -scale_i64
    return int(scale_i64), int(exponent - 31)


def apply_scale_shift_i32_round(acc: int, scale_i32: int, shift_i32: int) -> tuple[int, dict]:
    diag = {
        "rounded_to_zero": False,
        "sat_i32": False,
        "mul_overflow_guard_hit": False,
    }
    if acc == 0 or scale_i32 == 0:
        return 0, diag

    product_is_neg = (acc < 0) ^ (scale_i32 < 0)
    abs_product = abs(int(acc)) * abs(int(scale_i32))
    sat_limit = 0x8000_0000 if product_is_neg else 0x7FFF_FFFF

    if shift_i32 >= 0:
        if shift_i32 >= 64:
            diag["mul_overflow_guard_hit"] = abs_product != 0
            diag["sat_i32"] = abs_product != 0
            return (-0x8000_0000 if product_is_neg else 0x7FFF_FFFF), diag
        limit = sat_limit >> shift_i32
        if abs_product > limit:
            diag["mul_overflow_guard_hit"] = True
            diag["sat_i32"] = True
            return (-0x8000_0000 if product_is_neg else 0x7FFF_FFFF), diag
        abs_result = abs_product << shift_i32
    else:
        right_shift = -shift_i32
        if right_shift >= 64:
            diag["rounded_to_zero"] = True
            return 0, diag
        quotient = abs_product >> right_shift
        if right_shift > 0:
            remainder_mask = (1 << right_shift) - 1
            remainder = abs_product & remainder_mask
            half_ulp = 1 << (right_shift - 1)
            if remainder > half_ulp or (remainder == half_ulp and (quotient & 1)):
                quotient += 1
        if quotient > sat_limit:
            diag["sat_i32"] = True
            return (-0x8000_0000 if product_is_neg else 0x7FFF_FFFF), diag
        abs_result = quotient

    if abs_result == 0:
        diag["rounded_to_zero"] = True
    if abs_result > sat_limit:
        diag["sat_i32"] = True
        return (-0x8000_0000 if product_is_neg else 0x7FFF_FFFF), diag
    if not product_is_neg:
        return int(abs_result), diag
    if abs_result == 0x8000_0000:
        return -0x8000_0000, diag
    return -int(abs_result), diag


def u64_to_fp32_bits_with_shift(abs_value: int, sign_bit: bool, value_shift: int) -> tuple[int, dict]:
    diag = {
        "rounded_to_zero": False,
        "sat_fp_exp": False,
        "flush_to_zero_exp": False,
    }
    if abs_value == 0:
        return (0x80000000 if sign_bit else 0), diag

    msb_idx = abs_value.bit_length() - 1
    exponent_unbiased = msb_idx + value_shift
    if exponent_unbiased > 127:
        diag["sat_fp_exp"] = True
        return ((0x80000000 if sign_bit else 0) | 0x7F800000), diag
    if exponent_unbiased < -126:
        diag["flush_to_zero_exp"] = True
        diag["rounded_to_zero"] = True
        return (0x80000000 if sign_bit else 0), diag

    if msb_idx <= 23:
        mantissa_24 = abs_value << (23 - msb_idx)
    else:
        right_shift = msb_idx - 23
        mantissa_24 = abs_value >> right_shift
        remainder_mask = (1 << right_shift) - 1
        remainder_bits = abs_value & remainder_mask
        half_ulp = 1 << (right_shift - 1)
        if remainder_bits > half_ulp or (remainder_bits == half_ulp and (mantissa_24 & 1)):
            mantissa_24 += 1
        if mantissa_24 == 0x1000000:
            exponent_unbiased += 1
            mantissa_24 = 0x800000
            if exponent_unbiased > 127:
                diag["sat_fp_exp"] = True
                return ((0x80000000 if sign_bit else 0) | 0x7F800000), diag

    exponent_bits = exponent_unbiased + 127
    return ((0x80000000 if sign_bit else 0) | (exponent_bits << 23) | (mantissa_24 & 0x7FFFFF)), diag


def simulate(mode: str, acc: int, scale: float) -> tuple[float, dict]:
    diag = {
        "scale_zero": False,
        "rounded_to_zero": False,
        "sat_i32": False,
        "sat_fp_exp": False,
        "flush_to_zero_exp": False,
        "mul_overflow_guard_hit": False,
    }

    if mode == "off":
        return float(np.float32(np.float32(acc) * np.float32(scale))), diag

    if mode == "versa_q8_24":
        scale_q = fp32_bits_to_q_fixed(f32_to_bits(scale), 24)
        diag["scale_zero"] = scale != 0.0 and scale_q == 0
        rounded, diag["sat_i32"] = fixed_mul_to_i32(acc, scale_q, 24)
        diag["rounded_to_zero"] = acc != 0 and scale_q != 0 and rounded == 0
        return bits_to_f32(i32_to_fp32_bits_exact(rounded)), diag

    if mode == "versa_q8_24_fp_reconstruct":
        scale_q = fp32_bits_to_q_fixed(f32_to_bits(scale), 24)
        diag["scale_zero"] = scale != 0.0 and scale_q == 0
        if acc == 0 or scale_q == 0:
            return 0.0, diag
        product = int(acc) * int(scale_q)
        fp_bits, fp_diag = u64_to_fp32_bits_with_shift(abs(product), product < 0, -24)
        diag.update(fp_diag)
        return bits_to_f32(fp_bits), diag

    scale_i32, shift_i32 = dequant_scale_to_scale_shift(scale)
    diag["scale_zero"] = scale != 0.0 and scale_i32 == 0

    if mode == "scale_shift_i32_round":
        rounded, round_diag = apply_scale_shift_i32_round(acc, scale_i32, shift_i32)
        diag.update(round_diag)
        return bits_to_f32(i32_to_fp32_bits_exact(rounded)), diag

    if mode == "scale_shift_fp_reconstruct":
        if acc == 0 or scale_i32 == 0:
            return 0.0, diag
        product = int(acc) * int(scale_i32)
        fp_bits, fp_diag = u64_to_fp32_bits_with_shift(abs(product), product < 0, shift_i32)
        diag.update(fp_diag)
        return bits_to_f32(fp_bits), diag

    raise ValueError(f"unsupported mode: {mode}")


def load_scales(gguf_path: Path) -> list[float]:
    reader = gguf.GGUFReader(str(gguf_path), "r")
    tensor_count = int(reader.get_field("aicas.w8a8.tensor_count").contents())
    scales: list[float] = []
    for idx in range(tensor_count):
        prefix = f"aicas.w8a8.tensor.{idx}."
        policy = str(reader.get_field(prefix + "policy").contents())
        if policy != "W8A8":
            continue
        act_scale = float(reader.get_field(prefix + "act_scale").contents())
        weight_scale = reader.get_field(prefix + "weight_scale").contents()
        for item in weight_scale:
            scales.append(abs(act_scale * float(item)))
    return scales


def sample_acc(rng: random.Random) -> int:
    bucket = rng.random()
    if bucket < 0.70:
        return rng.randint(-200000, 200000)
    if bucket < 0.95:
        return rng.randint(-(1 << 24), (1 << 24))
    return rng.randint(-(1 << 31), (1 << 31) - 1)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(mode: str, scales: list[float], samples: int, seed: int, rel_floor: float) -> dict:
    rng = random.Random(seed)
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    diag_totals = {
        "scale_zero": 0,
        "rounded_to_zero": 0,
        "sat_i32": 0,
        "sat_fp_exp": 0,
        "flush_to_zero_exp": 0,
        "mul_overflow_guard_hit": 0,
    }

    for _ in range(samples):
        scale = scales[rng.randrange(len(scales))]
        acc = sample_acc(rng)
        ideal = float(np.float32(np.float32(acc) * np.float32(scale)))
        approx, diag = simulate(mode, acc, scale)
        abs_err = abs(ideal - approx)
        abs_errors.append(abs_err)
        if abs(ideal) > rel_floor:
            rel_errors.append(abs_err / abs(ideal))
        for key in diag_totals:
            diag_totals[key] += int(bool(diag[key]))

    return {
        "mode": mode,
        "samples": samples,
        "abs_mean": float(sum(abs_errors) / len(abs_errors)),
        "abs_p90": pct(abs_errors, 90),
        "abs_p99": pct(abs_errors, 99),
        "abs_max": max(abs_errors),
        "rel_mean": float(sum(rel_errors) / len(rel_errors)) if rel_errors else float("nan"),
        "rel_p90": pct(rel_errors, 90),
        "rel_p99": pct(rel_errors, 99),
        "rel_max": max(rel_errors) if rel_errors else float("nan"),
        "diag": diag_totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Numeric dequant analysis for strict per-tensor mmproj modes.")
    parser.add_argument("--gguf", default="AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1-pt-strict.gguf")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--rel-floor", type=float, default=1e-12)
    parser.add_argument("--output", default="AICAS/artifacts/strict_pt_dequant_numeric_analysis.json")
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    scales = load_scales(gguf_path)
    if not scales:
        raise SystemExit(f"no W8A8 scales found in {gguf_path}")

    rows = [
        summarize("versa_q8_24", scales, args.samples, args.seed, args.rel_floor),
        summarize("versa_q8_24_fp_reconstruct", scales, args.samples, args.seed, args.rel_floor),
        summarize("scale_shift_i32_round", scales, args.samples, args.seed, args.rel_floor),
        summarize("scale_shift_fp_reconstruct", scales, args.samples, args.seed, args.rel_floor),
    ]

    payload = {
        "schema": "aicas.mmproj.strict-pt-dequant-numeric-analysis.v1",
        "gguf": str(gguf_path.resolve()),
        "samples": args.samples,
        "seed": args.seed,
        "scale_count": len(scales),
        "scale_min": min(scales),
        "scale_max": max(scales),
        "rows": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
