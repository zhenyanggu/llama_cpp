#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>

static inline uint16_t aicas_rtl_fp16_bits(ggml_fp16_t v) {
    return (uint16_t) v;
}

static inline ggml_fp16_t aicas_rtl_fp16_make(uint16_t bits) {
    return (ggml_fp16_t) bits;
}

static inline bool aicas_rtl_fp16_is_normal(uint16_t bits) {
    const uint16_t exp = (bits >> 10) & 0x1f;
    return exp != 0 && exp != 0x1f;
}

static inline uint16_t aicas_rtl_fp16_zero_if_non_normal(uint16_t bits) {
    return aicas_rtl_fp16_is_normal(bits) ? bits : 0;
}

static inline uint16_t aicas_rtl_fp16_from_f32_flush(float value) {
    const uint16_t bits = aicas_rtl_fp16_bits(ggml_fp32_to_fp16(value));
    const uint16_t exp = (bits >> 10) & 0x1f;
    return exp == 0 ? 0 : bits;
}

static inline float aicas_rtl_fp16_to_f32(uint16_t bits) {
    return ggml_fp16_to_fp32(aicas_rtl_fp16_make(bits));
}

static inline uint16_t aicas_rtl_fp16_from_int9(int value) {
    return aicas_rtl_fp16_bits(ggml_fp32_to_fp16((float) value));
}

static inline uint16_t aicas_rtl_fp16_pack_product_rounded(
        bool     active,
        bool     sign,
        int      exp_base,
        uint16_t mant_pre_i,
        bool     round_bit,
        bool     sticky_bit) {
    if (!active) {
        return 0;
    }

    uint16_t mant_pre = mant_pre_i;
    const uint16_t incr = (round_bit && (sticky_bit || (mant_pre & 1))) ? 1 : 0;
    const uint16_t mant_round = (uint16_t) mant_pre + incr;
    if (mant_round & 0x0800) {
        ++exp_base;
        mant_pre = 0x0400;
    } else {
        mant_pre = mant_round & 0x07ff;
    }

    if (exp_base >= 31) {
        return (uint16_t) ((sign ? 0x8000 : 0) | 0x7c00);
    }
    if (exp_base <= 0) {
        return 0;
    }
    return (uint16_t) ((sign ? 0x8000 : 0) | ((uint16_t) exp_base << 10) | (mant_pre & 0x03ff));
}

static inline uint16_t aicas_rtl_fp16_mul(uint16_t a, uint16_t b) {
    a = aicas_rtl_fp16_zero_if_non_normal(a);
    b = aicas_rtl_fp16_zero_if_non_normal(b);
    if (a == 0 || b == 0) {
        return 0;
    }

    const bool sign = ((a ^ b) & 0x8000) != 0;
    int exp_base = (int) ((a >> 10) & 0x1f) + (int) ((b >> 10) & 0x1f) - 15;
    const uint32_t mant_a = 0x400u | (uint32_t) (a & 0x03ff);
    const uint32_t mant_b = 0x400u | (uint32_t) (b & 0x03ff);
    const uint32_t product = mant_a * mant_b;

    uint16_t mant_pre;
    bool round_bit;
    bool sticky_bit;
    if (product & (1u << 21)) {
        ++exp_base;
        mant_pre = (uint16_t) ((product >> 11) & 0x07ff);
        round_bit = ((product >> 10) & 1u) != 0;
        sticky_bit = (product & 0x03ffu) != 0;
    } else {
        mant_pre = (uint16_t) ((product >> 10) & 0x07ff);
        round_bit = ((product >> 9) & 1u) != 0;
        sticky_bit = (product & 0x01ffu) != 0;
    }

    return aicas_rtl_fp16_pack_product_rounded(true, sign, exp_base, mant_pre, round_bit, sticky_bit);
}

static inline uint16_t aicas_rtl_fp16_add(uint16_t a, uint16_t b) {
    a = aicas_rtl_fp16_zero_if_non_normal(a);
    b = aicas_rtl_fp16_zero_if_non_normal(b);
    return aicas_rtl_fp16_from_f32_flush(aicas_rtl_fp16_to_f32(a) + aicas_rtl_fp16_to_f32(b));
}

static inline uint64_t aicas_rtl_rshift_sticky_u64(uint64_t value, int shift) {
    if (shift <= 0) {
        return value;
    }
    if (shift >= 64) {
        return value != 0 ? 1 : 0;
    }
    const uint64_t lost_mask = (UINT64_C(1) << shift) - 1;
    const bool sticky = (value & lost_mask) != 0;
    value >>= shift;
    return sticky ? (value | 1u) : value;
}

static inline int aicas_rtl_leading_one_u64(uint64_t value) {
    if (value == 0) {
        return 0;
    }
#if defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(value);
#else
    int lead = 0;
    while (value >>= 1) {
        ++lead;
    }
    return lead;
#endif
}

static inline uint16_t aicas_rtl_fp16_from_aligned_sum(bool sign, uint64_t abs_sum, uint16_t emax) {
    constexpr int max_align_shift = 24;
    constexpr int fp16_guard_bits = 6;
    constexpr int fp16_target_bit = fp16_guard_bits + 10;

    if (abs_sum == 0 || emax == 0) {
        return 0;
    }

    const int lead = aicas_rtl_leading_one_u64(abs_sum);
    int exp_norm = (int) emax + lead - 10 - max_align_shift;
    uint64_t norm_sum;
    if (lead > fp16_target_bit) {
        norm_sum = aicas_rtl_rshift_sticky_u64(abs_sum, lead - fp16_target_bit);
    } else {
        norm_sum = abs_sum << (fp16_target_bit - lead);
    }

    if (exp_norm <= 0) {
        return 0;
    }
    if (exp_norm >= 31) {
        return (uint16_t) ((sign ? 0x8000 : 0) | 0x7c00);
    }

    const uint16_t mant_pre = (uint16_t) ((norm_sum >> fp16_guard_bits) & 0x07ff);
    const bool round_bit = ((norm_sum >> (fp16_guard_bits - 1)) & 1u) != 0;
    const bool sticky_bit = (norm_sum & ((UINT64_C(1) << (fp16_guard_bits - 1)) - 1)) != 0;
    uint16_t mant_round = mant_pre + (uint16_t) ((round_bit && (sticky_bit || (mant_pre & 1))) ? 1 : 0);
    if (mant_round & 0x0800) {
        ++exp_norm;
        if (exp_norm >= 31) {
            return (uint16_t) ((sign ? 0x8000 : 0) | 0x7c00);
        }
        return (uint16_t) ((sign ? 0x8000 : 0) | ((uint16_t) exp_norm << 10));
    }

    return (uint16_t) ((sign ? 0x8000 : 0) | ((uint16_t) exp_norm << 10) | (mant_round & 0x03ff));
}

template <typename WeightFn, typename ActFn>
static inline uint16_t aicas_rtl_dp128_tile(WeightFn weight_fn, ActFn act_fn, int valid_elems) {
    constexpr int tile_elems = 128;
    constexpr int max_align_shift = 24;

    uint16_t products[tile_elems];
    uint16_t emax = 0;
    for (int lane = 0; lane < tile_elems; ++lane) {
        const uint16_t w = lane < valid_elems ? weight_fn(lane) : 0;
        const uint16_t a = lane < valid_elems ? act_fn(lane) : 0;
        const uint16_t product = aicas_rtl_fp16_mul(w, a);
        products[lane] = product;
        if (aicas_rtl_fp16_is_normal(product)) {
            emax = std::max<uint16_t>(emax, (uint16_t) ((product >> 10) & 0x1f));
        }
    }

    int64_t sum = 0;
    for (int lane = 0; lane < tile_elems; ++lane) {
        const uint16_t product = products[lane];
        if (!aicas_rtl_fp16_is_normal(product)) {
            continue;
        }
        const int diff = (int) emax - (int) ((product >> 10) & 0x1f);
        if (diff < 0 || diff > max_align_shift) {
            continue;
        }
        const int32_t mant = 0x400 + (int32_t) (product & 0x03ff);
        const int64_t signed_mant = (product & 0x8000) ? -mant : mant;
        sum += signed_mant << (max_align_shift - diff);
    }

    const bool sign = sum < 0;
    const uint64_t abs_sum = sign ? (uint64_t) -sum : (uint64_t) sum;
    return aicas_rtl_fp16_from_aligned_sum(sign, abs_sum, emax);
}

static inline uint16_t aicas_rtl_kv_scale_from_max_abs_f32(float max_abs) {
    return aicas_rtl_fp16_bits(ggml_fp32_to_fp16(max_abs / 128.0f));
}
