#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <arm_neon.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
    const char * name;
    int rows;
    int dim;
};

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

int get_env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

void fill_input(std::vector<float> & data) {
    uint32_t state = 0x12345678u;
    for (float & value : data) {
        state = state * 1664525u + 1013904223u;
        const int centered = static_cast<int>((state >> 9) & 0x7ffu) - 1024;
        value = static_cast<float>(centered) * (1.0f / 128.0f);
    }
}

float max_abs_neon(const float * data, int n) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vabsq_f32(vld1q_f32(data + i));
        vmax = vmaxq_f32(vmax, v);
    }
    float maxv = std::max(
        std::max(vgetq_lane_f32(vmax, 0), vgetq_lane_f32(vmax, 1)),
        std::max(vgetq_lane_f32(vmax, 2), vgetq_lane_f32(vmax, 3)));
    for (; i < n; ++i) {
        maxv = std::max(maxv, std::fabs(data[i]));
    }
    return maxv;
}

float bench_per_tensor_scale(const float * data, int n, int iters) {
    volatile float guard = 0.0f;
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        const float maxv = max_abs_neon(data, n);
        guard += maxv > 0.0f ? 127.0f / maxv : 1.0f;
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345.0f) {
        std::printf("guard=%f\n", static_cast<float>(guard));
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

float bench_per_row_scale(const float * data, float * scales, int rows, int dim, int iters) {
    volatile float guard = 0.0f;
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < rows; ++r) {
            const float maxv = max_abs_neon(data + static_cast<size_t>(r) * dim, dim);
            const float scale = maxv > 0.0f ? 127.0f / maxv : 1.0f;
            scales[r] = scale;
            guard += scale;
        }
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345.0f) {
        std::printf("guard=%f\n", static_cast<float>(guard));
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

float bench_per_row_scale_quant(const float * data, float * scales, int8_t * out,
                                int rows, int dim, int iters) {
    volatile int guard = 0;
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < rows; ++r) {
            const float * row = data + static_cast<size_t>(r) * dim;
            const float maxv = max_abs_neon(row, dim);
            const float scale = maxv > 0.0f ? 127.0f / maxv : 1.0f;
            scales[r] = scale;
            int8_t * dst = out + static_cast<size_t>(r) * dim;
            for (int c = 0; c < dim; ++c) {
                int q = static_cast<int>(std::nearbyint(row[c] * scale));
                q = std::max(-127, std::min(127, q));
                dst[c] = static_cast<int8_t>(q);
            }
            guard += dst[0];
        }
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345) {
        std::printf("guard=%d\n", guard);
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

float bench_row_block_scale(const float * data, float * scales, int rows, int dim,
                            int block_rows, int iters) {
    volatile float guard = 0.0f;
    const int groups = (rows + block_rows - 1) / block_rows;
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int g = 0; g < groups; ++g) {
            const int r0 = g * block_rows;
            const int r1 = std::min(rows, r0 + block_rows);
            const int n = (r1 - r0) * dim;
            const float maxv = max_abs_neon(data + static_cast<size_t>(r0) * dim, n);
            const float scale = maxv > 0.0f ? 127.0f / maxv : 1.0f;
            scales[g] = scale;
            guard += scale;
        }
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345.0f) {
        std::printf("guard=%f\n", static_cast<float>(guard));
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

float bench_row_block_scale_quant(const float * data, float * scales, int8_t * out,
                                  int rows, int dim, int block_rows, int iters) {
    volatile int guard = 0;
    const int groups = (rows + block_rows - 1) / block_rows;
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int g = 0; g < groups; ++g) {
            const int r0 = g * block_rows;
            const int r1 = std::min(rows, r0 + block_rows);
            const int n = (r1 - r0) * dim;
            const float * src = data + static_cast<size_t>(r0) * dim;
            const float maxv = max_abs_neon(src, n);
            const float scale = maxv > 0.0f ? 127.0f / maxv : 1.0f;
            scales[g] = scale;
            int8_t * dst = out + static_cast<size_t>(r0) * dim;
            for (int i = 0; i < n; ++i) {
                int q = static_cast<int>(std::nearbyint(src[i] * scale));
                q = std::max(-127, std::min(127, q));
                dst[i] = static_cast<int8_t>(q);
            }
            guard += dst[0];
        }
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345) {
        std::printf("guard=%d\n", guard);
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

float max_abs_col_block_neon(const float * data, int rows, int dim, int c0, int c1) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int c = c0;
    for (; c + 4 <= c1; c += 4) {
        for (int r = 0; r < rows; ++r) {
            const float32x4_t v = vabsq_f32(vld1q_f32(data + static_cast<size_t>(r) * dim + c));
            vmax = vmaxq_f32(vmax, v);
        }
    }
    float maxv = std::max(
        std::max(vgetq_lane_f32(vmax, 0), vgetq_lane_f32(vmax, 1)),
        std::max(vgetq_lane_f32(vmax, 2), vgetq_lane_f32(vmax, 3)));
    for (; c < c1; ++c) {
        for (int r = 0; r < rows; ++r) {
            maxv = std::max(maxv, std::fabs(data[static_cast<size_t>(r) * dim + c]));
        }
    }
    return maxv;
}

float bench_col_block_scale(const float * data, float * scales, int rows, int dim,
                            int block_cols, int iters) {
    volatile float guard = 0.0f;
    const int groups = (dim + block_cols - 1) / block_cols;
    std::vector<float> max_vals(groups);
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(max_vals.begin(), max_vals.end(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            const float * row = data + static_cast<size_t>(r) * dim;
            for (int g = 0; g < groups; ++g) {
                const int c0 = g * block_cols;
                const int c1 = std::min(dim, c0 + block_cols);
                float32x4_t vmax = vdupq_n_f32(max_vals[g]);
                int c = c0;
                for (; c + 4 <= c1; c += 4) {
                    const float32x4_t v = vabsq_f32(vld1q_f32(row + c));
                    vmax = vmaxq_f32(vmax, v);
                }
                float maxv = std::max(
                    std::max(vgetq_lane_f32(vmax, 0), vgetq_lane_f32(vmax, 1)),
                    std::max(vgetq_lane_f32(vmax, 2), vgetq_lane_f32(vmax, 3)));
                for (; c < c1; ++c) {
                    maxv = std::max(maxv, std::fabs(row[c]));
                }
                max_vals[g] = maxv;
            }
        }
        for (int g = 0; g < groups; ++g) {
            const float scale = max_vals[g] > 0.0f ? 127.0f / max_vals[g] : 1.0f;
            scales[g] = scale;
            guard += scale;
        }
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345.0f) {
        std::printf("guard=%f\n", static_cast<float>(guard));
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

float bench_col_block_scale_quant(const float * data, float * scales, int8_t * out,
                                  int rows, int dim, int block_cols, int iters) {
    volatile int guard = 0;
    const int groups = (dim + block_cols - 1) / block_cols;
    std::vector<float> max_vals(groups);
    const auto begin = Clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(max_vals.begin(), max_vals.end(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            const float * row = data + static_cast<size_t>(r) * dim;
            for (int g = 0; g < groups; ++g) {
                const int c0 = g * block_cols;
                const int c1 = std::min(dim, c0 + block_cols);
                float32x4_t vmax = vdupq_n_f32(max_vals[g]);
                int c = c0;
                for (; c + 4 <= c1; c += 4) {
                    const float32x4_t v = vabsq_f32(vld1q_f32(row + c));
                    vmax = vmaxq_f32(vmax, v);
                }
                float maxv = std::max(
                    std::max(vgetq_lane_f32(vmax, 0), vgetq_lane_f32(vmax, 1)),
                    std::max(vgetq_lane_f32(vmax, 2), vgetq_lane_f32(vmax, 3)));
                for (; c < c1; ++c) {
                    maxv = std::max(maxv, std::fabs(row[c]));
                }
                max_vals[g] = maxv;
            }
        }
        for (int g = 0; g < groups; ++g) {
            scales[g] = max_vals[g] > 0.0f ? 127.0f / max_vals[g] : 1.0f;
        }
        for (int r = 0; r < rows; ++r) {
            const float * src = data + static_cast<size_t>(r) * dim;
            int8_t * dst = out + static_cast<size_t>(r) * dim;
            for (int g = 0; g < groups; ++g) {
                const float scale = scales[g];
                const int c0 = g * block_cols;
                const int c1 = std::min(dim, c0 + block_cols);
                for (int c = c0; c < c1; ++c) {
                    int q = static_cast<int>(std::nearbyint(src[c] * scale));
                    q = std::max(-127, std::min(127, q));
                    dst[c] = static_cast<int8_t>(q);
                }
            }
            guard += dst[0];
        }
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if (guard == 12345) {
        std::printf("guard=%d\n", guard);
    }
    return static_cast<float>(ns) / static_cast<float>(iters) / 1000.0f;
}

} // namespace

int main() {
    const int base_iters = get_env_int("QKV_SCALE_BENCH_ITERS", 2000);
    const Case cases[] = {
        {"text_prefill_q_T501_D960", 501, 960},
        {"text_prefill_k_T501_D320", 501, 320},
        {"text_prefill_v_T501_D320", 501, 320},
        {"text_decode_q_T1_D960", 1, 960},
        {"text_decode_k_T1_D320", 1, 320},
        {"text_decode_v_T1_D320", 1, 320},
        {"mmproj_vision_q_T1024_D768", 1024, 768},
        {"mmproj_vision_k_T1024_D768", 1024, 768},
        {"mmproj_vision_v_T1024_D768", 1024, 768},
    };

    std::puts("name,rows,dim,elements,iters,per_tensor_scale_us,per_row_scale_us,per_row_scale_quant_us,row32_scale_us,row32_scale_quant_us,col32_scale_us,col32_scale_quant_us");
    for (const Case & tc : cases) {
        const int elems = tc.rows * tc.dim;
        const int iters = std::max(50, base_iters * 480960 / std::max(1, elems));
        std::vector<float> data(elems);
        std::vector<float> scales(std::max(tc.rows, tc.dim));
        std::vector<int8_t> quant(elems);
        fill_input(data);

        const float tensor_us = bench_per_tensor_scale(data.data(), elems, iters);
        const float row_us = bench_per_row_scale(data.data(), scales.data(), tc.rows, tc.dim, iters);
        const float row_quant_us = bench_per_row_scale_quant(
            data.data(), scales.data(), quant.data(), tc.rows, tc.dim, iters);
        const float row32_us = bench_row_block_scale(data.data(), scales.data(), tc.rows, tc.dim, 32, iters);
        const float row32_quant_us = bench_row_block_scale_quant(
            data.data(), scales.data(), quant.data(), tc.rows, tc.dim, 32, iters);
        const float col32_us = bench_col_block_scale(data.data(), scales.data(), tc.rows, tc.dim, 32, iters);
        const float col32_quant_us = bench_col_block_scale_quant(
            data.data(), scales.data(), quant.data(), tc.rows, tc.dim, 32, iters);

        std::printf("%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    tc.name, tc.rows, tc.dim, elems, iters,
                    tensor_us, row_us, row_quant_us,
                    row32_us, row32_quant_us, col32_us, col32_quant_us);
    }
    return 0;
}
