#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

void fill(std::vector<float> & v, uint32_t salt) {
    for (size_t i = 0; i < v.size(); ++i) {
        const int x = static_cast<int>((i * 17u + salt * 29u) % 31u) - 15;
        v[i] = static_cast<float>(x) * (1.0f / 16.0f);
    }
}

double bench_gemm(uint32_t m, uint32_t k, uint32_t n, int repeats) {
    std::vector<float> a(static_cast<size_t>(m) * k);
    std::vector<float> b(static_cast<size_t>(k) * n);
    std::vector<float> c(static_cast<size_t>(m) * n);
    fill(a, 1);
    fill(b, 2);
    std::vector<double> samples;
    for (int r = 0; r < repeats; ++r) {
        std::fill(c.begin(), c.end(), 0.0f);
        const auto begin = Clock::now();
        for (uint32_t mm = 0; mm < m; ++mm) {
            for (uint32_t kk = 0; kk < k; ++kk) {
                const float av = a[static_cast<size_t>(mm) * k + kk];
                const float * bp = b.data() + static_cast<size_t>(kk) * n;
                float * cp = c.data() + static_cast<size_t>(mm) * n;
                for (uint32_t nn = 0; nn < n; ++nn) {
                    cp[nn] += av * bp[nn];
                }
            }
        }
        const auto end = Clock::now();
        g_sink += c[(static_cast<size_t>(m) * n) / 2u];
        samples.push_back(elapsed_ms(begin, end));
    }
    return median(samples);
}

double bench_gemv(uint32_t rows, uint32_t cols, int repeats) {
    std::vector<float> x(cols);
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    std::vector<float> y(rows);
    fill(x, 3);
    fill(w, 4);
    std::vector<double> samples;
    for (int r = 0; r < repeats; ++r) {
        const auto begin = Clock::now();
        for (uint32_t row = 0; row < rows; ++row) {
            const float * wp = w.data() + static_cast<size_t>(row) * cols;
            float acc = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                acc += wp[col] * x[col];
            }
            y[row] = acc;
        }
        const auto end = Clock::now();
        g_sink += y[rows / 2u];
        samples.push_back(elapsed_ms(begin, end));
    }
    return median(samples);
}

double bench_prefill_attention(uint32_t tokens, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, int repeats) {
    const uint32_t gqa = q_heads / kv_heads;
    std::vector<float> q(static_cast<size_t>(tokens) * head_dim);
    std::vector<float> k(static_cast<size_t>(kv_heads) * tokens * head_dim);
    std::vector<float> v(static_cast<size_t>(kv_heads) * tokens * head_dim);
    std::vector<float> scores(tokens);
    std::vector<float> out(static_cast<size_t>(q_heads) * tokens * head_dim);
    fill(q, 5);
    fill(k, 6);
    fill(v, 7);
    std::vector<double> samples;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int r = 0; r < repeats; ++r) {
        const auto begin = Clock::now();
        for (uint32_t h = 0; h < q_heads; ++h) {
            const uint32_t kv = h / gqa;
            const float * kp_base = k.data() + static_cast<size_t>(kv) * tokens * head_dim;
            const float * vp_base = v.data() + static_cast<size_t>(kv) * tokens * head_dim;
            for (uint32_t t = 0; t < tokens; ++t) {
                const float * qp = q.data() + static_cast<size_t>(t) * head_dim;
                float max_score = -1.0e30f;
                for (uint32_t j = 0; j <= t; ++j) {
                    const float * kp = kp_base + static_cast<size_t>(j) * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        dot += qp[d] * kp[d];
                    }
                    scores[j] = dot * scale;
                    max_score = std::max(max_score, scores[j]);
                }
                float denom = 0.0f;
                for (uint32_t j = 0; j <= t; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    denom += scores[j];
                }
                float * op = out.data() + (static_cast<size_t>(h) * tokens + t) * head_dim;
                std::fill(op, op + head_dim, 0.0f);
                for (uint32_t j = 0; j <= t; ++j) {
                    const float p = scores[j] / denom;
                    const float * vp = vp_base + static_cast<size_t>(j) * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        op[d] += p * vp[d];
                    }
                }
            }
        }
        const auto end = Clock::now();
        g_sink += out[out.size() / 2u];
        samples.push_back(elapsed_ms(begin, end));
    }
    return median(samples);
}

double bench_decode_attention(uint32_t past_tokens, uint32_t hidden, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, int repeats) {
    std::vector<float> hidden_vec(hidden);
    std::vector<float> q_w(static_cast<size_t>(hidden) * hidden);
    std::vector<float> k_w(static_cast<size_t>(kv_heads) * head_dim * hidden);
    std::vector<float> v_w(static_cast<size_t>(kv_heads) * head_dim * hidden);
    std::vector<float> o_w(static_cast<size_t>(hidden) * hidden);
    std::vector<float> q(hidden);
    std::vector<float> k_new(static_cast<size_t>(kv_heads) * head_dim);
    std::vector<float> v_new(static_cast<size_t>(kv_heads) * head_dim);
    std::vector<float> k_cache(static_cast<size_t>(kv_heads) * past_tokens * head_dim);
    std::vector<float> v_cache(static_cast<size_t>(kv_heads) * past_tokens * head_dim);
    std::vector<float> scores(past_tokens);
    std::vector<float> attn(hidden);
    std::vector<float> out(hidden);
    fill(hidden_vec, 8);
    fill(q_w, 9);
    fill(k_w, 10);
    fill(v_w, 11);
    fill(o_w, 12);
    fill(k_cache, 13);
    fill(v_cache, 14);

    auto linear = [](const std::vector<float> & w, const std::vector<float> & x, std::vector<float> & y, uint32_t rows, uint32_t cols) {
        for (uint32_t row = 0; row < rows; ++row) {
            const float * wp = w.data() + static_cast<size_t>(row) * cols;
            float acc = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                acc += wp[col] * x[col];
            }
            y[row] = acc;
        }
    };

    std::vector<double> samples;
    const uint32_t gqa = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int r = 0; r < repeats; ++r) {
        const auto begin = Clock::now();
        linear(q_w, hidden_vec, q, hidden, hidden);
        linear(k_w, hidden_vec, k_new, kv_heads * head_dim, hidden);
        linear(v_w, hidden_vec, v_new, kv_heads * head_dim, hidden);
        for (uint32_t h = 0; h < q_heads; ++h) {
            const uint32_t kv = h / gqa;
            const float * qp = q.data() + static_cast<size_t>(h) * head_dim;
            const float * kp_base = k_cache.data() + static_cast<size_t>(kv) * past_tokens * head_dim;
            const float * vp_base = v_cache.data() + static_cast<size_t>(kv) * past_tokens * head_dim;
            float max_score = -1.0e30f;
            for (uint32_t t = 0; t < past_tokens; ++t) {
                const float * kp = kp_base + static_cast<size_t>(t) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dot += qp[d] * kp[d];
                }
                scores[t] = dot * scale;
                max_score = std::max(max_score, scores[t]);
            }
            float denom = 0.0f;
            for (uint32_t t = 0; t < past_tokens; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                denom += scores[t];
            }
            float * ap = attn.data() + static_cast<size_t>(h) * head_dim;
            std::fill(ap, ap + head_dim, 0.0f);
            for (uint32_t t = 0; t < past_tokens; ++t) {
                const float p = scores[t] / denom;
                const float * vp = vp_base + static_cast<size_t>(t) * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    ap[d] += p * vp[d];
                }
            }
        }
        linear(o_w, attn, out, hidden, hidden);
        const auto end = Clock::now();
        g_sink += out[hidden / 2u] + k_new[0] + v_new[0];
        samples.push_back(elapsed_ms(begin, end));
    }
    return median(samples);
}

void print_result(const char * op, const char * shape, double ms, uint64_t ops, int repeats) {
    const double gops = ms > 0.0 ? static_cast<double>(ops) / (ms * 1000000.0) : 0.0;
    std::printf("cpu_operator_perf,operator=%s,shape=%s,median_ms=%.6f,effective_gops=%.6f,repeats=%d,baseline=fp16_reference\n",
                op, shape, ms, gops, repeats);
}

int parse_repeats(int argc, char ** argv) {
    int repeats = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::max(1, std::atoi(argv[++i]));
        }
    }
    return repeats;
}

} // namespace

int main(int argc, char ** argv) {
    const int repeats = parse_repeats(argc, argv);

    const double gemm_ms = bench_gemm(768, 768, 2048, repeats);
    print_result("gemm_prefill", "M768_K768_N2048", gemm_ms, 2ull * 768ull * 768ull * 2048ull, repeats);

    const double prefill_attn_ms = bench_prefill_attention(1024, 15, 5, 64, repeats);
    print_result("attention_prefill", "T1024_QH15_KVH5_D64_causal", prefill_attn_ms,
                 4ull * 15ull * 1024ull * 1024ull * 64ull / 2ull, repeats);

    const double gemv_ms = bench_gemv(960, 960, repeats);
    print_result("gemv_decode", "M960_K960_N1", gemv_ms, 2ull * 960ull * 960ull, repeats);

    const double decode_attn_ms = bench_decode_attention(992, 960, 15, 5, 64, repeats);
    const uint64_t decode_attn_ops =
        2ull * 960ull * 960ull +
        2ull * 320ull * 960ull +
        2ull * 320ull * 960ull +
        4ull * 15ull * 992ull * 64ull +
        2ull * 960ull * 960ull;
    print_result("attention_decode", "T992_H960_QH15_KVH5_D64", decode_attn_ms, decode_attn_ops, repeats);

    if (g_sink == 123456789.0) {
        std::fprintf(stderr, "sink=%f\n", g_sink);
    }
    return 0;
}
