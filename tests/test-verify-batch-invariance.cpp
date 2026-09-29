// Batch invariance of the small-batch matmul paths on one backend (default Vulkan0): for a 2..8-token
// batch (a speculative verify), every token's result must equal, bit for bit, what the same op computes
// for that token alone (single-token decode). Covers MUL_MAT_ID (routed experts) and MUL_MAT (dense
// weights, and the f32 multi-head scorer whose column count is heads x tokens).
//   usage: test-verify-batch-invariance [backend name]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(1234);

static void fill_tensor(ggml_backend_t, ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        return;  // set by the caller
    }
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(n);
    for (auto & x : f) x = u(rng);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
        return;
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    const int64_t n_per_row = t->ne[0];
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / n_per_row, n_per_row, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

struct result { int n_fail = 0; int n_case = 0; };

static bool compare(const char * what, const std::vector<float> & a, const std::vector<float> & b, result & r) {
    r.n_case++;
    if (a.size() != b.size() || memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        double md = 0;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) md = std::max(md, (double) std::fabs(a[i] - b[i]));
        printf("  FAIL %s (max |diff| %.3e)\n", what, md);
        r.n_fail++;
        return false;
    }
    return true;
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// MUL_MAT_ID: as [k, m, E], b [k, ne11, n], ids [n_used, n]; per-token views of b and ids
static void test_mmid(ggml_backend_t be, ggml_type type, int64_t k, int64_t m, int64_t E, int64_t n_used, int64_t ne11, int n, result & r) {
    const size_t mem = ggml_tensor_overhead() * (8 + 4 * n) + ggml_graph_overhead_custom(64, false);
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, k, m, E);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, ne11, n);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    std::vector<ggml_tensor *> outs;
    for (int t = 0; t < n; ++t) {
        ggml_tensor * bt = ggml_view_3d(ctx, b, k, ne11, 1, b->nb[1], b->nb[2], t * b->nb[2]);
        ggml_tensor * it = ggml_view_2d(ctx, ids, n_used, 1, ids->nb[1], t * ids->nb[1]);
        outs.push_back(ggml_mul_mat_id(ctx, as, bt, it));
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);
    for (auto * o : outs) ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    fill_tensor(be, as);
    fill_tensor(be, b);
    // tokens share part of their experts (a small pool), distinct within a token
    std::vector<int32_t> idv(n_used * n);
    for (int t = 0; t < n; ++t) {
        std::vector<int32_t> pool;
        for (int e = 0; e < E; ++e) pool.push_back(e);
        std::shuffle(pool.begin(), pool.begin() + std::min<int64_t>(E, n_used * 2), rng);
        for (int s = 0; s < n_used; ++s) idv[t * n_used + s] = pool[s];
    }
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
    ggml_backend_graph_compute(be, gf);
    const std::vector<float> all = get(out);
    const size_t per = (size_t) m * n_used;
    for (int t = 0; t < n; ++t) {
        std::vector<float> row(all.begin() + t * per, all.begin() + (t + 1) * per);
        char what[160];
        snprintf(what, sizeof(what), "MUL_MAT_ID %s k=%lld m=%lld ne11=%lld n=%d token %d", ggml_type_name(type), (long long) k, (long long) m, (long long) ne11, n, t);
        compare(what, row, get(outs[t]), r);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// MUL_MAT: w [k, m], x [k, cols_per_tok * n]; per-token column views
static void test_mm(ggml_backend_t be, ggml_type type, int64_t k, int64_t m, int cols_per_tok, int n, result & r) {
    const size_t mem = ggml_tensor_overhead() * (8 + 4 * n) + ggml_graph_overhead_custom(64, false);
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, (int64_t) cols_per_tok * n);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    std::vector<ggml_tensor *> outs;
    for (int t = 0; t < n; ++t) {
        ggml_tensor * xt = ggml_view_2d(ctx, x, k, cols_per_tok, x->nb[1], (size_t) t * cols_per_tok * x->nb[1]);
        outs.push_back(ggml_mul_mat(ctx, w, xt));
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);
    for (auto * o : outs) ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    fill_tensor(be, w);
    fill_tensor(be, x);
    ggml_backend_graph_compute(be, gf);
    const std::vector<float> all = get(out);
    const size_t per = (size_t) m * cols_per_tok;
    for (int t = 0; t < n; ++t) {
        std::vector<float> row(all.begin() + t * per, all.begin() + (t + 1) * per);
        char what[160];
        snprintf(what, sizeof(what), "MUL_MAT %s k=%lld m=%lld cols/token=%d n=%d token %d", ggml_type_name(type), (long long) k, (long long) m, cols_per_tok, n, t);
        compare(what, row, get(outs[t]), r);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * name = argc > 1 ? argv[1] : "Vulkan0";
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(name, nullptr);
    if (!be) {
        fprintf(stderr, "backend %s not found\n", name);
        return 1;
    }
    printf("backend %s\n", ggml_backend_name(be));
    result r;
    const ggml_type expert_types[] = { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    for (ggml_type t : expert_types) {
        for (int n = 2; n <= 8; ++n) {
            // gate/up: shared activation row (ne11 = 1); down: one row per expert slot (ne11 = n_used)
            test_mmid(be, t, 2560, 640, 16, 6, 1, n, r);
            if (640 % ggml_blck_size(t) == 0) {
                test_mmid(be, t, 640, 2560, 16, 6, 6, n, r);
            }
        }
    }
    for (int n = 2; n <= 8; ++n) {
        test_mm(be, GGML_TYPE_Q8_0, 2560, 1024, 1, n, r);
        test_mm(be, GGML_TYPE_Q6_K, 2560, 1024, 1, n, r);
        test_mm(be, GGML_TYPE_F32, 2560, 320, 1, n, r);
        // the QSA indexer scorer: keys [128, n_kv] x queries [128, 4 heads x n tokens]
        test_mm(be, GGML_TYPE_F32, 128, 1792, 4, n, r);
    }
    printf("%d/%d token rows bit-identical to single-token results\n", r.n_case - r.n_fail, r.n_case);
    ggml_backend_free(be);
    return r.n_fail == 0 ? 0 : 1;
}
