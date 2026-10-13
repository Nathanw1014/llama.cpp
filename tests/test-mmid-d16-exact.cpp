// MUL_MAT_ID(+MUL)+CPY(f16) exactness: the Vulkan backend runs the expert matmul, the router-weight MUL and the f16 cast as
// one f16-output MUL_MAT_ID (the _f16_d16 id pipelines). The contract is BIT-EXACT with the unfused chain on the same
// device: the same id kernel storing f32, the MUL, then the CPY f32 -> f16. The fused and the reference graph share every
// input; in the reference, the matmul (and MUL) outputs are graph outputs, so nothing fuses. Per shape: byte-equal f16
// outputs, fused-dispatch counts 1 / 0, the fused output equal to the host RTE conversion of the reference's f32 product,
// a run-to-run hash, and a positive control (one perturbed B value must change the fused output).
// Skips (exit 0) when there is no Vulkan device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

typedef uint64_t (*fusion_count_t)(const char *);
typedef void     (*fusion_reset_t)(void);

struct shape { ggml_type type; int64_t m, k, nt; int n_mats, n_used; bool with_mul; bool b16; };

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t *) p)[i]; h *= 1099511628211ull; }
    return h;
}

static int run_shape(ggml_backend_t be, ggml_backend_reg_t reg, const shape & sh) {
    auto count = (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_fusion_count");
    auto reset = (fusion_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_fusion_count_reset");
    if (!count || !reset) { fprintf(stderr, "FAIL: no fusion counters\n"); return 1; }
    const char * fname = sh.with_mul ? "MUL_MAT_ID_MUL_CPY16" : "MUL_MAT_ID_CPY16";

    ggml_init_params ip = { 32 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const ggml_type tb = sh.b16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, sh.type, sh.k, sh.m, sh.n_mats);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sh.n_used, sh.nt);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, tb, sh.k, sh.n_used, sh.nt);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, sh.n_used, sh.nt);

    // fused: MUL_MAT_ID (-> MUL) -> CPY(f16)
    ggml_tensor * f = ggml_mul_mat_id(ctx, as, b, ids);
    if (sh.with_mul) f = ggml_mul(ctx, f, w);
    ggml_tensor * out_f = ggml_cast(ctx, f, GGML_TYPE_F16);
    ggml_set_output(out_f);
    // reference: the same chain with the intermediates as graph outputs (never fused)
    ggml_tensor * r_mm = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_set_output(r_mm);
    ggml_tensor * r = r_mm;
    if (sh.with_mul) { r = ggml_mul(ctx, r, w); ggml_set_output(r); }
    ggml_tensor * out_r = ggml_cast(ctx, r, GGML_TYPE_F16);
    ggml_set_output(out_r);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_f);
    ggml_cgraph * gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gr, out_r);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    std::mt19937 rng(4321 + (int) sh.type * 131 + (int) sh.m + (int) sh.nt);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    // the expert weights, quantized once per (type, m, k): iq3_xxs quantization is slow on the host
    static std::vector<uint8_t> aq;
    static int64_t aq_key = -1;
    const int64_t key = (int64_t) sh.type * 1000000007ll + sh.m * 100003ll + sh.k * 7 + sh.n_mats;
    if (aq_key != key) {
        std::mt19937 rw(99 + (int) sh.type);
        std::vector<float> af(sh.k * sh.m * sh.n_mats);
        for (auto & v : af) v = u(rw);
        aq.assign(ggml_nbytes(as), 0);
        ggml_quantize_chunk(sh.type, af.data(), aq.data(), 0, sh.m * sh.n_mats, sh.k, nullptr);
        aq_key = key;
    }
    std::vector<int32_t> idv(sh.n_used * sh.nt);
    std::vector<int32_t> perm(sh.n_mats);
    for (int64_t t = 0; t < sh.nt; ++t) {
        for (int e = 0; e < sh.n_mats; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int j = 0; j < sh.n_used; ++j) idv[t * sh.n_used + j] = perm[j];
    }
    std::vector<float> bv(sh.k * sh.n_used * sh.nt), wv(sh.n_used * sh.nt);
    for (auto & v : bv) v = u(rng);
    for (auto & v : wv) v = 0.05f + 0.5f * (u(rng) + 1.0f);
    auto set_b = [&]() {
        if (sh.b16) {
            std::vector<ggml_fp16_t> h(bv.size());
            for (size_t i = 0; i < bv.size(); ++i) h[i] = ggml_fp32_to_fp16(bv[i]);
            ggml_backend_tensor_set(b, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
        }
    };
    ggml_backend_tensor_set(as, aq.data(), 0, aq.size());
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
    set_b();

    const size_t ne = (size_t) sh.m * sh.n_used * sh.nt;
    std::vector<ggml_fp16_t> o_f(ne), o_f2(ne), o_r(ne), o_p(ne);
    std::vector<float> r32(ne);
    uint64_t n_f = 0, n_r = 0;
    auto run = [&](ggml_cgraph * g, ggml_tensor * out, std::vector<ggml_fp16_t> & o, uint64_t * n) {
        reset();
        ggml_backend_graph_compute(be, g);
        if (n) *n = count(fname);
        ggml_backend_tensor_get(out, o.data(), 0, ne * sizeof(ggml_fp16_t));
    };
    run(gf, out_f, o_f, &n_f);
    run(gr, out_r, o_r, &n_r);
    ggml_backend_tensor_get(r, r32.data(), 0, ne * sizeof(float));
    run(gf, out_f, o_f2, nullptr);

    int fails = 0;
    size_t ndiff = 0, n_rte = 0;
    for (size_t i = 0; i < ne; ++i) {
        ndiff += o_f[i] != o_r[i];
        n_rte += o_f[i] == ggml_fp32_to_fp16(r32[i]);   // ggml's host conversion rounds to nearest even
    }
    if (ndiff) { fprintf(stderr, "FAIL %s: fused != unfused in %zu of %zu values\n", ggml_type_name(sh.type), ndiff, ne); fails++; }
    if (n_rte != ne) { fprintf(stderr, "FAIL %s: fused == RTE(reference f32) in only %zu of %zu values\n", ggml_type_name(sh.type), n_rte, ne); fails++; }
    if (n_f != 1 || n_r != 0) {
        fprintf(stderr, "FAIL %s: %s dispatches fused=%llu reference=%llu (expected 1 / 0)\n", ggml_type_name(sh.type), fname,
                (unsigned long long) n_f, (unsigned long long) n_r);
        fails++;
    }
    const uint64_t h1 = fnv1a(o_f.data(), ne * 2), h2 = fnv1a(o_f2.data(), ne * 2);
    if (h1 != h2) { fprintf(stderr, "FAIL %s: run-to-run hash %016llx vs %016llx\n", ggml_type_name(sh.type), (unsigned long long) h1, (unsigned long long) h2); fails++; }
    // positive control: one perturbed B value must change the fused output (the comparison can fail)
    bv[0] += 0.5f;
    set_b();
    run(gf, out_f, o_p, nullptr);
    const bool ctl = memcmp(o_f.data(), o_p.data(), ne * 2) != 0;
    if (!ctl) { fprintf(stderr, "FAIL %s: positive control, a perturbed input left the output unchanged\n", ggml_type_name(sh.type)); fails++; }
    printf("%s %-7s m=%lld k=%lld nt=%lld experts=%d/%d mul=%d b=%s: byte-equal=%s rte=%zu/%zu fused on/ref=%llu/%llu hash=%016llx run2=%s control=%s\n",
           fails ? "FAIL" : "ok", ggml_type_name(sh.type), (long long) sh.m, (long long) sh.k, (long long) sh.nt, sh.n_used, sh.n_mats,
           (int) sh.with_mul, sh.b16 ? "f16" : "f32", ndiff ? "NO" : "yes", n_rte, ne, (unsigned long long) n_f, (unsigned long long) n_r,
           (unsigned long long) h1, h1 == h2 ? "same" : "DIFF", ctl ? "differs" : "SAME");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return fails;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) { printf("test-mmid-d16-exact: no Vulkan0 device, skipped\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    int fails = 0;
    // 35B-A3B shapes: down 2048 x 512 with the router weight (MUL), gate/up 512 x 2048 without; 512 tokens over 32 experts
    // top-8 (~128 rows per expert), a ragged 77-token batch, and f16 B as LLAMA_MOE_F16 / LLAMA_NORM_F16B hand it over.
    // q2_K / q3_K / q5_K: the d16 pipelines added for the UD-Q4_K_XL experts; q4_K / q6_K / iq3_xxs / iq4_nl: the older ones.
    for (ggml_type t : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL }) {
        const shape shapes[] = {
            { t, 2048,  512, 512, 32, 8, true,  false },
            { t, 2048,  512,  77, 32, 8, true,  false },
            { t, 2048,  512, 512, 32, 8, true,  true  },
            { t,  512, 2048, 512, 32, 8, false, false },
            { t,  512, 2048, 512, 32, 8, false, true  },
        };
        for (const auto & sh : shapes) fails += run_shape(be, reg, sh);
    }
    ggml_backend_free(be);
    printf("test-mmid-d16-exact: %s (%d failing checks)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
