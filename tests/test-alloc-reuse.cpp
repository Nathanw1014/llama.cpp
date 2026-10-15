// test-alloc-reuse: the graph allocator's reuse hold-back (GGML_ALLOC_REUSE_DELAY / _HOLD_MIB with the _MAX_FRAC / _MAX_MIB caps).
// The allocator reads its settings once per process, so this test fixes them before the first plan: GGML_ALLOC_REUSE_DELAY=64 and
// GGML_ALLOC_REUSE_DELAY_MAX_FRAC=10 (unset caps off). On the CPU buffer type, no backend needed:
//   1. a layer-shaped graph planned with the hold-back: no tensor's range overlaps another tensor still read after it (live ranges)
//   2. the checker can fail: an overlap planted into its input is caught
//   3. reserve(pp), reserve(chain), reserve(pp): both pp plans are equal (a reduction forced on one graph must not carry over)
//   4. after those reserves, ALLOCATING a graph with a smaller no-hold-back plan than the worst case (the runtime case: fewer outputs
//      than the reserve's graph) keeps the delay: the allocator prints no re-plan line (its stderr is captured around the call)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>

static ggml_cgraph * build_layers(ggml_context * ctx, int64_t T) {
    const int64_t E = 256;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    ggml_set_input(x);
    ggml_tensor * cur = x;
    for (int l = 0; l < 24; l++) {
        ggml_tensor * n   = ggml_rms_norm(ctx, cur, 1e-6f);
        ggml_tensor * a   = ggml_scale(ctx, n, 0.5f);
        ggml_tensor * big = ggml_concat(ctx, a, ggml_concat(ctx, a, a, 0), 0);
        ggml_tensor * g   = ggml_silu(ctx, big);
        ggml_tensor * v   = ggml_view_2d(ctx, g, E, T, g->nb[1], 0);
        ggml_tensor * h   = ggml_mul(ctx, ggml_cont(ctx, v), n);
        cur = ggml_add(ctx, cur, h);
        if (l % 3 == 2) {
            cur = ggml_add(ctx, cur, ggml_sqr(ctx, n));
        }
    }
    ggml_set_output(cur);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// a chain of non-in-place matmuls: no hold-back keeps ~2 tensors live, a delay of N keeps ~N, so the hold-back costs far more relative
// to its base than in the layer graph
static ggml_cgraph * build_chain(ggml_context * ctx, int64_t T) {
    const int64_t E = 256;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, E);
    ggml_set_input(w);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    ggml_set_input(x);
    ggml_tensor * cur = x;
    for (int i = 0; i < 300; i++) {
        cur = ggml_mul_mat(ctx, w, cur);
    }
    ggml_set_output(cur);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

static ggml_tensor * root(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

// overlaps between a newly placed tensor and any tensor still read after its node; plant = check the first such pair as if the new
// tensor sat on the other's bytes
static int count_overlaps(ggml_cgraph * gf, bool plant) {
    std::map<ggml_tensor *, int> born, last;
    const int nn = ggml_graph_n_nodes(gf);
    for (int i = 0; i < nn; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!born.count(root(t))) {
            born[root(t)] = i;
        }
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
            last[root(t->src[s])] = i;
        }
    }
    last[root(ggml_graph_node(gf, nn - 1))] = nn;
    int overlaps = 0;
    for (int i = 0; i < nn; i++) {
        ggml_tensor * t = root(ggml_graph_node(gf, i));
        if (born[t] != i || t->data == nullptr) {
            continue;
        }
        const char * a0 = (const char *) t->data;
        const char * a1 = a0 + ggml_nbytes(t);
        for (auto & [o, b] : born) {
            if (o == t || o->data == nullptr || b > i || last[o] <= i) {
                continue;   // o is not read after node i (a last read at node i may be in place)
            }
            const char * b0 = (const char *) o->data;
            const char * b1 = b0 + ggml_nbytes(o);
            if (plant) {
                a0 = b0;
                a1 = b0 + ggml_nbytes(t);
                plant = false;
            }
            if (a0 < b1 && b0 < a1) {
                overlaps++;
            }
        }
    }
    return overlaps;
}

int main() {
    setenv("GGML_ALLOC_REUSE_DELAY", "64", 1);
    setenv("GGML_ALLOC_REUSE_DELAY_MAX_FRAC", "10", 1);
    unsetenv("GGML_ALLOC_REUSE_DELAY_MAX_MIB");
    unsetenv("GGML_ALLOC_REUSE_HOLD_MIB");
    unsetenv("GGML_ALLOC_DRY_RUN");

    int fails = 0;
    ggml_init_params ip = { ggml_tensor_overhead() * 16384 + 4 * ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    // 1 + 2
    {
        ggml_cgraph * gf = build_layers(ctx, 64);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            printf("FAIL: alloc\n");
            return 1;
        }
        const int ov = count_overlaps(gf, false);
        const int pc = count_overlaps(gf, true);
        printf("%s live-range overlaps with the hold-back: %d\n", ov == 0 ? "ok" : "FAIL", ov);
        printf("%s planted-overlap control: %d (must be > 0)\n", pc > 0 ? "ok" : "FAIL", pc);
        fails += ov != 0;
        fails += pc == 0;
        ggml_gallocr_free(ga);
    }

    // 3 + 4
    {
        ggml_cgraph * pp = build_layers(ctx, 64);
        ggml_cgraph * ch = build_chain(ctx, 64);
        ggml_cgraph * small = build_chain(ctx, 64);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        size_t s0 = 0, s1 = 0, s2 = 0;
        ggml_gallocr_reserve_n_size(ga, pp, nullptr, nullptr, &s0);
        ggml_gallocr_reserve_n_size(ga, ch, nullptr, nullptr, &s1);
        ggml_gallocr_reserve_n_size(ga, pp, nullptr, nullptr, &s2);
        printf("%s pp / chain / pp plans: %zu / %zu / %zu bytes (the two pp plans must match)\n", s0 == s2 ? "ok" : "FAIL", s0, s1, s2);
        fails += s0 != s2;

        // capture stderr around the allocation of the smaller-base graph
        fflush(stderr);
        FILE * tmp = tmpfile();
        const int saved = dup(fileno(stderr));
        dup2(fileno(tmp), fileno(stderr));
        const bool ok_alloc = ggml_gallocr_alloc_graph(ga, small);
        fflush(stderr);
        dup2(saved, fileno(stderr));
        close(saved);
        std::string err;
        rewind(tmp);
        char line[512];
        while (fgets(line, sizeof(line), tmp)) {
            err += line;
        }
        fclose(tmp);
        const bool replanned = err.find("re-planning") != std::string::npos;
        printf("%s smaller-base graph allocated after the reserves: %s\n", ok_alloc && !replanned ? "ok" : "FAIL",
               !ok_alloc ? "alloc failed" : replanned ? "RE-PLANNED (the cap followed the graph, not the worst case)" : "kept the delay");
        fails += !ok_alloc || replanned;
        ggml_gallocr_free(ga);
    }

    ggml_free(ctx);
    printf("test-alloc-reuse: %s (%d failing checks)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
