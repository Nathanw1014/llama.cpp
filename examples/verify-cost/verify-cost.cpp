// Debug tool: cost and exactness of an n-token speculative verify batch vs single-token decode.
//   VC_TOKENS=file   : "n0\n t0 t1 ..." (prompt tokens followed by the AR continuation)
//   VC_NS=list       : batch sizes to time (default 1,2,3,4,8)
//   VC_REPS=n        : timed repetitions per batch size (default 10)
//   VC_NEXTN=1       : run the target with embeddings_nextn (as the MTP server does; default on)
// Each batch decodes the continuation tokens at [n0, n0+n) with logits on every row, from a checkpoint
// of the n0-token context (restored after every batch, as the server's verify step does). Rows are
// compared with single-token decodes of the same positions (max |diff|, 0 = bit-exact).
// A "@@VC n=<n> rep=<r>" line goes to stderr before each timed decode, for GGML_VK_PERF_LOGGER parsing.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool dec(llama_context * ctx, const std::vector<llama_token> & toks, int a, int b, bool all_logits) {
    llama_batch batch = llama_batch_init(b - a, 0, 1);
    for (int i = a; i < b; ++i) {
        common_batch_add(batch, toks[i], i, { 0 }, all_logits || i == b - 1);
    }
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) { fprintf(stderr, "decode failed %d\n", ret); return false; }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    // VC_RS=n: recurrent snapshots for n-token rollback (as the MTP server requests) and a rollback check
    const int vc_rs = getenv("VC_RS") ? atoi(getenv("VC_RS")) : 0;
    if (vc_rs > 0) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = vc_rs;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    llama_model * model = init->model();
    if (!ctx) { fprintf(stderr, "no ctx\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_batch = llama_n_batch(ctx);

    if (!getenv("VC_NEXTN") || atoi(getenv("VC_NEXTN"))) {
        llama_set_embeddings_nextn(ctx, true, false);
    }

    std::ifstream f(getenv("VC_TOKENS"));
    int n0; f >> n0;
    std::vector<llama_token> toks; int t; while (f >> t) toks.push_back(t);

    std::vector<int> ns;
    { std::stringstream ss(getenv("VC_NS") ? getenv("VC_NS") : "1,2,3,4,8"); std::string m; while (std::getline(ss, m, ',')) ns.push_back(atoi(m.c_str())); }
    const int reps = getenv("VC_REPS") ? atoi(getenv("VC_REPS")) : 10;
    const int nmax = *std::max_element(ns.begin(), ns.end());
    GGML_ASSERT((int) toks.size() >= n0 + nmax);

    llama_memory_clear(llama_get_memory(ctx), true);
    for (int a = 0; a < n0; a += n_batch) {
        if (!dec(ctx, toks, a, std::min(a + n_batch, n0), false)) return 1;
    }
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    std::vector<uint8_t> ck(llama_state_seq_get_size_ext(ctx, 0, fl));
    llama_state_seq_get_data_ext(ctx, ck.data(), ck.size(), 0, fl);
    auto restore = [&]() {
        llama_state_seq_set_data_ext(ctx, ck.data(), ck.size(), 0, fl);
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n0, -1);
    };

    // single-token reference rows for positions n0 .. n0+nmax-1
    std::vector<std::vector<float>> ref(nmax);
    for (int r = 0; r < nmax; ++r) {
        if (!dec(ctx, toks, n0 + r, n0 + r + 1, true)) return 1;
        const float * l = llama_get_logits_ith(ctx, 0);
        ref[r].assign(l, l + n_vocab);
    }
    restore();

    printf("ctx n0=%d\n", n0);
    printf("%3s %9s %9s %9s %8s  %s\n", "n", "med_ms", "min_ms", "ms/tok", "vs_n1", "row max|diff| vs single-token decode");
    double t1 = 0;
    for (int n : ns) {
        // warm-up (pipeline compile, graph build) and exactness rows
        if (!dec(ctx, toks, n0, n0 + n, true)) return 1;
        std::vector<double> dif(n);
        for (int r = 0; r < n; ++r) {
            const float * l = llama_get_logits_ith(ctx, r);
            double md = 0;
            for (int k = 0; k < n_vocab; ++k) md = std::max(md, (double) fabs(l[k] - ref[r][k]));
            dif[r] = md;
        }
        restore();
        std::vector<double> ms;
        for (int rep = 0; rep < reps; ++rep) {
            fprintf(stderr, "@@VC n=%d rep=%d\n", n, rep);
            const auto a = std::chrono::steady_clock::now();
            if (!dec(ctx, toks, n0, n0 + n, true)) return 1;
            llama_synchronize(ctx);
            const auto b = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            fprintf(stderr, "@@VC end\n");
            restore();
        }
        std::vector<double> s = ms; std::sort(s.begin(), s.end());
        const double med = s[s.size() / 2];
        if (n == 1) t1 = med;
        printf("%3d %9.2f %9.2f %9.2f %8.3f ", n, med, s[0], med / n, t1 > 0 ? med / t1 : 0.0);
        for (double d : dif) printf(" %.2e", d);
        printf("\n");
        fflush(stdout);
    }
    if (vc_rs > 0) {
        // decode an n-token batch, roll the last r tokens back (seq_rm), then decode the first rolled-back
        // token alone: its row must match the single-token decode of the same context
        printf("rollback check (n_rs_seq=%u): n r max|diff| of the re-decoded row vs single-token decode\n", llama_n_rs_seq(ctx));
        for (int n : ns) {
            for (int r = 1; r < n && r <= vc_rs; ++r) {
                restore();
                if (!dec(ctx, toks, n0, n0 + n, true)) return 1;
                const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, n0 + n - r, -1);
                if (!dec(ctx, toks, n0 + n - r, n0 + n - r + 1, true)) return 1;
                const float * l = llama_get_logits_ith(ctx, 0);
                double md = 0;
                for (int k = 0; k < n_vocab; ++k) md = std::max(md, (double) fabs(l[k] - ref[n - r][k]));
                printf("  n=%d r=%d seq_rm=%d maxdiff %.3e\n", n, r, (int) ok, md);
            }
        }
    }
    llama_backend_free();
    return 0;
}
