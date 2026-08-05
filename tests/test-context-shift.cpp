// Test that the KV cache layout does not change what a context shift produces.
//
// The shift rotates the cached K in place, so it has to walk the cache with the same strides the
// read path uses. A row pad changes those strides ([TAG_KV_ROW_PAD] in llama-kv-cache.cpp), so run
// the same shift under two layouts and require the same logits.
//
// Note a shift is NOT equivalent to rebuilding the cache from the remaining tokens: layers past the
// first cached hidden states that attended to the dropped tokens. So the second check below is only
// a sanity bound, not an equality.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int n_tokens = 96;
static const int n_shift  = 32;

static std::vector<llama_token> make_tokens(int n_vocab) {
    std::vector<llama_token> res(n_tokens + 1);
    for (int i = 0; i < (int) res.size(); ++i) {
        res[i] = 1 + (i*7919) % (n_vocab - 1);
    }

    return res;
}

// fill the cache, drop the oldest cells, shift the rest down, then decode one more token
static bool run_shifted(llama_context * ctx, const std::vector<llama_token> & tokens, std::vector<float> & logits) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    auto decode = [&](int first, int last, int pos0, bool want_logits) {
        common_batch_clear(batch);
        for (int i = first; i < last; ++i) {
            common_batch_add(batch, tokens[i], pos0 + i - first, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = want_logits;

        return llama_decode(ctx, batch) == 0;
    };

    bool ok = decode(0, n_tokens, 0, false);

    if (ok) {
        llama_memory_t mem = llama_get_memory(ctx);

        llama_memory_seq_rm (mem, 0, 0, n_shift);
        llama_memory_seq_add(mem, 0, n_shift, -1, -n_shift);

        ok = decode(n_tokens, n_tokens + 1, n_tokens - n_shift, true);
    }

    if (ok) {
        logits.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    }

    llama_batch_free(batch);

    return ok;
}

// the same suffix, decoded at the shifted positions from the start
static bool run_rebuilt(llama_context * ctx, const std::vector<llama_token> & tokens, std::vector<float> & logits) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    common_batch_clear(batch);
    for (int i = n_shift; i < n_tokens + 1; ++i) {
        common_batch_add(batch, tokens[i], i - n_shift, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    const bool ok = llama_decode(ctx, batch) == 0;

    if (ok) {
        logits.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    }

    llama_batch_free(batch);

    return ok;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float res = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        res = std::max(res, std::fabs(a[i] - b[i]));
    }

    return res;
}

static int arg_max(const std::vector<float> & v) {
    int res = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] > v[res]) {
            res = (int) i;
        }
    }

    return res;
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified    = true;
    params.n_ctx         = 512;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();

    if (model == nullptr || llama_init->context() == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    const std::vector<llama_token> tokens = make_tokens(llama_vocab_n_tokens(llama_model_get_vocab(model)));

    // the pad is read when the cache is built, so each layout needs its own context
    std::vector<float> logits_pad[2];
    std::vector<float> logits_rebuilt;

    for (int arm = 0; arm < 2; ++arm) {
        setenv("LLAMA_KV_ROW_PAD", arm == 0 ? "0" : "16", 1);

        llama_context * ctx = llama_init_from_model(model, common_context_params_to_llama(params));
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to create the context\n", __func__);
            return 1;
        }

        bool ok = run_shifted(ctx, tokens, logits_pad[arm]);

        if (ok && arm == 0) {
            ok = run_rebuilt(ctx, tokens, logits_rebuilt);
        }

        llama_free(ctx);

        if (!ok) {
            fprintf(stderr, "%s : failed to decode, arm %d\n", __func__, arm);
            return 1;
        }
    }

    const float diff_pad     = max_diff(logits_pad[0], logits_pad[1]);
    const float diff_rebuilt = max_diff(logits_pad[0], logits_rebuilt);

    fprintf(stderr, "%s : pad off vs on: top token %d vs %d, max logit diff %.4f\n",
            __func__, arg_max(logits_pad[0]), arg_max(logits_pad[1]), diff_pad);
    fprintf(stderr, "%s : shifted vs rebuilt: max logit diff %.4f\n", __func__, diff_rebuilt);

    // the two layouts hold the same values, so only the order of the reductions can differ
    if (arg_max(logits_pad[0]) != arg_max(logits_pad[1]) || diff_pad > 0.05f) {
        fprintf(stderr, "%s : FAILED, the KV row pad changed the result of a context shift\n", __func__);
        return 1;
    }

    // a shift that walks the cache with a wrong stride reads unrelated tokens and lands far outside
    if (diff_rebuilt > 10.0f) {
        fprintf(stderr, "%s : FAILED, the shifted cache does not resemble the rebuilt one\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : OK\n", __func__);

    return 0;
}
