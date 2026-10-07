#include <sys/resource.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <thread>
#include <atomic>
#include <cstring>
#include <numeric>
#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

// A shared-head draft (Unsloth "shared" MTP files, converter --mtp-shared-embd) ships without
// token_embd / output and borrows them from the target it drafts for, the eagle3 pattern.
static const llama_model & qwen4exp_shared_model(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("QWEN4EXP MTP: this draft head has no '%s' of its own; "
                                        "load it as a draft of its target model (-md), not on its own", name));
    }
    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("QWEN4EXP MTP: draft and target disagree on the shape of '%s'", name));
    }
    return other;
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // NextN/MTP draft head, the deepseek4 pattern: the KV is optional and a missing tensor
    // downgrades it, so mainline GGUFs (whose converter drops the head) load unchanged and
    // a sidecar drafter (mtp-only file, e.g. blk.48 for the 48-layer model) is recognised.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn > 0 && hparams.n_layer_nextn < hparams.n_layer_all) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
            hparams.n_layer_nextn = 0;
        }
    }
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);


    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;


    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            GGML_ASSERT(il < hparams.n_layer_all);
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        GGML_ASSERT(hparams.ple_ngram_size >= 2 && hparams.ple_ngram_size <= LLAMA_MAX_PLE_NGRAM);
        GGML_ASSERT(hparams.ple_n_heads > 0 && hparams.ple_n_heads <= LLAMA_MAX_PLE_HEADS);

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file writes the head ranges as uint64 arrays, so read them at that width and
        // narrow; hparams keeps them at the int32 width the row gather actually uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            GGML_ASSERT(head_offsets[h] + head_vocab_sizes[h] <= INT32_MAX &&
                        "PLE head range does not fit the int32 row index");
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // A sidecar drafter file carries ONLY the NextN block plus the shared head/embedding
    // (deepseek4's DSpark shape): trunk tensors become optional there, and the NextN tensors
    // load only when the context asked for them.
    const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr || ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only    ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    // A sidecar may also leave these out (shared-head files) and borrow them from the target.
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);

    // there is no output_norm: the final hyper-connection mixer carries it. Older sidecars
    // carry the draft head's own mixer under these names; newer ones use blk.N.nextn.hc_head_*.
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    // tie_word_embeddings is false here: never tie to a token_embd a borrowing draft lacks
    if (output == NULL && tok_embd != NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto * ple_w = ml.get_weight(ple_name.c_str());
        if (ple_w != nullptr) {
            const int64_t ple_rows = ple_w->tensor->ne[1];
            // the PLE/engram table is gathered 16 random rows per token and never read densely, so
            // it wants MADV_RANDOM and no eager pull-in. See --tensor-read-lazy.
            per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                               { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
        } else {
            // per-head split of the same table (see llama_model::per_layer_tok_embd_h): head h holds
            // rows [ple_head_offsets[h], + ple_head_vocab_sizes[h]) of the single tensor
            for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
                const std::string suffix = "h" + std::to_string(h) + ".weight";
                per_layer_tok_embd_h.push_back(create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, suffix.c_str()),
                        { hparams.ple_head_dim, (int64_t) hparams.ple_head_vocab_sizes[h] }, TENSOR_READ_LAZY | TENSOR_READ_LAZY_SMALL));
            }
            GGML_ASSERT(!per_layer_tok_embd_h.empty() && "qwen4exp is missing the PLE n-gram table");
        }
    }

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        const bool is_mtp_layer = il >= (int) n_layer_main;
        const int  flags        = is_mtp_layer ? mtp_flags : trunk_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        // the NextN block is always a full-attention QSA layer; is_recr() is derived from
        // full_attention_interval and would misclassify it
        if (is_mtp_layer || !hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);


            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (!is_mtp_layer && hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (is_mtp_layer) {
            // eh_proj fuses [enorm(embd(next_tok)) ; collapsed hidden] -> n_embd. Note hnorm is
            // hc-space (hc_dim), unlike deepseek4's: the draft head consumes the target's
            // 4-stream hyper-connection state, collapsed by the (shared) head mixer.
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);

            // the draft head's own head mixer (upstream #28243 naming); absent in older sidecars,
            // which store the same weights as output_hc_* above
            layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { hc_dim }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags | TENSOR_NOT_REQUIRED);

            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
// Hyper-connection fast path (inject mat-vec, hc_post combine, no stream copy). Measured 2026-09-13
// on Vulkan (Q3KEXP ub512 d0): the glue was 22% of prefill GPU time, the w_inject matmul alone 4.8%,
// and the fast path is +8.9% prefill; see ~/strix-results/pwilkin-review-20260913/REVIEW.md.
// The HIP backend (halo-box tree) fuses the stock chain instead (grouped inject/down matvec,
// combine+norm kernel), so the default is on only when no CUDA/HIP device is in the model's
// device list. LLAMA_HC_FASTPATH=1/0 forces it either way.
// LLAMA_HC_NORM3D=0 restores the reshape-then-mul order of the hc norm (unfusable on Vulkan).
static bool qwen4exp_hc_norm3d() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_HC_NORM3D");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// LLAMA_HC_XN16=0 keeps the hc norm output in f32 (default: cast to f16 right after the norm; Vulkan
// fuses RMS_NORM+MUL+CPY into one kernel and the two consumers, the down GEMM and the inject mat-vec,
// take f16 B/A directly, so the GEMM's own 84 MB -> 42 MB conversion pass per mix disappears).
static bool qwen4exp_hc_xn16() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_HC_XN16");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// A bf16 weight cannot take an f16 B operand anywhere: ggml-vulkan's supports_op refuses the
// bf16 x f16 pair outright ("we currently don't have a bf16 x f16 shader, or an fp16->bf16 copy
// shader") and the CPU mul_mat converts its B operand from f32, so such a node gets NO backend and
// ggml_backend_sched_split_graph aborts on it. The f16 mixed/normed streams below feed exactly the
// matmuls listed in build_hc_mix, so if any of those weights is bf16 the whole layer keeps f32.
// Reachable with a stock community file, not just a hand-built one: AnonimousA's Flash-Next
// REAP-320 GGUF ships blk.N.indexer.{q,k}_proj.weight in bf16 (2026-09-15).
// Model-wide rather than per-layer: build_hc_mix is also called for the MTP draft head with
// il = -1 (it mixes with model.hc_head_*, not a layer's weights), so there is no layer to ask.
// Answering once for the whole model is also the conservative direction - a file with bf16 in
// only some layers keeps f32 everywhere rather than half the graph.
static bool qwen4exp_takes_f16_b(const llama_model & model) {
    for (const auto & layer : model.layers) {
        // the QSA indexer projections are not listed: qwen4exp_indexer_in gives a bf16 one an f32 B
        const ggml_tensor * consumers[] = {
            layer.wqkv, layer.wqkv_gate, layer.wq, layer.wk, layer.wv,  // attention / GDN in-projections
            layer.ffn_gate_inp,                              // MoE router
            layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_gate_up_exps, layer.ffn_down_exps,
        };
        for (const ggml_tensor * w : consumers) {
            if (w != nullptr && w->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }
    return true;
}

// Community Flash-Next GGUFs ship the indexer projections in bf16, and no backend takes bf16 x f16. Only
// those matmuls get an f32 copy of their input, so the rest of the graph keeps the f16 hc chain.
static ggml_tensor * qwen4exp_indexer_in(ggml_context * ctx, const ggml_tensor * w, ggml_tensor * cur) {
    return w->type == GGML_TYPE_BF16 && cur->type == GGML_TYPE_F16 ? ggml_cast(ctx, cur, GGML_TYPE_F32) : cur;
}

// LLAMA_HC_GATE16=0 keeps the hc gate logits in f32 (default: the up-GEMM writes f16 and the mix reads f16;
// needs the xn16 path, since the f16-gate mix kernel reads f16 xn and writes f16).
static bool qwen4exp_hc_gate16() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_HC_GATE16");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// LLAMA_HC_MIXOP=0 restores the unfused mix collapse (sigmoid, mul, hc-1 adds, scale).
static bool qwen4exp_hc_mixop() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_HC_MIXOP");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// LLAMA_HC_POST_GATE=0 restores the old combine node order. Default: the combine emits its scatter-weight
// chain (scale -> sigmoid -> scale) directly in front of the DSV4_HC_POST that reads it, with the block
// output, the residual and the identity comb expanded first, so a backend can fold the chain into the
// hc_post kernel (Vulkan HC_POST_GATE; the idea of upstream #29520). The identity comb is built once per
// graph and token count instead of once per combine (upstream #28901 drops it for the same reason).
static bool qwen4exp_hc_post_gate() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_HC_POST_GATE");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// Keep `t` allocated until `node` has been allocated, by listing it as an extra source of `node` that no
// backend reads (ggml-alloc frees a tensor after its last consumer). A consumer-less keep-alive view instead
// never releases its source: it pins the tensor to the end of the graph.
static void qwen4exp_keep_until(ggml_tensor * node, ggml_tensor * t) {
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (node->src[s] == t) {
            return;
        }
        if (node->src[s] == nullptr) {
            node->src[s] = t;
            return;
        }
    }
    GGML_ABORT("no free source slot");
}

// LLAMA_HC_KEEP_SRCS=1 keeps the inputs of a combine (block output, residual, inject weights) allocated until
// the next mix's f16 norm cast is. Their last reader is the DSV4_HC_POST that Vulkan fuses with that norm
// (HC_POST_NORM_CPY), so ggml-alloc was free to put the f16 norm output on their bytes; the fused kernel would
// then overwrite rows other workgroups have not read yet, and the overlap check dropped the fusion on 32 of 96
// combines at d0 and 50 at d16384 (REAP-320 ub2048). The hc gate cast keeps the low-rank input the same way
// instead of through a consumer-less view, which pinned it to the end of the graph.
static bool qwen4exp_hc_keep_srcs() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_HC_KEEP_SRCS");   // default on (2026-10-05); =0 restores the keep-alive view
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

static bool qwen4exp_hc_fastpath(const llama_model & model) {
    static const bool on = [&]() {
        if (const char * e = getenv("LLAMA_HC_FASTPATH")) {
            return atoi(e) != 0;
        }
        for (const auto & d : model.devices) {
            if (d.dev == nullptr) {
                continue;
            }
            const std::string reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d.dev));
            if (reg == "CUDA" || reg == "ROCm" || reg == "HIP") {
                return false;
            }
        }
        return true;
    }();
    return on;
}

// LLAMA_HC_UP_IL=1 reorders the rows of every hc up-projection weight once, on the first graph build, from
// stream-major (row c*n_embd + i) to hc-interleaved (row i*hc + c), and build_hc_mix tells DSV4_HC_MIX so.
// Each gate logit is the same row dot product as before, so the mix is the same bits; what changes is that a
// GEMM row tile of the up-projection then holds whole elements (all hc streams of each), which lets the Vulkan
// backend apply the mix in the up-GEMM epilogue (HC_UP_MIX) instead of writing and re-reading the
// [n_embd*hc, nt] gate: 84 MB of traffic per mix at ub2048, 95 mixes per graph. Needs the mix op path and
// every hc up weight in a device buffer that is not a host mapping (the rows are rewritten in place);
// otherwise it stays off. Default off.
// LLAMA_HC_DOWN_INJECT=1: the hc inject projection (f32 [hc_dim, hc]) reads xn a second time through a mat-vec
// (Flash-Next REAP-320 pp2048 ub2048: 95 calls, 42.8 ms per ubatch at ~94 GB/s). Appending its rows, quantized to
// q8_0, to the q8_0 down weight makes the down GEMM produce both: m 320 -> 324 fits the same 128-row tiles. The down
// rows and their outputs are unchanged; the inject weights are q8_0-rounded (and multiply the f16 xn like the down
// rows do), so the scatter weights differ at that rounding. Built once, on the first build with loaded weights.
ggml_tensor * llama_model_qwen4exp::hc_down_inject(const ggml_tensor * w_down) const {
    std::lock_guard<std::mutex> lock(hc_dinj.mutex);
    if (!hc_dinj.tried) {
        [this] {
            const char * e = getenv("LLAMA_HC_DOWN_INJECT");
            if (e == nullptr || atoi(e) == 0) {
                hc_dinj.tried = true;
                return;
            }
            std::vector<std::pair<ggml_tensor *, ggml_tensor *>> pairs;
            for (const auto & layer : layers) {
                if (layer.hc_attn_down && layer.hc_attn_inject) { pairs.emplace_back(layer.hc_attn_down, layer.hc_attn_inject); }
                if (layer.hc_ffn_down  && layer.hc_ffn_inject)  { pairs.emplace_back(layer.hc_ffn_down,  layer.hc_ffn_inject);  }
            }
            for (const auto & pr : pairs) {
                if (pr.first->data == nullptr || pr.second->data == nullptr) {
                    return;   // a no-alloc model (the -fit probe): decide on a later build, with the weights loaded
                }
            }
            hc_dinj.tried = true;
            if (pairs.empty()) {
                return;
            }
            for (const auto & pr : pairs) {
                const ggml_tensor * d = pr.first, * inj = pr.second;
                if (d->type != GGML_TYPE_Q8_0 || inj->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || !ggml_is_contiguous(inj) ||
                    d->ne[0] != inj->ne[0] || d->ne[2] != 1 || inj->ne[2] != 1 || d->buffer == nullptr) {
                    LLAMA_LOG_WARN("%s: LLAMA_HC_DOWN_INJECT: %s / %s are not q8_0 / f32 [hc_dim, *] weights, keeping the inject mat-vec\n",
                            __func__, ggml_get_name(d), ggml_get_name(inj));
                    return;
                }
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(pairs[0].first->buffer));
            if (dev == nullptr) {
                return;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
            ggml_init_params ip = { pairs.size() * ggml_tensor_overhead(), nullptr, true };
            hc_dinj.ctx = ggml_init(ip);
            std::vector<ggml_tensor *> out;
            for (const auto & pr : pairs) {
                out.push_back(ggml_new_tensor_2d(hc_dinj.ctx, GGML_TYPE_Q8_0, pr.first->ne[0], pr.first->ne[1] + pr.second->ne[1]));
            }
            hc_dinj.buf = ggml_backend_alloc_ctx_tensors_from_buft(hc_dinj.ctx, buft);
            if (hc_dinj.buf == nullptr) {
                LLAMA_LOG_WARN("%s: LLAMA_HC_DOWN_INJECT: buffer allocation failed, keeping the inject mat-vec\n", __func__);
                ggml_free(hc_dinj.ctx);
                hc_dinj.ctx = nullptr;
                return;
            }
            ggml_backend_buffer_set_usage(hc_dinj.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<uint8_t> dbytes, qbytes;
            std::vector<float>   inj;
            for (size_t i = 0; i < pairs.size(); ++i) {
                const ggml_tensor * d = pairs[i].first, * w = pairs[i].second;
                dbytes.resize(ggml_nbytes(d));
                ggml_backend_tensor_get(d, dbytes.data(), 0, dbytes.size());
                inj.resize(ggml_nelements(w));
                ggml_backend_tensor_get(w, inj.data(), 0, inj.size() * sizeof(float));
                qbytes.resize(ggml_row_size(GGML_TYPE_Q8_0, w->ne[0]) * w->ne[1]);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, inj.data(), qbytes.data(), 0, w->ne[1], w->ne[0], nullptr);
                ggml_backend_tensor_set(out[i], dbytes.data(), 0, dbytes.size());
                ggml_backend_tensor_set(out[i], qbytes.data(), dbytes.size(), qbytes.size());
                hc_dinj.merged[d] = out[i];
            }
            LLAMA_LOG_INFO("%s: LLAMA_HC_DOWN_INJECT: %zu hc down weights merged with their inject rows (%.1f MiB, %s)\n", __func__,
                    pairs.size(), ggml_backend_buffer_get_size(hc_dinj.buf) / 1048576.0, ggml_backend_buffer_name(hc_dinj.buf));
        }();
    }
    auto it = hc_dinj.merged.find(w_down);
    return it == hc_dinj.merged.end() ? nullptr : it->second;
}

bool llama_model_qwen4exp::hc_up_interleaved() const {
    std::lock_guard<std::mutex> lock(hc_up_il_mutex);
    if (hc_up_il_tried) {
        return hc_up_il;
    }
    [this] {
        // default OFF again (2026-10-06): with the HC_UP_MIX epilogue the cached multi-turn path and a full re-prefill
        // diverged beyond the release gate's tie threshold (FN REAP-320, Mesa 26.2.4 and mesa-main, deterministic); =1 enables
        const char * e = getenv("LLAMA_HC_UP_IL");
        if (e == nullptr || atoi(e) == 0 || !qwen4exp_hc_fastpath(*this) || !qwen4exp_hc_mixop()) {
            hc_up_il_tried = true;
            return;
        }
        const int64_t hc     = hparams.dsv4_hc_mult;
        const int64_t n_embd = hparams.n_embd;
        std::vector<ggml_tensor *> ws;
        for (const auto & layer : layers) {
            for (ggml_tensor * w : { layer.hc_attn_up, layer.hc_ffn_up, layer.nextn.hc_head_up }) {
                if (w != nullptr) {
                    ws.push_back(w);
                }
            }
        }
        if (hc_head_up != nullptr) {
            ws.push_back(hc_head_up);
        }
        for (const ggml_tensor * w : ws) {
            if (w->data == nullptr) {
                return;   // a no-alloc model (the -fit probe): decide on a later build, with the weights loaded
            }
        }
        hc_up_il_tried = true;
        for (const ggml_tensor * w : ws) {
            ggml_backend_buffer_t buf = w->buffer;
            ggml_backend_dev_t    dev = buf ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf)) : nullptr;
            ggml_backend_dev_props props;
            if (dev != nullptr) {
                ggml_backend_dev_get_props(dev, &props);
            }
            if (buf == nullptr || ggml_backend_buffer_is_host(buf) || dev == nullptr || props.caps.buffer_from_host_ptr ||
                !ggml_is_contiguous(w) || w->ne[1] != hc * n_embd || w->ne[2] != 1 || w->ne[3] != 1) {
                LLAMA_LOG_WARN("%s: LLAMA_HC_UP_IL: %s is not a contiguous device-resident [*, %" PRId64 "] weight, keeping the stream-major layout\n",
                        __func__, ggml_get_name(w), hc * n_embd);
                return;
            }
        }
        std::vector<uint8_t> src, dst;
        for (ggml_tensor * w : ws) {
            const size_t row = w->nb[1];
            src.resize(ggml_nbytes(w));
            dst.resize(ggml_nbytes(w));
            ggml_backend_tensor_get(w, src.data(), 0, src.size());
            for (int64_t c = 0; c < hc; ++c) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    memcpy(dst.data() + (i*hc + c)*row, src.data() + (c*n_embd + i)*row, row);
                }
            }
            ggml_backend_tensor_set(w, dst.data(), 0, dst.size());
        }
        hc_up_il = true;
        LLAMA_LOG_INFO("%s: LLAMA_HC_UP_IL: %zu hc up weights reordered to hc-interleaved rows\n", __func__, ws.size());
    }();
    return hc_up_il;
}

// w_inject is [hc_dim, hc]: hc (4) output rows. ggml_mul_mat(w_inject, xn) is a quantized GEMM with a
// 4-row A operand, which the Vulkan coopmat path pads to a full tile: 1.05 ms per call at 512 tokens
// (40 GFLOPS), 48 calls per graph. Swapping the operands makes it a mat-vec over xn with hc columns,
// which wants the hc weight rows in f32; get_rows dequantises them (Vulkan CPY cannot cast K-quants),
// and the row index is this input so it lives on the backend with the rest of the graph inputs.
// The combine, cur = residual + block_out (x) w, is ggml_dsv4_hc_post with an identity mixing matrix:
// one pass over the residual instead of repeat_4d + mul + add (six full-width passes). The identity
// is the second constant here; both are graph inputs so they live with the other inputs on the
// backend and are written once per ubatch (hc ints and hc*hc floats).
class llama_model_qwen4exp::llm_graph_input_hc_consts : public llm_graph_input_i {
public:
    explicit llm_graph_input_hc_consts(int64_t hc) : hc(hc) {}
    virtual ~llm_graph_input_hc_consts() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        // either constant can be dead in a trimmed graph (the MTP KV-only catch-up), and a tensor
        // no node reads is never allocated
        if (iota && iota->buffer) {
            std::vector<int32_t> v(hc);
            for (int64_t i = 0; i < hc; ++i) {
                v[i] = (int32_t) i;
            }
            ggml_backend_tensor_set(iota, v.data(), 0, hc*sizeof(int32_t));
        }

        if (eye && eye->buffer) {
            std::vector<float> e(hc*hc, 0.0f);
            for (int64_t i = 0; i < hc; ++i) {
                e[i*hc + i] = 1.0f;
            }
            ggml_backend_tensor_set(eye, e.data(), 0, hc*hc*sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return true;
    }

    ggml_tensor * iota = nullptr;   // I32 [hc]
    ggml_tensor * eye  = nullptr;   // F32 [hc, hc]
    const int64_t hc;
};

llama_model_qwen4exp::llm_graph_input_hc_consts * llama_model_qwen4exp::graph::build_hc_consts() {
    if (hc_consts == nullptr) {
        const int64_t hc = hparams.dsv4_hc_mult;
        auto in = std::make_unique<llm_graph_input_hc_consts>(hc);
        in->iota = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hc);
        in->eye  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc, hc);
        ggml_set_input(in->iota);
        ggml_set_input(in->eye);
        hc_consts = in.get();
        res->add_input(std::move(in));
    }
    return hc_consts;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    // LLAMA_HC_XN_PAD=N (f16 elements, multiple of 8): give the f16 xn rows a stride of hc_dim + N. hc_dim 10240 f16 is
    // 80 x 256 B, so unpadded every token row starts on the same DRAM channel of 16 for the down GEMM, the inject
    // mat-vec and the mix, which all read xn row-strided (Strata pads the same buffer: K + 64). The norm writes the
    // padded rows directly (RMS_NORM_MUL_CPY with a row-strided destination); the consumers take the stride.
    static const int64_t xn_pad = [] { const char * e = getenv("LLAMA_HC_XN_PAD"); return e ? (int64_t) (atoi(e) & ~7) : 0; }();
    bool xn_padded = false;
    if (qwen4exp_hc_norm3d()) {
        // Apply gamma in the [n_embd, hc, nt] shape so the graph is RMS_NORM directly followed by MUL
        // and the backend fuses them (Vulkan RMS_NORM_MUL); a RESHAPE node between the two blocks the
        // pattern and costs a full 84 MB pass per mix at ub2048. The [hc_dim] gamma becomes an
        // [n_embd, hc] view, expanded into the graph here so its RESHAPE node lands before the norm.
        ggml_tensor * wn = ggml_reshape_2d(ctx0, w_norm, n_embd, hc);
        ggml_build_forward_expand(gf, wn);
        xn = ggml_mul(ctx0, xn, wn);
        if (qwen4exp_hc_fastpath(model) && qwen4exp_hc_mixop() && qwen4exp_hc_xn16() && loras->empty() && nt >= 32 &&
            qwen4exp_takes_f16_b(model)) {
            // before the reshape: the cast must directly follow the MUL for the backend fusion.
            // prefill only (nt >= 32): at decode the f16-B mat-vec paths are slower than the f32 ones
            // and there is no conversion pass to save (2026-09-14: tg 27.2 -> 25.2 with the cast at N=1)
            if (xn_pad > 0) {
                ggml_tensor * buf = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, hc_dim + xn_pad, nt);
                ggml_tensor * dst = ggml_view_3d(ctx0, buf, n_embd, hc, nt,
                        n_embd * ggml_element_size(buf), buf->nb[1], 0);
                xn = ggml_cpy(ctx0, xn, dst);
                xn_padded = true;
            } else {
                xn = ggml_cast(ctx0, xn, GGML_TYPE_F16);
            }
            if (x->op == GGML_OP_DSV4_HC_POST && qwen4exp_hc_keep_srcs()) {
                // the combine's inputs stay allocated until the cast is (see qwen4exp_hc_keep_srcs)
                for (int s = 0; s < 3; ++s) {
                    qwen4exp_keep_until(xn, x->src[s]);
                }
            }
        }
        // a view of the copy (not of the buffer) so every consumer depends on the write
        xn = xn_padded ? ggml_view_2d(ctx0, xn, hc_dim, nt, xn->nb[2], 0) : ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    } else {
        xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
        xn = ggml_mul(ctx0, xn, w_norm);
    }
    cb(xn, "hc_norm", il);

    // LLAMA_HC_DOWN_INJECT: one GEMM for the down rows and the inject rows (prefill, no LoRA)
    ggml_tensor * w_dinj = (inject && qwen4exp_hc_fastpath(model) && loras->empty() && nt >= 32)
            ? static_cast<const llama_model_qwen4exp &>(model).hc_down_inject(w_down) : nullptr;
    ggml_tensor * dinj = nullptr;
    ggml_tensor * lo;
    if (w_dinj) {
        dinj = ggml_mul_mat(ctx0, w_dinj, xn);   // [n_down + hc, nt]
        // the down rows are a strided view (row stride n_down + hc): SCALE needs a contiguous source on Vulkan (and
        // the CPU fallback asserts), so copy the 2.6 MB out first
        lo = ggml_cont(ctx0, ggml_view_2d(ctx0, dinj, w_down->ne[1], nt, dinj->nb[1], 0));
    } else {
        lo = build_lora_mm(w_down, xn);
    }
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate_logits = build_lora_mm(w_up, lo);
    if (qwen4exp_hc_fastpath(model) && qwen4exp_hc_mixop() && qwen4exp_hc_xn16() && qwen4exp_hc_gate16() && loras->empty() && nt >= 32 &&
        qwen4exp_takes_f16_b(model) &&                 // no bf16 consumer: there is no bf16 x f16 shader
        llm_graph_weights_on_gpu(w_up->buffer)) {      // and a CPU matmul cannot take the f16 gate downstream
        // f16 gate logits: the up-GEMM writes its result as f16 through the MUL_MAT+CPY(f16) fusion and
        // DSV4_HC_MIX reads the f16 gate; the 84 MB f32 gate tensor is neither written nor read (prefill only)
        gate_logits = ggml_cast(ctx0, gate_logits, GGML_TYPE_F16);
        // the low-rank input stays referenced past the cast: the fused matmul reads it while it writes the
        // cast's destination, so the allocator must not place the two on the same bytes
        if (qwen4exp_hc_keep_srcs()) {
            qwen4exp_keep_until(gate_logits, lo);
        } else {
            ggml_build_forward_expand(gf, ggml_view_1d(ctx0, lo, 1, 0));
        }
    }

    ggml_tensor * mixed;
    if (qwen4exp_hc_fastpath(model) && qwen4exp_hc_mixop()) {
        // sigmoid gate, per-stream product and the mean over streams in one pass (DSV4_HC_MIX):
        // the unfused chain was five full-width passes per mix, 2.2 ms of 4.7 at ub2048
        // f16 result at prefill: every consumer of the mixed stream is a matmul B operand (attention and
        // GDN in-projections, indexer projections, MoE router and experts), so the GEMMs' own f32->f16
        // conversion passes disappear and the write halves; f32 at decode (f16-B mat-vec paths are slower)
        const ggml_type mix_type = (qwen4exp_hc_xn16() && nt >= 32 && qwen4exp_takes_f16_b(model))
                                   ? GGML_TYPE_F16 : GGML_TYPE_F32;
        const bool gate_il = static_cast<const llama_model_qwen4exp &>(model).hc_up_interleaved();
        if (gate_il) {
            for (const auto & lora : *loras) {
                if (lora.first->get_weight(w_up) != nullptr) {
                    GGML_ABORT("LLAMA_HC_UP_IL: a LoRA adapter on %s would add a stream-major delta to the interleaved gate", ggml_get_name(w_up));
                }
            }
        }
        ggml_tensor * xn3 = xn_padded ? ggml_view_3d(ctx0, xn, n_embd, hc, nt, n_embd * ggml_element_size(xn), xn->nb[1], 0)
                                      : ggml_reshape_3d(ctx0, xn, n_embd, hc, nt);
        mixed = ggml_dsv4_hc_mix_ext(ctx0, xn3, gate_logits, 1.0f / (float) hc, mix_type, gate_il);
        cb(mixed, "hc_mixed", il);
        if (gate_il && gate_logits->op == GGML_OP_CPY) {
            // HC_UP_MIX reads the low-rank input in the GEMM while its epilogue writes the mixed stream: keep lo
            // allocated until the mix is, so the allocator cannot hand the mixed stream lo's freed bytes
            qwen4exp_keep_until(mixed, lo);
        }
    } else {
        ggml_tensor * gate = ggml_sigmoid(ctx0, gate_logits);
        cb(gate, "hc_gate", il);

        ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        if (!qwen4exp_hc_fastpath(model)) {
            // ggml_add takes a strided src0 on every backend (the result is a fresh contiguous
            // tensor), so this copy of one stream per mix was a full-width pass for nothing
            mixed = ggml_cont(ctx0, mixed);
        }
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
        cb(mixed, "hc_mixed", il);
    }

    if (inject) {
        if (dinj) {
            *inject = ggml_cont(ctx0, ggml_view_2d(ctx0, dinj, hc, nt, dinj->nb[1], w_down->ne[1] * ggml_element_size(dinj)));   // [hc, nt]
        } else if (qwen4exp_hc_fastpath(model) && loras->empty()) {
            ggml_tensor * w_rows = ggml_get_rows(ctx0, w_inject, build_hc_consts()->iota); // f32 [hc_dim, hc]
            ggml_tensor * inj_t  = ggml_mul_mat(ctx0, xn, w_rows);              // [nt, hc]
            *inject = ggml_cont(ctx0, ggml_transpose(ctx0, inj_t));             // [hc, nt]
        } else {
            *inject = build_lora_mm(w_inject, xn);
        }
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // decode and verify-sized graphs only (nt < 32): at prefill the reorder moves the allocation, which moves two
    // allocation-dependent fusions (HC_POST_NORM_CPY 64 -> 66, TOPK_MOE 14 -> 15 of 96/47 on REAP-320 ub2048);
    // their kernels round differently from the unfused chains, so prefill was no longer bit-identical
    // (KLD 0.0118 at c4096x4, the same with the Vulkan gate fusion off) for a +1.3% pp2048 inside noise
    if (qwen4exp_hc_fastpath(model) && qwen4exp_hc_post_gate() && nt < 32 && inject->ne[0] == hc && inject->ne[1] == nt &&
        ggml_n_dims(inject) <= 2) {
        // the same combine as below, ordered for HC_POST_GATE: every other operand is in the graph before
        // the chain, and w needs no reshape, so scale, sigmoid, scale and hc_post are consecutive nodes
        ggml_tensor * x = ggml_reshape_2d(ctx0, block_out, n_embd, nt);
        ggml_tensor *& comb = hc_comb_by_nt[nt];
        if (comb == nullptr) {
            comb = ggml_repeat_4d(ctx0, build_hc_consts()->eye, hc, hc, nt, 1);
        }
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, x);
        ggml_build_forward_expand(gf, comb);
        ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
        w = ggml_scale(ctx0, w, 2.0f);
        ggml_tensor * cur = ggml_dsv4_hc_post(ctx0, x, residual, w, comb);
        cb(cur, "hc_combine", il);
        return cur;
    }

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);

    ggml_tensor * cur;
    if (qwen4exp_hc_fastpath(model)) {
        // cur[i, h, t] = block_out[i, t]*w[h, t] + sum_s residual[i, s, t]*I[h, s]: the DeepSeek V4
        // hc_post op with an identity mix. One kernel reading the residual once, instead of
        // repeat_4d + mul + add materialising hc copies of block_out (96 combines per graph).
        ggml_tensor * x    = ggml_reshape_2d(ctx0, block_out, n_embd, nt);
        ggml_tensor * comb = ggml_repeat_4d(ctx0, build_hc_consts()->eye, hc, hc, nt, 1);
        cur = ggml_dsv4_hc_post(ctx0, x, residual, ggml_reshape_2d(ctx0, w, hc, nt), comb);
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);
        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);
        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    // An MTP sidecar file carries only the NextN block; the trunk tensors are absent by design
    // and this graph cannot be built from it. Without this check the first hc_mix segfaults.
    if (model.hparams.n_layer_nextn > 0 && model.layers[0].hc_attn_norm == nullptr) {
        GGML_ABORT("this file is an MTP draft sidecar - load it as a draft model (-md) with --spec-type draft-mtp, not as a standalone model");
    }

    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the wide residual starts as hc identical copies of the embedding
    // set when the last layer gathered everything down to the output rows; the
    // post-stack gather and the h_nextn mask must then not gather a second time
    bool trimmed = false;

    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), mctx_hyb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids &&
            (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            // everything from here on is per token, so drop the rows that produce no output.
            // the reference port (upstream 6c84c7d5d) gathers here too; without it the last
            // layer's hc-combine/mix and MoE run over every ubatch row, and the output of a
            // request depends on how many requests the server has already served.
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
            trimmed = true;
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // hand the drafter the 4-stream residual, pre-collapse: nextn.hnorm is hc-space
    if (cparams.embeddings_nextn) {
        // export the REAL node, not a reshape view: the scheduler assigns no backend to a
        // naked view and the extraction then hits GGML_ASSERT(backend_h). Contiguous
        // [n_embd, hc, T] has the same memory layout as the flat rows the reader expects.
        ggml_tensor * h_nextn = res_hc;
        if (!trimmed && cparams.embeddings_nextn_masked && inp_out_ids) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, n_tokens);
            h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
        }
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
        ggml_build_forward_expand(gf, h_nextn);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    if (inp_out_ids && !trimmed) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu.
    // LLAMA_GDN_NORM_F16=1: emit RMS_NORM, MUL(gamma), SIGMOID(z), MUL, CPY(f16) back to back, which the Vulkan
    // backend runs as one pass (RMS_NORM_MUL_SIGMUL_CPY) writing only the f16 B operand of ssm_out: no sigmoid
    // pass, no f32 norm output and no B conversion in the GEMM (Strata's GDN_NOY). The GEMM's B was rounded to
    // f16 before either way. Needs a backend that takes f16 B for the out projection (not the CPU).
    static const bool norm_f16 = [] { const char * e = getenv("LLAMA_GDN_NORM_F16"); return e != nullptr && atoi(e) != 0; }();
    if (norm_f16) {
        ggml_build_forward_expand(gf, gate);   // keep gate's own view nodes ahead of the chain
        ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
        ggml_tensor * out = ggml_mul(ctx0, normalized, ggml_sigmoid(ctx0, gate));
        // LLAMA_GDN_NORM_PAD=N (default 128, 0 off): write the f16 rows with a stride of value_dim + N. value_dim 6144 f16 is
        // 48 x 256 B, so packed every token row starts on the same DRAM channel for the ssm_out GEMM (Flash-Next q8_0
        // 2560x2048x6144: 40.8 TFLOPS from the padded staging copy, 31.7 reading the packed f16 rows directly)
        static const int64_t pad = [] { const char * e = getenv("LLAMA_GDN_NORM_PAD"); return e ? (int64_t) (atoi(e) & ~7) : 128; }();
        const int64_t row = out->ne[0] * out->ne[1];
        if (pad > 0 && (row * 2) % 256 == 0 && std::gcd<int64_t>(row * 2 / 256, 16) >= 8 && out->ne[3] == 1) {
            ggml_tensor * buf = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, row + pad, out->ne[2]);
            ggml_tensor * dst = ggml_view_4d(ctx0, buf, out->ne[0], out->ne[1], out->ne[2], 1,
                    out->ne[0] * ggml_element_size(buf), buf->nb[1], buf->nb[1] * out->ne[2], 0);
            return ggml_cpy(ctx0, out, dst);
        }
        return ggml_cast(ctx0, out, GGML_TYPE_F16);
    }
    // The sigmoid is expanded first so the graph reads RMS_NORM, MUL(gamma), MUL(gate) back to back
    // and the Vulkan backend fuses the three (RMS_NORM_MUL_MUL) instead of a separate 50 MB pass.
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);
    ggml_build_forward_expand(gf, gated);
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);

    return ggml_mul(ctx0, normalized, gated);
}

// llama_kv_cache::get_n_kv pads n_kv to at least 256 cells, so n_kv grows in steps of 256 and
// the block count in steps of 256/ratio. The pooled-key window has to cover that jump.
static constexpr uint32_t QSA_N_PAD_KV = 256;

// Dense shortcut (LLAMA_QSA_DENSE_SHORTCUT=0 disables): when the cache holds no more cells than the
// budget, the top-k selects EVERY cell and sparse attention is the dense causal attention, exactly.
static bool qwen4exp_qsa_dense_shortcut() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_QSA_DENSE_SHORTCUT");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// Complete-block selection: pick indexer_top_k/ratio whole blocks that lie before the query's own block and
// append the query's own partial block as the tail (up to ratio-1 cells), -1 where nothing is visible. The
// selected rows then carry the visibility themselves, so attention runs without a mask. Only the selected-key
// attention kernels understand that layout (F16 K/V, head 256, single stream), so the selection needs
// selected_key_attn: every QSA layer's device took the probe op (llama_memory_hybrid_idx). The selection carries
// cells through f32, hence the 2^24 bound on n_kv.
static bool qwen4exp_use_block_selection(bool blk_bias, int64_t n_stream, int64_t ratio, int64_t n_kv,
        const llama_ubatch & ubatch, const llama_cparams & cparams, const llama_hparams & hparams,
        ggml_type type_k, ggml_type type_v, bool selected_key_attn) {
    return selected_key_attn && blk_bias && cparams.causal_attn && n_stream == 1 && ratio > 1 &&
        hparams.indexer_top_k % ratio == 0 &&
        n_kv > (int64_t) hparams.indexer_top_k + ratio - 1 && n_kv <= 16777216 && ubatch.token &&
        cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias == 0.0f &&
        !hparams.attn_soft_cap && hparams.n_embd_head_k() == 256 && hparams.n_embd_head_v() == 256 &&
        type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16;
}

// the selection decision for one ratio: the gate above, no SWA, and a cache whose visibility the selection
// encodes exactly (qsa_scalar_visibility: one-axis positions, no block split over sequence sets). Elsewhere the
// masked block-expanded top-k runs, whose set_rows would write row -1 for the selection's sentinels.
static bool qwen4exp_block_selection(const llama_memory_hybrid_idx_context * mctx, bool blk_bias, int64_t n_stream,
        int64_t ratio, int64_t n_kv, const llama_ubatch & ubatch, const llama_cparams & cparams,
        const llama_hparams & hparams) {
    const auto * attn = mctx->get_attn();
    return attn != nullptr && hparams.n_swa == 0 &&
        qwen4exp_use_block_selection(blk_bias, n_stream, ratio, n_kv, ubatch, cparams, hparams,
                attn->type_k(), attn->type_v(), mctx->qsa_selected_key_attn()) &&
        mctx->qsa_scalar_visibility(ubatch, (uint32_t) ratio);
}

// top block_budget blocks by score, in ascending block order, expanded to their cells (-1 for an invisible block:
// its score is -inf, so step(score + 1) is 0), followed by the tail cells: [block_budget*r + r-1, n_query, 1, n_stream]
static ggml_tensor * qwen4exp_select_complete_blocks(ggml_context * ctx0, ggml_tensor * score,
        ggml_tensor * block_cells, ggml_tensor * tail, int64_t block_budget, int64_t r) {
    const int64_t n_blocks = score->ne[0], n_query = score->ne[1], n_stream = score->ne[2];
    GGML_ASSERT(score->type == GGML_TYPE_F32 && ggml_is_contiguous(score));
    GGML_ASSERT(block_cells->type == GGML_TYPE_I32 && block_cells->ne[0] == r*n_blocks && block_cells->ne[1] == n_stream);
    GGML_ASSERT(tail->type == GGML_TYPE_I32 && tail->ne[0] == r-1 && tail->ne[1] == n_query && tail->ne[3] == n_stream);
    GGML_ASSERT(n_blocks >= block_budget);
    ggml_tensor * blocks = ggml_cont(ctx0, ggml_top_k(ctx0, score, block_budget));
    ggml_tensor * order  = ggml_argsort(ctx0, ggml_cast(ctx0, blocks, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
    ggml_tensor * blocks_view = ggml_view_4d(ctx0, blocks, 1, block_budget, n_query, n_stream,
            blocks->nb[0], blocks->nb[1], blocks->nb[2], 0);
    blocks = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, blocks_view, order), block_budget, n_query, n_stream);
    ggml_tensor * score_view = ggml_view_4d(ctx0, score, 1, n_blocks, n_query, n_stream,
            score->nb[0], score->nb[1], score->nb[2], 0);
    ggml_tensor * picked_score = ggml_get_rows(ctx0, score_view, blocks);
    ggml_tensor * valid = ggml_step(ctx0, ggml_scale_bias(ctx0, picked_score, 1.0f, 1.0f));
    ggml_tensor * cells_view  = ggml_reshape_3d(ctx0, block_cells, r, n_blocks, n_stream);
    ggml_tensor * flat_blocks = ggml_reshape_2d(ctx0, blocks, block_budget*n_query, n_stream);
    ggml_tensor * cells = ggml_get_rows(ctx0, cells_view, flat_blocks);
    cells = ggml_reshape_4d(ctx0, cells, r, block_budget, n_query, n_stream);
    cells = ggml_scale_bias(ctx0, ggml_cast(ctx0, cells, GGML_TYPE_F32), 1.0f, 1.0f);
    cells = ggml_mul(ctx0, cells, valid);
    cells = ggml_cast(ctx0, ggml_scale_bias(ctx0, cells, 1.0f, -1.0f), GGML_TYPE_I32);
    cells = ggml_reshape_4d(ctx0, cells, block_budget*r, n_query, 1, n_stream);
    return ggml_concat(ctx0, cells, tail, 0);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, each scored by one
// mean-pooled indexer key, plus the incomplete tail. set_input resolves the cache layout.
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, uint32_t top_k,
            bool shortcut, bool maskless) :
        mctx(mctx), ratio(ratio), top_k(top_k), shortcut(shortcut), maskless(maskless), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias,
                            pool_idxs, pool_cells, pool_pos, tail_idxs, ubatch, ratio, blk_bias);
    }

    bool can_reuse(const llm_graph_params & params) override;

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]
    // the pooled-key window; null when every block is recomputed inline
    ggml_tensor * pool_idxs  = nullptr;  // I64 [n_recomp]
    ggml_tensor * pool_cells = nullptr;  // I32 [ratio*n_recomp]
    ggml_tensor * pool_pos   = nullptr;  // I32 [4*n_recomp]
    // complete-block selection only: the query's own partial block, -1 padded
    ggml_tensor * tail_idxs  = nullptr;  // I32 [ratio-1, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;
    const uint32_t top_k;
    const bool     shortcut;   // built without cell_blk / bias: the dense shortcut was taken
    // complete-block selection (qwen4exp_block_selection): no cell_blk, blk_cells always on the graph, and
    // attention without a mask, the selected rows carry the visibility
    const bool     maskless;
    int64_t        n_kv_built = 0;   // cell_blk pins n_kv otherwise

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;
};

// Without this the base class returns false and the whole graph is rebuilt every token.
// n_kv, n_stream and n_tokens pin every shape here: n_blocks is ceil(n_kv/ratio), the top-k width
// is min(n_kv, indexer_top_k + ratio - 1), and ratio is fixed per layer. blk_bias is pinned too --
// it turns on the kq_mask matching those same three, and causal_attn / use_alibi are fixed for the
// context. n_kv is padded, so this holds between padding steps. set_input still runs on the reuse
// path, so only topology is certified here.
bool llama_model_qwen4exp::llm_graph_input_qsa::can_reuse(const llm_graph_params & params) {
    const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

    this->mctx = m;

    const llama_kv_cache_context * midx = m->get_idx();
    if (midx == nullptr) {
        return false;
    }

    const int64_t n_kv     = midx->get_n_kv();
    const int64_t n_stream = m->get_n_stream();
    const int64_t n_tokens = params.ubatch.n_tokens;

    if (n_stream <= 0 || n_tokens % n_stream != 0) {
        return false;
    }

    const int64_t n_blocks = (n_kv + (int64_t) ratio - 1)/(int64_t) ratio;

    bool res = true;
    res &= k_idxs   != nullptr && k_idxs->buffer   != nullptr && k_idxs->ne[0]   == n_tokens;
    // a shortcut graph has no cell_blk / bias (nothing reads them); it can only be reused while the
    // shortcut decision holds for the new n_kv, and a scoring graph only while it does not
    const bool want_shortcut = qwen4exp_qsa_dense_shortcut() && n_kv <= (int64_t) top_k + (int64_t) ratio - 1;
    res &= shortcut == want_shortcut;
    // the selection mode is baked into the graph (no mask, a different top-k), so it must hold for the new ubatch
    const bool want_maskless = !want_shortcut &&
        qwen4exp_block_selection(m, blk_bias, n_stream, ratio, n_kv, params.ubatch, params.cparams, params.hparams);
    res &= maskless == want_maskless;
    if (maskless) {
        res &= n_kv_built == n_kv;
        res &= blk_cells != nullptr && blk_cells->buffer != nullptr && blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= tail_idxs != nullptr && tail_idxs->buffer != nullptr && tail_idxs->ne[1] == n_tokens/n_stream;
    }
    if (!shortcut && !maskless) {
        res &= cell_blk != nullptr && cell_blk->buffer != nullptr && cell_blk->ne[0] == n_kv;
        res &= cell_blk != nullptr && cell_blk->ne[1] == n_stream;
    }
    if (!shortcut) {
        res &= bias     != nullptr && bias->buffer     != nullptr;
        res &= bias     != nullptr && bias->ne[0] == (blk_bias ? n_blocks : n_kv);
        res &= bias     != nullptr && bias->ne[1] == n_tokens/n_stream;
    }

    // the window is sized from whether the pooled cache was valid when the graph was built,
    // so a graph built over a valid cache must not be reused after something dropped it
    if (pool_cells != nullptr) {
        const int64_t want = (int64_t) ratio *
            m->qsa_pool_n_recomp(ratio, (uint32_t) n_tokens, (uint32_t) n_kv, QSA_N_PAD_KV);
        res &= pool_cells->buffer != nullptr && pool_cells->ne[0] == want;
    }

    return res;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // the bias is per cell, but only its "which block is visible" half varies per block; the rest
    // is the plain visible/not test the attention mask already carries over the same cells. where
    // the two tests agree, upload the per-block half only: that is 1/ratio of the cells.
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out.
    // the mask also holds an mrope rule for cells of the query's own position, but it compares a
    // text cell against itself and so never fires; only 2d image positions can differ there.
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set.
    // set_input_qsa is an O(n_kv) per-cell scan (about 865 us at 33k context) and every QSA
    // layer was paying it for byte-identical data: 12 scans per ubatch instead of one.
    llm_graph_input_qsa * inp = nullptr;

    const bool shortcut = qwen4exp_qsa_dense_shortcut() && n_kv <= (int64_t) hparams.indexer_top_k + r - 1;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        // complete-block selection, decided once per ratio (the layers of a ratio share these inputs) and pinned in
        // can_reuse. The shortcut and the selection exclude each other: the gate wants n_kv past the budget.
        const bool maskless = !shortcut &&
            qwen4exp_block_selection(mctx_hyb, blk_bias, n_stream, r, n_kv, ubatch, cparams, hparams);

        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, hparams.indexer_top_k, shortcut, maskless);
        qsa->n_kv_built = n_kv;

        // the pooled-key cache is addressed by block with no stream offset, so it serves a
        // single-stream cache only; everything else keeps recomputing every block inline.
        // the test is on the cache, not on the ubatch: a slot of a multi-stream cache gets
        // n_stream == 1 too, but its rows start at sinfo.s0 rather than at zero.
        llama_kv_cache * mem_pool = mctx_hyb->get_mem_pool();

        const bool use_pool = mem_pool != nullptr && mem_pool->get_n_stream() == 1;

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        if (!shortcut) {
            // the shortcut graph never reads these: an input no node reads gets no buffer from
            // ggml-alloc, which would then fail can_reuse every token (graph rebuilt per token,
            // -3.5 t/s decode, 2026-09-14) and write through a null pointer in set_input.
            // the selection gathers cells by block and never reads cell_blk either
            if (!maskless) {
                qsa->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
                ggml_set_input(qsa->cell_blk);
            }
            qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

            ggml_set_input(qsa->bias);
        }

        if (maskless) {
            qsa->tail_idxs = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, r-1, n_tps, n_stream);
            ggml_set_input(qsa->tail_idxs);
        }

        // ggml-alloc gives data only to tensors some node reads, so an input the graph has no
        // use for keeps data == nullptr and set_input then writes through a null pointer. With
        // the pooled-key cache the graph reads the recompute window, never the whole block
        // table, so these two are not created and set_input_qsa keeps the table in host scratch.
        if (!use_pool) {
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);

            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
        } else {
            const int64_t n_recomp = mctx_hyb->qsa_pool_n_recomp(
                    (uint32_t) r, (uint32_t) n_tokens, (uint32_t) n_kv, QSA_N_PAD_KV);

            qsa->pool_idxs  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_recomp);
            qsa->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r*n_recomp);
            qsa->pool_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_recomp);

            ggml_set_input(qsa->pool_idxs);
            ggml_set_input(qsa->pool_cells);
            ggml_set_input(qsa->pool_pos);

            // the selection expands the picked blocks to their cells, so it reads the whole block table
            // (the positions still come from the window)
            if (maskless) {
                qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
                ggml_set_input(qsa->blk_cells);
            }
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, qwen4exp_indexer_in(ctx0, model.layers[il].index_k_proj, cur));
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // A full block never changes again: its cells are written once, and pooling, normalisation
    // and rotation are all position-determined. So with the pooled-key cache only the tail of
    // the block table is recomputed and written back; the rest is read straight out of the
    // cache. n_recomp is n_blocks while the cache is invalid, which makes this the same code
    // as the inline path, just writing its result out as well.
    const int64_t n_recomp = inp->pool_cells ? inp->pool_cells->ne[0]/r : n_blocks;

    ggml_tensor * blk_src = inp->pool_cells ? inp->pool_cells : inp->blk_cells;

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, blk_src);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_recomp, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows.
    // the slices are strided views of members. ggml_add has no contiguity requirement (the Vulkan
    // backend gates it on type alone) and ggml_dup_tensor gives the sum a contiguous home, so the
    // per-slice ggml_cont was materialising the whole of members a second time for nothing.
    // the addition order is unchanged, so the arithmetic is identical.
    ggml_tensor * fresh = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_view_3d(ctx0, members, idx_dim, n_recomp, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]);
        fresh = fresh ? ggml_add(ctx0, fresh, slice) : slice;
    }
    fresh = ggml_scale(ctx0, fresh, 1.0f/(float) r);
    cb(fresh, "indexer_k_pooled", il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    fresh = ggml_reshape_3d(ctx0, fresh, idx_dim, n_recomp*n_stream, 1);
    fresh = build_norm(fresh, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    fresh = ggml_reshape_3d(ctx0, fresh, idx_dim, 1, n_recomp*n_stream);
    fresh = ggml_rope_multi(ctx0, fresh, inp->pool_pos ? inp->pool_pos : inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    ggml_tensor * pooled = nullptr;

    if (inp->pool_cells) {
        // write the window back, then read the whole block range out of the cache. the write is
        // expanded first, the same ordering the KV cache relies on for its own write-then-read.
        ggml_tensor * pk = mctx_hyb->get_mem_pool()->get_k_storage(il);

        ggml_build_forward_expand(gf, ggml_set_rows(ctx0,
                ggml_reshape_2d(ctx0, pk, pk->ne[0], pk->ne[1]*pk->ne[2]),
                ggml_reshape_2d(ctx0, fresh, idx_dim, n_recomp*n_stream),
                inp->pool_idxs));

        pooled = ggml_view_3d(ctx0, pk, idx_dim, n_blocks, n_stream, pk->nb[1], pk->nb[2], 0);
    } else {
        pooled = ggml_reshape_3d(ctx0, fresh, idx_dim, n_blocks, n_stream);
    }
    cb(pooled, "indexer_k", il);

    // Dense shortcut: the top-k below would select EVERY cell (width == n_kv), so skip the scoring, the
    // top-k and the mask rebuild and let the caller take the dense path; the indexer key cache and the
    // pooled-key window above are still written, so later ubatches score against a complete history.
    if (shortcut) {
        return nullptr;
    }

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, qwen4exp_indexer_in(ctx0, model.layers[il].index_q_proj, cur));
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the maskless selection scores whole blocks: n_blocks x rows, a quarter of the masked chain, so it stays untiled
    if (inp->maskless) {
        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
                ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
        score = ggml_relu(ctx0, score);

        // the heads sit side by side on ne[1] and there are only a few of them
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }

        score = summed;
        cb(score, "indexer_score", il);

        // one value per block, so it is cheaper to bias here than after the cells are expanded
        if (blk_bias) {
            score = ggml_add(ctx0, score, inp->bias);
        }

        ggml_tensor * tail = ggml_reshape_4d(ctx0, inp->tail_idxs, r-1, n_tps, 1, n_stream);
        ggml_tensor * top_k = qwen4exp_select_complete_blocks(ctx0, score, inp->blk_cells, tail,
                (int64_t) hparams.indexer_top_k/r, r);
        cb(top_k, "indexer_top_k", il);
        return top_k;
    }

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * q3 = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream);

    // score, expand, mask and select for query rows [t0, t0 + nt) of every stream
    auto select_rows = [&](int64_t t0, int64_t nt) {
        const bool all = t0 == 0 && nt == n_tps;
        ggml_tensor * qr = all ? q3 : ggml_view_3d(ctx0, q3, idx_dim, n_idx_h*nt, n_stream,
                q3->nb[1], q3->nb[2], t0*n_idx_h*q3->nb[1]);
        ggml_tensor * bias = all ? inp->bias : ggml_view_3d(ctx0, inp->bias, inp->bias->ne[0], nt, n_stream,
                inp->bias->nb[1], inp->bias->nb[2], t0*inp->bias->nb[1]);
        ggml_tensor * mask_rows = !blk_bias || all ? kq_mask : ggml_view_4d(ctx0, kq_mask, n_kv, nt, 1, n_stream,
                kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], t0*kq_mask->nb[1]);
        if (!all) {
            // put the row views in the graph ahead of the chain: Vulkan fuses the scorer (QSA_SCORE) and the
            // selection (TOPK_QSA) by matching the exact node order of the untiled chain, and a view visited in
            // the middle of either breaks the match (-12% pp at 32k with two tiles, measured 2026-10-04)
            ggml_build_forward_expand(gf, bias);
            if (blk_bias) {
                ggml_build_forward_expand(gf, mask_rows);
            }
        }

        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        ggml_tensor * score = ggml_mul_mat(ctx0, pooled, qr);
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, nt, n_stream);
        score = ggml_relu(ctx0, score);

        // the heads sit side by side on ne[1] and there are only a few of them
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, nt, n_stream,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }

        score = summed;
        cb(score, "indexer_score", il);

        // one value per block, so it is cheaper to bias here than after the cells are expanded
        if (blk_bias) {
            score = ggml_add(ctx0, score, bias);
        }

        // give every token of a block the block score; the budget is a whole number of
        // blocks, so the top-k cut still lands on a block boundary
        ggml_tensor * expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

        if (blk_bias) {
            // flash attention keeps the mask in f16; the scores are f32
            ggml_tensor * mask = mask_rows->type == GGML_TYPE_F32 && all ? mask_rows : ggml_cast(ctx0, mask_rows, GGML_TYPE_F32);
            expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, nt, n_stream));
        } else {
            expanded = ggml_add(ctx0, expanded, bias);
        }
        cb(expanded, "indexer_score_tokens", il);

        return ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
    };

    // Every query row selects on its own, so a large ubatch is scored in row tiles: the f32 tensors
    // above are n_kv x rows (4 B per cell and query, six of them), which passes the 4 GiB single
    // buffer limit of RADV at ub x n_kv = 2^30 (ub 8192 at 128k) and sets the compute-buffer peak.
    // Tiles keep each one under LLAMA_QSA_IDX_TILE_MB (default 1024, 0 = off), split evenly: a short tail tile
    // costs a whole pass of the chain (ub 2048 at 32k as 1927 + 121 rows was -4.4% pp). The selection is the same.
    static const int64_t tile_mb = [] { const char * e = getenv("LLAMA_QSA_IDX_TILE_MB"); return e ? atoll(e) : 1024; }();
    int64_t tile = n_tps;
    if (tile_mb > 0 && n_stream == 1) {
        const int64_t max_rows = std::max<int64_t>(256, (tile_mb << 20) / (4*n_kv));
        const int64_t n_tiles  = (n_tps + max_rows - 1)/max_rows;
        tile = (n_tps + n_tiles - 1)/n_tiles;
    }

    ggml_tensor * top_k = nullptr;
    if (tile >= n_tps) {
        top_k = select_rows(0, n_tps);
    } else {
        std::vector<ggml_tensor *> parts;
        for (int64_t t0 = 0; t0 < n_tps; t0 += tile) {
            parts.push_back(select_rows(t0, std::min(tile, n_tps - t0)));
        }
        // pairwise, so each row is copied log2(tiles) times rather than once per later tile
        while (parts.size() > 1) {
            std::vector<ggml_tensor *> next;
            for (size_t i = 0; i + 1 < parts.size(); i += 2) {
                next.push_back(ggml_concat(ctx0, parts[i], parts[i + 1], 1));
            }
            if (parts.size() % 2) {
                next.push_back(parts.back());
            }
            parts.swap(next);
        }
        top_k = parts[0];
    }

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    // complete-block selection lists only the cells a query may see (-1 elsewhere), so the selected-key kernels
    // need no mask; a backend without them refuses the op in supports_op
    const auto shared_qsa = qsa_inps.find((uint32_t) hparams.dsv4_compress_ratios[il]);
    if (shared_qsa != qsa_inps.end() && shared_qsa->second->maskless) {
        ggml_tensor * k = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v = mctx_cur->get_v(ctx0, il);
        ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, nullptr, nullptr, nullptr, kq_scale, il, top_k, 0);
        cb(cur, "kqv_out", il);
        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }
        return cur;
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // rebuild the mask and attend for query rows [t0, t0 + nt) of every stream
    auto attn_rows = [&](int64_t t0, int64_t nt) {
        const bool all = t0 == 0 && nt == q_cur->ne[2];
        ggml_tensor * q      = all ? q_cur : ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], nt,
                q_cur->nb[1], q_cur->nb[2], t0*q_cur->nb[2]);
        ggml_tensor * mask   = all ? kq_mask : ggml_view_4d(ctx0, kq_mask, kq_mask->ne[0], nt, 1, kq_mask->ne[3],
                kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], t0*kq_mask->nb[1]);
        ggml_tensor * sel    = all ? top_k : ggml_view_4d(ctx0, top_k, top_k->ne[0], nt, 1, top_k->ne[3],
                top_k->nb[1], top_k->nb[2], top_k->nb[3], t0*top_k->nb[1]);

        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, mask, -INFINITY);

        // reshape KQ mask into tensor with rows of size 1:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, sel, sel->ne[0], sel->ne[1], sel->ne[3], 1, sel->nb[1], sel->nb[2], sel->ne[3]*sel->nb[3], 0);

        // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
        // this will be our source of zero values for unmasking top k mask elements
        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        // modify KQ mask by unmasking elements that are in top_k indices
        // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
        ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        // reshape to restore the original shape of KQ mask:
        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        // combine with the original kq mask
        kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, mask);

        // Hand the selection to flash attention as well as to the mask. The mask alone still makes
        // the backend attend over the whole cache and merely discard what it read, which is O(n_kv)
        // per token; with top_k attached, a backend that can compact the active set (the Vulkan
        // gather-compact path) costs O(n_top_k) instead. n_kv_raw is 0: unlike DeepSeek V4 this
        // cache has no dense prefix, every attended cell comes from the selection. Backends without
        // that path ignore the extra argument and read the same mask they do today.
        return build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il, sel, 0);
    };

    // The rebuilt mask is n_kv x rows (2 B per cell and query for flash attention's f16, three of them), and
    // passes the 4 GiB single-buffer limit of RADV at ub x n_kv = 2^31 (ub 8192 at 256k), where the chain and
    // the attention fall back to the CPU. Query rows attend on their own, so a large ubatch runs in row tiles
    // under the indexer's LLAMA_QSA_IDX_TILE_MB budget, split evenly; the result is the same.
    static const int64_t tile_mb = [] { const char * e = getenv("LLAMA_QSA_IDX_TILE_MB"); return e ? atoll(e) : 1024; }();
    const int64_t n_rows = q_cur->ne[2];
    const int64_t n_kv   = kq_mask->ne[0];
    int64_t tile = n_rows;
    if (tile_mb > 0 && kq_mask->ne[3] == 1 && (int64_t) ggml_row_size(kq_mask->type, n_kv)*n_rows > (tile_mb << 20)) {
        const int64_t max_rows = std::max<int64_t>(256, (tile_mb << 20) / (int64_t) ggml_row_size(kq_mask->type, n_kv));
        const int64_t n_tiles  = (n_rows + max_rows - 1)/max_rows;
        tile = (n_rows + n_tiles - 1)/n_tiles;
    }

    ggml_tensor * cur = nullptr;
    if (tile >= n_rows) {
        cur = attn_rows(0, n_rows);
    } else {
        std::vector<ggml_tensor *> parts;
        for (int64_t t0 = 0; t0 < n_rows; t0 += tile) {
            parts.push_back(attn_rows(t0, std::min(tile, n_rows - t0)));
        }
        // pairwise, so each row is copied log2(tiles) times rather than once per later tile
        while (parts.size() > 1) {
            std::vector<ggml_tensor *> next;
            for (size_t i = 0; i + 1 < parts.size(); i += 2) {
                next.push_back(ggml_concat(ctx0, parts[i], parts[i + 1], 1));
            }
            if (parts.size() % 2) {
                next.push_back(parts.back());
            }
            parts.swap(next);
        }
        cur = parts[0];
    }
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb != nullptr && mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);
    // the projection stays referenced past the conv: the Vulkan CONCAT_SSM_CONV_SILU fusion reads it directly
    // while writing the conv output, so the allocator must not hand the output the projection's freed bytes
    // (the backend's overlap check otherwise disables the fusion on every layer)
    ggml_build_forward_expand(gf, conv_output_silu);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // LLAMA_GDN_CONV_L2=1: one l2 norm over the q and k heads together (they are the leading columns of the conv
    // output), expanded right after the conv so the Vulkan backend runs conv + silu + norm as one pass
    // (CONCAT_SSM_CONV_SILU_L2, Strata's GDN_CONVL2): the q/k columns are never written out and read back.
    // q and k are then views of the normalised tensor. Same arithmetic as the two separate norms.
    static const bool conv_l2 = [] { const char * e = getenv("LLAMA_GDN_CONV_L2"); return e != nullptr && atoi(e) != 0; }();
    ggml_tensor * qk_norm = nullptr;
    if (conv_l2) {
        ggml_tensor * qk_view = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, 2 * num_k_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
        qk_norm = build_gdn_l2_norm(ctx0, qk_view, hparams.f_norm_rms_eps);
        ggml_build_forward_expand(gf, qk_norm);
    }
    ggml_build_forward_expand(gf, ggml_view_1d(ctx0, qkv_mixed, 1, 0));

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    if (qk_norm) {
        q_conv = ggml_view_4d(ctx0, qk_norm, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                qk_norm->nb[1], qk_norm->nb[2], qk_norm->nb[3], 0);
        k_conv = ggml_view_4d(ctx0, qk_norm, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                qk_norm->nb[1], qk_norm->nb[2], qk_norm->nb[3], num_k_heads * qk_norm->nb[1]);
    } else {
        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }



    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // a padded f16 norm output (LLAMA_GDN_NORM_PAD) is viewed, not reshaped: ssm_out reads its rows through the stride
    ggml_tensor * final_output = ggml_is_contiguous(attn_out_norm)
            ? ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs)
            : ggml_view_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs,
                    attn_out_norm->nb[2], attn_out_norm->nb[3], 0);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

llama_model_qwen4exp::~llama_model_qwen4exp() {
    if (ple_hot.buf) { ggml_backend_buffer_free(ple_hot.buf); }
    if (ple_hot.ctx) { ggml_free(ple_hot.ctx); }
    if (hc_dinj.buf) { ggml_backend_buffer_free(hc_dinj.buf); }
    if (hc_dinj.ctx) { ggml_free(hc_dinj.ctx); }
}

// LLAMA_PLE_HOT=<sidecar> (scripts/flash-next/make_ple_hot.py): load the hot rows into a buffer of
// buft once. Any mismatch with the loaded table (type, row size, row count, or the bytes of sampled
// rows) disables the cache with a warning, so a sidecar can never change the output.
const ggml_tensor * llama_model_qwen4exp::ple_hot_table(ggml_backend_buffer_type_t buft) const {
    if (ple_hot.tried) {
        return ple_hot.tbl;
    }
    ple_hot.tried = true;
    const char * path = getenv("LLAMA_PLE_HOT");
    const ggml_tensor * t = per_layer_tok_embd;
    if (path == nullptr || *path == 0 || t == nullptr || t->data == nullptr) {
        return nullptr;
    }
    const int64_t t0 = ggml_time_us();
    const size_t row_sz = ggml_row_size(t->type, t->ne[0]);
    FILE * f = fopen(path, "rb");
    auto fail = [&](const char * why) -> const ggml_tensor * {
        LLAMA_LOG_WARN("%s: LLAMA_PLE_HOT=%s not used: %s\n", __func__, path, why);
        if (f) { fclose(f); }
        ple_hot.ids.clear();
        return nullptr;
    };
    if (!f) { return fail("cannot open"); }
    char magic[8];
    uint32_t hdr_row = 0, hdr_type = 0;
    uint64_t hdr_rows = 0, n_hot = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PLEHOT01", 8) != 0 ||
        fread(&hdr_row, 4, 1, f) != 1 || fread(&hdr_type, 4, 1, f) != 1 ||
        fread(&hdr_rows, 8, 1, f) != 1 || fread(&n_hot, 8, 1, f) != 1) {
        return fail("bad header");
    }
    if (hdr_row != row_sz || hdr_type != (uint32_t) t->type || hdr_rows != (uint64_t) t->ne[1] || n_hot == 0 || n_hot > (uint64_t) INT32_MAX) {
        return fail("sidecar was built for a different table");
    }
    ple_hot.ids.resize(n_hot);
    if (fread(ple_hot.ids.data(), sizeof(int32_t), n_hot, f) != n_hot) { return fail("short read (ids)"); }
    for (size_t i = 0; i < n_hot; ++i) {
        if (ple_hot.ids[i] < 0 || ple_hot.ids[i] >= t->ne[1] || (i > 0 && ple_hot.ids[i] <= ple_hot.ids[i-1])) {
            return fail("ids are not ascending table rows");
        }
    }
    const long data_off = (long) ((24 + 4 * n_hot + 63) / 64 * 64);
    if (fseek(f, data_off, SEEK_SET) != 0) { return fail("seek"); }

    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ple_hot.ctx = ggml_init(ip);
    ple_hot.tbl = ggml_new_tensor_2d(ple_hot.ctx, t->type, t->ne[0], (int64_t) n_hot);
    ggml_set_name(ple_hot.tbl, "per_layer_token_embd.hot");
    ple_hot.buf = ggml_backend_alloc_ctx_tensors_from_buft(ple_hot.ctx, buft);
    if (!ple_hot.buf) { ple_hot.tbl = nullptr; return fail("buffer allocation failed"); }
    ggml_backend_buffer_set_usage(ple_hot.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // copy in chunks, checking a spread of rows byte for byte against the mapped table
    const size_t chunk_rows = std::max<size_t>(1, (64u << 20) / row_sz);
    std::vector<uint8_t> buf(chunk_rows * row_sz);
    const size_t check_every = std::max<size_t>(1, n_hot / 1024);
    size_t n_checked = 0;
    for (size_t r0 = 0; r0 < n_hot; r0 += chunk_rows) {
        const size_t nr = std::min<size_t>(chunk_rows, n_hot - r0);
        if (fread(buf.data(), row_sz, nr, f) != nr) { ple_hot.tbl = nullptr; return fail("short read (rows)"); }
        for (size_t r = r0 + (check_every - r0 % check_every) % check_every; r < r0 + nr; r += check_every) {
            const uint8_t * ref = (const uint8_t *) t->data + (size_t) ple_hot.ids[r] * row_sz;
            if (memcmp(ref, buf.data() + (r - r0) * row_sz, row_sz) != 0) {
                ple_hot.tbl = nullptr;
                return fail("row bytes differ from the table");
            }
            ++n_checked;
        }
        ggml_backend_tensor_set(ple_hot.tbl, buf.data(), r0 * row_sz, nr * row_sz);
    }
    fclose(f);
    LLAMA_LOG_INFO("%s: LLAMA_PLE_HOT: %zu rows (%.1f MiB) in %s, %zu rows verified against the table, %.2f s\n", __func__,
            (size_t) n_hot, n_hot * row_sz / 1048576.0, ggml_backend_buffer_name(ple_hot.buf), n_checked, (ggml_time_us() - t0) / 1e6);
    return ple_hot.tbl;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // LLAMA_PLE_HOST_GATHER=0 restores the in-graph gather for A/B.
    ggml_tensor * emb  = nullptr;  // F32 [ple_head_dim * ple_n_heads, n_tokens], host gather
    ggml_tensor * rows = nullptr;  // I32 [ple_n_heads * n_tokens], in-graph get_rows

    // LLAMA_PLE_HOT: with a hot table, emb carries only the misses (compacted to the front) and
    //   out[k] = concat(get_rows(hot, slot), emb)[sel[k]]
    ggml_tensor * slot = nullptr;  // I32 [ple_n_heads * n_tokens], hot slot per lookup (0 for a miss)
    ggml_tensor * sel  = nullptr;  // I32 [ple_n_heads * n_tokens], k for a hit, n_lookups + miss# for a miss

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

// The table is a CPU tensor far too big to offload, so ggml_get_rows on it puts a CPU node in
// the middle of the graph: the scheduler splits there and every token pays a GPU->CPU->GPU
// round trip inside this layer. prefetch_rows already queues the faults; doing the gather
// here as well removes the split, and the hash it depends on is host-side anyway.
// LLAMA_PLE_HOST_GATHER=0 restores the in-graph gather.
static bool ple_host_gather() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_PLE_HOST_GATHER");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// Topology is fixed by the token count alone: table, head count and hash constants never
// change. set_input still refreshes the values on the reuse path.
bool llm_graph_input_ple::can_reuse(const llm_graph_params & params) {
    const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

    // set_input reads the predecessor tokens out of the attention cells through this pointer, and
    // the memory context is rebuilt for every decode. A reused graph keeps the input object alive
    // across those rebuilds, so re-point it at the live context or set_input dereferences a freed
    // one -- which segfaults once the allocator hands that memory out again.
    this->mctx = m->get_attn();

    const int64_t n_tokens = params.ubatch.n_tokens;

    if (emb != nullptr) {
        return emb->buffer != nullptr && emb->ne[1] == n_tokens &&
               (slot == nullptr || (slot->buffer != nullptr && sel->buffer != nullptr &&
                                    slot->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * n_tokens));
    }
    if (rows != nullptr) {
        return rows->buffer != nullptr &&
               rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * n_tokens;
    }
    return false;
}

// PLE row ids for one token: n-gram hashes of the token with its predecessors. window[s] is the
// predecessor s positions back (index 0 unused); an EOS in the window resets everything at or
// before it, and a missing predecessor (LLAMA_TOKEN_NULL: before the sequence start, or no cached
// cell) reads as EOS. The EOS of the token itself does not cut its own context, as in the reference.
// Shared by the per-ubatch input (exact, predecessors from the KV cells) and the batch-level
// readahead (predecessors taken from the batch order).
static void qwen4exp_ple_rows(const llama_hparams & hp, llama_token tok, const llama_token * window, int32_t * out) {
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;

    int64_t ctx[LLAMA_MAX_PLE_NGRAM];
    ctx[0] = tok;
    bool cut = false;
    for (int64_t s = 1; s < n_gram; ++s) {
        const llama_token t = cut ? LLAMA_TOKEN_NULL : window[s];
        cut = cut || t < 0 || t == eos;
        ctx[s] = cut ? eos : t;
    }

    for (int64_t n = 2; n <= n_gram; ++n) {
        uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
        for (int64_t j = 1; j < n; ++j) {
            mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
        }
        const int64_t base = (n - 2) * per_gram;
        for (int64_t g = 0; g < per_gram; ++g) {
            const int64_t h_i = base + g;
            out[h_i] = (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
        }
    }
}

// Batch-level readahead (LLAMA_PLE_PREFETCH_BATCH=0 disables). Measured 2026-09-13 on the pruned
// 320-expert model at 32k context: set_inputs was 640-690 ms of a 4.6 s ubatch, nearly all of it
// the PLE gather paging its rows from NVMe, because the 96 GB f16 table cannot be page-cached next
// to the GPU-resident weights on a 64 GB box. The batch's tokens are known before its first ubatch
// runs, so the rows of every later ubatch can be in flight while the first one computes.
void llama_model_qwen4exp::prefetch_batch_rows(const llama_token * tokens, uint32_t n_tokens) const {
    // OPT-IN (LLAMA_PLE_PREFETCH_BATCH=1). Measured 2026-09-13, pruned model, -b 8192 -ub 2048: the
    // 131k advice calls cost 2.2-3.4 s synchronously before the first ubatch and the in-flight reads
    // slowed compute ~0.5 s per ubatch; set_inputs fell 650 -> 100 ms but the call was 14% slower at
    // d0 (555 vs 645 pp8192) and only won at 32k because the plain arm hit multi-second disk stalls.
    // The paging itself (~23% of the decode wall on this 64 GB box) wants a direct-read thread pool
    // (pwilkin's on-direct reader), not page-fault advice. Kept for that comparison.
    static const bool off = !(getenv("LLAMA_PLE_PREFETCH_BATCH") && atoi(getenv("LLAMA_PLE_PREFETCH_BATCH")) != 0);
    const auto & hp = hparams;
    if (off || !tokens || (per_layer_tok_embd == nullptr && per_layer_tok_embd_h.empty()) || hp.ple_n_heads == 0 || hp.ple_ngram_size == 0 || n_tokens < 2) {
        return;
    }
    const int64_t n_heads = hp.ple_n_heads;
    const int64_t n_gram  = hp.ple_ngram_size;
    std::vector<int32_t> idx((size_t) n_heads * n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        llama_token window[LLAMA_MAX_PLE_NGRAM];
        for (int64_t s = 1; s < n_gram; ++s) {
            window[s] = (int64_t) i - s >= 0 ? tokens[i - s] : LLAMA_TOKEN_NULL;
        }
        qwen4exp_ple_rows(hp, tokens[i], window, idx.data() + (size_t) i * n_heads);
    }
    if (per_layer_tok_embd != nullptr) {
        prefetch_rows(per_layer_tok_embd, idx.data(), idx.size());
        return;
    }
    std::vector<int32_t> loc(n_tokens);
    for (int64_t h = 0; h < n_heads; ++h) {
        for (uint32_t i = 0; i < n_tokens; ++i) {
            loc[i] = idx[(size_t) i*n_heads + h] - (int32_t) hp.ple_head_offsets[h];
        }
        prefetch_rows(per_layer_tok_embd_h[h], loc.data(), n_tokens);
    }
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // An image is decoded as an embeddings-only batch, so ubatch->token is null and the
    // placeholder ids are not available. The hash must still give every position a row,
    // because this input feeds ggml_get_rows. Stand in the configured image token id, as
    // the reference hashes the placeholder, or EOS if the file has no such key.
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() has already stored the
    // current ubatch, so predecessors within this very ubatch are covered as well
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
        llama_token window[LLAMA_MAX_PLE_NGRAM];
        for (int64_t s = 1; s < n_gram; ++s) {
            window[s] = prev[i*n_prev + (n_prev - s)];
        }
        qwen4exp_ple_rows(hp, tok_of(i), window, idx.data() + i * n_heads);
    }

    // the table is far too big to offload, so it is gathered straight out of the mapping: one
    // fault per row, 16 per token, no two of them on the same page. left to the get_rows those
    // faults happen one at a time; queued here they are in flight before the graph even runs.
    const bool split = pmodel.per_layer_tok_embd == nullptr;

    // LLAMA_PLE_DEBUG=1: time the three host-side phases of the gather (prefill ubatches only)
    static const bool ple_dbg = getenv("LLAMA_PLE_DEBUG") != nullptr && atoi(getenv("LLAMA_PLE_DEBUG")) != 0;
    const int64_t t_pf0 = ple_dbg ? ggml_time_us() : 0;
    struct rusage ru0 = {}; if (ple_dbg) getrusage(RUSAGE_SELF, &ru0);

    // LLAMA_PLE_DUMP=<file>: append every ubatch's global row ids (int32, token-major) for checking
    // an offline reimplementation of the hash against the model
    static FILE * ple_dump = [] {
        const char * e = getenv("LLAMA_PLE_DUMP");
        return e ? fopen(e, "ab") : (FILE *) nullptr;
    }();
    if (ple_dump) {
        fwrite(idx.data(), sizeof(int32_t), idx.size(), ple_dump);
        fflush(ple_dump);
    }

    // LLAMA_PLE_HOT: resolve every lookup to a hot slot or a miss; only the misses are read from the
    // mapping (and hinted), compacted to the front of emb in lookup order
    const auto & hot_ids = pmodel.ple_hot.ids;
    const bool   hot     = slot != nullptr && !split;
    std::vector<int32_t>  miss_idx;   // global row ids of the misses
    std::vector<uint32_t> miss_k;     // lookup position of each miss
    if (hot) {
        const size_t n_lk = idx.size();
        std::vector<int32_t> slot_v(n_lk), sel_v(n_lk);
        miss_idx.reserve(n_lk);
        miss_k.reserve(n_lk);
        for (size_t k = 0; k < n_lk; ++k) {
            auto it = std::lower_bound(hot_ids.begin(), hot_ids.end(), idx[k]);
            if (it != hot_ids.end() && *it == idx[k]) {
                slot_v[k] = (int32_t) (it - hot_ids.begin());
                sel_v[k]  = (int32_t) k;
            } else {
                slot_v[k] = 0;
                sel_v[k]  = (int32_t) (n_lk + miss_k.size());
                miss_k.push_back((uint32_t) k);
                miss_idx.push_back(idx[k]);
            }
        }
        ggml_backend_tensor_set(slot, slot_v.data(), 0, n_lk*sizeof(int32_t));
        ggml_backend_tensor_set(sel,  sel_v.data(),  0, n_lk*sizeof(int32_t));
    }
    // the lookups the host has to read: all of them, or only the hot misses
    const std::vector<int32_t> & need = hot ? miss_idx : idx;

    // per-head tables take head-local ids in head-major order
    std::vector<int32_t> idx_h;
    if (split) {
        idx_h.resize(idx.size());
        for (int64_t i = 0; i < n_tokens; ++i) {
            for (int64_t h = 0; h < n_heads; ++h) {
                idx_h[h*n_tokens + i] = idx[i*n_heads + h] - (int32_t) hp.ple_head_offsets[h];
            }
        }
    }
    // LLAMA_PLE_PREFETCH: how the scattered table rows reach the host gather. The table is far larger
    // than RAM and a prompt's rows are effectively random, so every 2048-token ubatch is ~32k reads of
    // one page each. 1 = one posix_madvise(WILLNEED) per row page then fault (measured 2026-09-15: the
    // hints alone take 550-700 ms per ubatch, ~20 us of submission each, single-threaded), 0 = no hint,
    // faults taken on the gather pool, 2 = the hint only for pages mincore() reports absent (no better:
    // the probe costs the same syscall), 3 = pread() the row pages through an O_DIRECT descriptor from
    // the pool, at NVMe queue depth. Measured 2026-09-15 on the Intel 660p: modes 1, 2 and 3 all take
    // ~550 ms per 32k rows with 16, 32 or 64 threads = the drive's ~60k random-read IOPS, so the reads
    // are the cost, not the syscalls; mode 1 stays the default because its pages stay in the page
    // cache for the next prompt that hits the same rows, while O_DIRECT reads keep nothing.
    static const int ple_prefetch = [] {
        const char * e = getenv("LLAMA_PLE_PREFETCH");
        return e ? atoi(e) : 1;
    }();
    if (ple_prefetch == 1 || (ple_prefetch >= 2 && !rows)) {
        // modes 2/3 are handled inside the gather (it knows the row addresses); the in-graph get_rows
        // path keeps the plain hint
        // LLAMA_PLE_HINT_THREADS: threads issuing the page hints (default: the gather pool size; 1 =
        // the serial pass). Each hint is ~10 us of syscall even when the page is cached, so ~15-30k
        // hints per ubatch were 200-350 ms on one thread. LLAMA_PLE_HINT_PROBE=1 hints only the pages
        // mincore() reports absent.
        static const int hint_threads = [] {
            const char * e = getenv("LLAMA_PLE_HINT_THREADS");
            const int hw = (int) std::thread::hardware_concurrency();
            return e ? std::max(1, atoi(e)) : std::max(1, std::min(32, 2 * hw));
        }();
        static const bool hint_probe = getenv("LLAMA_PLE_HINT_PROBE") && atoi(getenv("LLAMA_PLE_HINT_PROBE")) != 0;
        if (ple_prefetch == 1 || rows) {
            if (split) {
                for (int64_t h = 0; h < n_heads; ++h) {
                    pmodel.prefetch_rows(pmodel.per_layer_tok_embd_h[h], idx_h.data() + h*n_tokens, n_tokens, hint_threads, hint_probe);
                }
            } else {
                pmodel.prefetch_rows(pmodel.per_layer_tok_embd, need.data(), need.size(), hint_threads, hint_probe);
            }
        }
    }

    const int64_t t_pf1 = ple_dbg ? ggml_time_us() : 0;

    if (rows) {
        const std::vector<int32_t> & src = split ? idx_h : idx;
        ggml_backend_tensor_set(rows, src.data(), 0, src.size()*ggml_element_size(rows));
        return;
    }

    // Gather host-side. Head varies fastest within a token, the layout ggml_get_rows produced for
    // the same index vector, so the flattened [head_dim * n_heads] row per token is unchanged.
    const ggml_tensor * tbl0     = split ? pmodel.per_layer_tok_embd_h[0] : pmodel.per_layer_tok_embd;
    const int64_t       head_dim = tbl0->ne[0];
    const size_t        row_sz   = ggml_row_size(tbl0->type, head_dim);
    const ggml_type_traits * traits = tbl0->type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(tbl0->type);
    GGML_ASSERT(tbl0->type == GGML_TYPE_F32 || (traits->to_float && "PLE table type has no to_float"));

    // get_rows dequantised to F32; keep that so the downstream matmuls are bit-identical.
    // The rows are scattered over a mapping far larger than RAM: every row is its own page, so the
    // copy is bound by faults, not bytes. Gather on a pool so the faults (minor when cached, disk
    // reads when not) overlap; with mode 2 each thread first hints the pages mincore() says are
    // absent, so a cold run still gets asynchronous readahead without paying the hint on warm pages.
    // with a hot table vals[j] is the j-th miss (lookup miss_k[j]); otherwise vals[k] is lookup k
    const size_t n_rows_all = need.size();
    std::vector<float> vals((size_t) head_dim * std::max<size_t>(n_rows_all, 1));
    const int64_t page = (int64_t) sysconf(_SC_PAGESIZE);
    // mode 3: an O_DIRECT descriptor + file offset per table tensor (falls back to mode 1 if the table
    // is not a lazily mapped range, e.g. --tensor-read-lazy off)
    const int64_t n_tbl = split ? n_heads : 1;
    std::vector<int>    dfd(n_tbl, -1);
    std::vector<size_t> doff(n_tbl, 0);
    bool direct = ple_prefetch == 3;
    if (direct) {
        for (int64_t h = 0; h < n_tbl; ++h) {
            const ggml_tensor * t = split ? pmodel.per_layer_tok_embd_h[h] : tbl0;
            if (!pmodel.direct_row_source(t, dfd[h], doff[h])) { direct = false; break; }
        }
        if (!direct) {
            static bool warned = false;
            if (!warned) { warned = true; LLAMA_LOG_WARN("%s: PLE table is not a lazily mapped range, direct row reads unavailable: using the page hint\n", __func__); }
            if (split) {
                for (int64_t h = 0; h < n_heads; ++h) {
                    pmodel.prefetch_rows(pmodel.per_layer_tok_embd_h[h], idx_h.data() + h*n_tokens, n_tokens);
                }
            } else {
                pmodel.prefetch_rows(pmodel.per_layer_tok_embd, need.data(), need.size());
            }
        }
    }
    // the head of a gathered row: from the lookup position (split tables are never hot)
    auto head_of = [&](size_t k) -> int64_t { return split ? (int64_t) (k % n_heads) : 0; };
    auto gather_range = [&](size_t k0, size_t k1) {
        if (direct) {
            // one or two 4 KiB pages per row through pread(); the pool gives the device its queue depth
            const size_t bsz = 2 * (size_t) page + (size_t) page;
            void * mem = nullptr;
            if (posix_memalign(&mem, (size_t) page, bsz) != 0) { mem = nullptr; }
            char * buf = (char *) mem;
            for (size_t k = k0; k < k1; ++k) {
                const int64_t h = head_of(k);
                const int64_t row = split ? need[k] - (int32_t) hp.ple_head_offsets[h] : need[k];
                const size_t foff = doff[h] + (size_t) row * row_sz;
                const size_t p0   = foff & ~((size_t) page - 1);
                const size_t len  = ((foff + row_sz + (size_t) page - 1) & ~((size_t) page - 1)) - p0;
                const char * src  = nullptr;
                if (buf != nullptr) {
                    ssize_t got = 0;
                    while (got < (ssize_t) len) {
                        const ssize_t r = pread(dfd[h], buf + got, len - got, (off_t) (p0 + got));
                        if (r <= 0) { break; }
                        got += r;
                    }
                    if (got >= (ssize_t) (foff - p0 + row_sz)) { src = buf + (foff - p0); }
                }
                if (src == nullptr) {
                    // direct read failed: fall back to the mapping for this row
                    const char * base = (const char *) (split ? pmodel.per_layer_tok_embd_h[h]->data : tbl0->data);
                    src = base + (size_t) row * row_sz;
                }
                if (traits == nullptr) {
                    memcpy(vals.data() + k*head_dim, src, head_dim*sizeof(float));
                } else {
                    traits->to_float(src, vals.data() + k*head_dim, head_dim);
                }
            }
            free(mem);
            return;
        }
        if (ple_prefetch == 2) {
            unsigned char vec[2];
            for (size_t k = k0; k < k1; ++k) {
                const int64_t h = head_of(k);
                const char * base = (const char *) (split ? pmodel.per_layer_tok_embd_h[h]->data : tbl0->data);
                const int64_t row = split ? need[k] - (int32_t) hp.ple_head_offsets[h] : need[k];
                const char * p0 = base + (size_t) row*row_sz;
                const uintptr_t a0 = ((uintptr_t) p0) & ~(uintptr_t)(page - 1);
                const size_t len = ((uintptr_t) p0 + row_sz) - a0;
                if (mincore((void *) a0, len, vec) == 0 && !(vec[0] & 1)) {
                    posix_madvise((void *) a0, len, POSIX_MADV_WILLNEED);
                }
            }
        }
        for (size_t k = k0; k < k1; ++k) {
            const int64_t h = head_of(k);
            const char * base = (const char *) (split ? pmodel.per_layer_tok_embd_h[h]->data : tbl0->data);
            const int64_t row = split ? need[k] - (int32_t) hp.ple_head_offsets[h] : need[k];
            if (traits == nullptr) {
                memcpy(vals.data() + k*head_dim, base + (size_t) row*row_sz, head_dim*sizeof(float));
            } else {
                traits->to_float(base + (size_t) row*row_sz, vals.data() + k*head_dim, head_dim);
            }
        }
    };
    static const int ple_threads = [] {
        const char * e = getenv("LLAMA_PLE_GATHER_THREADS");
        const int hw = (int) std::thread::hardware_concurrency();
        return e ? std::max(1, atoi(e)) : std::max(1, std::min(32, 2 * hw));
    }();
    if (ple_threads <= 1 || n_rows_all < 4096) {
        gather_range(0, n_rows_all);
    } else {
        std::vector<std::thread> pool;
        const size_t chunk = (n_rows_all + ple_threads - 1) / ple_threads;
        for (int t = 0; t < ple_threads; ++t) {
            const size_t k0 = (size_t) t * chunk, k1 = std::min(n_rows_all, k0 + chunk);
            if (k0 < k1) pool.emplace_back(gather_range, k0, k1);
        }
        for (auto & th : pool) th.join();
    }

    const int64_t t_g1 = ple_dbg ? ggml_time_us() : 0;
    if (n_rows_all > 0) {
        ggml_backend_tensor_set(emb, vals.data(), 0, (size_t) head_dim * n_rows_all * sizeof(float));
    }
    if (ple_dbg && n_tokens >= 1024) {
        const int64_t t_s1 = ggml_time_us();
        struct rusage ru1 = {}; getrusage(RUSAGE_SELF, &ru1);
        fprintf(stderr, "PLE_GATHER n_tokens=%lld rows=%zu read=%zu prefetch=%.1f ms gather=%.1f ms upload=%.1f ms minflt=%ld majflt=%ld\n",
                (long long) n_tokens, idx.size(), n_rows_all, (t_pf1 - t_pf0) / 1e3, (t_g1 - t_pf1) / 1e3, (t_s1 - t_g1) / 1e3,
                ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt);
    }
    if (hot) {
        static std::atomic<uint64_t> n_look{0}, n_miss{0};
        const uint64_t l = n_look += idx.size();
        const uint64_t m = n_miss += n_rows_all;
        if (ple_dbg && n_tokens >= 1024) {
            fprintf(stderr, "PLE_HOT ubatch hit %.1f%%, cumulative hit %.1f%% of %llu lookups\n",
                    100.0 * (double) (idx.size() - n_rows_all) / (double) idx.size(), 100.0 * (double) (l - m) / (double) l,
                    (unsigned long long) l);
        }
    }
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // keep the last state_cols columns for the next ubatch
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);

    ggml_tensor * tail;
    if (x->ne[1] >= state_cols) {
        // the tail is the last state_cols tokens of x itself; taking it from x rather than from
        // conv_input leaves the concat with one consumer (the conv), which is what lets the Vulkan
        // backend fuse CONCAT+SSM_CONV+SILU and never write the 84 MB transposed concat (2026-09-14)
        tail = ggml_transpose(ctx0, ggml_view_3d(ctx0, x,
                channels, state_cols, n_seqs,
                x->nb[1], x->nb[2],
                (x->ne[1] - state_cols) * x->nb[1]));
    } else {
        tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols));
    }

    ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
            state_cols * channels, n_seqs,
            conv_states_all->nb[1],
            kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));

    // [TAG_RECURRENT_ROLLBACK_SPLITS] with n_rs_seq > 0, snapshot group s holds the history as it was
    // s tokens before the end of this ubatch (split_equal keeps the trailing n_rs_seq + 1 tokens of a
    // sequence in one ubatch), so a speculative verify can roll back a rejected suffix without a
    // checkpoint restore and a replay. The snapshots read the old history and x, never conv_input,
    // so the concat keeps its single consumer and the fused conv path is the same as without them.
    const int64_t n_seq_tokens = x->ne[1];
    const int64_t mem_size     = mctx_cur->get_size();
    for (int64_t s = 1; s <= (int64_t) cparams.n_rs_seq && s <= n_seq_tokens; ++s) {
        // columns [c0, c0 + state_cols) of [history | x^T], copied straight into the group's row
        const int64_t c0   = n_seq_tokens - s;
        const size_t  row0 = ((size_t) s * mem_size + kv_head) * row_size;
        const size_t  esz  = ggml_element_size(conv_states_all);
        if (c0 >= state_cols) {
            ggml_tensor * xs = ggml_transpose(ctx0, ggml_view_3d(ctx0, x,
                    channels, state_cols, n_seqs,
                    x->nb[1], x->nb[2],
                    (c0 - state_cols) * x->nb[1]));
            ggml_tensor * d = ggml_view_3d(ctx0, conv_states_all,
                    state_cols, channels, n_seqs,
                    state_cols * esz, conv_states_all->nb[1], row0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, xs, d));
            continue;
        }
        // the part still in the old history
        ggml_tensor * old = ggml_view_3d(ctx0, state,
                state_cols - c0, channels, n_seqs,
                state->nb[1], state->nb[2],
                c0 * state->nb[0]);
        ggml_tensor * d_old = ggml_view_3d(ctx0, conv_states_all,
                state_cols - c0, channels, n_seqs,
                state_cols * esz, conv_states_all->nb[1], row0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, old, d_old));
        if (c0 > 0) {
            // and the first c0 tokens of this ubatch
            ggml_tensor * xs = ggml_transpose(ctx0, ggml_view_3d(ctx0, x,
                    channels, c0, n_seqs,
                    x->nb[1], x->nb[2], 0));
            ggml_tensor * d_x = ggml_view_3d(ctx0, conv_states_all,
                    c0, channels, n_seqs,
                    state_cols * esz, conv_states_all->nb[1], row0 + (state_cols - c0) * esz);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, xs, d_x));
        }
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *        hidden,
        int                  il) {
    GGML_UNUSED(inp);

    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    // heads lie slowest within a token either way, as the reference does.
    // The host gather only applies to a host-resident table: with the table (or its per-head split)
    // on a device buffer the rows are gathered in-graph, whatever LLAMA_PLE_HOST_GATHER says, so a
    // GPU-resident table needs no env to be safe (the host path would read a device pointer).
    const ggml_tensor * tbl0 = model.per_layer_tok_embd != nullptr ? model.per_layer_tok_embd
                             : (model.per_layer_tok_embd_h.empty() ? nullptr : model.per_layer_tok_embd_h[0]);
    const bool tbl_host = tbl0 != nullptr && tbl0->buffer != nullptr && ggml_backend_buffer_is_host(tbl0->buffer);
    static bool logged = false;
    if (!logged) {
        logged = true;
        LLAMA_LOG_INFO("%s: PLE table is %s-resident: %s gather\n", __func__, tbl_host ? "host" : "device",
                (ple_host_gather() && tbl_host) ? "host" : "in-graph");
    }
    ggml_tensor * emb = nullptr;
    // LLAMA_PLE_HOT: the hot rows go next to this layer's weights (the device the PLE matmuls run on)
    const auto & qmodel = static_cast<const llama_model_qwen4exp &>(model);
    const ggml_tensor * hot_tbl = nullptr;
    if (ple_host_gather() && tbl_host && model.per_layer_tok_embd != nullptr && model.layers[il].ple_key->buffer != nullptr) {
        hot_tbl = qmodel.ple_hot_table(ggml_backend_buffer_get_type(model.layers[il].ple_key->buffer));
    }
    if (ple_host_gather() && tbl_host) {
        ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb);
        emb = ple_inp->emb;
        if (hot_tbl != nullptr) {
            // exact merge: the hot rows are dequantised by get_rows (as the in-graph path does), the
            // misses arrive dequantised by the host gather, and a second F32 get_rows picks per lookup
            const int64_t n_lk = n_heads * n_tokens;
            ple_inp->slot = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_lk);
            ple_inp->sel  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_lk);
            ggml_set_input(ple_inp->slot);
            ggml_set_input(ple_inp->sel);
            ggml_tensor * hot_rows = ggml_get_rows(ctx0, const_cast<ggml_tensor *>(hot_tbl), ple_inp->slot); // [head_dim, n_lk]
            ggml_tensor * miss     = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim, n_lk);
            ggml_tensor * both     = ggml_concat(ctx0, hot_rows, miss, 1);                              // [head_dim, 2*n_lk]
            emb = ggml_get_rows(ctx0, both, ple_inp->sel);
            emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
        }
        res->add_input(std::move(ple_inp));
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));

        if (model.per_layer_tok_embd != nullptr) {
            emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
            emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
        } else {
            // per-head tables: rows is head-major [n_tokens per head] with head-local ids; gather each
            // head and lay the heads out slowest within a token, as the single get_rows did
            ggml_tensor * stack = nullptr;
            for (int64_t h = 0; h < n_heads; ++h) {
                ggml_tensor * rows_h = ggml_view_1d(ctx0, rows, n_tokens, h * n_tokens * ggml_element_size(rows));
                ggml_tensor * e_h    = ggml_get_rows(ctx0, model.per_layer_tok_embd_h[h], rows_h);   // [head_dim, n_tokens]
                e_h = ggml_reshape_3d(ctx0, e_h, hparams.ple_head_dim, 1, n_tokens);
                stack = stack ? ggml_concat(ctx0, stack, e_h, 1) : e_h;
            }
            emb = ggml_reshape_2d(ctx0, ggml_cont(ctx0, stack), hparams.ple_head_dim * n_heads, n_tokens);
        }
    }
    cb(emb, "ple_embd", il);

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // Depthwise causal conv dilated by the n-gram size, as a sum of shifted copies, because
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}


// NextN/MTP draft graph: one full-attention QSA block fed by [enorm(embd(tok)) ; hnorm(h)]
// through eh_proj, closing with the shared head mixer. Mirrors deepseek4::graph_mtp; the
// differences are dictated by the format: hnorm is hc-space (the drafter consumes the target's
// 4-stream residual, exported by the mainline graph under cparams.embeddings_nextn), the block
// is HC-wrapped, and the head mixer doubles as the output norm.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, mtp_tag{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm &&
            "MTP block tensors absent - was the draft context created with load_mtp?");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp_h = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp_h->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_h->tokens);

    inp_h->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp_h->embd);

    inp_h->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp_h->h);
    ggml_set_name(inp_h->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    if (tok_embd_w == nullptr) {
        tok_embd_w = qwen4exp_shared_model(cparams, model, "token_embd.weight").tok_embd;
        GGML_ASSERT(tok_embd_w && "QWEN4EXP MTP: the target model has no token embedding to borrow");
    }
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp_h->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp_h->h, n_embd, hc, n_tokens);
    res->add_input(std::move(inp_h));

    // catch-up batches (the drafter consuming the target's prompt / accepted rows) request no
    // outputs: nothing downstream of this block's KV write is ever read, because the next draft
    // step restarts from the target's own hidden row. Store K/V for every row and stop, the way
    // Gufo trims its prefill MTP pass. Same K/V ops as build_layer_attn, so the cache is identical.
    // GGML_MTP_PF_TRIM=0 keeps the full block.
    static const bool kv_only_enabled = [] {
        const char * e = getenv("GGML_MTP_PF_TRIM");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool kv_only = kv_only_enabled && n_outputs == 0;

    ggml_tensor * inp_pos     = build_inp_pos();
    // an unused out_ids input is never allocated, and setting it would abort
    ggml_tensor * inp_out_ids = kv_only ? nullptr : build_inp_out_ids();

    // the MTP context holds a plain attention cache over the nextn layer(s) only, the
    // deepseek32 pattern: the draft runs dense (no indexer cache, no recurrent state)
    auto * inp_attn = build_attn_inp_kv();
    const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;

    // hnorm is the same grouped RMSNorm as every HC norm: rms over one stream, flat gamma
    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = ggml_rms_norm(ctx0, tok_embd, hparams.f_norm_rms_eps);
    e_norm = ggml_mul(ctx0, e_norm, layer.nextn.enorm);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    // eh_proj runs per stream: src1 is [2*n_embd, hc, n_tokens], which the backends treat as
    // n_tokens batched hc-column mat-vecs and so re-read the weight once per token (~114 ms for
    // a 2k-token prompt ubatch on gfx1151). Flattened to [2*n_embd, hc*n_tokens] it is one GEMM.
    // Kept to prompt-sized batches so draft steps and small verify catch-ups keep their kernel.
    // GGML_MTP_EH_GEMM=0 disables.
    static const bool eh_gemm_enabled = [] {
        const char * e = getenv("GGML_MTP_EH_GEMM");
        return e == nullptr || atoi(e) != 0;
    }();
    ggml_tensor * res_hc = nullptr;
    if (eh_gemm_enabled && n_tokens >= 32) {
        ggml_tensor * concat_2d = ggml_reshape_2d(ctx0, concat, concat->ne[0], hc*n_tokens);
        res_hc = build_lora_mm(layer.nextn.eh_proj, concat_2d);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, n_tokens);
    } else {
        res_hc = build_lora_mm(layer.nextn.eh_proj, concat);
    }
    cb(res_hc, "mtp_eh_proj", il);

    // one HC-wrapped full-attention QSA block, the mainline loop body minus PLE/GDN
    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    ggml_build_forward_expand(gf, cur);

    if (kv_only) {
        const int64_t n_embd_head = hparams.n_embd_head_v();

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = ggml_rope_multi(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Kcur, "Kcur", il);

        if (inp_attn->self_k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
        }
        if (inp_attn->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
        }

        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);

        const auto * mctx_cur = inp_attn->mctx;
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));

        res->t_h_nextn = nullptr;
        res->t_embd    = nullptr;
        res->t_logits  = nullptr;
        return;
    }

    cur = build_layer_attn(inp_attn, mctx_hyb, cur, inp_pos, sections, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    // chained-draft export: the drafter's own hc state, gathered to the output rows
    ggml_tensor * h_nextn = res_hc;   // real node; see the export note in the mainline graph
    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, res_hc, hc_dim, n_tokens);
        h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
    }
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;
    ggml_build_forward_expand(gf, h_nextn);

    // the head mixer is the output norm. It is the draft head's own: blk.N.nextn.hc_head_* in
    // current files, output_hc_* in older sidecars (same weights under the trunk's names).
    ggml_tensor * hm_norm = layer.nextn.hc_head_norm ? layer.nextn.hc_head_norm : model.hc_head_norm;
    ggml_tensor * hm_down = layer.nextn.hc_head_down ? layer.nextn.hc_head_down : model.hc_head_down;
    ggml_tensor * hm_up   = layer.nextn.hc_head_up   ? layer.nextn.hc_head_up   : model.hc_head_up;
    GGML_ASSERT(hm_norm && hm_down && hm_up &&
            "QWEN4EXP MTP: draft head has no head mixer (nextn.hc_head_* or output_hc_*)");
    cur = build_hc_mix(res_hc, hm_norm, hm_down, hm_up, nullptr, nullptr, -1);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    if (head_w == nullptr) {
        const llama_model & other = qwen4exp_shared_model(cparams, model, "output.weight");
        head_w = other.output;
        head_s = other.output_s;
        GGML_ASSERT(head_w && "QWEN4EXP MTP: the target model has no LM head to borrow");
    }
    // --spec-draft-mtp-vocab: score only the subset rows (copied at context creation from this head, own or
    // borrowed) and scatter them into a full-vocab row of -inf. Draft steps output one row; other batches
    // (prompt catch-up with several outputs) keep the full head.
    if (mtp_draft != nullptr && layer.nextn.shared_head_head == nullptr && n_outputs == 1) {
        GGML_ASSERT(mtp_draft->n_keep == cparams.mtp_draft_vocab);
        GGML_ASSERT(mtp_draft->head->type == head_w->type && mtp_draft->head->ne[0] == head_w->ne[0]);
        const int64_t n_sel = mtp_draft->head->ne[1];
        const int64_t n_row = head_w->ne[1];
        ggml_tensor * sub = build_lora_mm(mtp_draft->head, cur, head_s);
        cur = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_row), -INFINITY);
        cur = ggml_set_rows(ctx0, cur, ggml_reshape_2d(ctx0, sub, 1, n_sel), mtp_draft->ids);
        cur = ggml_reshape_2d(ctx0, cur, n_row, 1);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
