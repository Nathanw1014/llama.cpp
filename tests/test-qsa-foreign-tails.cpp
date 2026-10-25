// QSA complete-block selection with another sequence's image cells in the same stream (halo-box #165).
// A unified cache holds seq 1's image cells, which repeat one position, next to the text of seqs 0 and 2.
// Text positions are broadcast to all 4 axes, so the scalar-visibility gate takes a batch of seqs 0 and 2;
// filling their tail lists must skip the image cells instead of asserting on their shared slot.
// Runs the production scan on host tensors, without model weights or a device.

#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model.h"
#include "../src/llama-batch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

// one token per entry, all n_pos axes equal to the linear position (the fork's text layout)
struct qsa_ubatch {
    std::vector<llama_token> tok; std::vector<llama_pos> pos; std::vector<int32_t> n_seq_id;
    std::vector<std::vector<llama_seq_id>> ids; std::vector<llama_seq_id *> seq_id; llama_ubatch u{};
    qsa_ubatch(std::vector<std::pair<llama_pos, std::vector<llama_seq_id>>> tokens, uint32_t n_pos) {
        const uint32_t n = tokens.size();
        tok.assign(n, 0); pos.resize(n*n_pos); ids.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t a = 0; a < n_pos; ++a) {
                pos[i + a*n] = tokens[i].first;
            }
            ids[i] = tokens[i].second;
            n_seq_id.push_back(ids[i].size());
            seq_id.push_back(ids[i].data());
        }
        u.n_tokens = n; u.n_pos = n_pos; u.token = tok.data(); u.pos = pos.data(); u.n_seq_id = n_seq_id.data(); u.seq_id = seq_id.data();
    }
};

int main() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    model->hparams = {};
    llama_memory_hybrid_idx mem(*model, GGML_TYPE_F16, GGML_TYPE_F16, false, 64, 1, 0, LLAMA_SWA_TYPE_NONE,
                                GGML_TYPE_F32, GGML_TYPE_F32, 4, 3, 0, false, true,
                                nullptr, nullptr, [](int32_t) { return false; });
    GGML_ASSERT(mem.get_mem_idx() != nullptr);
    auto & cells = const_cast<llama_kv_cells &>(mem.get_mem_idx()->get_cells(0));
    // seq 0: positions 0..9 in cells 0..9, the first block shared with seq 2
    for (uint32_t j = 0; j < 10; ++j) {
        cells.pos_set(j, j);
        cells.seq_add(j, 0);
        if (j < 4) {
            cells.seq_add(j, 2);
        }
    }
    // seq 1: an image, 8 cells at position 4
    for (uint32_t j = 16; j < 24; ++j) {
        cells.pos_set(j, 4);
        cells.seq_add(j, 1);
        cells.ext_set(j, {4, llama_pos(j - 15)});
    }
    // seq 2: positions 4..9 in cells 32..37
    for (uint32_t j = 32; j < 38; ++j) {
        cells.pos_set(j, j - 28);
        cells.seq_add(j, 2);
    }
    qsa_ubatch ub({{8, {0}}, {9, {2}}}, 4);

    ggml_context_ptr ctx(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
    auto * cell_blk  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 64, 1);
    auto * blk_cells = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 64, 1);
    auto * blk_pos   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 64);
    auto * bias      = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 2, 1);
    auto * tails     = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, 3, 2, 1);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(buffer);

    mem.set_input_qsa(cell_blk, blk_cells, blk_pos, bias, nullptr, nullptr, nullptr, tails, &ub.u, 4, true, 64, 1);

    const int32_t expected_tail[] = {8, -1, -1, 36, 37, -1};
    GGML_ASSERT(std::equal(expected_tail, expected_tail + 6, (int32_t *) tails->data));
    for (int query = 0; query < 2; ++query) {
        std::vector<int32_t> selected;
        for (int b = 0; b < 16; ++b) {
            if (((float *) bias->data)[query*16 + b] == 0.0f) {
                const auto * block = (int32_t *) blk_cells->data + 4*b;
                selected.insert(selected.end(), block, block + 4);
            }
        }
        std::sort(selected.begin(), selected.end());
        const std::vector<int32_t> expected = query == 0 ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                                                         : std::vector<int32_t>{0, 1, 2, 3, 32, 33, 34, 35};
        GGML_ASSERT(selected == expected);
    }
    printf("test-qsa-foreign-tails: OK\n");
    return 0;
}
