// test-sched-split-fusion: a fused GPU op whose input the scheduler copied in from the CPU (a split input, GGML_OP_NONE).
// ggml-alloc can place the fused op's output on that copy once its last consumer is allocated, so the backend's fusion
// overlap check has to look at split inputs too. 2026-10-25: Qwen3.5 EXL3 with the EXL3 linears on the CPU and the rest on
// Vulkan gave PPL 8.10-8.14 instead of 7.965, from CONCAT_SSM_CONV_SILU reading the CPU in_proj output (the copy of
// transpose(x)) while it wrote the SiLU over it.
// Graph, as qwen35's delta-net conv: x = scale(x0) pinned to the CPU, then concat(state, transpose(x), 0) -> ssm_conv ->
// silu on the first GPU device. The result is compared with the same graph on the CPU backend alone.
// No GPU device: prints "skipped" and passes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct graph_inputs {
    ggml_tensor * x0;
    ggml_tensor * state;
    ggml_tensor * kern;
};

static ggml_tensor * build_conv(ggml_context * ctx, const graph_inputs & in, ggml_tensor ** x_out) {
    ggml_tensor * x  = ggml_scale(ctx, in.x0, 1.0f);   // [ch, n_t, 1], the node the test pins to the CPU
    ggml_tensor * xt = ggml_transpose(ctx, x);         // [n_t, ch, 1]
    ggml_tensor * sx = ggml_concat(ctx, in.state, xt, 0);
    ggml_tensor * y  = ggml_silu(ctx, ggml_ssm_conv(ctx, sx, in.kern));
    ggml_set_output(y);
    *x_out = x;
    return y;
}

static graph_inputs new_inputs(ggml_context * ctx, int64_t ch, int64_t n_t) {
    graph_inputs in;
    in.x0    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ch, n_t, 1);
    in.state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, ch, 1);
    in.kern  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, ch);
    ggml_set_input(in.x0);
    return in;
}

static void set_inputs(const graph_inputs & in, const std::vector<float> & x0, const std::vector<float> & state, const std::vector<float> & kern) {
    ggml_backend_tensor_set(in.x0,    x0.data(),    0, ggml_nbytes(in.x0));
    ggml_backend_tensor_set(in.state, state.data(), 0, ggml_nbytes(in.state));
    ggml_backend_tensor_set(in.kern,  kern.data(),  0, ggml_nbytes(in.kern));
}

static std::vector<float> run_cpu(ggml_backend_t cpu, int64_t ch, int64_t n_t, const std::vector<float> & x0,
                                  const std::vector<float> & state, const std::vector<float> & kern) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    graph_inputs in = new_inputs(ctx, ch, n_t);
    ggml_tensor * x;
    ggml_tensor * y = build_conv(ctx, in, &x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    set_inputs(in, x0, state, kern);
    ggml_backend_graph_compute(cpu, gf);
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static std::vector<float> run_split(ggml_backend_t gpu, ggml_backend_t cpu, int64_t ch, int64_t n_t, const std::vector<float> & x0,
                                    const std::vector<float> & state, const std::vector<float> & kern) {
    // state and kernel live on the GPU (weights), x0 on the CPU; the op nodes are placed by the scheduler
    ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w  = ggml_init(ip);
    ggml_context * ctx_in = ggml_init(ip);
    graph_inputs in;
    in.x0    = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, ch, n_t, 1);
    in.state = ggml_new_tensor_3d(ctx_w,  GGML_TYPE_F32, 3, ch, 1);
    in.kern  = ggml_new_tensor_2d(ctx_w,  GGML_TYPE_F32, 4, ch);
    ggml_set_input(in.x0);
    ggml_backend_buffer_t buf_w  = ggml_backend_alloc_ctx_tensors(ctx_w, gpu);
    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, cpu);
    set_inputs(in, x0, state, kern);

    ggml_init_params gp = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x;
    ggml_tensor * y = build_conv(ctx, in, &x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    // views (the transpose) follow their source, as in llama: the GPU then reads a copy of transpose(x)
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->view_src == nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, node, node == x ? cpu : gpu);
        }
    }
    std::vector<float> out(ggml_nelements(y));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_w);
    ggml_free(ctx_in);
    return out;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (dev == nullptr) {
        printf("test-sched-split-fusion: no GPU device, skipped\n");
        return 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    printf("test-sched-split-fusion: %s + CPU\n", ggml_backend_dev_name(dev));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int n_fail = 0;
    const int64_t shapes[][2] = { { 128, 8 }, { 256, 64 }, { 1024, 64 }, { 2048, 512 } };
    for (const auto & s : shapes) {
        const int64_t ch = s[0], n_t = s[1];
        std::vector<float> x0(ch*n_t), state(3*ch), kern(4*ch);
        for (float & v : x0)    v = dist(rng);
        for (float & v : state) v = dist(rng);
        for (float & v : kern)  v = dist(rng);
        const std::vector<float> ref = run_cpu(cpu, ch, n_t, x0, state, kern);
        const std::vector<float> got = run_split(gpu, cpu, ch, n_t, x0, state, kern);
        double err = 0.0, nrm = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            err += (got[i] - ref[i])*(double) (got[i] - ref[i]);
            nrm += ref[i]*(double) ref[i];
        }
        const double nmse = err/(nrm > 0.0 ? nrm : 1.0);
        const bool ok = std::isfinite(nmse) && nmse < 1e-6;
        printf("  ch %5lld n_t %4lld: NMSE %.3e %s\n", (long long) ch, (long long) n_t, nmse, ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("test-sched-split-fusion: %s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
