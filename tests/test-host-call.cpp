// Tests for GGML_OP_HOST_CALL, on every backend that supports it: a graph with device work between several host calls
// is computed many times with new inputs, so that a backend that records the graph once and replays it (CUDA graphs)
// answers the calls from inside the replayed graph. Each result must equal the same arithmetic done on the host.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstring>
#include <vector>

struct call_state {
    int     n_calls = 0; // calls made, to check that a replayed graph calls every time
    int     offset  = 0; // changes between computes: the reply must not be recorded with the graph
    ggml_tensor * side = nullptr; // if set, fn also writes this tensor of the backend, as a cache filling its misses does
};

// dst = src/2 + offset, exact in float for the small integers used here
static void host_fn(const void * src, size_t src_size, void * dst, size_t dst_size, void * user_data) {
    call_state & st = *(call_state *) user_data;
    const float * a = (const float *) src;
    float * b = (float *) dst;
    for (size_t i = 0; i < dst_size/sizeof(float); ++i) {
        b[i] = 0.5f*a[i % (src_size/sizeof(float))] + (float) st.offset;
    }
    if (st.side != nullptr) {
        std::vector<uint8_t> bytes(ggml_nbytes(st.side), (uint8_t) st.n_calls);
        ggml_backend_tensor_set(st.side, bytes.data(), 0, bytes.size());
    }
    st.n_calls++;
}

static bool test_host_call(ggml_backend_t backend, int64_t ne, int n_layers, int n_iter, bool side = false) {
    ggml_init_params params_static = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx_static(ggml_init(params_static));
    ggml_tensor * x = ggml_new_tensor_1d(ctx_static.get(), GGML_TYPE_F32, ne);
    ggml_set_input(x);
    ggml_tensor * side_t = side ? ggml_new_tensor_1d(ctx_static.get(), GGML_TYPE_F32, 1 << 18) : nullptr;

    const size_t graph_size = 8*n_layers + 16;
    ggml_init_params params_compute = { graph_size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true };
    ggml_context_ptr ctx_compute(ggml_init(params_compute));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute.get(), graph_size, false);

    // device work between the calls: x -> call -> add itself -> call -> ...
    call_state st;
    st.side = side_t;
    ggml_tensor * cur = x;
    for (int l = 0; l < n_layers; ++l) {
        cur = ggml_host_call(ctx_compute.get(), cur, GGML_TYPE_F32, ne, 1, host_fn, &st);
        cur = ggml_add(ctx_compute.get(), cur, cur);
    }
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    char desc[128];
    snprintf(desc, sizeof(desc), "  %-8s ne=%-6lld layers=%-3d iter=%-3d side=%d ", ggml_backend_name(backend), (long long) ne, n_layers, n_iter, (int) side);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) {
            printf("%snot supported\n", desc);
            return true;
        }
    }

    ggml_backend_buffer_ptr buf_static(ggml_backend_alloc_ctx_tensors(ctx_static.get(), backend));
    ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    ggml_gallocr_alloc_graph(galloc.get(), gf);

    bool ok = true;
    std::vector<float> in(ne), out(ne), want(ne);
    for (int it = 0; it < n_iter && ok; ++it) {
        for (int64_t i = 0; i < ne; ++i) {
            in[i] = (float) ((i*7 + it*13) % 101) - 50.0f;
        }
        st.offset = it - 3;
        ggml_backend_tensor_set(x, in.data(), 0, ggml_nbytes(x));

        const int n_calls0 = st.n_calls;
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(cur, out.data(), 0, ggml_nbytes(cur));

        want = in;
        for (int l = 0; l < n_layers; ++l) {
            for (auto & v : want) {
                v = 2.0f*(0.5f*v + (float) st.offset);
            }
        }
        ok = ok && st.n_calls - n_calls0 == n_layers && out == want;
        if (!ok) {
            printf("%sFAIL at iteration %d (calls %d of %d)\n", desc, it, st.n_calls - n_calls0, n_layers);
        }
    }
    if (ok) {
        // the cost of one call inside a graph, against the same graph with the calls on the host between graphs
        const int64_t t0 = ggml_time_us();
        for (int it = 0; it < n_iter; ++it) {
            ggml_backend_graph_compute(backend, gf);
        }
        printf("%sOK  %.1f us per call\n", desc, (double) (ggml_time_us() - t0)/n_iter/n_layers);
    }
    return ok;
}

int main() {
    ggml_backend_load_all();

    int n_fail = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_ptr backend(ggml_backend_dev_init(ggml_backend_dev_get(i), nullptr));
        if (backend == nullptr) {
            continue;
        }
        n_fail += !test_host_call(backend.get(), 10, 1, 8);
        n_fail += !test_host_call(backend.get(), 10, 48, 20);
        n_fail += !test_host_call(backend.get(), 320, 48, 20);
        n_fail += !test_host_call(backend.get(), 4096, 4, 8);
        n_fail += !test_host_call(backend.get(), 10, 48, 20, true);
    }
    printf("%s\n", n_fail == 0 ? "all passed" : "FAILED");
    return n_fail > 0;
}
