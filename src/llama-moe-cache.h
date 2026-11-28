#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llm_graph_context;

// the experts of a MoE layer, as llm_graph_context::build_moe_ffn passes them to the MoE cache
struct llama_moe_cache_layer {
    int32_t       il;
    ggml_tensor * up_exps;
    ggml_tensor * gate_exps;
    ggml_tensor * down_exps;
    ggml_tensor * gate_up_exps;
    bool          weight_before_ffn; // the router weights scale the expert input
};

// the lookup of the selected experts of a layer in the MoE cache, see llama_moe_cache::build_lookup
struct llama_moe_cache_lookup {
    int32_t       il      = -1;
    ggml_tensor * slots   = nullptr; // [n_expert_used, n_tokens] slots of the selected experts in the MoE cache, nullptr if the layer does not read the cache
    ggml_tensor * cpu_ids = nullptr; // [n_expert_used, n_tokens] misses computed from the host experts, -1 for the other experts, or nullptr
    ggml_tensor * cpu_in  = nullptr; // EXP: the expert input when the host computes the misses inside the graph (see build_experts)
};

// builds the experts selected by mm_ids up to the down projection, from the MoE cache if cached: [n_embd, n_expert_used, n_tokens]
// cur is the expert input and ids are the selected experts, for the biases and scales (see llm_graph_context::build_moe_ffn)
// how the cache serves its misses, see llama_context_params
struct llama_moe_cache_options {
    bool    host_call   = false; // lookup inside the graph where the device supports GGML_OP_HOST_CALL
    int32_t cpu_threads = 0;     // with host_call: a CPU worker computes the CPU share of the misses inside the graph
    int32_t read_ahead  = 0;     // experts per token of the next layer predicted and filled while this layer computes
    bool    async_admit = false; // with host_call and cpu_threads: small batches compute every miss on the CPU, the cache admits in the background
    bool    lend        = false; // the banks live in the compute buffer past the bytes small batches use, see llama_moe_cache::bind
};

using llama_moe_cache_build_experts_t = ggml_tensor * (*)(ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached, void * user_data);

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share,
            const llama_moe_cache_options & opts = {});

    // lending (EXPERIMENTAL): the compute buffer of a device is reserved at least bank_bytes past what small batches use,
    // the banks are bound there, and a large batch takes that memory: the cache is emptied before it (begin_ubatch) and
    // refills from the misses of the small batches after, so prefill and generation share one reservation
    bool lends() const;
    static int64_t max_small_batch();               // the largest batch that reads and fills the cache
    size_t bank_bytes(ggml_backend_t backend) const; // the bytes the banks of backend take
    void park();                                     // unbind from the compute buffer before the scheduler is replaced
    bool bind(ggml_backend_t backend, ggml_backend_buffer_t buf, size_t offset); // false: no room, the banks got their own buffer
    void begin_ubatch(int64_t n_tokens);             // before each ubatch
    ~llama_moe_cache();

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the graph of a MoE layer, built by llm_graph_context::build_moe_ffn in two steps:

    // 1. the lookup of the selected experts in the cache, built after the expert input and before the expert matmuls
    //    returns no slots if the layer does not read its experts from the cache for this ubatch
    llama_moe_cache_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts, ggml_tensor * cur = nullptr) const;

    // 2. the experts of a layer with slots, from the cache and from the host experts: [n_embd, n_expert_used, n_tokens]
    // calls build synchronously; does not retain user_data
    ggml_tensor * build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
            ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // for large batches, where the copy dst of w is device memory the host writes in place: the experts of w that are
    // not in the cache are filled straight into dst like the cache's misses (the fill function, on its lanes), instead
    // of being uploaded from w->data; false if this does not apply, and then nothing was written
    bool can_fill_experts(const ggml_tensor * w, const ggml_tensor * dst) const;

    // layers that run another layer's expert tensors (looped blocks, shared weights) read, in a batch the cache does not
    // serve, a twin of each: a tensor of its own over the same bytes, so that the scheduler's copy of it lives for that
    // layer only. twin() is nullptr for any other layer; experts_of() maps a twin back to the experts it stands for.
    ggml_tensor * twin(int32_t il, const ggml_tensor * w) const;
    const ggml_tensor * experts_of(const ggml_tensor * w) const;
    void fill_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, const std::vector<int32_t> & experts);

    // fill the misses in place with fn instead of uploading them, false if a cache buffer has no host address
    bool set_fill(llama_moe_cache_fill_fn fn, void * user_data);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
