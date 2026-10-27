#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu >= 0 computes the missing experts on the CPU and uploads at most cpu of them per layer and batch, see get_cpu_maps
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int32_t cpu = -1);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the data of the slot map of layer il if the graph reads its experts from the cache for n_tokens tokens, nullptr otherwise
    // an expert selected by the batch that has a slot here when the layer's lookup starts keeps it through that layer's plan
    const int32_t * get_slots(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // if the missing experts are computed on the CPU, the maps of layer il, looked up like the slot map:
    // I32 cpu_ids (the expert, -1 if it is in the cache), F32 dev_mask (1 if it is in the cache) and cpu_mask (1 - dev_mask)
    // the slot map then gives slot 0 to the missing experts, so the device computes them with any weights and dev_mask drops them
    bool get_cpu_maps(int32_t il, ggml_tensor ** cpu_ids, ggml_tensor ** dev_mask, ggml_tensor ** cpu_mask) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
