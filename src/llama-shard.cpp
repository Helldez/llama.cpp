// ShardLLM seam — in-core implementation of the narrow hooks declared in include/llama.h.
//
// The out-of-tree engine drives block-by-block ("partial forward") execution and supplies
// layer weights on demand. This file holds:
//   - the thread-local partial-forward range read by the graph clamp (see llama-graph.cpp);
//   - the plain-C entry points the seam target forwards to (llama_shard_*);
//   - the on-demand weight-source setter (stored on llama_model; the materialize/evict call
//     site is in llama-context.cpp).
//
// Copyright 2026 ShardLLM contributors. Apache-2.0.

#include "llama.h"
#include "llama-shard.h"
#include "llama-model.h"
#include "llama-arch.h"

#include <climits>

// The clamp range is THREAD-LOCAL so two contexts can be decoded concurrently (a displaced
// pipeline) without racing. Default (0, INT32_MAX) == full forward.
static thread_local int32_t t_shard_pf_begin = 0;
static thread_local int32_t t_shard_pf_end   = INT32_MAX;

// Internal getter (also declared in llama-shard.h for the graph clamp / context hook).
void llama_shard_get_partial_forward(int32_t * begin, int32_t * end) {
    if (begin) { *begin = t_shard_pf_begin; }
    if (end)   { *end   = t_shard_pf_end;   }
}

void llama_shard_set_partial_forward(int32_t begin, int32_t end) {
    t_shard_pf_begin = begin;
    t_shard_pf_end   = end;
}

void llama_shard_clear_partial_forward(void) {
    t_shard_pf_begin = 0;
    t_shard_pf_end   = INT32_MAX;
}

int llama_shard_supports_partial_forward(const struct llama_model * model) {
    if (!model) { return 0; }
    // Only architectures whose graph applies the partial-forward clamp.
    return (model->arch == LLM_ARCH_QWEN2 || model->arch == LLM_ARCH_QWEN3) ? 1 : 0;
}

int llama_shard_abi_version(void) {
    return 1; // keep in sync with SHARDLLM_SEAM_ABI_VERSION in the parent seam header
}

void llama_shard_set_weight_source(
        struct llama_model * model,
        void * user_data,
        int  (*materialize)(void * user_data, int layer),
        void (*evict)(void * user_data, int layer)) {
    if (!model) { return; }
    model->shard_ws_user_data  = user_data;
    model->shard_ws_materialize = materialize;
    model->shard_ws_evict       = evict;
}

int llama_shard_layer_tensors(
        const struct llama_model * model,
        int          layer,
        const void ** out_tensors,
        uint64_t    * out_file_offsets,
        uint64_t    * out_nbytes,
        int           cap) {
    (void) model; (void) layer; (void) out_tensors;
    (void) out_file_offsets; (void) out_nbytes; (void) cap;
    // TODO(M2): enumerate blk.<layer>.* weight tensors with their gguf file offsets + byte
    // sizes. The offset map lives on the transient llama_model_loader (weights_map) and is not
    // retained on the model after load; wiring it through is deferred to the streamer work.
    // The clamp and the materialize/evict call site do not depend on this accessor.
    return 0;
}
