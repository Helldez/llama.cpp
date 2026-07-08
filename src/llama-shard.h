#pragma once

#include <cstdint>

// Internal ShardLLM seam hook shared between llama-shard.cpp (definition),
// llama-graph.cpp (the partial-forward clamp) and llama-context.cpp (materialize/evict).
// The public entry points are declared in include/llama.h (llama_shard_*).
void llama_shard_get_partial_forward(int32_t * begin, int32_t * end);
