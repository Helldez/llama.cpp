#pragma once

#include "common.cuh"

// the largest src and dst of a host call, a larger one is not supported and the scheduler gives the op to the CPU
#define GGML_CUDA_HOST_CALL_MAX_BYTES (1024*1024)

// in mapped pinned memory: the device writes the request, the host writes the reply
struct ggml_cuda_host_mailbox {
    volatile uint64_t req; // sequence number of the last request, written by the device
    volatile uint64_t ack; // sequence number of the last reply, written by the host
    uint64_t fn;
    uint64_t userdata;
    uint64_t src_size;
    uint64_t dst_size;
    alignas(16) uint8_t src[GGML_CUDA_HOST_CALL_MAX_BYTES];
    alignas(16) uint8_t dst[GGML_CUDA_HOST_CALL_MAX_BYTES];
};

bool ggml_cuda_host_call_supported(const ggml_tensor * op);

void ggml_cuda_op_host_call(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// answers the n host calls of a launched CUDA graph, in order, then returns
void ggml_cuda_host_call_serve(ggml_backend_cuda_context & ctx, int n);

