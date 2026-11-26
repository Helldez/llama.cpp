#include "host-call.cuh"

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>

bool ggml_cuda_host_call_supported(const ggml_tensor * op) {
    const size_t n_src = GGML_PAD(ggml_nbytes(op->src[0]), 16) + (op->src[1] ? ggml_nbytes(op->src[1]) : 0);
    return ggml_is_contiguous(op->src[0]) && (op->src[1] == nullptr || ggml_is_contiguous(op->src[1])) &&
           n_src <= GGML_CUDA_HOST_CALL_MAX_BYTES && ggml_nbytes(op) <= GGML_CUDA_HOST_CALL_MAX_BYTES;
}

static ggml_cuda_host_mailbox * host_mailbox(ggml_backend_cuda_context & ctx) {
    if (ctx.host_mailbox == nullptr) {
        CUDA_CHECK(cudaHostAlloc((void **) &ctx.host_mailbox, sizeof(ggml_cuda_host_mailbox), cudaHostAllocMapped | cudaHostAllocPortable));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &ctx.host_mailbox_dev, ctx.host_mailbox, 0));
        ctx.host_mailbox->req = 0;
        ctx.host_mailbox->ack = 0;
    }
    return ctx.host_mailbox;
}

// publishes src, waits for the host to reply with the same sequence number, then copies the reply to dst
// one block: the kernels after it on the stream start only when it retires
static __global__ void k_host_call(ggml_cuda_host_mailbox * mb, const uint8_t * src, const uint64_t src0_size,
        const uint8_t * src1, const uint64_t src1_off, const uint64_t src_size, uint8_t * dst,
        const uint64_t dst_size, const uint64_t fn, const uint64_t userdata) {
    for (uint64_t i = threadIdx.x; i < src0_size; i += blockDim.x) {
        mb->src[i] = src[i];
    }
    for (uint64_t i = threadIdx.x; src1 != nullptr && i < src_size - src1_off; i += blockDim.x) {
        mb->src[src1_off + i] = src1[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mb->fn       = fn;
        mb->userdata = userdata;
        mb->src_size = src_size;
        mb->dst_size = dst_size;
        const uint64_t seq = mb->req + 1;
        __threadfence_system(); // the data before the request
        mb->req = seq;
        while (mb->ack < seq) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(256);
#endif
        }
        __threadfence_system(); // the reply after the ack
    }
    __syncthreads();

    const volatile uint8_t * reply = mb->dst;
    for (uint64_t i = threadIdx.x; i < dst_size; i += blockDim.x) {
        dst[i] = reply[i];
    }
}

void ggml_cuda_op_host_call(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src  = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const size_t src1_off = GGML_PAD(ggml_nbytes(src), 16);
    const size_t src_size = src1 ? src1_off + ggml_nbytes(src1) : ggml_nbytes(src);
    const ggml_host_call_op_params * p = (const ggml_host_call_op_params *) dst->op_params;
    ggml_cuda_host_mailbox * mb = host_mailbox(ctx);

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));

    if (status == cudaStreamCaptureStatusNone) {
        // evaluated node by node: wait here and call fn directly
        CUDA_CHECK(cudaMemcpyAsync(mb->src, src->data, ggml_nbytes(src), cudaMemcpyDeviceToHost, ctx.stream()));
        if (src1 != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(mb->src + src1_off, src1->data, ggml_nbytes(src1), cudaMemcpyDeviceToHost, ctx.stream()));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        p->fn(mb->src, src_size, mb->dst, ggml_nbytes(dst), p->userdata);
        CUDA_CHECK(cudaMemcpyAsync(dst->data, mb->dst, ggml_nbytes(dst), cudaMemcpyHostToDevice, ctx.stream()));
        // mb->dst is written again only after the next call synchronizes the stream
        return;
    }

    // captured: the host answers after the launch, see ggml_cuda_host_call_serve
    k_host_call<<<1, 256, 0, ctx.stream()>>>(ctx.host_mailbox_dev, (const uint8_t *) src->data, ggml_nbytes(src),
        src1 ? (const uint8_t *) src1->data : nullptr, src1_off, src_size, (uint8_t *) dst->data, ggml_nbytes(dst), (uint64_t) p->fn, (uint64_t) p->userdata);
    CUDA_CHECK(cudaGetLastError());
    ctx.host_calls_captured++;
}

void ggml_cuda_host_call_serve(ggml_backend_cuda_context & ctx, int n) {
    ggml_cuda_host_mailbox * mb = host_mailbox(ctx);

    for (int i = 0; i < n; ++i) {
        const uint64_t seq = ctx.host_mailbox_seq + 1;
        // spin: the device waits for this reply with the rest of the graph behind it
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t spins = 0;
        bool reported = false;
        while (mb->req < seq) {
            if (!reported && (spins & 0xfffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(3)) {
                reported = true;
                GGML_LOG_ERROR("%s: stalled: call %d of %d, waiting seq %llu, req %llu, ack %llu, captured kernels %d\n", __func__, i, n,
                    (unsigned long long) seq, (unsigned long long) mb->req, (unsigned long long) mb->ack, ctx.host_calls_captured);
            }
            if ((++spins & 0xfffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) {
                GGML_ABORT("%s: no host call request from the device after 60 s (%d of %d answered)", __func__, i, n);
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        GGML_ASSERT(mb->req == seq);

        const ggml_host_call_t fn = (ggml_host_call_t) mb->fn;
        fn(mb->src, mb->src_size, mb->dst, mb->dst_size, (void *) mb->userdata);

        std::atomic_thread_fence(std::memory_order_release);
        mb->ack = seq;
        ctx.host_mailbox_seq = seq;
    }
    ctx.host_calls_captured = 0;
}
