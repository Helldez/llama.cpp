#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

// Overwrite a single expert slice of an already uploaded MoE weight tensor, so a routed expert
// can be streamed into a residency slot between graph evaluations. `data` is one slice worth of
// bytes in the original gguf layout; `slot` indexes the ne02 dimension of `tensor`.
// Returns false if the tensor type or its on-device layout does not admit a per-slice write.
GGML_BACKEND_API bool ggml_backend_opencl_set_expert_slice(
        struct ggml_tensor * tensor, int slot, const void * data, size_t size);

// Substitutes for one MUL_MAT_ID dispatch: a residency pool holding a subset of the experts, and
// the routed ids rewritten from expert indices to indices into that pool.
struct ggml_opencl_moe_slots {
    const struct ggml_tensor * weights;
    const struct ggml_tensor * ids;
};

// Called before each MUL_MAT_ID dispatch. Returning true makes the dispatch read `out` instead of
// the model's own expert tensor, which lets an external engine keep only a few experts resident on
// a device that cannot hold them all. Both substitutes must live in OpenCL buffers the caller owns
// and keeps alive for the dispatch. Returning false leaves the dispatch untouched.
typedef bool (*ggml_opencl_moe_slot_hook_t)(
        const struct ggml_tensor  * weights,
        const struct ggml_tensor  * ids,
        struct ggml_opencl_moe_slots * out,
        void * user_data);

// Process-global registration; pass NULL to unregister. Register before compute starts.
GGML_BACKEND_API void ggml_backend_opencl_set_moe_slot_hook(
        ggml_opencl_moe_slot_hook_t hook, void * user_data);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
