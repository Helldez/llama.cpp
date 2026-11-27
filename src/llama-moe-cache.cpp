#include <chrono>
#include "llama-moe-cache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }

    // give slots to the predicted experts of layer il that are not cached, at most max_fills, in the order given
    // the experts the last plan() selected (of layer `keep`) are not evicted: that layer is about to read them
    // an admitted expert gets no use: it enters the list of its class as of its last use, and is a hit if selected
    void admit(int32_t il, const int32_t * pred, size_t n_pred, int32_t keep, size_t max_fills, std::vector<fill> & fills) {
        fills.clear();
        int32_t * slots = slot_map[il];
        for (size_t i = 0; i < n_pred && fills.size() < max_fills; ++i) {
            const int32_t e = pred[i];
            if (e < 0 || e >= n_expert || slots[e] < n_slots) {
                continue;
            }
            const int32_t s = victim(keep);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, crf[key] >= 1.0f ? class_of(key) : 0);
            fills.push_back({ e, s });
        }
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

// fills slots the host writes in place (ggml_backend_buffer_get_host_base) by copying the host experts, when the caller
// gave no fill function of its own: same contract as llama_moe_cache_fill_fn, the async fills run on one worker thread
struct moe_cache_memcpy_fill {
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<llama_moe_cache_fill> queued;
    bool busy = false;
    bool quit = false;

    static void run(const llama_moe_cache_fill * fills, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            const size_t size = fills[i].w->nb[2];
            memcpy(fills[i].dst, (const uint8_t *) fills[i].w->data + fills[i].expert*size, size);
        }
    }

    void loop() {
        std::unique_lock<std::mutex> lock(mtx);
        while (true) {
            cv.wait(lock, [&] { return quit || !queued.empty(); });
            if (quit) {
                return;
            }
            std::vector<llama_moe_cache_fill> jobs;
            jobs.swap(queued);
            busy = true;
            lock.unlock();
            run(jobs.data(), jobs.size());
            lock.lock();
            busy = false;
            cv.notify_all();
        }
    }

    void fill(const llama_moe_cache_fill * fills, size_t n, bool async) {
        if (async) {
            if (n == 0) {
                return;
            }
            std::lock_guard<std::mutex> lock(mtx);
            if (!worker.joinable()) {
                worker = std::thread([this] { loop(); });
            }
            queued.insert(queued.end(), fills, fills + n);
            cv.notify_all();
            return;
        }
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !busy && queued.empty(); });
        }
        run(fills, n);
    }

    static void fill_fn(void * user_data, const llama_moe_cache_fill * fills, size_t n, bool async) {
        ((moe_cache_memcpy_fill *) user_data)->fill(fills, n, async);
    }

    ~moe_cache_memcpy_fill() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                quit = true;
            }
            cv.notify_all();
            worker.join();
        }
    }
};

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // a second backend of the device, whose queue uploads the predicted experts while the first computes
        ggml_backend_t prefetch = nullptr;
        ggml_backend_event_t prefetch_done = nullptr;
        ggml_backend_t upload = nullptr; // a backend of the device whose queue the host calls upload the misses on

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
        bool direct = false; // the host writes the banks in place (ggml_backend_buffer_get_host_base), no upload
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        std::vector<uint8_t *> banks_host; // host address of each bank, if the device is direct
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        ggml_tensor * router = nullptr;     // the router weights of this layer, to predict its experts from the layer before
        int32_t next = -1;                  // the next cached layer, whose experts this layer predicts
        ggml_tensor * pred = nullptr;       // [n_pred, n_tokens] experts of the next cached layer predicted in the last graph, or nullptr
        std::vector<int32_t> predicted;     // the experts this layer was predicted to select, by the layer before
        bool prefetched = false;            // experts of this layer are being uploaded on the prefetch queue
        std::vector<int32_t> prefetched_experts;
        struct impl * self = nullptr;       // user data of the host call that looks up the slots of this layer
        int32_t il = -1;

        // the misses of this layer computed on the host while the device computes the hits, inside one graph:
        // the same experts the graph builds, built again on host tensors and computed by the CPU worker
        struct side_graph {
            ggml_context_ptr ctx;
            ggml_cgraph * gf  = nullptr;
            ggml_tensor * cur = nullptr; // [n_embd, 1, n_tokens]
            ggml_tensor * ids = nullptr; // [n_expert_used, n_tokens] the selected experts
            ggml_tensor * mm  = nullptr; // [n_expert_used, n_tokens] the misses computed here, -1 for the others
            ggml_tensor * out = nullptr; // [n_embd, n_expert_used, n_tokens]
            ggml_gallocr_ptr alloc;
            ggml_context_ptr ctx_in;     // cur, ids, mm: allocated on their own, the graph may not read all of them
            ggml_backend_buffer_ptr buf_in;
            bool ok      = false; // built and readable on the host: the lookup may leave misses to it
            bool running = false; // the worker computes it for the current lookup
        } side;
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    // how well the router of a layer, applied to the input of the layer before, predicts its experts
    struct {
        size_t selected = 0, selected_pred = 0; // selected experts, and those predicted
        size_t misses = 0, misses_pred = 0;     // misses, and those predicted
        size_t n_pred = 0;
        size_t uploads = 0, layers = 0;         // predicted experts not cached when predicted, and the predictions made
        size_t prefetched = 0, prefetched_used = 0;
    } stats_pred;
    int32_t n_pred = 0;
    size_t  max_prefetch = 0; // uploads of predicted experts per layer, 0 = predict only
    std::vector<moe_cache_lru::fill> prefetch_fills;
    std::vector<float> pred_logits;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;

    // layers that run the very expert tensors of an earlier layer (a looped block, weights shared between layers) are
    // that layer for the cache: one slot map, one set of slots, and its experts counted once. alias[il] is the earlier
    // layer, -1 for a layer of its own; shared[il] is true for every layer of such a set.
    std::vector<int32_t> alias;
    std::vector<bool>    shared;
    int32_t canon(int32_t il) const {
        return il >= 0 && il < (int32_t) alias.size() && alias[il] >= 0 ? alias[il] : il;
    }
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    // fills the misses in place where the host writes the cache memory (see llama_set_moe_cache_fill)
    llama_moe_cache_fill_fn fill_fn = nullptr;
    void * fill_ud = nullptr;
    std::vector<llama_moe_cache_fill> fill_jobs;
    moe_cache_memcpy_fill memcpy_fill; // fills the direct devices when no fill function was set

    // the slots of group g are written in place by the host, through fill() -- never with an upload
    bool direct(const group & g) const {
        return devices[g.id].direct;
    }
    void fill(const llama_moe_cache_fill * fills, size_t n, bool async) {
        if (fill_fn != nullptr) {
            fill_fn(fill_ud, fills, n, async);
        } else {
            memcpy_fill.fill(fills, n, async);
        }
    }
    // the fill jobs of the experts fs of the layer whose host experts are `experts`, into the banks of g
    size_t push_fills(const group & g, const std::vector<ggml_tensor *> & experts, const std::vector<moe_cache_lru::fill> & fs) {
        size_t bytes = 0;
        for (size_t ip = 0; ip < experts.size(); ++ip) {
            const size_t expert_size = experts[ip]->nb[2];
            for (const auto & f : fs) {
                fill_jobs.push_back({ experts[ip], f.expert, g.banks_host[ip] + f.slot*expert_size });
                bytes += expert_size;
            }
        }
        return bytes;
    }
    // where the copy callback spends its time
    double t_wait_pref = 0, t_sync = 0, t_host = 0; size_t n_copy = 0;
    size_t n_host_call = 0;
    size_t n_side = 0; double t_side_wait = 0;

    // with host calls and a CPU share (llama_moe_cache_options::cpu_threads): the CPU worker that computes the side graphs
    ggml_backend_ptr cpu_backend;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    layer * job = nullptr;
    bool job_done = true;
    bool quit = false;

    void worker_loop() {
        for (;;) {
            layer * l = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return quit || job != nullptr; });
                if (quit) {
                    return;
                }
                l = job;
            }
            ggml_backend_graph_compute(cpu_backend.get(), l->side.gf);
            {
                std::lock_guard<std::mutex> lock(mtx);
                job = nullptr;
                job_done = true;
            }
            cv.notify_all();
        }
    }

    void start_side(layer & l) {
        std::lock_guard<std::mutex> lock(mtx);
        GGML_ASSERT(job_done);
        job = &l;
        job_done = false;
        cv.notify_all();
    }

    void wait_side() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return job_done; });
    }
    std::vector<int32_t> n_expert_used_of; // [n_layer] experts per token of the last lookup built, for its host call

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // twins of the experts that layers share (see llama_moe_cache::twin): headers only, over the experts' bytes
    ggml_context_ptr ctx_twins;
    std::map<std::pair<int32_t, const ggml_tensor *>, ggml_tensor *> twins; // (layer, experts) -> twin
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> twin_of;  // twin -> experts
    const ggml_tensor * base_of(const ggml_tensor * w) const {
        const auto it = twin_of.find(w);
        return it != twin_of.end() ? it->second : w;
    }

    bool use_host_call = false; // the lookup is a host call where the device supports it

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share,
            const llama_moe_cache_options & opts) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        use_host_call = opts.host_call;
        n_expert_used_of.assign(layers.size(), n_expert_used);
        for (size_t il = 0; il < layers.size(); ++il) {
            layers[il].self = this;
            layers[il].il   = (int32_t) il;
        }
        if (opts.host_call && opts.cpu_threads > 0 && this->cpu_share > 0.0f) {
            ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu != nullptr) {
                cpu_backend.reset(ggml_backend_dev_init(cpu, nullptr));
                auto * reg = ggml_backend_dev_backend_reg(cpu);
                auto * set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (set_n_threads != nullptr) {
                    set_n_threads(cpu_backend.get(), opts.cpu_threads);
                }
                worker = std::thread([this] { worker_loop(); });
            }
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        alias.assign(model.layers.size(), -1);
        shared.assign(model.layers.size(), false);
        std::vector<std::pair<std::vector<ggml_tensor *>, int32_t>> seen; // expert tensors of each layer cached so far
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            // the same tensors as a layer already cached: this layer reads that layer's slots
            const auto same = std::find_if(seen.begin(), seen.end(), [&](const auto & e) { return e.first == experts; });
            if (same != seen.end()) {
                alias[il]            = same->second;
                shared[il]           = true;
                shared[same->second] = true;
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            seen.emplace_back(experts, (int32_t) il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        // a twin of each expert tensor a layer shares with an earlier one, for the batches the cache does not serve
        {
            size_t n_twins = 0;
            for (size_t il = 0; il < alias.size(); ++il) {
                n_twins += alias[il] >= 0 ? llama_moe_cache_layer_experts(model.layers[il]).size() : 0;
            }
            if (n_twins > 0) {
                ggml_init_params ip = { n_twins*ggml_tensor_overhead(), nullptr, true };
                ctx_twins.reset(ggml_init(ip));
                for (size_t il = 0; il < alias.size(); ++il) {
                    if (alias[il] < 0) {
                        continue;
                    }
                    for (ggml_tensor * t : llama_moe_cache_layer_experts(model.layers[il])) {
                        ggml_tensor * tw = ggml_dup_tensor(ctx_twins.get(), t);
                        GGML_ASSERT(tw->nb[1] == t->nb[1] && tw->nb[2] == t->nb[2]);
                        tw->data   = t->data;   // the same bytes; a copy reads them through experts_of at the time it is made
                        tw->buffer = t->buffer;
                        ggml_set_name(tw, t->name);
                        twins[{ (int32_t) il, t }] = tw;
                        twin_of[tw] = t;
                    }
                }
            }
        }
        if (opts.host_call) {
            // the misses of a host call go out on a queue of their own, the queue of the graph waits for the reply
            for (device & d : devices) {
                d.upload = ggml_backend_dev_init(ggml_backend_get_device(d.backend), nullptr);
            }
        }
        // read-ahead: the next cached layer's router on this layer's input names the experts to fill while this layer computes
        if (opts.read_ahead > 0) {
            max_prefetch = (size_t) opts.read_ahead;
            n_pred = opts.read_ahead;
            for (device & d : devices) {
                if (max_prefetch == 0) {
                    break;
                }
                ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.events) {
                    continue;
                }
                d.prefetch      = ggml_backend_dev_init(dev, nullptr);
                d.prefetch_done = d.prefetch ? ggml_backend_event_new(dev) : nullptr;
            }
            // per layer of the graph: a layer that runs another layer's experts predicts and is predicted on its own
            int32_t prev = -1;
            for (size_t il = 0; il < model.layers.size(); ++il) {
                const int32_t ic = canon((int32_t) il);
                const bool cached = std::any_of(groups.begin(), groups.end(), [&](const group & g) { return std::find(g.layers.begin(), g.layers.end(), ic) != g.layers.end(); });
                if (!cached) {
                    continue;
                }
                layers[il].router = model.layers[il].ffn_gate_inp;
                if (prev >= 0 && layers[il].router != nullptr && layers[il].router->ne[1] == model.hparams.n_expert) {
                    layers[prev].next = (int32_t) il;
                }
                prev = (int32_t) il;
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
            for (size_t il = 0; il < alias.size(); ++il) {
                n_tensors_host += alias[il] >= 0 && std::find(g.layers.begin(), g.layers.end(), alias[il]) != g.layers.end(); // a slot map of its own
            }
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            // a layer that runs the experts of a layer of this group: a slot map of its own, which the graph copies to the
            // device on its own, filled from that layer's (see copy); the slots and their plan stay that layer's
            for (size_t ia = 0; ia < alias.size(); ++ia) {
                if (alias[ia] < 0 || std::find(g.layers.begin(), g.layers.end(), alias[ia]) == g.layers.end()) {
                    continue;
                }
                layer & la = layers[ia];
                la.ig       = (int32_t) ig;
                la.experts  = layers[alias[ia]].experts;
                la.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(la.slot_map, "moe_cache.slot_map-%d", (int) ia);
                layer_of[la.slot_map] = { (int32_t) ia, false };
                buf_host_size += GGML_PAD(ggml_nbytes(la.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (device & d : devices) {
                d.direct = d.buf != nullptr && ggml_backend_buffer_get_host_base(d.buf.get()) != nullptr;
            }
            for (group & g : groups) {
                if (devices[g.id].direct) {
                    for (ggml_tensor * bank : g.banks) {
                        g.banks_host.push_back((uint8_t *) ggml_backend_tensor_host_ptr(bank));
                    }
                }
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        g.lru.slot_map[il] = slots;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                quit = true;
            }
            cv.notify_all();
            worker.join();
        }
        log_stats();
        for (device & d : devices) {
            if (d.prefetch_done) {
                ggml_backend_event_free(d.prefetch_done);
            }
            if (d.upload) {
                ggml_backend_free(d.upload);
            }
            if (d.prefetch) {
                ggml_backend_free(d.prefetch);
            }
        }
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    bool can_fill_experts(const ggml_tensor * w_in, const ggml_tensor * dst) const {
        const ggml_tensor * w = base_of(w_in);
        const auto it = bindings.find(w);
        if (it == bindings.end() || !devices[groups[layers[it->second.il].ig].id].direct) {
            return false;
        }
        return ggml_backend_tensor_host_ptr(dst) != nullptr;
    }

    void fill_experts(ggml_backend_t backend, const ggml_tensor * w_in, ggml_tensor * dst, const std::vector<int32_t> & experts) {
        const ggml_tensor * w = base_of(w_in);
        if (experts.empty()) {
            return;
        }
        // the host writes dst: nothing queued may still read the bytes it held for an earlier graph
        ggml_backend_synchronize(backend);
        uint8_t * dst_host = (uint8_t *) ggml_backend_tensor_host_ptr(dst);
        GGML_ASSERT(dst_host != nullptr);
        const size_t expert_size = w->nb[2];
        fill_jobs.clear();
        for (int32_t e : experts) {
            fill_jobs.push_back({ w, e, dst_host + (size_t) e*expert_size });
        }
        fill(fill_jobs.data(), fill_jobs.size(), false);
        stats_copy.misses += experts.size();
        stats_copy.bytes  += experts.size()*expert_size;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w_in, ggml_tensor * dst, int64_t e, int64_t last) {
        const ggml_tensor * w = base_of(w_in);
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        // a layer that runs another layer's experts plans that layer's slots, and gets its slot map
        const int32_t il      = canon(it->second.first);
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        // the prediction and the read-ahead belong to this layer of the graph, the slots to the layer it reads them from
        layer & lo = layers[it->second.first];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        device & dv = devices[g.id];
        const auto tc0 = std::chrono::steady_clock::now();
        if (lo.prefetched) {
            if (direct(g)) {
                fill(nullptr, 0, false); // the predicted experts of this layer are in their slots
            } else {
                ggml_backend_event_wait(backend, dv.prefetch_done);
            }
            lo.prefetched = false;
        }

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_lookup

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        std::vector<int32_t> pred_next;
        const bool want_pred = lo.pred != nullptr && lo.next >= 0;
        if (want_pred) {
            pred_logits.resize(ggml_nelements(lo.pred));
        }
        // where the host addresses the memory, the results are read in place once the device is idle; else queued copies
        const void * sel_host  = ggml_backend_tensor_host_ptr(sel);
        const void * pred_host = want_pred ? ggml_backend_tensor_host_ptr(lo.pred) : nullptr;
        if (sel_host == nullptr) {
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        }
        if (want_pred && pred_host == nullptr) {
            ggml_backend_tensor_get_async(backend, lo.pred, pred_logits.data(), 0, ggml_nbytes(lo.pred));
        }
        const auto tc1 = std::chrono::steady_clock::now();
        ggml_backend_synchronize(backend);
        const auto tc2 = std::chrono::steady_clock::now();
        if (sel_host != nullptr) {
            memcpy(ids.data(), sel_host, ggml_nbytes(sel));
        }
        if (pred_host != nullptr) {
            memcpy(pred_logits.data(), pred_host, ggml_nbytes(lo.pred));
        }
        if (lo.pred != nullptr && lo.next >= 0) {
            pred_next.resize(pred_logits.size());
            for (size_t i = 0; i < pred_next.size(); ++i) {
                pred_next[i] = (int32_t) i;
            }
            const size_t k = std::min<size_t>(n_pred, pred_next.size());
            std::partial_sort(pred_next.begin(), pred_next.begin() + k, pred_next.end(), [&](int32_t a, int32_t b) { return pred_logits[a] > pred_logits[b]; });
            pred_next.resize(k);
        }

        // score the prediction of this layer's experts, made by the layer before, against the cache before this plan
        if (!lo.predicted.empty() && sel->src[0]->ne[1] == 1) {
            const int32_t * slots_now = g.lru.slot_map[il];
            for (int32_t e : ids) {
                const bool pred_hit = std::find(lo.predicted.begin(), lo.predicted.end(), e) != lo.predicted.end();
                stats_pred.selected++;
                stats_pred.selected_pred += pred_hit;
                if (slots_now[e] >= g.n_slots) {
                    stats_pred.misses++;
                    stats_pred.misses_pred += pred_hit;
                }
            }
        }
        for (int32_t e : lo.prefetched_experts) {
            stats_pred.prefetched_used += std::find(ids.begin(), ids.end(), e) != ids.end();
        }
        lo.prefetched_experts.clear();
        lo.predicted.clear();
        if (!pred_next.empty()) {
            layers[lo.next].predicted = pred_next;
            const group & gn = groups[layers[lo.next].ig];
            stats_pred.layers++;
            for (int32_t e : pred_next) {
                stats_pred.uploads += gn.lru.slot_map[canon(lo.next)][e] >= gn.n_slots;
            }
        }

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        if (direct(g)) {
            // the backend is synchronized above: no kernel reads a slot while the host writes it
            fill_jobs.clear();
            bytes = push_fills(g, l.experts, fills);
            if (!fill_jobs.empty()) {
                fill(fill_jobs.data(), fill_jobs.size(), false);
            }
        }
        for (size_t ip = 0; !direct(g) && ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        // nothing queued reads dst now (synchronized above): where the host writes it in place, a store is the copy
        if (void * dst_host = ggml_backend_tensor_host_ptr(dst)) {
            memcpy(dst_host, l.slot_map->data, ggml_nbytes(l.slot_map));
        } else {
            ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));
        }

        // upload the predicted experts of the next layer of the same group while this layer computes
        if (max_prefetch > 0 && !pred_next.empty() && layers[lo.next].ig == l.ig && (dv.prefetch != nullptr || direct(g))) {
            layer & ln = layers[lo.next];
            g.lru.admit(canon(lo.next), pred_next.data(), pred_next.size(), il, max_prefetch, prefetch_fills);
            if (!prefetch_fills.empty() && direct(g)) {
                // written while this layer computes: the victims are no slot this layer reads
                fill_jobs.clear();
                push_fills(g, ln.experts, prefetch_fills);
                fill(fill_jobs.data(), fill_jobs.size(), true);
                ln.prefetched = true;
                for (const auto & f : prefetch_fills) {
                    ln.prefetched_experts.push_back(f.expert);
                }
                stats_pred.prefetched += prefetch_fills.size();
            } else if (!prefetch_fills.empty()) {
                for (size_t ip = 0; ip < ln.experts.size(); ++ip) {
                    const ggml_tensor * w = ln.experts[ip];
                    const size_t expert_size = w->nb[2];
                    for (const auto & f : prefetch_fills) {
                        ggml_backend_tensor_set_async(dv.prefetch, g.banks[ip], (const uint8_t *) w->data + f.expert*expert_size, f.slot*expert_size, expert_size);
                    }
                }
                ggml_backend_event_record(dv.prefetch_done, dv.prefetch);
                ln.prefetched = true;
                for (const auto & f : prefetch_fills) {
                    ln.prefetched_experts.push_back(f.expert);
                }
                stats_pred.prefetched += prefetch_fills.size();
            }
        }

        {
            const auto tc3 = std::chrono::steady_clock::now();
            t_wait_pref += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
            t_sync      += std::chrono::duration<double, std::milli>(tc2 - tc1).count();
            t_host      += std::chrono::duration<double, std::milli>(tc3 - tc2).count();
            n_copy++;
        }
        return true;
    }

    // build the experts of layer l again on host tensors, for the CPU worker; not ok if any weight is not in host memory
    void build_side(const llm_graph_context & g, layer & l, ggml_tensor * cur, ggml_tensor * sel, llama_moe_cache_build_experts_t build, void * user_data) {
        auto & sd = l.side;
        GGML_ASSERT(!sd.running);
        sd = {};
        constexpr size_t max_nodes = 256;
        ggml_init_params ip = { max_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false), nullptr, true };
        sd.ctx.reset(ggml_init(ip));
        ggml_context * c = sd.ctx.get();
        // the inputs live in their own buffer: the host writes all three before each run, whichever the graph reads
        ggml_init_params ip_in = { 3*ggml_tensor_overhead(), nullptr, true };
        sd.ctx_in.reset(ggml_init(ip_in));
        sd.cur = ggml_new_tensor_3d(sd.ctx_in.get(), GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2]);
        sd.ids = ggml_new_tensor_2d(sd.ctx_in.get(), GGML_TYPE_I32, sel->ne[0], sel->ne[1]);
        sd.mm  = ggml_new_tensor_2d(sd.ctx_in.get(), GGML_TYPE_I32, sel->ne[0], sel->ne[1]);
        sd.buf_in.reset(ggml_backend_alloc_ctx_tensors_from_buft(sd.ctx_in.get(), ggml_backend_get_default_buffer_type(cpu_backend.get())));
        if (sd.buf_in == nullptr) {
            return;
        }
        for (ggml_tensor * t : { sd.cur, sd.ids, sd.mm }) {
            ggml_set_input(t);
        }
        // the graph builder reads its context from g: build these nodes into the side context instead
        auto & gm = const_cast<llm_graph_context &>(g);
        ggml_context * saved = gm.ctx0;
        gm.ctx0 = c;
        sd.out = build(sd.cur, sd.ids, sd.mm, false, user_data);
        gm.ctx0 = saved;
        if (sd.out == nullptr || sd.out->type != GGML_TYPE_F32 || !ggml_is_contiguous(sd.out)) {
            return;
        }
        ggml_set_output(sd.out);
        sd.gf = ggml_new_graph_custom(c, max_nodes, false);
        ggml_build_forward_expand(sd.gf, sd.out);
        for (int i = 0; i < ggml_graph_n_nodes(sd.gf); ++i) {
            ggml_tensor * node = ggml_graph_node(sd.gf, i);
            if (!ggml_backend_supports_op(cpu_backend.get(), node)) {
                return;
            }
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                const ggml_tensor * s = node->src[j];
                if (s == nullptr || s == sd.cur || s == sd.ids || s == sd.mm || s->op != GGML_OP_NONE) {
                    continue;
                }
                const ggml_tensor * base = s->view_src ? s->view_src : s;
                if (base->buffer == nullptr || !ggml_backend_buffer_is_host(base->buffer)) {
                    return; // a weight only the device holds (a bias, a scale): this layer keeps its misses on the device
                }
            }
        }
        sd.alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend.get())));
        if (!ggml_gallocr_alloc_graph(sd.alloc.get(), sd.gf)) {
            return;
        }
        sd.ok = true;
    }

    // the lookup inside the graph (ggml_host_call): the device waits while the host plans the selected experts of a layer,
    // uploads the misses on its own queue and replies with their slots, so the layer needs no split and no host sync
    // the stream of the graph runs nothing else meanwhile: no kernel reads a slot while it is written

    static void host_call(const void * src, size_t src_size, void * dst, size_t dst_size, void * user_data) {
        const layer & lu = *(const layer *) user_data;
        // with a side graph the inputs are the ids, then the expert input at the next 16 bytes (ggml_host_call_2)
        const size_t n_ids = dst_size/sizeof(int32_t);
        const float * cur = n_ids*sizeof(int32_t) < src_size ? (const float *) ((const uint8_t *) src + GGML_PAD(n_ids*sizeof(int32_t), 16)) : nullptr;
        lu.self->serve(lu.il, (const int32_t *) src, n_ids, (int32_t *) dst, n_ids, cur);
    }

    // the misses the CPU worker computed for this layer, zero for the other experts
    static void host_collect(const void * src, size_t src_size, void * dst, size_t dst_size, void * user_data) {
        GGML_UNUSED(src);
        GGML_UNUSED(src_size);
        layer & l = *(layer *) user_data;
        impl & self = *l.self;
        if (!l.side.running) {
            memset(dst, 0, dst_size);
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        self.wait_side();
        self.t_side_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        l.side.running = false;
        GGML_ASSERT(ggml_nbytes(l.side.out) == dst_size);
        memcpy(dst, l.side.out->data, dst_size);
    }

    void serve(int32_t il, const int32_t * sel, size_t n_ids, int32_t * slots_out, size_t n_out, const float * cur_in = nullptr) {
        GGML_ASSERT(n_out == n_ids);
        layer & l = layers[il];
        group & g = groups[l.ig];
        const auto tc0 = std::chrono::steady_clock::now();

        ids.assign(sel, sel + n_ids);
        const int32_t n_used = n_expert_used_of[il];
        const int64_t n_tokens = (int64_t) n_ids/n_used;

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        size_t n_hit = 0;
        // misses left to the host only with a side graph, and only once the cache is full (as for the CPU part of a split)
        const bool side = cur_in != nullptr && l.side.ok && cpu_backend != nullptr;
        const uint32_t share_q = side && g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        if (!g.lru.plan(il, ids.data(), ids.size(), n_tokens, n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (!cpu.empty()) {
            // the host computes these misses while the device computes the rest, the collect call adds them
            GGML_ASSERT(side && ggml_nelements(l.side.ids) == (int64_t) n_ids);
            memcpy(l.side.cur->data, cur_in, ggml_nbytes(l.side.cur));
            memcpy(l.side.ids->data, ids.data(), n_ids*sizeof(int32_t));
            int32_t * mm = (int32_t *) l.side.mm->data;
            for (size_t i = 0; i < n_ids; ++i) {
                mm[i] = -1;
                for (const auto & f : cpu) {
                    if (f.expert == ids[i]) {
                        mm[i] = ids[i];
                        break;
                    }
                }
            }
            l.side.running = true;
            start_side(l);
            n_side++;
        }

        size_t bytes = 0;
        if (direct(g)) {
            // the device waits for this reply: no kernel reads a slot while the host writes it
            fill_jobs.clear();
            bytes = push_fills(g, l.experts, fills);
            if (!fill_jobs.empty()) {
                fill(fill_jobs.data(), fill_jobs.size(), false);
            }
        } else {
            // on the upload queue when there is one, synchronized once below; else one synchronous set each
            // never on the queue of the graph, which waits for this reply
            ggml_backend_t up = devices[g.id].upload;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w    = l.experts[ip];
                ggml_tensor       * bank = g.banks[ip];
                const size_t expert_size = w->nb[2];
                for (size_t i = 0; i < fills.size();) {
                    size_t n = 1;
                    while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                        n++;
                    }
                    if (up != nullptr) {
                        ggml_backend_tensor_set_async(up, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                    } else {
                        ggml_backend_tensor_set(bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                    }
                    bytes += n*expert_size;
                    i += n;
                }
            }
            if (up != nullptr && !fills.empty()) {
                ggml_backend_synchronize(up);
            }
        }

        const int32_t * slot_map = g.lru.slot_map[il];
        for (size_t i = 0; i < n_ids; ++i) {
            slots_out[i] = slot_map[ids[i]];
        }

        stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        t_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
        n_host_call++;
    }

    void log_stats() const {
        if (n_host_call > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: host calls %zu: host %.1f ms, side graphs %zu, waited %.1f ms\n", n_host_call, t_host, n_side, t_side_wait);
        }
        if (n_copy > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: copy callbacks %zu: wait prefetch %.1f ms, wait device %.1f ms, host %.1f ms\n", n_copy, t_wait_pref, t_sync, t_host);
        }
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        if (stats_pred.selected > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: predicted with top-%d of the next router on the layer input: %.1f%% of the selected experts, %.1f%% of the misses (%zu misses)\n",
                n_pred, 100.0*stats_pred.selected_pred/stats_pred.selected, 100.0*stats_pred.misses_pred/std::max<size_t>(1, stats_pred.misses), stats_pred.misses);
            if (stats_pred.prefetched > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: prefetched %zu experts, %.1f%% of them selected by their layer\n", stats_pred.prefetched, 100.0*stats_pred.prefetched_used/stats_pred.prefetched);
            }
            LLAMA_LOG_INFO("llama_moe_cache: predicted experts not cached: %.2f per layer (misses: %.2f per layer)\n",
                (double) stats_pred.uploads/std::max<size_t>(1, stats_pred.layers), (double) stats_pred.misses*n_expert_used/std::max<size_t>(1, stats_pred.selected));
        }
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share,
        const llama_moe_cache_options & opts) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share, opts)) {
}

llama_moe_cache::~llama_moe_cache() = default;

bool llama_moe_cache::set_fill(llama_moe_cache_fill_fn fn, void * user_data) {
    bool any = false;
    for (const auto & d : pimpl->devices) {
        if (d.buf != nullptr) {
            if (!d.direct) {
                return false; // the host has no address for this cache memory: its misses are uploaded
            }
            any = true;
        }
    }
    if (!any) {
        return false;
    }
    pimpl->fill_fn = fn;
    pimpl->fill_ud = user_data;
    return true;
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts, ggml_tensor * cur) const {
    // a layer that runs an earlier layer's experts reads that layer's cache (see impl::alias) through a slot map of its
    // own; the graph's nodes keep the layer's own index in their names
    const int32_t il = pimpl->canon(layer.il);
    ggml_context * ctx0 = g.ctx0;

    // the experts of the next cached layer of the graph, predicted with its router on this layer's expert input
    if (layer.il >= 0 && layer.il < (int32_t) pimpl->layers.size()) {
        auto & l = pimpl->layers[layer.il]; // the prediction is this layer of the graph's, not the layer whose slots it reads
        l.pred = nullptr;
        if (l.next >= 0 && cur != nullptr && selected_experts->ne[1] == 1) {
            ggml_tensor * router = pimpl->layers[l.next].router;
            // the host ranks the logits while it plans the lookup: a sort on the device costs more than the matmul
            ggml_tensor * logits = ggml_mul_mat(ctx0, router, ggml_reshape_2d(ctx0, cur, cur->ne[0], ggml_nelements(cur)/cur->ne[0]));
            ggml_set_output(logits);
            ggml_build_forward_expand(g.gf, logits);
            l.pred = logits;
        }
    }

    // part of the misses may be computed from the host experts, but not with weights before the FFN, LoRAs or rotated experts
    bool cpu_part = !layer.weight_before_ffn && g.loras->empty();
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        cpu_part = cpu_part && !(w != nullptr && g.hdmd != nullptr && g.hdmd->rot.count(w) > 0);
    }

    ggml_tensor * cpu_map  = cpu_part ? pimpl->get_cpu_map(layer.il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = pimpl->get_slot_map(layer.il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    if (slot_map == nullptr) {
        return {};
    }
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = pimpl->get_experts(w);
        if (experts == nullptr) {
            return {};
        }
        // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == layer.down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return {};
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the lookup as a host call inside the graph, when the device can wait for the host in a graph
    // without a CPU part or a prediction, which read the host slot map and the router logits
    const bool use_host_call = pimpl->use_host_call;
    ggml_tensor * slots = nullptr;
    // with a CPU worker the misses left to the host are computed inside the graph too (see build_experts)
    bool side = use_host_call && cpu_map != nullptr && pimpl->cpu_backend != nullptr && cur != nullptr;
    if (side) {
        // the CPU worker's misses come back through a host call as wide as the experts' output, which a backend
        // bounds (a mailbox of fixed size): a graph whose output does not fit keeps the CPU part as a split
        ggml_tensor * probe = ggml_host_call(ctx0, ids, GGML_TYPE_F32, cur->ne[0], selected_experts->ne[0]*selected_experts->ne[1],
                &impl::host_collect, &pimpl->layers[il]);
        side = ggml_backend_supports_op(pimpl->backend(il), probe);
    }
    ggml_tensor * cpu_in = nullptr;
    if (use_host_call && (cpu_map == nullptr || side) && pimpl->layers[layer.il].pred == nullptr) {
        cpu_in = side ? (ggml_is_contiguous(cur) ? cur : ggml_cont(ctx0, cur)) : nullptr;
        slots = ggml_host_call_2(ctx0, ids, cpu_in, GGML_TYPE_I32, ggml_nelements(ids), 1, &impl::host_call, &pimpl->layers[il]);
        if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
            slots  = nullptr;
            cpu_in = nullptr;
        } else {
            pimpl->n_expert_used_of[il] = (int32_t) selected_experts->ne[0];
        }
    }
    if (slots == nullptr) {
        // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
        // the callback reads the selected experts, uploads the missing ones and updates the slot map
        slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
        if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
            return {};
        }
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", layer.il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(g.gf, slots);

    // add the views of the cached experts with it, so that no view lands between the expert matmuls and keeps a backend from fusing them
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w != nullptr) {
            ggml_build_forward_expand(g.gf, pimpl->get_experts(w));
        }
    }

    // the misses computed from the host experts, read after the lookup has picked them
    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr && cpu_in == nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids); // [1, n_expert_used*n_tokens]
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    return { layer.il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids, cpu_in };
}

ggml_tensor * llama_moe_cache::build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const {
    GGML_ASSERT(lookup.slots != nullptr);

    ggml_tensor * experts = build(cur, selected_experts, lookup.slots, true, user_data);
    if (lookup.cpu_ids != nullptr) {
        // the cached experts and the misses computed from the host experts, each reads the others as zero
        // the cached experts come first in the graph, so that they are computed while the misses are
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.cpu_ids, false, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }
    if (lookup.cpu_in != nullptr) {
        // the misses left to the host, computed by the CPU worker while the device computes these experts,
        // collected after them and added in the same order as the CPU part of a split
        auto & l = pimpl->layers[pimpl->canon(lookup.il)];
        pimpl->build_side(g, l, cur, selected_experts, build, user_data);
        if (l.side.ok) {
            ggml_tensor * after = ggml_view_1d(g.ctx0, experts, 1, 0);
            ggml_tensor * cpu = ggml_host_call(g.ctx0, after, GGML_TYPE_F32, experts->ne[0], experts->ne[1]*experts->ne[2], &impl::host_collect, &l);
            ggml_backend_sched_set_tensor_backend(g.sched, cpu, pimpl->backend(pimpl->canon(lookup.il)));
            cpu = ggml_reshape_3d(g.ctx0, cpu, experts->ne[0], experts->ne[1], experts->ne[2]);
            experts = ggml_add(g.ctx0, experts, cpu);
            g.cb(experts, "ffn_moe_down_merged", lookup.il);
        }
    }

    return experts;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

bool llama_moe_cache::can_fill_experts(const ggml_tensor * w, const ggml_tensor * dst) const {
    return pimpl->can_fill_experts(w, dst);
}

ggml_tensor * llama_moe_cache::twin(int32_t il, const ggml_tensor * w) const {
    const auto it = pimpl->twins.find({ il, w });
    return it != pimpl->twins.end() ? it->second : nullptr;
}

const ggml_tensor * llama_moe_cache::experts_of(const ggml_tensor * w) const {
    return pimpl->base_of(w);
}

void llama_moe_cache::fill_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, const std::vector<int32_t> & experts) {
    pimpl->fill_experts(backend, w, dst, experts);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
