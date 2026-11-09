// رافد - expert store: routed MoE experts stay on disk, Rafid keeps a RAM cache of them.
//
// How it plugs into llama.cpp (public C API only):
//   * the expert tensors (blk.N.ffn_*_exps) are placed with tensor_buft_overrides into a custom host buffer
//     type whose memory is only *reserved* address space ("home"); the model is loaded with mmap, so
//     llama.cpp never copies expert data into it
//   * the cache is a fixed set of slots per layer ("pool"), allocated once and reused: no memory is
//     requested from or returned to the OS while the model generates (on Windows that costs ~1 ms per expert)
//   * the graph evaluation callback sees the router's choice ("ffn_moe_topk-<layer>") before the expert
//     matmuls of that layer run; on_route() makes the experts resident in the layer's slots and points the
//     layer's expert tensors at the pool; at "ffn_moe_weights-<layer>" (after the routing weights were taken
//     with the real ids) the ids are rewritten to slot numbers, so the matmuls read the slots directly
//   * a prompt batch needs more experts than a layer's slots: they are gathered in a resident full-layer
//     buffer ("staging", read straight from the disk or copied from the pool) and the layer points there;
//     while one layer computes, the next one is gathered in the second staging buffer
//   * while a layer computes, the experts the next layer will probably need are read into its slots:
//       "gate"    - the next layer's own router applied to the current hidden state (new in Rafid)
//       "history" - transitions between layers learned from your fingerprint trace
//   * safety net (Windows): an access to the home memory loads that expert there (fault handler)
#pragma once

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml-backend-impl.h"

#include "gguf_info.h"

namespace rafid {

struct StreamConfig {
    uint64_t    budget_bytes = 2ull << 30;  // RAM for routed experts
    // "lru" = least recently used goes first; "freq" = the lowest use count with decay (half-life freq_half
    // tokens) goes first; "fingerprint" = hot experts pinned + LRU
    std::string policy       = "freq";
    double      pin_frac     = 0.7;         // share of the slots given to pinned experts ("fingerprint")
    double      freq_half    = 32;          // "freq": tokens after which a use counts half
    std::string prefetch     = "gate";      // "off" | "history" | "gate"
    int         prefetch_n   = 0;           // experts prefetched per layer, 0 = experts used per token
    // cache-aware routing (changes the result slightly): "off" | "light" | "medium" | "strong". When a
    // low-ranked expert of a token is not in RAM, an expert that is in RAM and that the router weighs nearly as
    // much takes its place instead of waiting for the disk
    std::string swap         = "off";
};

// "on"/true -> "gate", "false"/"none" -> "off"; unknown values -> "gate"
std::string normalize_prefetch_mode(const std::string & s);
std::string normalize_policy(const std::string & s);   // lru | freq | fingerprint
std::string normalize_swap_mode(const std::string & s); // off | light | medium | strong
// for k experts per token: first rank (1 = highest weight) that may be replaced (0 = off), and how much weight
// the replacement needs relative to the expert it replaces
void swap_params(const std::string & mode, int k, int & from_rank, double & ratio);

struct StreamStats {
    uint64_t routed = 0;                       // router decisions seen (layers x batches)
    uint64_t accesses = 0;                     // experts needed
    uint64_t hits = 0;                         // already in RAM
    uint64_t inflight = 0;                     // being read already (mostly prefetches)
    uint64_t misses = 0;                       // had to be read now
    uint64_t prefetch_issued = 0, prefetch_used = 0, prefetch_cancelled = 0;
    uint64_t pred_total = 0, pred_hits = 0;    // next-layer prediction: predicted experts / of them really used
    uint64_t pred2_total = 0, pred2_hits = 0;  // same, two layers ahead (measured only, "gate" mode)
    uint64_t pred_late = 0;                    // predictions that arrived after the layer was routed
    uint64_t bulk_issued = 0;                  // experts gathered ahead during prompt processing
    uint64_t routes_pool = 0, routes_staged = 0; // layers computed from the slots / from a staging buffer
    uint64_t bytes_read = 0, reads = 0, bytes_copied = 0, evictions = 0;
    uint64_t over_budget = 0, fault_loads = 0, io_errors = 0;
    uint64_t swap_candidates = 0, swapped = 0; // cache-aware routing: missing experts it could replace / replaced
    uint64_t ahead_issued = 0, ahead_used = 0; // prompts: experts gathered ahead from the gate's guesses / needed
    double   wait_sec = 0;                     // time the model waited for the disk
    double   io_busy_sec = 0;                  // wall time with at least one read in flight

    StreamStats delta(const StreamStats & before) const;
    nlohmann::json to_json() const;
};

class ExpertStore {
public:
    // simd: the CPU has AVX2 + FMA + F16C (fast router matvec for the gate predictor)
    explicit ExpertStore(const ModelInfo & mi, bool simd = false);
    ~ExpertStore();
    ExpertStore(const ExpertStore &) = delete;
    ExpertStore & operator=(const ExpertStore &) = delete;

    ggml_backend_buffer_type_t buft() { return &buft_; }

    // after llama_model_load_from_file: checks that every expert tensor was placed in this store
    bool finalize(std::string & err);

    // budget / policy / prefetch; counts = fingerprint [S*E] (may be empty), trace = .rftrace path (may be empty)
    void configure(const StreamConfig & cfg, const std::vector<uint64_t> & counts, const std::string & trace_path);
    void   reset_cache();             // forget every unpinned expert (waits until the readers are idle)
    double preload(bool wait);        // pinned experts + the most used ones into the free slots; returns seconds waited
    void   prompt_done();             // after a prompt: big staging buffers are given back to the OS
    void   shutdown_io();             // stop the reader threads (before the model is freed)

    // graph callback, router output of one layer (ids[n_tok][k]): makes the experts resident and points the
    // layer's expert tensors at them. hidden: for one token, the residual stream entering this layer's FFN norm
    // (hidden_normed = false) or the normalized router input (true); used by the "gate" predictor, may be null
    // probs: the router's weights of all experts for the token (n_tok == 1), needed by cache-aware routing; when
    // it replaces experts, ids is rewritten and true is returned (the caller writes ids back into the graph)
    // hidden: hidden_rows rows of n_embd (the token for generation, the batch for a prompt)
    bool on_route(int layer, int32_t * ids, int k, int64_t n_tok, const float * hidden, int64_t hidden_rows,
                  bool hidden_normed, const float * probs = nullptr);
    bool wants_probs() const { return swap_from_.load() > 0; }
    bool wants_batch_hidden() const { return gate_on_.load(); } // prompt read-ahead guesses from every token
    // graph callback after the routing weights were taken: ids (n values, as on_route saw them) are rewritten
    // to slot numbers when the layer computes from the pool; returns true when ids was changed
    bool on_weights(int layer, int32_t * ids, int64_t n);

    // the gate predictor: router weights of a layer ([n_exp][n_embd]) and its FFN norm weight (may be null)
    bool wants_router(int layer);
    bool router_ready(int s);
    void set_router(int layer, const float * gate, int64_t n_embd, int64_t n_exp, const float * norm_w);

    StreamStats    stats();
    StreamConfig   config();
    nlohmann::json status();
    uint64_t       total_bytes() const { return total_bytes_; }
    uint64_t       min_budget() const;
    int64_t        n_embd() const { return n_embd_; }

    // Windows: access to the home memory of an expert (safety net, counted in fault_loads).
    // Other threads may read a faulted expert while it is still being filled, so the engine recomputes
    // any decode during which fault_count() changed.
    bool contains(uintptr_t addr) const;
    bool handle_fault(uintptr_t addr);
    uint64_t fault_count() const { return fault_count_.load(); }

private:
    struct MTensor {
        std::string   name;
        ggml_tensor * t      = nullptr;
        int      layer   = -1;
        int      slot    = -1;
        int      part    = -1;      // same role (gate / up / down / gate_up) has the same index in every layer
        bool     managed = false;   // routed experts, read on demand
        uint8_t * home   = nullptr; // reserved address space; (home - foff) is page aligned
        uint64_t nbytes  = 0;
        uint64_t nb2     = 0;       // bytes of one expert
        int64_t  n_exp   = 0;
        int      file    = 0;
        uint64_t foff    = 0;       // file offset of the tensor data
    };
    struct Region {
        ExpertStore * store = nullptr;
        uint8_t * base = nullptr;
        size_t    size = 0;
    };
    // one cache slot of one layer
    struct Entry {
        int32_t  expert = -1;
        uint8_t  state = 0;         // FREE / LOADING / READY
        uint8_t  parts_left = 0;
        uint8_t  prefetched = 0, demote = 0, started = 0, err = 0, pinned = 0, in_lru = 0;
        uint16_t pins = 0;          // staging copies reading this slot right now
        uint32_t use_epoch = 0, gen = 0;
        int32_t  prev = -1, next = -1;
        double   fkey = -INFINITY;  // "freq": log of the uses weighted by exp(lambda * time); lowest goes first
    };
    struct Staging {
        uint8_t * block = nullptr;
        size_t    size = 0;
        std::vector<uint8_t *> base; // [P] page aligned start of each part
        int       slot = -1;         // layer whose experts it holds
        uint32_t  gen = 0;           // bumped when it is given to another layer (stale reads are skipped)
        std::vector<uint8_t> state, left, err; // [E]
        int       active = 0;        // reads / copies into this buffer that are running right now
        bool      claiming = false;  // being emptied for another layer (by the main or the predictor thread)
    };
    enum : uint8_t { J_POOL = 0, J_STAGE_READ = 1, J_STAGE_COPY = 2 };
    struct Run {
        uint8_t  kind = 0;
        int32_t  slot = -1, part = -1, e0 = 0, e1 = 0;
        int32_t  entry = -1;        // J_POOL: target slot entry, J_STAGE_COPY: source entry
        int32_t  buf = -1;          // staging buffer (J_STAGE_*)
        uint32_t gen = 0;
        int32_t  jobs_left = 0;
        uint8_t  error = 0;
    };
    struct Job {
        int32_t  run;
        uint64_t f0, f1;            // page aligned file range (reads)
    };
    struct Router {
        std::vector<uint16_t> gate; // [E][n_embd] fp16
        std::vector<float>    norm; // [n_embd], empty = none
        bool ok = false;
    };
    struct PredReq {
        int slot = -1;
        uint64_t seq = 0;
        bool hidden_ok = false, normed = false;
        bool prompt = false;           // a batch: guess the union of the next layer's experts for these rows
        int  rows = 1;                 // hidden holds rows * n_embd floats
        std::vector<float> hidden;
        std::vector<int>   chosen;
    };

    enum : uint8_t { FREE = 0, LOADING = 1, READY = 2 };
    enum : uint8_t { MODE_NONE = 0, MODE_POOL = 1, MODE_STAGE = 2 };

    // ggml buffer type / buffer callbacks
    static const char *          cb_buft_name(ggml_backend_buffer_type_t);
    static ggml_backend_buffer_t cb_buft_alloc(ggml_backend_buffer_type_t, size_t);
    static size_t                cb_buft_align(ggml_backend_buffer_type_t);
    static size_t                cb_buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor *);
    static bool                  cb_buft_is_host(ggml_backend_buffer_type_t);
    static void             cb_free(ggml_backend_buffer_t);
    static void *           cb_base(ggml_backend_buffer_t);
    static enum ggml_status cb_init_tensor(ggml_backend_buffer_t, ggml_tensor *);
    static void             cb_memset(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t);
    static void             cb_set(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t);
    static void             cb_get(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t);
    static bool             cb_cpy(ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *);
    static void             cb_clear(ggml_backend_buffer_t, uint8_t);

    MTensor * tensor_of(const ggml_tensor * t);
    MTensor * tensor_at_home(uintptr_t addr);
    const MTensor * part_tensor(int slot, int p) const { const int ti = slot_part_[(size_t) slot * P_ + p]; return ti >= 0 ? &tensors_[(size_t) ti] : nullptr; }
    int  key_of(int slot, int e) const { return slot * E_ + e; }
    uint8_t * slot_mem(int entry, int p) const;
    uint8_t * stage_ptr(int b, int slot, int p) const;

    // all of the following: caller holds mu_
    void lru_remove(int q);
    void lru_push_front(int q);
    void lru_push_back(int q);
    int  pick_victim(int slot, bool prefetch, const std::vector<uint8_t> * needed);
    void assign_entry(int q, int slot, int e);
    void load_entry(int q, int prio);                   // reads the entry's expert into its slot
    void stage_experts(int b, int slot, const std::vector<int> & ex, int prio, bool copy_from_pool);
    bool ensure_staging(std::unique_lock<std::mutex> & lk);
    void claim_staging(int b, int slot, std::unique_lock<std::mutex> & lk);
    void promote_slot(int slot, const std::vector<uint8_t> & needed);
    int32_t new_run();
    void push_jobs(int32_t rid, uint64_t f0, uint64_t f1, int prio);
    void finish_job(const Job & j, bool ran, bool ok);
    void entry_loaded(int q);
    void cancel_entry(int q);
    bool run_stale(const Run & r) const;
    bool idle_locked() const;
    void quiesce(std::unique_lock<std::mutex> & lk);   // drops queued work and waits for running reads
    void set_pointers(int slot, uint8_t mode, int b);
    void restore_home_pointers();
    bool alloc_pool(std::string & err);
    void free_pool();
    void free_staging();
    void io_begin();
    void io_end();

    void worker_main(int id);
    void predictor_main();
    void build_transitions(const std::string & trace_path);
    void predict_history(int slot_next, const std::vector<int> & chosen, int n, std::vector<int> & out);
    bool predict_gate(int slot_next, const float * hidden, bool normed, int slot_from, int n, std::vector<int> & out);
    void run_batch_guess(const PredReq & rq, int sn, std::unique_lock<std::mutex> & lk);
    bool read_home_sync(int slot, int e);

    ModelInfo mi_;
    struct ggml_backend_buffer_type buft_;
    std::vector<Region *> regions_;
    std::vector<MTensor>  tensors_;
    std::vector<std::string> roles_;     // part index -> tensor role ("ffn_up_exps.weight", ...)
    std::vector<int>      slot_of_layer_;
    std::vector<int>      slot_part_;    // [S*P] tensor index or -1
    int S_ = 0, E_ = 0, K_ = 0, P_ = 0;
    uint64_t total_bytes_ = 0, max_key_bytes_ = 0;
    std::vector<uint64_t> key_bytes_;    // [S] bytes of one expert of each layer
    bool finalized_ = false;

    // cache state (guarded by mu_)
    mutable std::mutex mu_;
    std::condition_variable cv_work_, cv_done_, cv_pred_;
    std::vector<Entry>    ent_;          // all slots of all layers
    std::vector<int32_t>  ent0_;         // [S+1] first entry of each layer
    std::vector<int32_t>  where_;        // [S*E] entry that holds (or loads) the expert, -1 = none
    std::vector<int32_t>  lru_head_, lru_tail_; // per layer, READY unpinned entries, head = most recent
    std::vector<uint8_t*> pool_mem_;     // [S*P] start of the slots of a layer / part
    uint8_t * pool_block_ = nullptr;
    size_t    pool_size_ = 0;
    bool      pool_locked_ = false;
    Staging   stg_[2];
    int       n_stg_ = 0;                // staging buffers in use (0 = not allocated yet)
    size_t    stg_bytes_ = 0;            // size of one staging buffer
    bool      stg_locked_ = false;
    std::vector<uint8_t>  mode_;         // [S] how the layer computes in the current step
    std::vector<uint8_t>  remap_ok_;     // [S] the ids hook was seen working for this layer
    std::vector<int16_t>  remap_;        // [S*E] expert -> slot for the current step
    std::vector<std::vector<int32_t>> last_ids_; // [S] ids seen by on_route (checked in on_weights)
    std::vector<std::vector<int>> pf_entries_;   // per layer: entries filled by a guess for it
    std::vector<Run>     runs_;
    std::vector<int32_t> free_runs_;
    std::deque<Job>      dq_, pq_;       // demand / background work
    uint32_t epoch_ = 0;
    StreamConfig cfg_;
    bool   freq_ = false;              // cfg_.policy == "freq"
    double freq_lambda_ = 0.0216;      // ln 2 / half-life
    std::atomic<int> swap_from_{0};    // cache-aware routing: first rank (1-based) that may be replaced, 0 = off
    std::atomic<bool> gate_on_{false}; // cfg_.prefetch == "gate"
    double swap_ratio_ = 0.5;
    void freq_touch(int q) {
        Entry & x = ent_[(size_t) q];
        const double now = freq_lambda_ * (double) tok_seq_;
        x.fkey = x.fkey == -INFINITY ? now : std::max(x.fkey, now) + std::log1p(std::exp(-std::fabs(x.fkey - now)));
    }
    StreamStats  stats_;
    int    io_active_ = 0, jobs_active_ = 0;
    double io_busy_start_ = 0;
    uint64_t chunk_small_ = 256 << 10, chunk_big_ = 1 << 20;
    int      bulk_tok_ = 16;             // batches this big gather the whole next layer ahead
    bool     warned_remap_ = false;
    bool     remap_safe_ = true;         // false when other tensors are also indexed by the expert ids (expert biases)

    // per token bookkeeping for predictions (guarded by mu_)
    uint64_t tok_seq_ = 1;
    int      last_slot_ = -1;
    std::vector<uint64_t> pred_seq_, pred2_seq_;
    std::vector<std::vector<int>> pred_set_, pred2_set_;

    // fingerprint data for pinning, warm start and the history predictor
    std::vector<double>   pop_;     // [S*E]
    std::vector<uint32_t> trans_;   // [(S-1)*E*E] counts: expert a at slot s -> expert b at slot s+1
    std::string trans_src_;

    // gate predictor
    std::mutex router_mu_;
    std::vector<Router> routers_;   // per slot
    std::atomic<int> routers_ok_{0};
    int64_t n_embd_ = 0;
    bool simd_ = false;
    std::thread pred_thread_;
    bool pred_has_ = false;         // a request is waiting (guarded by mu_)
    PredReq pred_req_;              // guarded by mu_

    // reader threads
    int n_io_ = 16;
    std::vector<std::thread> workers_;
    bool stop_ = false;
    bool direct_io_ = true;
    bool debug_no_hook_ = false;
    int  debug_layer_ms_ = 0;

    // fault path
    std::atomic<uint64_t> fault_count_{0};
    std::mutex fault_mu_;

    std::vector<int> scratch_ids_, scratch_miss_, scratch_list_;
    std::vector<uint8_t> scratch_seen_;
};

} // namespace rafid
