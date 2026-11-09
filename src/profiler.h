// رافد - expert fingerprint: records which experts the router picks in every MoE layer.
//
// llama.cpp names the router's choice "ffn_moe_topk-<layer>" in the compute graph.
// The engine hooks the graph evaluation callback, reads that small int32 tensor and passes it here; we keep:
//   * cumulative counts per (layer, expert)  -> the long-term fingerprint (.rfprof)
//   * the exact sequence for generated tokens -> the trace (.rftrace) used to simulate caches
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_info.h"

namespace rafid {

class ExpertProfiler {
public:
    enum Phase { PREFILL = 0, GEN = 1 };

    struct Snapshot {
        bool supported = false;
        int  S = 0, E = 0, K = 0;              // MoE layers, experts per layer, experts used per token
        std::vector<int> moe_layers;
        std::vector<uint64_t> counts_gen;      // [S * E]
        std::vector<uint64_t> counts_prefill;  // [S * E]
        uint64_t tokens_gen = 0, tokens_prefill = 0;
        std::string trace_path;
    };

    ~ExpertProfiler();

    // prepares the counters for a model and loads its saved fingerprint (if any)
    void configure(const ModelInfo & mi, const std::string & profiles_dir);
    void shutdown();                           // save + close files

    bool supported() const { return S_ > 0; }
    void set_enabled(bool e) { enabled_.store(e && supported()); }
    bool enabled() const { return enabled_.load(); }

    void set_phase(Phase p) { phase_.store((int) p); }
    void begin_decode();                       // call before llama_decode() of one generated token
    void end_decode();                         // call after it

    bool save(std::string & err);
    void reset();                              // forget everything for this model (also deletes the trace)

    Snapshot       snapshot();
    nlohmann::json summary();

    // router output of one layer (from the engine's graph callback): ids[n_tok][k]
    void on_route(int layer, const int32_t * ids, int k, int64_t n_tok);
    std::string trace_path();

private:
    void write_trace_record();
    bool load_profile();
    bool save_locked(std::string & err);

    std::mutex mu_;
    std::atomic<bool> enabled_{false};
    std::atomic<int>  phase_{PREFILL};

    int S_ = 0, E_ = 0, K_ = 0, n_layer_ = 0;
    std::vector<int> moe_layers_;
    std::vector<int> slot_of_layer_;           // layer id -> slot, -1 if not MoE

    std::vector<uint64_t> counts_gen_, counts_prefill_;
    uint64_t tokens_gen_ = 0, tokens_prefill_ = 0;
    uint64_t unsaved_ = 0;

    std::vector<uint16_t> cur_;                // [S * K] experts of the token being decoded
    std::vector<uint8_t>  seen_;               // [S]

    std::string profile_path_, trace_path_;
    FILE *   trace_ = nullptr;
    uint64_t trace_bytes_ = 0;
    static constexpr uint64_t TRACE_MAX_BYTES = 1ull << 30; // 1 GB of raw sequences is plenty
};

} // namespace rafid
