// رافد - the planner: decides where every part of the model lives (VRAM / RAM / disk)
// and estimates the generation speed from the measured memory bandwidths.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_info.h"
#include "hw_probe.h"

namespace rafid {

struct PlanOptions {
    int      n_ctx        = 8192;
    int      gpu_slot     = -1;    // index into HwInfo::gpus, -1 = best discrete GPU
    int      n_cpu_moe    = -1;    // MoE layers whose experts stay on the CPU, -1 = automatic
    int      n_gpu_layers = -1;    // layers offloaded to the GPU, -1 = automatic
    bool     moe_cache    = false; // experts in RAM + a GPU cache of hot experts (llama.cpp, experimental)
    uint64_t moe_cache_bytes = 0;  // 0 = all VRAM that is left
    double   cache_hit_hint  = 0.5;// expected cache hit rate (from the expert fingerprint when known)
    uint64_t vram_margin  = 0;     // 0 = automatic
    uint64_t ram_margin   = 0;     // 0 = automatic
    int      n_threads       = -1;
    int      n_threads_batch = -1;

    // experts on disk + Rafid's RAM cache (expert store)
    bool        stream_experts      = false;
    uint64_t    stream_budget_bytes = 0;   // 0 = 20% of the expert bytes
    std::string stream_policy       = "freq";
    std::string stream_prefetch     = "gate"; // off | history | gate
    std::string stream_swap         = "off";  // cache-aware routing: off | light | medium | strong
    double      stream_hit_hint     = -1;  // expected cache hit rate (from the fingerprint), <0 = unknown
};

struct Plan {
    std::string mode;              // gpu_full | hybrid_moe | hybrid_moe_cache | hybrid_layers | cpu_only
    int  gpu_slot        = -1;
    std::string gpu_name;
    int  n_layer         = 0;      // layers used for decoding
    int  n_gpu_layers    = 0;      // value passed to llama.cpp (n_layer + 1 = everything incl. output)
    int  n_moe_layers    = 0;
    int  n_cpu_moe       = 0;      // MoE layers whose experts stay in RAM
    bool output_on_gpu   = false;
    std::vector<int> cpu_expert_layers;
    std::vector<int> layer_place;  // per layer: 0 = CPU, 1 = GPU, 2 = GPU with experts in RAM
    std::string override_pattern;  // regex: expert tensors forced to the CPU

    uint64_t vram_free = 0, gpu_budget = 0, gpu_weights = 0, gpu_kv = 0, gpu_compute = 0, gpu_cache = 0;
    uint64_t ram_avail = 0, ram_budget = 0, cpu_weights = 0, cpu_kv = 0, cpu_compute = 0;
    uint64_t disk_overflow = 0;    // weights that do not fit in RAM and are paged from disk
    bool     use_mmap = false;

    bool        stream = false;    // experts on disk, Rafid's RAM cache
    uint64_t    stream_budget = 0;
    uint64_t    stream_experts_bytes = 0;
    uint64_t    stream_staging = 0;// resident buffers for reading prompts (full layers of experts)
    uint64_t    stream_budget_fit = 0; // the biggest cache that fits in the RAM left by everything else
    uint64_t    cpu_snapshots = 0; // snapshots of the recurrent state (hybrid models) kept between turns
    double      stream_hit = 0;    // hit rate used for the estimate
    bool        stream_hit_known = false;
    std::string stream_policy;
    std::string stream_prefetch;
    std::string stream_swap;

    // memory traffic for one generated token
    double tok_gpu_bytes = 0, tok_cpu_bytes = 0, tok_pcie_bytes = 0, tok_disk_bytes = 0;
    double bw_gpu = 0, bw_cpu = 0, bw_pcie = 0, bw_disk = 0; // GB/s used for the estimate
    double t_gpu = 0, t_cpu = 0, t_pcie = 0, t_disk = 0, t_sync = 0; // seconds per token (mid estimate)
    double est_tps_low = 0, est_tps_high = 0;

    int n_ctx = 0, n_threads = 0, n_threads_batch = 0;
    int n_ubatch = 512;            // tokens per compute pass while reading a prompt
    std::vector<std::string> warnings;
};

Plan make_plan(const HwInfo & hw, const ModelInfo & mi, const PlanOptions & opt);
nlohmann::json plan_to_json(const Plan & p);

// Same plan with more of the model on the CPU, used when loading runs out of VRAM.
bool plan_step_back(const HwInfo & hw, const ModelInfo & mi, PlanOptions & opt, const Plan & failed);

} // namespace rafid
