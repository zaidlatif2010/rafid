#include "planner.h"
#include "util.h"

#include <algorithm>
#include <cmath>

namespace rafid {

static constexpr uint64_t MiB = 1ull << 20;

// rough size of llama.cpp's compute buffers for n_ubatch tokens per pass
static uint64_t compute_buffer_gpu(const ModelInfo & mi, int n_ctx, int n_ubatch = 512) {
    uint64_t b = 256 * MiB;
    b += (uint64_t) mi.n_embd * (uint64_t) n_ubatch * 4 * 12;
    b += (uint64_t) mi.n_vocab * 4 * 8;
    b += (uint64_t) n_ctx / 1024 * 8 * MiB * (uint64_t) std::max(1, n_ubatch / 512);
    return b;
}

static uint64_t compute_buffer_cpu(const ModelInfo & mi, int n_ubatch = 512) {
    return 256 * MiB + (uint64_t) mi.n_embd * (uint64_t) n_ubatch * 4 * 8 + (uint64_t) mi.n_vocab * 4 * 4;
}

static std::string expert_regex(const std::vector<int> & layers, int n_moe_total) {
    if (layers.empty()) return "";
    if ((int) layers.size() >= n_moe_total) return "\\.ffn_(gate|up|down|gate_up)_exps";
    std::string s = "^blk\\.(";
    for (size_t i = 0; i < layers.size(); i++) {
        if (i) s += "|";
        s += std::to_string(layers[i]);
    }
    s += ")\\.ffn_(gate|up|down|gate_up)_exps";
    return s;
}

Plan make_plan(const HwInfo & hw, const ModelInfo & mi, const PlanOptions & opt) {
    Plan p;
    p.n_ctx        = std::max(256, opt.n_ctx);
    p.n_layer      = std::max(1, mi.n_layer_used());
    p.n_moe_layers = (int) mi.moe_layers.size();
    const bool moe = mi.is_moe();
    const int  n   = p.n_layer;

    const int slot = opt.gpu_slot >= 0 ? opt.gpu_slot : hw.best_gpu();
    const GpuInfo * gpu = (slot >= 0 && slot < (int) hw.gpus.size()) ? &hw.gpus[(size_t) slot] : nullptr;

    p.bw_cpu  = hw.ram_bw_gbps > 1 ? hw.ram_bw_gbps : DEFAULT_RAM_BW;
    p.bw_gpu  = gpu && gpu->bw_gbps > 1 ? gpu->bw_gbps : DEFAULT_GPU_BW;
    p.bw_pcie = gpu && gpu->h2d_gbps > 0.5 ? gpu->h2d_gbps : DEFAULT_PCIE_BW;
    p.bw_disk = hw.disk_gbps > 0.05 ? hw.disk_gbps : DEFAULT_DISK_BW;

    // generation is memory bound: on hybrid Intel CPUs the P-cores alone are usually fastest,
    // prompt processing is compute bound and likes every core
    p.n_threads       = opt.n_threads > 0 ? opt.n_threads : (hw.p_cores > 0 && hw.e_cores > 0 ? hw.p_cores : hw.physical_cores);
    p.n_threads_batch = opt.n_threads_batch > 0 ? opt.n_threads_batch : hw.physical_cores;
    p.n_threads       = std::max(1, std::min(p.n_threads, std::max(1, hw.logical_cpus)));
    p.n_threads_batch = std::max(1, std::min(p.n_threads_batch, std::max(1, hw.logical_cpus)));

    // the recurrent state of hybrid models lives with its layer like the KV cache does
    const uint64_t kv_total = mi.kv_bytes_per_token(2) * (uint64_t) p.n_ctx + mi.recurrent_state_bytes();
    const uint64_t kv_layer = kv_total / (uint64_t) n;

    std::vector<uint64_t> nonexp((size_t) n, 0), exps((size_t) n, 0);
    std::vector<bool> is_moe_layer((size_t) n, false);
    uint64_t sum_nonexp = 0, sum_exps = 0;
    for (int l = 0; l < n && l < (int) mi.layers.size(); l++) {
        nonexp[(size_t) l] = mi.layers[(size_t) l].nonexp();
        exps[(size_t) l]   = mi.layers[(size_t) l].experts;
        sum_nonexp += nonexp[(size_t) l];
        sum_exps   += exps[(size_t) l];
    }
    for (int l : mi.moe_layers) if (l < n) is_moe_layer[(size_t) l] = true;

    p.ram_avail = hw.ram_avail;
    const uint64_t ram_margin = opt.ram_margin ? opt.ram_margin : 1536 * MiB;
    p.ram_budget = p.ram_avail > ram_margin ? p.ram_avail - ram_margin : 0;

    if (gpu) {
        p.gpu_slot    = slot;
        p.gpu_name    = gpu->description;
        p.vram_free   = gpu->vram_free;
        p.gpu_compute = compute_buffer_gpu(mi, p.n_ctx);
        const uint64_t vram_margin = opt.vram_margin ? opt.vram_margin : 400 * MiB;
        p.gpu_budget = p.vram_free > vram_margin + p.gpu_compute ? p.vram_free - vram_margin - p.gpu_compute : 0;
    }
    const bool gpu_usable = gpu != nullptr && p.gpu_budget >= 256 * MiB;

    int  L       = 0;      // the last L layers live on the GPU
    bool out_gpu = false;
    std::vector<bool> exp_cpu((size_t) n, false);

    auto first_moe_layers_to_cpu = [&](int count) {
        int c = 0;
        for (int l : mi.moe_layers) {
            if (l >= n || c >= count) break;
            exp_cpu[(size_t) l] = true;
            c++;
        }
    };

    if (opt.stream_experts && moe) {
        // experts on disk with Rafid's RAM cache; attention, router and shared experts go to the GPU when they fit
        p.mode   = "stream_experts";
        p.stream = true;
        // the CPU computes every expert here: all cores help (measured on a 24-core laptop: 8 -> 24 threads, +44%)
        if (opt.n_threads <= 0) p.n_threads = std::max(1, std::min(hw.physical_cores > 0 ? hw.physical_cores : hw.logical_cpus, hw.logical_cpus));
        first_moe_layers_to_cpu(p.n_moe_layers);
        // a prompt pass reads (nearly) every expert from the disk, so fewer, bigger passes are much faster
        p.n_ubatch = gpu_usable ? 512 : 1024;
        if (gpu_usable) {
            for (int ub : { 2048, 1024 }) {
                // only when the attention part of every layer still fits on the GPU afterwards
                const uint64_t more = compute_buffer_gpu(mi, p.n_ctx, ub) - p.gpu_compute;
                if (p.gpu_budget > more + sum_nonexp + kv_total + mi.bytes_output + 512 * MiB) {
                    p.n_ubatch     = ub;
                    p.gpu_compute += more;
                    p.gpu_budget  -= more;
                    break;
                }
            }
        }
        if (!gpu_usable) {
            p.warnings.push_back(gpu ? "gpu_too_small" : "no_gpu");
        } else if (opt.n_gpu_layers >= 0) {
            L       = std::min(opt.n_gpu_layers, n);
            out_gpu = opt.n_gpu_layers > n;
        } else {
            uint64_t used = 0;
            for (int l = n - 1; l >= 0; l--) {
                const uint64_t cost = nonexp[(size_t) l] + kv_layer;
                if (used + cost > p.gpu_budget) break;
                used += cost;
                L++;
            }
            if (L == n && used + mi.bytes_output <= p.gpu_budget) out_gpu = true;
        }
    } else if (!gpu_usable) {
        p.mode = "cpu_only";
        p.warnings.push_back(gpu ? "gpu_too_small" : "no_gpu");
    } else if (opt.n_gpu_layers >= 0) {
        // manual layer count
        L       = std::min(opt.n_gpu_layers, n);
        out_gpu = opt.n_gpu_layers > n;
        if (moe) {
            if (opt.moe_cache)           first_moe_layers_to_cpu(p.n_moe_layers);
            else if (opt.n_cpu_moe >= 0) first_moe_layers_to_cpu(opt.n_cpu_moe);
            else if (L < n)              first_moe_layers_to_cpu(p.n_moe_layers);
        }
        if (L == 0)                              p.mode = "cpu_only";
        else if (moe && opt.moe_cache)           p.mode = "hybrid_moe_cache";
        else if (L >= n && out_gpu)              p.mode = "hybrid_moe";
        else                                     p.mode = "hybrid_layers";
    } else {
        const uint64_t all  = sum_nonexp + sum_exps + mi.bytes_output + kv_total;
        const uint64_t base = sum_nonexp + mi.bytes_output + kv_total;
        if (all <= p.gpu_budget && !opt.moe_cache && opt.n_cpu_moe <= 0) {
            p.mode = "gpu_full";
            L = n;
            out_gpu = true;
        } else if (moe && base <= p.gpu_budget) {
            L = n;
            out_gpu = true;
            uint64_t remaining = p.gpu_budget - base;
            if (opt.moe_cache) {
                p.mode = "hybrid_moe_cache";
                first_moe_layers_to_cpu(p.n_moe_layers);
                p.gpu_cache = opt.moe_cache_bytes ? std::min(opt.moe_cache_bytes, remaining) : remaining;
            } else {
                p.mode = "hybrid_moe";
                if (opt.n_cpu_moe >= 0) {
                    first_moe_layers_to_cpu(opt.n_cpu_moe);
                } else {
                    // fill the VRAM with whole expert layers, starting from the last layer
                    bool stop = false;
                    for (int i = (int) mi.moe_layers.size() - 1; i >= 0; i--) {
                        const int l = mi.moe_layers[(size_t) i];
                        if (l >= n) continue;
                        if (!stop && exps[(size_t) l] <= remaining) {
                            remaining -= exps[(size_t) l];
                        } else {
                            stop = true;
                            exp_cpu[(size_t) l] = true;
                        }
                    }
                }
            }
        } else {
            p.mode = "hybrid_layers";
            if (moe) first_moe_layers_to_cpu(p.n_moe_layers);
            uint64_t used = 0;
            for (int l = n - 1; l >= 0; l--) {
                const uint64_t cost = nonexp[(size_t) l] + (exp_cpu[(size_t) l] ? 0 : exps[(size_t) l]) + kv_layer;
                if (used + cost > p.gpu_budget) break;
                used += cost;
                L++;
            }
            if (L == n && used + mi.bytes_output <= p.gpu_budget) out_gpu = true;
            if (L == 0) p.mode = "cpu_only";
        }
    }
    if (p.mode == "cpu_only") {
        L = 0;
        out_gpu = false;
        std::fill(exp_cpu.begin(), exp_cpu.end(), false);
        p.gpu_cache = 0;
    }
    if (p.stream) {
        p.stream_experts_bytes = sum_exps;
        p.stream_policy  = opt.stream_policy == "fingerprint" || opt.stream_policy == "freq" ? opt.stream_policy : "lru";
        p.stream_prefetch = opt.stream_prefetch;
        p.stream_swap    = opt.stream_swap;
    }

    // ---- what goes where ----
    p.layer_place.assign((size_t) n, 0);
    const int first_gpu = n - L;
    for (int l = 0; l < n; l++) {
        const bool on_gpu = l >= first_gpu;
        const uint64_t ne = nonexp[(size_t) l], ex = exps[(size_t) l];
        const uint64_t ex_ram = p.stream ? 0 : ex; // streamed experts live on disk, their RAM is the cache budget
        if (on_gpu) {
            p.gpu_weights += ne;
            if (exp_cpu[(size_t) l]) { p.cpu_weights += ex_ram; p.layer_place[(size_t) l] = 2; }
            else                     { p.gpu_weights += ex;     p.layer_place[(size_t) l] = 1; }
        } else {
            p.cpu_weights += ne + ex_ram;
        }
        if (exp_cpu[(size_t) l] && is_moe_layer[(size_t) l]) p.cpu_expert_layers.push_back(l);
    }
    if (out_gpu) p.gpu_weights += mi.bytes_output;
    else if (!mi.output_tied) p.cpu_weights += mi.bytes_output;
    p.cpu_weights += mi.bytes_embed + mi.bytes_global;
    p.output_on_gpu = out_gpu;
    p.n_cpu_moe     = (int) p.cpu_expert_layers.size();
    p.gpu_kv        = kv_layer * (uint64_t) L;
    p.cpu_kv        = kv_total - p.gpu_kv;
    p.cpu_compute   = compute_buffer_cpu(mi, p.n_ubatch);
    p.cpu_snapshots = 2 * mi.recurrent_state_bytes(); // Engine keeps the last 2 (RAFID_CHECKPOINTS)

    if (p.stream) {
        // the cache gets the RAM that everything else leaves; prompts are gathered layer by layer in two
        // resident buffers (one when a layer is big next to the cache)
        uint64_t max_layer = 0;
        for (int l = 0; l < n; l++) if (is_moe_layer[(size_t) l]) max_layer = std::max(max_layer, exps[(size_t) l]);
        const uint64_t other = p.cpu_weights + p.cpu_kv + p.cpu_compute + p.cpu_snapshots;
        // never more than the machine can hold next to the system, however much RAM looks available
        const uint64_t room  = hw.ram_total > 0 ? std::min(p.ram_budget, hw.ram_total - std::min(hw.ram_total, os_ram_reserve(hw.ram_total)))
                                                : p.ram_budget;
        auto fit_with = [&](uint64_t staging) { return room > other + staging ? room - other - staging : 0; };
        const uint64_t want = opt.stream_budget_bytes ? std::min<uint64_t>(opt.stream_budget_bytes, sum_exps) : sum_exps / 5;
        uint64_t staging = 2 * max_layer;
        uint64_t fit     = fit_with(staging);
        if (staging > std::max<uint64_t>(2ull << 30, std::min(want, fit) / 2)) {
            staging = max_layer;
            fit     = fit_with(staging);
        }
        p.stream_staging    = staging;
        p.stream_budget_fit = std::min(fit, sum_exps);
        // never more than the RAM can hold: a cache that has to be paged out is slower than no cache
        p.stream_budget = std::min(want, p.stream_budget_fit);
        if (opt.stream_budget_bytes && p.stream_budget < want) p.warnings.push_back("stream_budget_clamped");
        // the expert store needs room for the experts of one token in every layer, plus two
        const uint64_t one_token = mi.n_expert > 0 ? (uint64_t) ((double) max_layer / mi.n_expert * (mi.n_expert_used + 2) * p.n_moe_layers) : 0;
        if (p.stream_budget < one_token) {
            if (p.stream_budget_fit < one_token) p.warnings.push_back("stream_ram_too_small");
            p.stream_budget = std::min(one_token, sum_exps);
        }
        p.stream_hit_known = opt.stream_hit_hint >= 0;
        p.stream_hit = p.stream_hit_known ? std::max(0.0, std::min(1.0, opt.stream_hit_hint))
                                          : (sum_exps ? (double) p.stream_budget / (double) sum_exps : 0.0); // random access baseline
    }
    if (p.mode != "cpu_only" && !p.cpu_expert_layers.empty() && L > 0) {
        p.override_pattern = expert_regex(p.cpu_expert_layers, p.n_moe_layers);
    }
    if (!gpu_usable || L == 0) p.gpu_compute = 0;
    p.n_gpu_layers = out_gpu ? 9999 : std::min(L + (L > 0 ? mi.n_nextn : 0), mi.n_layer);

    if (p.mode == "hybrid_moe_cache" && p.gpu_cache == 0) {
        const uint64_t used = p.gpu_weights + p.gpu_kv;
        const uint64_t rem  = p.gpu_budget > used ? p.gpu_budget - used : 0;
        p.gpu_cache = opt.moe_cache_bytes ? std::min(opt.moe_cache_bytes, rem) : rem;
    }
    if (p.mode == "hybrid_moe" && p.n_cpu_moe == 0 && L >= n && out_gpu) p.mode = "gpu_full";

    const uint64_t cpu_need = p.cpu_weights + p.cpu_kv + p.cpu_compute + p.cpu_snapshots + p.stream_budget + p.stream_staging;
    if (p.stream) {
        p.use_mmap = true; // required: llama.cpp must not copy the expert data
        if (cpu_need > p.ram_budget && std::find(p.warnings.begin(), p.warnings.end(), "stream_ram_too_small") == p.warnings.end()) {
            p.warnings.push_back("stream_budget_big");
        }
    } else if (cpu_need > p.ram_budget) {
        p.disk_overflow = cpu_need - p.ram_budget;
        p.use_mmap = true;
        p.warnings.push_back("disk_streaming");
    }

    // ---- memory traffic for one generated token ----
    const double frac = moe ? (double) mi.n_expert_used / (double) mi.n_expert : 1.0;
    double gpu_b = 0, cpu_b = 0, cpu_exp_active = 0;
    for (int l = 0; l < n; l++) {
        const double ne = (double) nonexp[(size_t) l];
        const double ex = (double) exps[(size_t) l] * (is_moe_layer[(size_t) l] ? frac : 1.0);
        if (l >= first_gpu) {
            gpu_b += ne;
            if (exp_cpu[(size_t) l]) { cpu_b += ex; cpu_exp_active += ex; }
            else gpu_b += ex;
        } else {
            cpu_b += ne + ex;
        }
    }
    if (out_gpu) gpu_b += (double) mi.bytes_output; else cpu_b += (double) mi.bytes_output;
    const double kv_read = (double) kv_total * 0.25; // the cache is a quarter full on average
    gpu_b += kv_read * (double) L / n;
    cpu_b += kv_read * (double) (n - L) / n;

    double pcie_b = 0;
    if (p.mode == "hybrid_moe_cache") {
        const double h = std::max(0.0, std::min(0.98, opt.cache_hit_hint));
        gpu_b  += h * cpu_exp_active;
        pcie_b  = (1.0 - h) * cpu_exp_active;
        cpu_b  -= cpu_exp_active;
    }
    double disk_b = 0;
    if (p.stream) {
        // every active expert is read from RAM by the CPU; the misses also come from the disk first
        double act_exp = 0;
        for (int l = 0; l < n; l++) if (is_moe_layer[(size_t) l]) act_exp += (double) exps[(size_t) l] * frac;
        disk_b = (1.0 - p.stream_hit) * act_exp;
    } else if (p.disk_overflow > 0 && p.cpu_weights > 0) {
        const double miss = std::min(1.0, (double) p.disk_overflow / (double) p.cpu_weights);
        disk_b = cpu_b * miss;
        cpu_b -= disk_b;
    }
    p.tok_gpu_bytes = gpu_b; p.tok_cpu_bytes = cpu_b; p.tok_pcie_bytes = pcie_b; p.tok_disk_bytes = disk_b;

    // GPU <-> CPU hand-overs per token
    int transitions = 0;
    if (p.mode == "stream_experts")        transitions = 2 * p.n_moe_layers;
    else if (p.mode == "hybrid_moe")       transitions = 2 * p.n_cpu_moe;
    else if (p.mode == "hybrid_moe_cache") transitions = p.n_moe_layers;
    else if (p.mode == "hybrid_layers")    transitions = 2 + 2 * (int) std::count_if(p.layer_place.begin(), p.layer_place.end(), [](int v) { return v == 2; });
    p.t_sync = transitions * 60e-6;

    // fraction of the raw bandwidth that real kernels reach (quantized matmuls on the CPU lose the most)
    auto tok_time = [&](double eg, double ec, double ep, double ed) {
        return gpu_b / (p.bw_gpu * 1e9 * eg) + cpu_b / (p.bw_cpu * 1e9 * ec) +
               pcie_b / (p.bw_pcie * 1e9 * ep) + disk_b / (p.bw_disk * 1e9 * ed) + p.t_sync;
    };
    const double t_hi = tok_time(0.75, 0.65, 0.85, 0.80);
    const double t_lo = tok_time(0.50, 0.40, 0.60, 0.35);
    p.est_tps_high = t_hi > 0 ? 1.0 / t_hi : 0;
    p.est_tps_low  = t_lo > 0 ? 1.0 / t_lo : 0;
    p.t_gpu  = gpu_b  / (p.bw_gpu  * 1e9 * 0.62);
    p.t_cpu  = cpu_b  / (p.bw_cpu  * 1e9 * 0.52);
    p.t_pcie = pcie_b / (p.bw_pcie * 1e9 * 0.72);
    p.t_disk = disk_b / (p.bw_disk * 1e9 * 0.57);

    // ---- warnings ----
    if (gpu_usable && kv_total > (p.gpu_budget + p.gpu_compute) * 3 / 10) p.warnings.push_back("kv_heavy");
    if (!moe && gpu_usable && L < n)                                    p.warnings.push_back("dense_partial");
    if (moe && p.mode == "hybrid_layers")                               p.warnings.push_back("moe_nonexp_partial");
    if (gpu_usable && p.gpu_weights + p.gpu_kv + p.gpu_cache > p.gpu_budget) p.warnings.push_back("override_too_big");
    if (mi.n_ctx_train > 0 && p.n_ctx > mi.n_ctx_train)                 p.warnings.push_back("ctx_above_train");
    const double t_total = p.t_gpu + p.t_cpu + p.t_pcie + p.t_disk + p.t_sync;
    if (!p.stream && t_total > 0 && p.t_cpu > 0.75 * t_total && gpu_usable) p.warnings.push_back("cpu_dominant");
    if (p.mode == "hybrid_moe_cache")                                   p.warnings.push_back("moe_cache_experimental");
    if (p.stream)                                                       p.warnings.push_back("stream_experimental");
    if (p.stream && !p.stream_hit_known)                                p.warnings.push_back("stream_no_fingerprint");
    return p;
}

bool plan_step_back(const HwInfo & hw, const ModelInfo & mi, PlanOptions & opt, const Plan & failed) {
    (void) hw;
    opt.vram_margin = (opt.vram_margin ? opt.vram_margin : 400 * MiB) + 256 * MiB;
    if (failed.mode == "cpu_only") return false;
    const int n = failed.n_layer;
    if (failed.mode == "hybrid_moe_cache") {
        opt.moe_cache_bytes = failed.gpu_cache / 2;
        if (opt.moe_cache_bytes < 256 * MiB) { opt.moe_cache = false; opt.moe_cache_bytes = 0; opt.n_cpu_moe = failed.n_moe_layers; }
        return true;
    }
    if (mi.is_moe() && (failed.mode == "gpu_full" || failed.mode == "hybrid_moe")) {
        const int step = std::max(1, failed.n_moe_layers / 8);
        if (failed.n_cpu_moe + step <= failed.n_moe_layers) {
            opt.n_cpu_moe = failed.n_cpu_moe + step;
            return true;
        }
        opt.n_cpu_moe    = failed.n_moe_layers;
        opt.n_gpu_layers = std::max(0, n - std::max(2, n / 8));
        return true;
    }
    const int L = failed.output_on_gpu ? n : std::min(failed.n_gpu_layers, n);
    if (L <= 0) return false;
    opt.n_gpu_layers = std::max(0, L - std::max(2, n / 8));
    return true;
}

nlohmann::json plan_to_json(const Plan & p) {
    return {
        { "mode", p.mode }, { "gpu_slot", p.gpu_slot }, { "gpu_name", p.gpu_name },
        { "n_layer", p.n_layer }, { "n_gpu_layers", p.n_gpu_layers }, { "n_moe_layers", p.n_moe_layers },
        { "n_cpu_moe", p.n_cpu_moe }, { "output_on_gpu", p.output_on_gpu },
        { "cpu_expert_layers", p.cpu_expert_layers }, { "layer_place", p.layer_place },
        { "override_pattern", p.override_pattern },
        { "vram_free", p.vram_free }, { "gpu_budget", p.gpu_budget }, { "gpu_weights", p.gpu_weights },
        { "gpu_kv", p.gpu_kv }, { "gpu_compute", p.gpu_compute }, { "gpu_cache", p.gpu_cache },
        { "ram_avail", p.ram_avail }, { "ram_budget", p.ram_budget }, { "cpu_weights", p.cpu_weights },
        { "cpu_kv", p.cpu_kv }, { "cpu_compute", p.cpu_compute }, { "disk_overflow", p.disk_overflow },
        { "use_mmap", p.use_mmap },
        { "tok_bytes", { { "gpu", p.tok_gpu_bytes }, { "cpu", p.tok_cpu_bytes }, { "pcie", p.tok_pcie_bytes }, { "disk", p.tok_disk_bytes } } },
        { "bw", { { "gpu", p.bw_gpu }, { "cpu", p.bw_cpu }, { "pcie", p.bw_pcie }, { "disk", p.bw_disk } } },
        { "tok_time", { { "gpu", p.t_gpu }, { "cpu", p.t_cpu }, { "pcie", p.t_pcie }, { "disk", p.t_disk }, { "sync", p.t_sync } } },
        { "est_tps_low", p.est_tps_low }, { "est_tps_high", p.est_tps_high },
        { "n_ctx", p.n_ctx }, { "n_threads", p.n_threads }, { "n_threads_batch", p.n_threads_batch },
        { "warnings", p.warnings },
        { "stream", p.stream }, { "stream_budget", p.stream_budget }, { "stream_experts_bytes", p.stream_experts_bytes },
        { "stream_hit", p.stream_hit }, { "stream_hit_known", p.stream_hit_known },
        { "stream_policy", p.stream_policy }, { "stream_prefetch", p.stream_prefetch }, { "stream_swap", p.stream_swap }, { "n_ubatch", p.n_ubatch },
        { "stream_staging", p.stream_staging }, { "stream_budget_fit", p.stream_budget_fit }, { "cpu_snapshots", p.cpu_snapshots },
    };
}

} // namespace rafid
