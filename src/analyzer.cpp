#include "analyzer.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace rafid {

namespace {

struct Trace {
    int S = 0, E = 0, K = 0;
    size_t T = 0;
    uint64_t total_in_file = 0;
    std::vector<uint16_t> data; // [T][S][K]
    const uint16_t * at(size_t t, int s) const { return &data[(t * (size_t) S + (size_t) s) * (size_t) K]; }
};

bool seek64(FILE * f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (long long) off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
}

bool load_trace(const std::string & path, int S, int E, int K, size_t max_tokens, Trace & tr) {
    FILE * f = fopen_utf8(path, "rb");
    if (!f) return false;
    char m[8];
    uint32_t h[4];
    bool ok = fread(m, 1, 8, f) == 8 && memcmp(m, "RFTRACE1", 8) == 0 && fread(h, 4, 4, f) == 4 &&
              h[1] == (uint32_t) S && h[2] == (uint32_t) E && h[3] == (uint32_t) K;
    if (!ok) { fclose(f); return false; }
    const uint64_t rec  = (uint64_t) S * K * 2;
    const uint64_t size = file_size(path);
    const uint64_t n    = size > 24 ? (size - 24) / rec : 0;
    const uint64_t take = std::min<uint64_t>(n, max_tokens);
    tr.S = S; tr.E = E; tr.K = K;
    tr.total_in_file = n;
    tr.data.resize((size_t) (take * (uint64_t) S * K));
    if (take > 0) {
        ok = seek64(f, 24 + (n - take) * rec) && fread(tr.data.data(), 2, tr.data.size(), f) == tr.data.size();
    }
    fclose(f);
    if (!ok) return false;
    tr.T = (size_t) take;
    return true;
}

// array based LRU over dense keys
struct Lru {
    std::vector<int32_t> prev, next;
    std::vector<uint8_t> in;
    int32_t head = -1, tail = -1; // head = most recent
    size_t size = 0, cap = 0;
    void init(size_t n_keys, size_t capacity) {
        prev.assign(n_keys, -1); next.assign(n_keys, -1); in.assign(n_keys, 0);
        head = tail = -1; size = 0; cap = capacity;
    }
    void unlink(int32_t k) {
        if (prev[(size_t) k] >= 0) next[(size_t) prev[(size_t) k]] = next[(size_t) k]; else head = next[(size_t) k];
        if (next[(size_t) k] >= 0) prev[(size_t) next[(size_t) k]] = prev[(size_t) k]; else tail = prev[(size_t) k];
        prev[(size_t) k] = next[(size_t) k] = -1;
    }
    void push_front(int32_t k) {
        prev[(size_t) k] = -1; next[(size_t) k] = head;
        if (head >= 0) prev[(size_t) head] = k;
        head = k;
        if (tail < 0) tail = k;
    }
    bool access(int32_t k) {
        if (in[(size_t) k]) { unlink(k); push_front(k); return true; }
        if (size >= cap && tail >= 0) { const int32_t v = tail; unlink(v); in[(size_t) v] = 0; size--; }
        push_front(k); in[(size_t) k] = 1; size++;
        return false;
    }
};

// the expert store's "freq" policy: per layer, the expert with the lowest use count (each use decays with a
// half-life in tokens) leaves first; the experts of the current token stay
struct FreqLayers {
    int S = 0, E = 0, cap = 0;
    double lam = 0;
    std::vector<double>  key;   // [S*E] log of sum(exp(lam * t_use)), -inf = never used
    std::vector<uint8_t> in;    // [S*E]
    std::vector<std::vector<int32_t>> items; // per layer, experts in cache
    std::vector<uint8_t> cur;   // [E] experts of the token being processed
    void init(int s, int e, int c, double half) {
        S = s; E = e; cap = c; lam = std::log(2.0) / half;
        key.assign((size_t) S * E, -INFINITY); in.assign((size_t) S * E, 0);
        items.assign((size_t) S, {});
        cur.assign((size_t) E, 0);
    }
    // one token of one layer; returns the hits
    int access(int s, const uint16_t * ids, int K, double t) {
        int hits = 0;
        const double now = lam * t;
        auto & it = items[(size_t) s];
        for (int i = 0; i < K; i++) cur[ids[i]] = 1;
        for (int i = 0; i < K; i++) {
            const size_t k = (size_t) s * E + ids[i];
            if (in[k]) { hits++; }
            else {
                if ((int) it.size() >= cap) {
                    int best = -1;
                    for (int j = 0; j < (int) it.size(); j++) {
                        const int x = it[(size_t) j];
                        if (cur[(size_t) x]) continue;
                        if (best < 0 || key[(size_t) s * E + x] < key[(size_t) s * E + it[(size_t) best]]) best = j;
                    }
                    if (best < 0) continue; // everything is needed by this token (cannot happen with cap > K)
                    in[(size_t) s * E + it[(size_t) best]] = 0;
                    it[(size_t) best] = it.back();
                    it.pop_back();
                }
                it.push_back(ids[i]);
                in[k] = 1;
            }
            const double a = key[k];
            key[k] = a == -INFINITY ? now : std::max(a, now) + std::log1p(std::exp(-std::fabs(a - now)));
        }
        for (int i = 0; i < K; i++) cur[ids[i]] = 0;
        return hits;
    }
};

// segmented LRU in the spirit of llama.cpp's MoE cache: hits are promoted to a protected half,
// new entries enter the probation segment and are evicted from there first
struct Slru {
    Lru prob, prot; // reuse the list mechanics; capacities are managed here
    std::vector<uint8_t> seg; // 0 none, 1 probation, 2 protected
    size_t cap = 0;
    void init(size_t n_keys, size_t capacity) {
        prob.init(n_keys, (size_t) -1); prot.init(n_keys, (size_t) -1);
        seg.assign(n_keys, 0); cap = capacity;
    }
    bool access(int32_t k) {
        uint8_t & s = seg[(size_t) k];
        if (s == 2) { prot.unlink(k); prot.push_front(k); return true; }
        if (s == 1) {
            prob.unlink(k); prob.size--;
            prot.push_front(k); prot.size++; s = 2;
            if (prot.size > cap / 2 && prot.tail >= 0) {
                const int32_t d = prot.tail; prot.unlink(d); prot.size--;
                prob.push_front(d); prob.size++; seg[(size_t) d] = 1;
            }
            return true;
        }
        if (prob.size + prot.size >= cap) {
            if (prob.tail >= 0)      { const int32_t v = prob.tail; prob.unlink(v); prob.size--; seg[(size_t) v] = 0; }
            else if (prot.tail >= 0) { const int32_t v = prot.tail; prot.unlink(v); prot.size--; seg[(size_t) v] = 0; }
        }
        prob.push_front(k); prob.size++; s = 1;
        return false;
    }
};

} // namespace

nlohmann::json analyze_profile(const ExpertProfiler::Snapshot & snap, const ModelInfo & mi, const HwInfo & hw, const AnalyzeOptions & opt) {
    using nlohmann::json;
    json out;
    out["supported"] = snap.supported;
    if (!snap.supported) return out;

    const int S = snap.S, E = snap.E, K = snap.K;
    out["n_moe_layers"]   = S;
    out["n_expert"]       = E;
    out["n_expert_used"]  = K;
    out["tokens_gen"]     = snap.tokens_gen;
    out["tokens_prefill"] = snap.tokens_prefill;
    out["moe_layers"]     = snap.moe_layers;

    // ---------------------------------------------------------------- counts
    const bool use_gen = snap.tokens_gen >= 50;
    std::vector<uint64_t> counts((size_t) S * E, 0);
    for (size_t i = 0; i < counts.size(); i++) {
        counts[i] = use_gen ? snap.counts_gen[i] : snap.counts_gen[i] + snap.counts_prefill[i];
    }
    out["counts_source"] = use_gen ? "generated" : "all";

    const double thr[5] = { 0.5, 0.8, 0.9, 0.95, 0.99 };
    double cov_sum[5] = { 0, 0, 0, 0, 0 };
    int layers_with_data = 0;
    json cov80_layers = json::array();
    json entropy_layers = json::array();
    std::vector<int> heat((size_t) S * E, 0);
    uint64_t unused = 0;
    double entropy_sum = 0;

    for (int s = 0; s < S; s++) {
        std::vector<uint64_t> c(counts.begin() + (ptrdiff_t) s * E, counts.begin() + (ptrdiff_t) (s + 1) * E);
        uint64_t tot = 0, mx = 0;
        for (int e = 0; e < E; e++) {
            const uint64_t all = snap.counts_gen[(size_t) s * E + e] + snap.counts_prefill[(size_t) s * E + e];
            if (all == 0) unused++;
            tot += c[(size_t) e];
            mx = std::max(mx, c[(size_t) e]);
        }
        for (int e = 0; e < E; e++) heat[(size_t) s * E + e] = mx ? (int) std::lround(255.0 * (double) c[(size_t) e] / (double) mx) : 0;
        if (tot == 0) { cov80_layers.push_back(nullptr); entropy_layers.push_back(nullptr); continue; }
        layers_with_data++;
        double h = 0;
        for (int e = 0; e < E; e++) {
            if (c[(size_t) e] == 0) continue;
            const double p = (double) c[(size_t) e] / (double) tot;
            h -= p * std::log(p);
        }
        const double hn = E > 1 ? h / std::log((double) E) : 1.0;
        entropy_sum += hn;
        entropy_layers.push_back(hn);
        std::sort(c.begin(), c.end(), std::greater<uint64_t>());
        for (int ti = 0; ti < 5; ti++) {
            uint64_t cum = 0;
            int m = 0;
            while (m < E && (double) cum < thr[ti] * (double) tot) cum += c[(size_t) m++];
            const double frac = (double) m / (double) E;
            cov_sum[ti] += frac;
            if (ti == 1) cov80_layers.push_back(frac);
        }
    }
    json coverage = json::object();
    coverage["thresholds"] = { 0.5, 0.8, 0.9, 0.95, 0.99 };
    json cov_avg = json::array();
    for (int ti = 0; ti < 5; ti++) cov_avg.push_back(layers_with_data ? json(cov_sum[ti] / layers_with_data) : json(nullptr));
    coverage["experts_frac"] = cov_avg;
    coverage["per_layer_80"] = cov80_layers;
    out["coverage"]       = coverage;
    out["entropy_norm"]   = layers_with_data ? entropy_sum / layers_with_data : 1.0;
    out["entropy_layers"] = entropy_layers;
    out["unused_frac"]    = (double) unused / (double) ((size_t) S * E);
    out["heatmap"]        = { { "rows", S }, { "cols", E }, { "data", heat } };

    // global concentration curve: share of all activations carried by the top x% of (layer, expert) pairs
    {
        std::vector<uint64_t> all(counts);
        std::sort(all.begin(), all.end(), std::greater<uint64_t>());
        const double tot = (double) std::accumulate(all.begin(), all.end(), (uint64_t) 0);
        json curve = json::array();
        uint64_t cum = 0;
        size_t idx = 0;
        for (int pct = 0; pct <= 100; pct += 5) {
            const size_t upto = (size_t) std::llround((double) all.size() * pct / 100.0);
            while (idx < upto && idx < all.size()) cum += all[idx++];
            curve.push_back(tot > 0 ? (double) cum / tot : 0.0);
        }
        out["concentration_curve"] = curve;
    }

    // ---------------------------------------------------------------- trace
    Trace tr;
    const bool have_trace = load_trace(snap.trace_path, S, E, K, (size_t) opt.max_trace_tokens, tr);
    json tj = json::object();
    tj["tokens_used"]    = have_trace ? tr.T : 0;
    tj["tokens_in_file"] = have_trace ? tr.total_in_file : 0;

    // expert sizes for this model (bytes of one expert per MoE slot)
    std::vector<double> eb((size_t) S, 0);
    double all_expert_bytes = 0, active_expert_bytes = 0;
    for (int s = 0; s < S; s++) {
        eb[(size_t) s] = mi.expert_bytes(snap.moe_layers[(size_t) s]);
        all_expert_bytes    += eb[(size_t) s] * E;
        active_expert_bytes += eb[(size_t) s] * K;
    }
    const double nonexp_bytes = std::max(0.0, (double) mi.active_bytes_per_token() - active_expert_bytes);

    const int gslot  = hw.best_gpu();
    const double bw_gpu  = gslot >= 0 && hw.gpus[(size_t) gslot].bw_gbps > 1 ? hw.gpus[(size_t) gslot].bw_gbps : DEFAULT_GPU_BW;
    const double bw_pcie = gslot >= 0 && hw.gpus[(size_t) gslot].h2d_gbps > 0.5 ? hw.gpus[(size_t) gslot].h2d_gbps : DEFAULT_PCIE_BW;
    const double bw_ram  = hw.ram_bw_gbps > 1 ? hw.ram_bw_gbps : DEFAULT_RAM_BW;
    const double bw_disk = hw.disk_gbps > 0.05 ? hw.disk_gbps : DEFAULT_DISK_BW;
    out["bandwidth"] = { { "gpu", bw_gpu }, { "pcie", bw_pcie }, { "ram", bw_ram }, { "disk", bw_disk }, { "disk_measured", hw.disk_gbps > 0.05 } };
    out["bytes"] = { { "experts_total", all_expert_bytes }, { "experts_active_per_token", active_expert_bytes }, { "nonexpert_per_token", nonexp_bytes } };

    if (have_trace && tr.T >= 64) {
        const size_t T = tr.T, half = T / 2;
        const size_t n_keys = (size_t) S * E;

        // temporal locality: how many experts of a layer are reused by the next token
        double ov = 0;
        size_t ov_n = 0;
        for (size_t t = half + 1; t < T; t++) {
            for (int s = 0; s < S; s++) {
                const uint16_t * a = tr.at(t, s);
                const uint16_t * b = tr.at(t - 1, s);
                int inter = 0;
                for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) if (a[i] == b[j]) { inter++; break; }
                ov += (double) inter / K;
                ov_n++;
            }
        }
        tj["temporal_overlap"] = ov_n ? ov / (double) ov_n : 0.0;

        // train counts (first half) for the static hot set and the popularity baseline
        std::vector<uint64_t> train((size_t) n_keys, 0);
        for (size_t t = 0; t < half; t++)
            for (int s = 0; s < S; s++)
                for (int i = 0; i < K; i++) train[(size_t) s * E + tr.at(t, s)[i]]++;
        std::vector<int32_t> order(n_keys);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return train[(size_t) a] > train[(size_t) b]; });

        const double fracs[6] = { 0.05, 0.10, 0.20, 0.30, 0.50, 0.75 };
        json sims = json::array();
        const double test_access = (double) (T - half) * S * K;
        for (double f : fracs) {
            const size_t cap = std::max<size_t>((size_t) K * 2, (size_t) std::llround(f * (double) n_keys));
            std::vector<uint8_t> hot(n_keys, 0);
            for (size_t i = 0; i < std::min(cap, n_keys); i++) hot[(size_t) order[i]] = 1;
            Lru lru;  lru.init(n_keys, cap);
            Slru slru; slru.init(n_keys, cap);
            FreqLayers fq; fq.init(S, E, std::max(K + 2, (int) std::floor(f * E)), 32.0);
            uint64_t h_static = 0, h_lru = 0, h_slru = 0, h_freq = 0;
            for (size_t t = 0; t < T; t++) {
                const bool test = t >= half;
                for (int s = 0; s < S; s++) {
                    const uint16_t * ids = tr.at(t, s);
                    for (int i = 0; i < K; i++) {
                        const int32_t key = (int32_t) ((size_t) s * E + ids[i]);
                        const bool a = lru.access(key);
                        const bool b = slru.access(key);
                        if (test) { h_lru += a; h_slru += b; h_static += hot[(size_t) key]; }
                    }
                    const int hf = fq.access(s, ids, K, (double) t);
                    if (test) h_freq += (uint64_t) hf;
                }
            }
            const double hs = h_static / test_access, hl = h_lru / test_access, hq = h_slru / test_access, hf = h_freq / test_access;
            const double best_ram = std::max(hs, std::max(hl, hq));
            const double cache_bytes = f * all_expert_bytes;
            // A) experts on the SSD, hot experts cached in RAM
            const double t_ssd = nonexp_bytes / (bw_gpu * 1e9 * 0.7) + best_ram * active_expert_bytes / (bw_ram * 1e9 * 0.65) +
                                 (1 - best_ram) * active_expert_bytes / (bw_disk * 1e9 * 0.6);
            // B) experts in RAM, hot experts cached in VRAM (llama.cpp MoE cache, SLRU, misses are uploaded over PCIe)
            const double t_vram = nonexp_bytes / (bw_gpu * 1e9 * 0.7) + hq * active_expert_bytes / (bw_gpu * 1e9 * 0.7) +
                                  (1 - hq) * active_expert_bytes / (bw_pcie * 1e9 * 0.75);
            sims.push_back({
                { "frac", f }, { "capacity", cap }, { "cache_bytes", cache_bytes },
                { "hit_static", hs }, { "hit_lru", hl }, { "hit_slru", hq }, { "hit_freq", hf },
                { "tps_ssd_ram_cache", t_ssd > 0 ? 1.0 / t_ssd : 0.0 },
                { "tps_ram_vram_cache", t_vram > 0 ? 1.0 / t_vram : 0.0 },
            });
        }
        tj["cache_sims"] = sims;
        const double t_cpu_experts = nonexp_bytes / (bw_gpu * 1e9 * 0.7) + active_expert_bytes / (bw_ram * 1e9 * 0.65);
        const double t_ssd_nocache = nonexp_bytes / (bw_gpu * 1e9 * 0.7) + active_expert_bytes / (bw_disk * 1e9 * 0.6);
        tj["tps_cpu_experts"]   = t_cpu_experts > 0 ? 1.0 / t_cpu_experts : 0.0;
        tj["tps_ssd_no_cache"]  = t_ssd_nocache > 0 ? 1.0 / t_ssd_nocache : 0.0;

        // cross-layer prediction: experts of layer s -> experts of layer s+1
        const size_t tsize = (size_t) std::max(0, S - 1) * (size_t) E * (size_t) E;
        if (S >= 2 && tsize <= (size_t) 80 * 1000 * 1000) {
            std::vector<uint32_t> trans(tsize, 0);
            for (size_t t = 0; t < half; t++) {
                for (int s = 0; s + 1 < S; s++) {
                    const uint16_t * a = tr.at(t, s);
                    const uint16_t * b = tr.at(t, s + 1);
                    for (int i = 0; i < K; i++) {
                        uint32_t * row = &trans[((size_t) s * E + a[i]) * (size_t) E];
                        for (int j = 0; j < K; j++) row[b[j]]++;
                    }
                }
            }
            // popularity baseline per layer
            std::vector<std::vector<int>> pop((size_t) S);
            for (int s = 0; s < S; s++) {
                std::vector<int> idx((size_t) E);
                std::iota(idx.begin(), idx.end(), 0);
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return train[(size_t) s * E + a] > train[(size_t) s * E + b]; });
                idx.resize((size_t) std::min(E, 2 * K));
                pop[(size_t) s] = idx;
            }
            const size_t n_test = std::min<size_t>(T - half, (size_t) opt.max_predict_tokens);
            const size_t step   = std::max<size_t>(1, (T - half) / std::max<size_t>(1, n_test));
            std::vector<double> score((size_t) E);
            std::vector<int> idx((size_t) E);
            const int k2 = std::min(E, 2 * K);
            uint64_t hk = 0, h2k = 0, bk = 0, b2k = 0, cases = 0;
            for (size_t t = half, done = 0; t < T && done < n_test; t += step, done++) {
                for (int s = 0; s + 1 < S; s++) {
                    std::fill(score.begin(), score.end(), 0.0);
                    const uint16_t * a = tr.at(t, s);
                    for (int i = 0; i < K; i++) {
                        const uint32_t * row = &trans[((size_t) s * E + a[i]) * (size_t) E];
                        for (int e = 0; e < E; e++) score[(size_t) e] += row[e];
                    }
                    // break ties with the popularity of the next layer
                    for (int e = 0; e < E; e++) score[(size_t) e] += 1e-7 * (double) train[(size_t) (s + 1) * E + e];
                    std::iota(idx.begin(), idx.end(), 0);
                    std::partial_sort(idx.begin(), idx.begin() + k2, idx.end(), [&](int x, int y) { return score[(size_t) x] > score[(size_t) y]; });
                    const uint16_t * actual = tr.at(t, s + 1);
                    const std::vector<int> & p = pop[(size_t) s + 1];
                    for (int j = 0; j < K; j++) {
                        const int e = actual[j];
                        for (int r = 0; r < k2; r++) if (idx[(size_t) r] == e) { h2k++; if (r < K) hk++; break; }
                        for (int r = 0; r < (int) p.size(); r++) if (p[(size_t) r] == e) { b2k++; if (r < K) bk++; break; }
                    }
                    cases++;
                }
            }
            const double denom = (double) cases * K;
            tj["predict"] = {
                { "recall_k", denom > 0 ? hk / denom : 0.0 }, { "recall_2k", denom > 0 ? h2k / denom : 0.0 },
                { "baseline_k", denom > 0 ? bk / denom : 0.0 }, { "baseline_2k", denom > 0 ? b2k / denom : 0.0 },
                { "test_tokens", n_test },
            };
        }
    }
    out["trace"] = tj;

    // ---------------------------------------------------------------- verdict numbers for the UI
    json v = json::object();
    v["enough_data"] = have_trace && tr.T >= 2000;
    v["coverage80"]  = layers_with_data ? cov_sum[1] / layers_with_data : 1.0;
    if (tj.contains("cache_sims")) {
        for (const auto & sim : tj["cache_sims"]) {
            if (std::fabs(sim["frac"].get<double>() - 0.20) < 1e-9) {
                v["cache_hit20"] = std::max(sim["hit_static"].get<double>(), std::max(sim["hit_lru"].get<double>(), sim["hit_slru"].get<double>()));
            }
        }
    }
    if (tj.contains("predict")) {
        v["predict_recall2k"] = tj["predict"]["recall_2k"];
        v["predict_gain"]     = tj["predict"]["recall_2k"].get<double>() - tj["predict"]["baseline_2k"].get<double>();
    }
    out["verdict"] = v;
    return out;
}

} // namespace rafid
