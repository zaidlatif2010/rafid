#include "profiler.h"
#include "util.h"

#include <algorithm>
#include <cstring>
#include <filesystem>

namespace rafid {

static const char     PROF_MAGIC[8]  = { 'R', 'F', 'P', 'R', 'O', 'F', '0', '1' };
static const char     TRACE_MAGIC[8] = { 'R', 'F', 'T', 'R', 'A', 'C', 'E', '1' };
static const uint32_t FORMAT_VERSION = 1;

static void remove_file(const std::string & p) {
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(p), ec);
}

ExpertProfiler::~ExpertProfiler() {
    shutdown();
}

void ExpertProfiler::configure(const ModelInfo & mi, const std::string & profiles_dir) {
    shutdown();
    std::lock_guard<std::mutex> lk(mu_);
    enabled_.store(false);
    S_ = E_ = K_ = 0;
    moe_layers_.clear();
    slot_of_layer_.clear();
    counts_gen_.clear();
    counts_prefill_.clear();
    tokens_gen_ = tokens_prefill_ = unsaved_ = 0;
    profile_path_.clear();
    trace_path_.clear();
    trace_bytes_ = 0;

    if (!mi.is_moe() || mi.n_expert > 65535 || mi.n_expert_used <= 0) return;

    S_       = (int) mi.moe_layers.size();
    E_       = mi.n_expert;
    K_       = mi.n_expert_used;
    n_layer_ = mi.n_layer;
    moe_layers_ = mi.moe_layers;
    slot_of_layer_.assign((size_t) n_layer_ + 1, -1);
    for (int s = 0; s < S_; s++) {
        const int l = moe_layers_[(size_t) s];
        if (l >= 0 && l < (int) slot_of_layer_.size()) slot_of_layer_[(size_t) l] = s;
    }
    counts_gen_.assign((size_t) S_ * E_, 0);
    counts_prefill_.assign((size_t) S_ * E_, 0);
    cur_.assign((size_t) S_ * K_, 0);
    seen_.assign((size_t) S_, 0);

    make_dirs(profiles_dir);
    const std::string key = profile_key(mi);
    profile_path_ = path_join(profiles_dir, key + ".rfprof");
    trace_path_   = path_join(profiles_dir, key + ".rftrace");

    load_profile();

    // keep the existing trace only if it belongs to the same layout
    if (FILE * f = fopen_utf8(trace_path_, "rb")) {
        char m[8];
        uint32_t h[4] = { 0, 0, 0, 0 };
        const bool ok = fread(m, 1, 8, f) == 8 && memcmp(m, TRACE_MAGIC, 8) == 0 && fread(h, 4, 4, f) == 4 &&
                        h[0] == FORMAT_VERSION && h[1] == (uint32_t) S_ && h[2] == (uint32_t) E_ && h[3] == (uint32_t) K_;
        fclose(f);
        if (!ok) {
            RWARN("expert trace %s has a different layout, starting a new one", trace_path_.c_str());
            remove_file(trace_path_);
        }
    }
    trace_bytes_ = file_size(trace_path_);
}

void ExpertProfiler::shutdown() {
    std::string err;
    bool need_save;
    {
        std::lock_guard<std::mutex> lk(mu_);
        need_save = S_ > 0 && unsaved_ > 0;
    }
    if (need_save && !save(err)) RWARN("could not save expert fingerprint: %s", err.c_str());
    std::lock_guard<std::mutex> lk(mu_);
    if (trace_) { fclose(trace_); trace_ = nullptr; }
}

void ExpertProfiler::on_route(int il, const int32_t * ids, int k, int64_t n_tok) {
    if (!enabled_.load() || S_ == 0 || k <= 0 || n_tok <= 0) return;
    if (il < 0 || il >= (int) slot_of_layer_.size()) return;
    const int s = slot_of_layer_[(size_t) il];
    if (s < 0) return;
    const int kk = std::min(k, K_);
    auto expert_at = [&](int i, int64_t j) -> int {
        const int32_t e = ids[j * k + i];
        return (e >= 0 && e < E_) ? e : -1;
    };

    if (phase_.load() == GEN) {
        const int64_t j = n_tok - 1;
        for (int i = 0; i < kk; i++) {
            const int e = expert_at(i, j);
            cur_[(size_t) s * K_ + i] = (uint16_t) (e < 0 ? 0 : e);
        }
        for (int i = kk; i < K_; i++) cur_[(size_t) s * K_ + i] = cur_[(size_t) s * K_];
        seen_[(size_t) s] = 1;
    } else {
        std::lock_guard<std::mutex> lk(mu_);
        for (int64_t j = 0; j < n_tok; j++) {
            for (int i = 0; i < kk; i++) {
                const int e = expert_at(i, j);
                if (e >= 0) counts_prefill_[(size_t) s * E_ + e]++;
            }
        }
        if (s == 0) { tokens_prefill_ += (uint64_t) n_tok; unsaved_++; }
    }
}

std::string ExpertProfiler::trace_path() {
    std::lock_guard<std::mutex> lk(mu_);
    if (trace_) fflush(trace_);
    return trace_path_;
}

void ExpertProfiler::begin_decode() {
    if (!enabled_.load() || S_ == 0) return;
    std::fill(seen_.begin(), seen_.end(), 0);
}

void ExpertProfiler::end_decode() {
    if (!enabled_.load() || phase_.load() != GEN || S_ == 0) return;
    for (int s = 0; s < S_; s++) {
        if (!seen_[(size_t) s]) return; // incomplete record (should not happen)
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (int s = 0; s < S_; s++) {
        for (int i = 0; i < K_; i++) counts_gen_[(size_t) s * E_ + cur_[(size_t) s * K_ + i]]++;
    }
    tokens_gen_++;
    unsaved_++;
    write_trace_record();
    if (unsaved_ >= 512) {
        std::string err;
        if (!save_locked(err)) RWARN("could not save expert fingerprint: %s", err.c_str());
    }
}

void ExpertProfiler::write_trace_record() {
    if (trace_path_.empty() || trace_bytes_ >= TRACE_MAX_BYTES) return;
    if (!trace_) {
        trace_ = fopen_utf8(trace_path_, "ab");
        if (!trace_) return;
        if (file_size(trace_path_) == 0) {
            const uint32_t h[4] = { FORMAT_VERSION, (uint32_t) S_, (uint32_t) E_, (uint32_t) K_ };
            fwrite(TRACE_MAGIC, 1, 8, trace_);
            fwrite(h, 4, 4, trace_);
            trace_bytes_ = 24;
        }
    }
    fwrite(cur_.data(), sizeof(uint16_t), cur_.size(), trace_);
    trace_bytes_ += cur_.size() * sizeof(uint16_t);
}

bool ExpertProfiler::save(std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    return save_locked(err);
}

bool ExpertProfiler::save_locked(std::string & err) {
    if (S_ == 0 || profile_path_.empty()) { err = "profiling is not available for this model"; return false; }
    if (trace_) fflush(trace_);

    const std::string tmp = profile_path_ + ".tmp";
    FILE * f = fopen_utf8(tmp, "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    const uint32_t h[5] = { FORMAT_VERSION, (uint32_t) S_, (uint32_t) E_, (uint32_t) K_, (uint32_t) n_layer_ };
    bool ok = fwrite(PROF_MAGIC, 1, 8, f) == 8;
    ok = ok && fwrite(h, 4, 5, f) == 5;
    std::vector<int32_t> layers(moe_layers_.begin(), moe_layers_.end());
    ok = ok && fwrite(layers.data(), 4, layers.size(), f) == layers.size();
    const uint64_t t[2] = { tokens_gen_, tokens_prefill_ };
    ok = ok && fwrite(t, 8, 2, f) == 2;
    ok = ok && fwrite(counts_gen_.data(), 8, counts_gen_.size(), f) == counts_gen_.size();
    ok = ok && fwrite(counts_prefill_.data(), 8, counts_prefill_.size(), f) == counts_prefill_.size();
    ok = (fclose(f) == 0) && ok;
    if (!ok) { remove_file(tmp); err = "write error"; return false; }

    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(profile_path_), ec);
    if (ec) {
        remove_file(profile_path_);
        std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(profile_path_), ec);
        if (ec) { err = ec.message(); return false; }
    }
    unsaved_ = 0;
    return true;
}

bool ExpertProfiler::load_profile() {
    FILE * f = fopen_utf8(profile_path_, "rb");
    if (!f) return false;
    char m[8];
    uint32_t h[5] = { 0, 0, 0, 0, 0 };
    bool ok = fread(m, 1, 8, f) == 8 && memcmp(m, PROF_MAGIC, 8) == 0 && fread(h, 4, 5, f) == 5 &&
              h[0] == FORMAT_VERSION && h[1] == (uint32_t) S_ && h[2] == (uint32_t) E_ && h[3] == (uint32_t) K_;
    std::vector<int32_t> layers((size_t) S_);
    ok = ok && fread(layers.data(), 4, layers.size(), f) == layers.size();
    for (int s = 0; ok && s < S_; s++) ok = layers[(size_t) s] == moe_layers_[(size_t) s];
    uint64_t t[2] = { 0, 0 };
    ok = ok && fread(t, 8, 2, f) == 2;
    std::vector<uint64_t> cg(counts_gen_.size()), cp(counts_prefill_.size());
    ok = ok && fread(cg.data(), 8, cg.size(), f) == cg.size();
    ok = ok && fread(cp.data(), 8, cp.size(), f) == cp.size();
    fclose(f);
    if (!ok) {
        RWARN("ignoring expert fingerprint %s (different model layout or damaged file)", profile_path_.c_str());
        return false;
    }
    counts_gen_     = cg;
    counts_prefill_ = cp;
    tokens_gen_     = t[0];
    tokens_prefill_ = t[1];
    RLOG("loaded expert fingerprint: %llu generated + %llu prompt tokens", (unsigned long long) t[0], (unsigned long long) t[1]);
    return true;
}

void ExpertProfiler::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    std::fill(counts_gen_.begin(), counts_gen_.end(), 0);
    std::fill(counts_prefill_.begin(), counts_prefill_.end(), 0);
    tokens_gen_ = tokens_prefill_ = unsaved_ = 0;
    if (trace_) { fclose(trace_); trace_ = nullptr; }
    if (!trace_path_.empty())   remove_file(trace_path_);
    if (!profile_path_.empty()) remove_file(profile_path_);
    trace_bytes_ = 0;
}

ExpertProfiler::Snapshot ExpertProfiler::snapshot() {
    std::lock_guard<std::mutex> lk(mu_);
    if (trace_) fflush(trace_);
    Snapshot s;
    s.supported      = S_ > 0;
    s.S              = S_;
    s.E              = E_;
    s.K              = K_;
    s.moe_layers     = moe_layers_;
    s.counts_gen     = counts_gen_;
    s.counts_prefill = counts_prefill_;
    s.tokens_gen     = tokens_gen_;
    s.tokens_prefill = tokens_prefill_;
    s.trace_path     = trace_path_;
    return s;
}

nlohmann::json ExpertProfiler::summary() {
    std::lock_guard<std::mutex> lk(mu_);
    return {
        { "supported", S_ > 0 }, { "enabled", enabled_.load() },
        { "n_moe_layers", S_ }, { "n_expert", E_ }, { "n_expert_used", K_ },
        { "tokens_gen", tokens_gen_ }, { "tokens_prefill", tokens_prefill_ },
        { "trace_bytes", trace_bytes_ }, { "trace_full", trace_bytes_ >= TRACE_MAX_BYTES },
        { "profile_path", profile_path_ },
    };
}

} // namespace rafid
