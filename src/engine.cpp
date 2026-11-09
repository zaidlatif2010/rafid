#include "engine.h"
#include "slim.h"
#include "util.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace rafid {

namespace {
// while the engine works the computer stays awake: an answer or a test run left alone is not cut by sleep
// (a night's test once waited five hours on one answer)
struct BusyGuard {
    std::atomic<bool> & flag;
    explicit BusyGuard(std::atomic<bool> & f) : flag(f) { flag.store(true); keep_awake(true); }
    ~BusyGuard() { flag.store(false); keep_awake(false); }
};

struct WaitGuard {
    std::atomic<int> & n;
    explicit WaitGuard(std::atomic<int> & c) : n(c) { n.fetch_add(1); }
    ~WaitGuard() { n.fetch_sub(1); }
};

// the part of a draft that will not change any more: up to its last complete word (the word being typed, and
// the spaces before it, may still change)
size_t complete_words_end(const std::string & s) {
    auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    size_t e = s.size();
    while (e > 0 && !space(s[e - 1])) e--;
    while (e > 0 && space(s[e - 1])) e--;
    return e;
}

const char * state_name(int s) {
    switch (s) {
        case Engine::EMPTY:   return "empty";
        case Engine::LOADING: return "loading";
        case Engine::READY:   return "ready";
        default:              return "failed";
    }
}
} // namespace

Engine::Engine(HwState & hw, std::string profiles_dir) : hws_(hw), profiles_dir_(std::move(profiles_dir)) {}

Engine::~Engine() {
    unload();
}

bool Engine::progress_cb(float p, void * ud) {
    auto * e = static_cast<Engine *>(ud);
    e->progress_.store(p);
    return !e->cancel_load_.load();
}

bool Engine::preview(const std::string & path, const PlanOptions & opt, ModelInfo & mi, Plan & plan, std::string & err) {
    if (!read_model_info(path, mi, err)) return false;
    HwInfo hw;
    {
        std::lock_guard<std::mutex> lk(hws_.mu);
        const bool idle = state() == EMPTY || state() == FAILED;
        if (idle) {
            refresh_memory(hws_.hw);
            hws_.baseline = hws_.hw;
        }
        hw = idle ? hws_.hw : hws_.baseline; // while a model is loaded, plan with the memory it would free
    }
    plan = make_plan(hw, mi, opt);
    return true;
}

bool Engine::start_load(const std::string & path, const PlanOptions & opt, std::string & err) {
    if (state() == LOADING) { err = "already_loading"; return false; }
    if (busy())             { err = "busy"; return false; }
    unload();
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        load_error_.clear();
        load_path_ = path;
        attempts_  = 0;
    }
    progress_.store(0.0f);
    cancel_load_.store(false);
    state_.store(LOADING);
    loader_ = std::thread(&Engine::load_worker, this, path, opt);
    return true;
}

void Engine::load_worker(std::string path, PlanOptions opt) {
    const double t0 = now_sec();
    std::string err;
    ModelInfo mi;
    if (!read_model_info(path, mi, err)) {
        std::lock_guard<std::mutex> lk(model_mu_);
        load_error_ = err;
        state_.store(FAILED);
        return;
    }
    HwInfo hw;
    {
        std::lock_guard<std::mutex> lk(hws_.mu);
        refresh_memory(hws_.hw);
        hws_.baseline = hws_.hw;
        hw = hws_.hw;
    }

    for (int attempt = 1; attempt <= 6 && !cancel_load_.load(); attempt++) {
        const Plan plan = make_plan(hw, mi, opt);
        {
            std::lock_guard<std::mutex> lk(model_mu_);
            attempts_ = attempt;
        }
        RLOG("loading %s (attempt %d): mode=%s gpu_layers=%d experts_on_cpu=%d/%d cache=%s ctx=%d mmap=%d",
             mi.name.c_str(), attempt, plan.mode.c_str(), plan.n_gpu_layers, plan.n_cpu_moe, plan.n_moe_layers,
             human_bytes(plan.gpu_cache).c_str(), plan.n_ctx, plan.use_mmap ? 1 : 0);
        progress_.store(0.0f);
        if (try_load(mi, plan, err)) {
            {
                std::lock_guard<std::mutex> lk(model_mu_);
                opt_ = opt;
            }
            profiler_.configure(mi, profiles_dir_);
            profiler_.set_enabled(true);
            if (store_ptr_.load()) configure_store(opt);
            {
                std::lock_guard<std::mutex> lk(hws_.mu);
                refresh_memory(hws_.hw);
            }
            load_sec_ = now_sec() - t0;
            state_.store(READY);
            RLOG("model ready in %.1f s", load_sec_.load());
            return;
        }
        RWARN("attempt %d failed: %s", attempt, err.c_str());
        if (cancel_load_.load()) break;
        {
            // the failed attempt has released its buffers: plan again with fresh numbers
            std::lock_guard<std::mutex> lk(hws_.mu);
            refresh_memory(hws_.hw);
            hw = hws_.hw;
        }
        if (!plan_step_back(hw, mi, opt, plan)) break;
    }
    std::lock_guard<std::mutex> lk(model_mu_);
    load_error_ = cancel_load_.load() ? "cancelled" : err;
    state_.store(FAILED);
}

bool Engine::try_load(const ModelInfo & mi, const Plan & plan, std::string & err) {
    HwInfo hw = hws_.get();

    // stream mode: every routed expert tensor goes into the expert store (reserved memory, read on demand)
    std::unique_ptr<ExpertStore> store;
    if (plan.stream) store = std::make_unique<ExpertStore>(mi, hw.avx2 && hw.fma && hw.f16c);

    ot_patterns_.clear();
    ot_.clear();
    if (store) {
        ot_patterns_.push_back("\\.ffn_(gate|up|down|gate_up)_exps");
    } else if (!plan.override_pattern.empty()) {
        ot_patterns_.push_back(plan.override_pattern);
    }
    for (const auto & p : ot_patterns_) ot_.push_back({ p.c_str(), store ? store->buft() : ggml_backend_cpu_buffer_type() });
    ot_.push_back({ nullptr, nullptr });

    devices_.clear();
    if (plan.mode != "cpu_only") {
        if (ggml_backend_dev_t d = gpu_device(hw, plan.gpu_slot)) devices_.push_back(d);
    }
    devices_.push_back(nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.devices                     = devices_.data();
    mp.n_gpu_layers                = devices_.size() > 1 ? plan.n_gpu_layers : 0;
    mp.tensor_buft_overrides       = ot_.size() > 1 ? ot_.data() : nullptr;
    mp.load_mode                   = plan.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    mp.progress_callback           = progress_cb;
    mp.progress_callback_user_data = this;

    // stream mode on Windows: llama.cpp maps a copy without the experts, so that the original file (where the
    // experts are read from) is not mapped; NTFS reads a mapped file unbuffered one request at a time
    std::string load_path = mi.path;
    if (store) {
        {
            std::lock_guard<std::mutex> lk(model_mu_);
            load_phase_ = "slim";
        }
        std::string serr;
        const std::string slim_dir = path_join(path_join(parent_dir(profiles_dir_), "slim"), sanitize_name(profile_key(mi)));
        const std::string slim = prepare_slim_model(mi, slim_dir, [this](double f) { progress_.store((float) f); return !cancel_load_.load(); }, serr);
        {
            std::lock_guard<std::mutex> lk(model_mu_);
            load_phase_.clear();
        }
        if (cancel_load_.load()) { err = "cancelled"; return false; }
        if (!slim.empty()) {
            load_path = slim;
            RLOG("expert store: llama.cpp loads %s, experts are read from %s", slim.c_str(), mi.path.c_str());
        } else {
#ifdef _WIN32
            RWARN("expert store: no slim copy (%s); expert reads may not run in parallel", serr.c_str());
#endif
        }
        progress_.store(0.0f);
    }
    llama_model * model = llama_model_load_from_file(load_path.c_str(), mp);
    if (!model) {
        err = cancel_load_.load() ? "cancelled" : "model load failed (not enough memory or unsupported model, see log)";
        return false;
    }
    if (store && !store->finalize(err)) {
        llama_model_free(model);
        err = "expert store: " + err;
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) plan.n_ctx;
    cp.n_ubatch        = (uint32_t) (plan.n_ubatch > 0 ? plan.n_ubatch : 512);
    cp.n_batch         = std::max<uint32_t>(2048, cp.n_ubatch);
    cp.n_seq_max       = 1;
    cp.n_threads       = plan.n_threads;
    cp.n_threads_batch = plan.n_threads_batch;
    cp.no_perf         = true;
    if (mi.is_moe()) {
        cp.cb_eval           = &Engine::route_cb;
        cp.cb_eval_user_data = this;
    }
    if (store) cp.op_offload = false; // experts must be computed by the CPU straight from the store
    if (plan.mode == "hybrid_moe_cache" && plan.gpu_cache > 0) cp.moe_cache_size = (size_t) plan.gpu_cache;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        err = "context creation failed (likely out of VRAM for the KV cache / compute buffers)";
        return false;
    }

    std::lock_guard<std::mutex> lk(model_mu_);
    store_  = std::move(store);
    store_ptr_.store(store_.get());
    model_  = model;
    ctx_    = ctx;
    vocab_  = llama_model_get_vocab(model);
    mi_     = mi;
    plan_   = plan;
    n_ctx_  = (int) llama_n_ctx(ctx);
    n_batch_ = (int) llama_n_batch(ctx);
    n_ubatch_ = (int) llama_n_ubatch(ctx);
    n_threads_       = plan.n_threads;
    n_threads_batch_ = plan.n_threads_batch;
    cache_tokens_.clear();
    ctx_used_.store(0);
    logits_at_    = 0;
    prefill_from_ = -1;
    prefill_sec_  = 0;
    const char * tmpl = llama_model_chat_template(model, nullptr);
    chat_template_ = tmpl ? tmpl : "";
    think_switch_  = chat_template_.find("enable_thinking") != std::string::npos;
    think_toks_.clear();
    if (think_switch_) {
        for (const char * s : { "<think>", "</think>" }) {
            const std::vector<int32_t> t = tokenize(s, false);
            if (t.size() == 1) think_toks_.push_back(t[0]);
        }
        if (!think_toks_.empty()) RLOG("answers without thinking: %zu thinking tag token(s) barred", think_toks_.size());
    }
    recurrent_     = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    ckpts_.clear();
    ckpt_count_.store(0);
    ckpt_bytes_.store(0);
    ckpt_restored_.store(0);
    if (const char * e = getenv("RAFID_CHECKPOINTS")) max_ckpts_ = std::max(0, atoi(e));
    if (recurrent_) RLOG("recurrent state: up to %d snapshots per conversation (RAFID_CHECKPOINTS)", max_ckpts_);
    char desc[256] = { 0 };
    llama_model_desc(model, desc, sizeof(desc));
    model_desc_ = desc;
    return true;
}

void Engine::free_model() {
    std::lock_guard<std::mutex> lk(model_mu_);
    store_ptr_.store(nullptr);
    if (store_) store_->shutdown_io();
    if (ctx_)   { llama_free(ctx_);         ctx_   = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; } // releases the store's buffers
    store_.reset();
    vocab_ = nullptr;
    think_toks_.clear();
    cache_tokens_.clear();
    logits_at_    = 0;
    prefill_from_ = -1;
    ckpts_.clear();
    ckpt_count_.store(0);
    ckpt_bytes_.store(0);
    ctx_used_.store(0);
}

void Engine::unload() {
    cancel_load_.store(true);
    stop_flag_.store(true);
    if (loader_.joinable()) loader_.join();
    cancel_load_.store(false);
    {
        WaitGuard w(waiters_);                  // a question being read while typed stops at its next step
        std::lock_guard<std::mutex> g(gen_mu_); // waits for a running generation to stop
        profiler_.set_enabled(false);
        profiler_.shutdown();
        free_model();
    }
    if (state() != FAILED) state_.store(EMPTY);
    std::lock_guard<std::mutex> lk(hws_.mu);
    refresh_memory(hws_.hw);
}

bool Engine::model_info(ModelInfo & mi) {
    std::lock_guard<std::mutex> lk(model_mu_);
    if (!model_) return false;
    mi = mi_;
    return true;
}

std::vector<int32_t> Engine::tokenize(const std::string & text, bool add_special) {
    const int n = -llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    if (n <= 0) return {};
    std::vector<int32_t> t((size_t) n);
    const int r = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), t.data(), n, add_special, true);
    if (r < 0) return {};
    t.resize((size_t) r);
    return t;
}

std::string Engine::token_piece(int32_t tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab_, tok, buf, (int32_t) sizeof(buf), 0, true);
    if (n >= 0) return std::string(buf, (size_t) n);
    std::string s((size_t) -n, '\0');
    n = llama_token_to_piece(vocab_, tok, &s[0], (int32_t) s.size(), 0, true);
    if (n < 0) return "";
    s.resize((size_t) n);
    return s;
}

bool Engine::build_prompt(const std::vector<ChatMessage> & msgs_in, std::string & out, bool think, size_t * stable) {
    // what these templates write for enable_thinking=false: an empty thinking block, then the answer
    static const std::string NO_THINK = "<think>\n\n</think>\n\n";
    const bool direct = !think && think_switch_;
    // earlier answers keep the empty block they were written after: the conversation in the cache then stays
    // the start of the next prompt, and nothing has to be read again
    std::vector<ChatMessage> msgs = msgs_in;
    if (direct) {
        for (auto & m : msgs) {
            if (m.role == "assistant" && m.content.find("<think>") == std::string::npos) m.content = NO_THINK + m.content;
        }
    }
    std::vector<llama_chat_message> cm;
    size_t total = 0;
    for (const auto & m : msgs) {
        cm.push_back({ m.role.c_str(), m.content.c_str() });
        total += m.content.size() + m.role.size();
    }
    const char * tmpl = chat_template_.empty() ? nullptr : chat_template_.c_str();
    std::vector<char> buf(total * 2 + 1024);
    int n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), true, buf.data(), (int32_t) buf.size());
    if (n > (int) buf.size()) {
        buf.resize((size_t) n + 1);
        n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), true, buf.data(), (int32_t) buf.size());
    }
    if (n < 0) {
        // template not recognised: fall back to ChatML, understood by most instruction models
        out.clear();
        for (const auto & m : msgs) out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
    } else {
        out.assign(buf.data(), (size_t) n);
    }
    if (stable) *stable = out.size();
    if (direct && out.size() >= 10 && out.compare(out.size() - 10, 10, "assistant\n") == 0) out += NO_THINK;
    return true;
}

void Engine::save_checkpoint(int kind) {
    if (!recurrent_ || max_ckpts_ <= 0 || cache_tokens_.empty()) return;
    for (Checkpoint & c : ckpts_) {
        if (c.tokens.size() == cache_tokens_.size() && std::equal(c.tokens.begin(), c.tokens.end(), cache_tokens_.begin())) {
            if (kind == CKPT_QUESTION || c.kind == CKPT_SPEC) c.kind = kind; // keep it as the longest-lived kind
            return;
        }
    }
    const size_t size = llama_state_seq_get_size_ext(ctx_, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) return;
    Checkpoint c;
    try {
        c.data.resize(size);
    } catch (...) {
        return;
    }
    const double t0 = now_sec();
    if (llama_state_seq_get_data_ext(ctx_, c.data.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) return;
    c.tokens = cache_tokens_;
    c.kind   = kind;
    // the newest speculative snapshot replaces the others (typing on continues from it); then, with no room,
    // the oldest of the first kind present gives way: speculative, end of a prompt, start of a question
    if (kind == CKPT_SPEC) {
        for (auto it = ckpts_.begin(); it != ckpts_.end();) it = it->kind == CKPT_SPEC ? ckpts_.erase(it) : it + 1;
    }
    while (!ckpts_.empty() && (int) ckpts_.size() >= max_ckpts_) {
        auto victim = ckpts_.end();
        for (int k : { CKPT_SPEC, CKPT_TURN, CKPT_QUESTION }) {
            victim = std::find_if(ckpts_.begin(), ckpts_.end(), [k](const Checkpoint & x) { return x.kind == k; });
            if (victim != ckpts_.end()) break;
        }
        ckpts_.erase(victim != ckpts_.end() ? victim : ckpts_.begin());
    }
    ckpts_.push_back(std::move(c));
    uint64_t bytes = 0;
    for (const Checkpoint & x : ckpts_) bytes += x.data.size();
    ckpt_count_.store((int) ckpts_.size());
    ckpt_bytes_.store(bytes);
    static const char * KIND[] = { "end of a prompt", "start of a question", "read ahead" };
    RLOG("snapshot of the recurrent state at token %zu, %s (%s, %.0f ms)", cache_tokens_.size(), KIND[kind < 0 || kind > 2 ? 0 : kind],
         human_bytes(size).c_str(), (now_sec() - t0) * 1e3);
}

bool Engine::restore_checkpoint(const std::vector<int32_t> & toks, size_t & n_keep) {
    // the furthest snapshot that ends at or before n_keep (the part of the cache shared with the new prompt)
    const Checkpoint * best = nullptr;
    for (const Checkpoint & c : ckpts_) {
        const size_t n = c.tokens.size();
        if (n == 0 || n > n_keep || n >= toks.size() || n > cache_tokens_.size()) continue;
        if (!std::equal(c.tokens.begin(), c.tokens.end(), cache_tokens_.begin())) continue;
        if (!best || n > best->tokens.size()) best = &c;
    }
    if (!best) return false;
    const size_t n = best->tokens.size();
    llama_memory_t mem = llama_get_memory(ctx_);
    if (llama_state_seq_set_data_ext(ctx_, best->data.data(), best->data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != best->data.size()) {
        RWARN("could not restore the snapshot of the recurrent state");
        return false;
    }
    // the attention part of the cache is still valid up to n: drop what follows
    if (!llama_memory_seq_rm(mem, 0, (llama_pos) n, -1)) return false;
    RLOG("continued from the snapshot at token %zu (shared with the new prompt: %zu)", n, n_keep);
    n_keep = n;
    ckpt_restored_.fetch_add(1);
    return true;
}

void Engine::drop_checkpoints(const std::vector<int32_t> * keep_prefix_of) {
    for (auto it = ckpts_.begin(); it != ckpts_.end();) {
        const bool keep = keep_prefix_of && it->tokens.size() <= keep_prefix_of->size() &&
                          std::equal(it->tokens.begin(), it->tokens.end(), keep_prefix_of->begin());
        it = keep ? it + 1 : ckpts_.erase(it);
    }
    uint64_t bytes = 0;
    for (const Checkpoint & x : ckpts_) bytes += x.data.size();
    ckpt_count_.store((int) ckpts_.size());
    ckpt_bytes_.store(bytes);
}

size_t Engine::match_cache(std::vector<int32_t> & toks) {
    size_t n_keep = 0;
    while (n_keep < cache_tokens_.size() && n_keep < toks.size() && cache_tokens_[n_keep] == toks[n_keep]) n_keep++;
    // the model sometimes writes a text with other tokens than the tokenizer would choose for it, and the next
    // prompt (tokenized again) then leaves the cache in the middle of the last answer. With a byte-level BPE
    // vocabulary the text is all that matters: when the prompt's text continues the cached tokens' text, the
    // prompt is built from the cached tokens plus the tokens of the new text.
    if (n_keep < cache_tokens_.size() && n_keep < toks.size() && llama_vocab_type(vocab_) == LLAMA_VOCAB_TYPE_BPE) {
        std::string rest;
        for (size_t i = n_keep; i < toks.size(); i++) rest += token_piece(toks[i]);
        size_t k = n_keep, used = 0;
        while (k < cache_tokens_.size()) {
            const std::string piece = token_piece(cache_tokens_[k]);
            if (piece.empty() || rest.compare(used, piece.size(), piece) != 0) break;
            used += piece.size();
            k++;
        }
        if (k > n_keep) {
            std::vector<int32_t> tail;
            if (used < rest.size()) tail = tokenize(rest.substr(used), false);
            if ((used == rest.size() || !tail.empty()) && (int) (k + tail.size()) < n_ctx_ - 8) {
                std::vector<int32_t> spliced(cache_tokens_.begin(), cache_tokens_.begin() + (ptrdiff_t) k);
                spliced.insert(spliced.end(), tail.begin(), tail.end());
                toks.swap(spliced);
                n_keep = k;
            }
        }
    }
    return n_keep;
}

void Engine::rewind_cache(const std::vector<int32_t> & toks, size_t & n_keep) {
    if (n_keep >= cache_tokens_.size()) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    // a recurrent state cannot be rewound (llama_memory_seq_rm refuses): go back to a snapshot instead
    const bool ok = recurrent_ ? restore_checkpoint(toks, n_keep) : llama_memory_seq_rm(mem, 0, (llama_pos) n_keep, -1);
    if (!ok) {
        llama_memory_clear(mem, true);
        n_keep = 0;
    }
    cache_tokens_.resize(n_keep);
    ctx_used_.store((int) n_keep);
    logits_at_ = 0;
}

bool Engine::lock_for_chat(std::unique_lock<std::mutex> & lk) {
    lk = std::unique_lock<std::mutex>(gen_mu_, std::defer_lock);
    for (int i = 0; i < 40; i++) {
        if (lk.try_lock()) return true;
        if (prefilling_.load()) {
            // a question being read while it is typed: it stops at its next step, and this continues from there
            WaitGuard w(waiters_);
            lk.lock();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

bool Engine::prefill(const std::vector<ChatMessage> & msgs_in, const GenParams & gp, bool speculate,
                     nlohmann::json & out, std::string & err) {
    if (waiters_.load() > 0) { err = "busy"; return false; }
    std::unique_lock<std::mutex> lk(gen_mu_, std::try_to_lock);
    if (!lk.owns_lock())           { err = "busy"; return false; }
    if (state() != READY || !ctx_) { err = "no_model"; return false; }
    if (msgs_in.empty() || msgs_in.back().role != "user") { err = "no_draft"; return false; }
    BusyGuard reading(prefilling_);

    // ---- what the prompt will start with ----
    // the template's text up to a mark put in place of what follows the draft: the conversation, the start of
    // the new message (head), and the draft up to its last complete word (safe)
    static const std::string MARK = "\x01rafid-draft\x01";
    std::vector<ChatMessage> msgs = msgs_in;
    const std::string draft = msgs.back().content;
    auto upto = [&](const std::string & content, std::string & prefix) {
        msgs.back().content = content + MARK;
        std::string p;
        build_prompt(msgs, p, gp.think);
        const size_t at = p.rfind(MARK);
        if (at == std::string::npos) return false;
        prefix = p.substr(0, at);
        return true;
    };
    std::string head, safe;
    if (!upto("", head) || !upto(draft.substr(0, complete_words_end(draft)), safe) || safe.compare(0, head.size(), head) != 0) {
        err = "template";
        return false;
    }
    const std::vector<int32_t> toks_head = tokenize(head, true);
    std::vector<int32_t> toks = tokenize(safe, true);
    if (toks_head.empty() || toks.size() < toks_head.size() || !std::equal(toks_head.begin(), toks_head.end(), toks.begin())) {
        err = "template";
        return false;
    }
    // the last tokens of a word can still change when more text follows (a newline joins the punctuation
    // before it, for one): two of them are left to the chat
    toks.resize(toks.size() - std::min<size_t>(2, toks.size() - toks_head.size()));
    // positions counted from the end: they stay right when match_cache rebuilds the start of the prompt
    size_t draft_from_end = toks.size() - toks_head.size(); // where the new message's text starts
    size_t spec_from_end  = 0;                              // where the part read on speculation starts
    if (speculate) {
        // the whole prompt as the chat will build it if the draft is sent as it is
        msgs.back().content = draft;
        std::string full;
        build_prompt(msgs, full, gp.think);
        std::vector<int32_t> all = tokenize(full, true);
        size_t common = 0;
        while (common < toks.size() && common < all.size() && toks[common] == all[common]) common++;
        if (common < toks_head.size() || common >= all.size()) {
            err = "template";
            return false;
        }
        draft_from_end = all.size() - toks_head.size();
        spec_from_end  = all.size() - common;
        toks.swap(all);
    }
    // a prompt the chat would shorten (leaving old messages out to fit the context) would not reuse this
    const int reserve = std::min(std::max(gp.max_tokens, 64), n_ctx_ / 2);
    if ((int) toks.size() + reserve + (speculate ? 0 : 16) > n_ctx_) {
        err = "too_long";
        return false;
    }

    // ---- what of it is in the cache already ----
    size_t n_keep = match_cache(toks);
    const size_t n_draft = toks.size() - std::min(toks.size(), draft_from_end);
    const size_t n_spec  = speculate ? toks.size() - std::min(toks.size(), spec_from_end) : 0;
    out = { { "ok", true }, { "speculative", speculate }, { "processed", 0 }, { "sec", 0.0 }, { "cancelled", false }, { "more", false } };
    auto finish = [&](size_t target) {
        out["ready_tokens"] = cache_tokens_.size() > n_draft ? (int) (std::min(cache_tokens_.size(), target) - n_draft) : 0;
        out["complete"]     = speculate && cache_tokens_.size() == toks.size() && logits_at_ == toks.size();
        out["cached"]       = (int) cache_tokens_.size();
    };
    if (n_keep >= toks.size()) {
        // read already (a pause read the whole question, and this asks for its start again)
        finish(toks.size());
        return true;
    }
    if (recurrent_) drop_checkpoints(&toks); // snapshots of other conversations and of text typed over
    rewind_cache(toks, n_keep);
    if (prefill_from_ < 0 || (long long) n_keep < prefill_from_) prefill_from_ = (long long) n_keep;

    // ---- reading ----
    // snapshots of a recurrent state: where the new message starts (typing over its text goes back there) and,
    // before the part read on speculation, where that part starts (typing on goes back there)
    const size_t snap_draft = recurrent_ ? n_draft : 0;
    const size_t snap_spec  = recurrent_ && n_spec > n_draft ? n_spec : 0;
    auto snapshot_at = [&](size_t i) {
        if (i == 0 || i != cache_tokens_.size()) return;
        if (i == snap_draft) save_checkpoint(CKPT_QUESTION);
        else if (i == snap_spec) save_checkpoint(CKPT_SPEC);
    };
    snapshot_at(n_keep);
    profiler_.set_phase(ExpertProfiler::PREFILL);
    ExpertStore * store = store_ptr_.load();
    const double t0 = now_sec();
    size_t done = 0;
    // one pass of the model at a time (a pass reads its experts once whatever the call size, so nothing is lost),
    // to give way quickly to a chat; and a long draft (a pasted text) over several calls of two passes each
    const size_t step   = (size_t) std::max(1, std::min(n_ubatch_ > 0 ? n_ubatch_ : n_batch_, n_batch_));
    const size_t budget = step * 2;
    for (size_t i = n_keep; i < toks.size();) {
        if (waiters_.load() > 0) { out["cancelled"] = true; break; } // a chat is waiting: it continues from here
        if (done >= budget)      { out["more"] = true; break; }
        size_t nn = std::min<size_t>(step, toks.size() - i);
        for (size_t at : { snap_draft, snap_spec }) {
            if (at > i && at < i + nn) nn = at - i;
        }
        const int n  = (int) nn;
        const int rc = decode(toks.data() + i, n);
        if (rc != 0) {
            llama_memory_clear(llama_get_memory(ctx_), true);
            cache_tokens_.clear();
            ctx_used_.store(0);
            logits_at_    = 0;
            prefill_from_ = -1;
            err = fmt("decode failed (%d) while reading the draft", rc);
            return false;
        }
        cache_tokens_.insert(cache_tokens_.end(), toks.begin() + (ptrdiff_t) i, toks.begin() + (ptrdiff_t) i + n);
        ctx_used_.store((int) cache_tokens_.size());
        logits_at_ = cache_tokens_.size();
        i += (size_t) n;
        done += (size_t) n;
        snapshot_at(i);
    }
    const double sec = now_sec() - t0;
    if (store) store->prompt_done();
    prefill_sec_ += sec;
    prefill_runs_.fetch_add(1);
    prefill_tokens_.fetch_add(done);
    out["processed"] = (int) done;
    out["sec"]       = sec;
    finish(toks.size());
    return true;
}

bool Engine::chat(const std::vector<ChatMessage> & msgs_in, const GenParams & gp,
                  const std::function<bool(const std::string &)> & on_text, GenStats & st, std::string & err) {
    std::unique_lock<std::mutex> lk;
    if (!lock_for_chat(lk))        { err = "busy"; return false; }
    if (state() != READY || !ctx_) { err = "no_model"; return false; }
    BusyGuard busy(busy_);
    stop_flag_.store(false);

    // ---- fit the conversation into the context window ----
    std::vector<ChatMessage> msgs = msgs_in;
    std::string prompt;
    std::vector<int32_t> toks;
    size_t stable_chars = 0;
    const int reserve = std::min(std::max(gp.max_tokens, 64), n_ctx_ / 2);
    for (;;) {
        build_prompt(msgs, prompt, gp.think, &stable_chars);
        toks = tokenize(prompt, true);
        if ((int) toks.size() + reserve <= n_ctx_) break;
        auto it = std::find_if(msgs.begin(), msgs.end(), [](const ChatMessage & m) { return m.role != "system"; });
        if (it == msgs.end() || std::distance(it, msgs.end()) <= 1) break;
        msgs.erase(it);
        st.dropped_messages++;
    }
    if (toks.empty())                     { err = "empty_prompt"; return false; }
    if ((int) toks.size() >= n_ctx_ - 8)  { err = "prompt_too_long"; return false; }
    st.prompt_tokens = (int) toks.size();

    // ---- reuse the KV cache for the common prefix with the previous turn (and with what was read while
    // this question was typed) ----
    llama_memory_t mem = llama_get_memory(ctx_);
    size_t n_keep = match_cache(toks);
    st.prompt_tokens = (int) toks.size();
    // read to its end while it was typed, and nothing ran since: the logits of the answer's first token are there
    const bool ready = n_keep == toks.size() && cache_tokens_.size() == toks.size() && logits_at_ == toks.size();
    if (!ready && n_keep >= toks.size()) n_keep = toks.size() - 1; // we need at least one token to get logits
    st.shared_tokens = (int) n_keep;
    if (recurrent_) drop_checkpoints(&toks); // snapshots of other conversations
    rewind_cache(toks, n_keep);
    st.cached_tokens = (int) n_keep;
    if (prefill_from_ >= 0 && (long long) n_keep > prefill_from_) {
        st.prefilled_tokens = (int) (n_keep - (size_t) prefill_from_);
        st.prefilled_sec    = prefill_sec_;
        prefill_used_.fetch_add((uint64_t) st.prefilled_tokens);
    }
    prefill_from_ = -1;
    prefill_sec_  = 0;

    // where the next turn's prompt will leave this one: after the generation prompt (before the "answer
    // directly" suffix). The snapshot is taken one token earlier so that asking again also finds it.
    size_t ckpt_at = 0;
    if (recurrent_ && max_ckpts_ > 0) {
        size_t stable_toks = toks.size();
        if (stable_chars < prompt.size()) {
            const std::vector<int32_t> suffix = tokenize(prompt.substr(stable_chars), false);
            stable_toks = suffix.size() < toks.size() && std::equal(suffix.begin(), suffix.end(), toks.end() - (ptrdiff_t) suffix.size())
                        ? toks.size() - suffix.size() : 0;
        }
        if (stable_toks >= 2) ckpt_at = stable_toks - 1;
        if (ckpt_at > 0 && ckpt_at == n_keep) save_checkpoint();
    }

    // ---- prompt processing ----
    profiler_.set_phase(ExpertProfiler::PREFILL);
    ExpertStore * store = store_ptr_.load();
    const StreamStats stream_pp0 = store ? store->stats() : StreamStats();
    const double t0 = now_sec();
    for (size_t i = n_keep; i < toks.size();) {
        size_t nn = std::min<size_t>((size_t) n_batch_, toks.size() - i);
        if (ckpt_at > i && ckpt_at < i + nn) nn = ckpt_at - i;
        const int n = (int) nn;
        const int rc = decode(toks.data() + i, n);
        if (rc != 0) {
            llama_memory_clear(mem, true);
            cache_tokens_.clear();
            ctx_used_.store(0);
            logits_at_ = 0;
            err = fmt("decode failed (%d) while reading the prompt", rc);
            st.stop_reason = "error";
            return false;
        }
        cache_tokens_.insert(cache_tokens_.end(), toks.begin() + (ptrdiff_t) i, toks.begin() + (ptrdiff_t) i + n);
        ctx_used_.store((int) cache_tokens_.size());
        logits_at_ = cache_tokens_.size();
        i += (size_t) n;
        if (i == ckpt_at) save_checkpoint();
        if (stop_flag_.load()) {
            st.prompt_sec  = now_sec() - t0;
            st.stop_reason = "stopped";
            return true;
        }
    }
    st.prompt_sec = now_sec() - t0;
    if (store) {
        st.stream_prompt = store->stats().delta(stream_pp0).to_json();
        store->prompt_done();
    }
    const int n_prompt_eval = (int) (toks.size() - n_keep);
    if (st.prompt_sec > 0 && n_prompt_eval > 0) last_pp_tps_ = n_prompt_eval / st.prompt_sec;

    if (getenv("RAFID_DEBUG_LOGITS")) {
        // tests: the logits the answer starts from, to compare ways of reading the same prompt
        const float * lg = llama_get_logits_ith(ctx_, -1);
        const int nv = llama_vocab_n_tokens(vocab_);
        if (lg && nv > 0) {
            std::vector<int> ids((size_t) nv);
            for (int i = 0; i < nv; i++) ids[(size_t) i] = i;
            const int k = std::min(8, nv);
            std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [lg](int a, int b) { return lg[a] > lg[b]; });
            st.first_logits = nlohmann::json::array();
            for (int i = 0; i < k; i++) st.first_logits.push_back({ ids[(size_t) i], lg[ids[(size_t) i]] });
        }
    }

    // ---- sampler ----
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // asked to answer directly, the model can still open a thinking block of its own after the empty one and
    // spend the whole answer in it (Qwen3.5 at temperature 0, a plain question in a long conversation): the
    // tags are barred, so what it writes is the answer
    if (!gp.think && think_switch_ && !think_toks_.empty()) {
        std::vector<llama_logit_bias> bias;
        for (int32_t t : think_toks_) bias.push_back({ t, -INFINITY });
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab_), (int32_t) bias.size(), bias.data()));
    }
    if (gp.repeat_penalty > 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab_), 64, gp.repeat_penalty, 0.0f, 0.0f));
    }
    if (gp.temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (gp.top_k > 0)     llama_sampler_chain_add(smpl, llama_sampler_init_top_k(gp.top_k));
        if (gp.top_p < 1.0f)  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(gp.top_p, 1));
        if (gp.min_p > 0.0f)  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(gp.min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(gp.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(gp.seed));
    }

    // ---- generation ----
    profiler_.set_phase(ExpertProfiler::GEN);
    const StreamStats stream0 = store ? store->stats() : StreamStats();
    std::string pending;
    bool ok = true;
    st.stop_reason = "length";
    const double t1 = now_sec();
    for (int i = 0; i < gp.max_tokens; i++) {
        int32_t tok = llama_sampler_sample(smpl, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, tok)) { st.stop_reason = "eos"; break; }

        pending += token_piece(tok);
        const size_t complete = utf8_complete_prefix(pending);
        if (complete > 0) {
            if (!on_text(pending.substr(0, complete))) stop_flag_.store(true); // client went away
            pending.erase(0, complete);
        }
        if ((int) cache_tokens_.size() + 1 >= n_ctx_) { st.stop_reason = "ctx_full"; break; }

        profiler_.begin_decode();
        const int rc = decode(&tok, 1);
        profiler_.end_decode();
        if (rc != 0) {
            err = fmt("decode failed (%d) during generation", rc);
            st.stop_reason = "error";
            ok = false;
            llama_memory_clear(mem, true);
            cache_tokens_.clear();
            logits_at_ = 0;
            break;
        }
        cache_tokens_.push_back(tok);
        ctx_used_.store((int) cache_tokens_.size());
        logits_at_ = cache_tokens_.size();
        st.gen_tokens++;
        if (stop_flag_.load()) { st.stop_reason = "stopped"; break; }
    }
    if (!pending.empty()) on_text(pending);
    st.gen_sec = now_sec() - t1;
    if (store) {
        st.stream = store->stats().delta(stream0).to_json();
        st.stream["swap"] = store->config().swap; // the routing this answer was made with (shown under the answer)
    }
    if (st.gen_sec > 0 && st.gen_tokens > 0) last_tps_ = st.gen_tokens / st.gen_sec;
    llama_sampler_free(smpl);
    profiler_.set_phase(ExpertProfiler::PREFILL);
    return ok;
}

void Engine::reset_conversation() {
    stop_flag_.store(true);
    WaitGuard w(waiters_); // a question being read while typed stops at its next step
    std::lock_guard<std::mutex> g(gen_mu_);
    if (ctx_) llama_memory_clear(llama_get_memory(ctx_), true);
    cache_tokens_.clear();
    logits_at_    = 0;
    prefill_from_ = -1;
    prefill_sec_  = 0;
    drop_checkpoints(nullptr);
    ctx_used_.store(0);
}

nlohmann::json Engine::bench_threads(std::string & err) {
    std::unique_lock<std::mutex> lk;
    if (!lock_for_chat(lk))        { err = "busy"; return nullptr; }
    if (state() != READY || !ctx_) { err = "no_model"; return nullptr; }
    BusyGuard busy(busy_);

    const HwInfo hw = hws_.get();
    std::vector<int> cands;
    auto add = [&](int v) {
        v = std::max(1, std::min(v, std::max(1, hw.logical_cpus)));
        if (std::find(cands.begin(), cands.end(), v) == cands.end()) cands.push_back(v);
    };
    add(hw.p_cores);
    add(hw.p_cores + hw.e_cores / 2);
    add(hw.physical_cores);
    add(hw.logical_cpus);
    add(std::max(1, hw.physical_cores / 2));
    add(n_threads_);
    std::sort(cands.begin(), cands.end());

    const bool was_enabled = profiler_.enabled();
    profiler_.set_enabled(false);
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_clear(mem, true);
    cache_tokens_.clear();
    ctx_used_.store(0);
    logits_at_ = 0;

    std::vector<int32_t> toks = tokenize("The quick brown fox jumps over the lazy dog.", true);
    if (toks.empty() || llama_decode(ctx_, llama_batch_get_one(toks.data(), (int32_t) toks.size())) != 0) {
        profiler_.set_enabled(was_enabled);
        err = "decode failed";
        return nullptr;
    }
    int32_t tok = toks.back();
    nlohmann::json results = nlohmann::json::array();
    int best = n_threads_;
    double best_tps = 0;
    for (int n : cands) {
        llama_set_n_threads(ctx_, n, n_threads_batch_);
        for (int i = 0; i < 2; i++) llama_decode(ctx_, llama_batch_get_one(&tok, 1));
        const int N = 12;
        const double t0 = now_sec();
        int done = 0;
        for (; done < N; done++) {
            if (llama_decode(ctx_, llama_batch_get_one(&tok, 1)) != 0) break;
        }
        const double dt = now_sec() - t0;
        const double tps = dt > 0 ? done / dt : 0;
        results.push_back({ { "threads", n }, { "tps", tps } });
        if (tps > best_tps) { best_tps = tps; best = n; }
    }
    llama_set_n_threads(ctx_, best, n_threads_batch_);
    {
        std::lock_guard<std::mutex> g(model_mu_);
        n_threads_       = best;
        plan_.n_threads  = best;
    }
    llama_memory_clear(mem, true);
    cache_tokens_.clear();
    ctx_used_.store(0);
    logits_at_ = 0;
    profiler_.set_enabled(was_enabled);
    return { { "results", results }, { "best", best }, { "best_tps", best_tps } };
}

int Engine::decode(int32_t * tokens, int n) {
    ExpertStore * st = store_ptr_.load();
    llama_memory_t mem = llama_get_memory(ctx_);
    const llama_pos pos0 = llama_memory_seq_pos_max(mem, 0) + 1;
    for (int attempt = 0;; attempt++) {
        const uint64_t f0 = st ? st->fault_count() : 0;
        const int rc = llama_decode(ctx_, llama_batch_get_one(tokens, n));
        if (rc != 0 || !st || st->fault_count() == f0 || attempt >= 2) return rc;
        // an expert was loaded by the fault handler while other threads may have read it half filled:
        // forget this step and compute it again, now with every expert resident
        RWARN("expert store: %llu fault loads during a decode, recomputing it", (unsigned long long) (st->fault_count() - f0));
        if (!llama_memory_seq_rm(mem, 0, pos0, -1)) {
            RWARN("expert store: this model's recurrent state cannot be rewound, keeping the step as computed");
            return rc;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// graph callback: the router's choice of experts, shared by the fingerprint and the expert store
// -------------------------------------------------------------------------------------------------
// a tensor's values as floats (weights of any type, graph results); false when it cannot be read back
static bool tensor_to_floats(const ggml_tensor * t, int64_t n, std::vector<float> & out) {
    if (!t || !t->data || n <= 0 || n > ggml_nelements(t)) return false;
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || !buf->iface.get_tensor) return false; // e.g. CPU_REPACK weights cannot be read back
    out.resize((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        if (t->nb[0] != sizeof(float) || (n > t->ne[0] && !ggml_is_contiguous(t))) return false;
        ggml_backend_tensor_get(t, out.data(), 0, (size_t) n * sizeof(float));
        return true;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float || !ggml_is_contiguous(t) || n != ggml_nelements(t)) return false;
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    tr->to_float(raw.data(), out.data(), n);
    return true;
}

// from "ffn_moe_topk-<il>" back to the router: topk = view(argsort(probs)), probs = f(logits), logits = gate_inp x cur
static ggml_tensor * find_router_mm(ggml_tensor * t, int depth) {
    if (!t || depth > 8) return nullptr;
    if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[1] && strstr(t->src[0]->name, "ffn_gate_inp") &&
        !strstr(t->src[0]->name, "shexp")) return t;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (ggml_tensor * r = find_router_mm(t->src[i], depth + 1)) return r;
    }
    return nullptr;
}

// the router's weight of every expert ("ffn_moe_probs-<il>"): topk = view(argsort(selection)), and selection is
// the probs themselves or probs + bias (the weights are always taken from the unbiased probs)
static ggml_tensor * find_probs(ggml_tensor * topk) {
    ggml_tensor * a = topk->view_src ? topk->view_src : topk->src[0];
    if (!a || a->op != GGML_OP_ARGSORT) return nullptr;
    ggml_tensor * sel = a->src[0];
    for (int d = 0; sel && d < 3; d++) {
        if (strncmp(sel->name, "ffn_moe_probs-", 14) == 0) return sel->type == GGML_TYPE_F32 ? sel : nullptr;
        sel = sel->op == GGML_OP_ADD ? sel->src[0] : nullptr;
    }
    return nullptr;
}

// the ids of a (possibly strided) [k, n_tok] I32 tensor, compacted; raw keeps the bytes as they are in the tensor
static bool read_ids(const ggml_tensor * t, std::vector<uint8_t> & raw, std::vector<int32_t> & ids) {
    if (!t || t->type != GGML_TYPE_I32 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != sizeof(int32_t)) return false;
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || !buf->iface.get_tensor || !t->data) return false;
    const int64_t k = t->ne[0], n_tok = t->ne[1];
    const size_t nb = ggml_nbytes(t);
    raw.resize(nb);
    ggml_backend_tensor_get(t, raw.data(), 0, nb);
    ids.resize((size_t) (k * n_tok));
    for (int64_t j = 0; j < n_tok; j++) {
        for (int64_t i = 0; i < k; i++) memcpy(&ids[(size_t) (j * k + i)], raw.data() + (size_t) i * t->nb[0] + (size_t) j * t->nb[1], sizeof(int32_t));
    }
    return true;
}

// "<prefix><layer>" exactly (views of a node get names like "<name> (reshaped)" and must not match)
static bool layer_node(const char * name, const char * prefix, size_t n) {
    if (strncmp(name, prefix, n) != 0 || name[n] < '0' || name[n] > '9') return false;
    for (const char * p = name + n; *p; p++) if (*p < '0' || *p > '9') return false;
    return true;
}

bool Engine::route_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * self = static_cast<Engine *>(ud);
    const bool is_topk = layer_node(t->name, "ffn_moe_topk-", 13);
    const bool is_weights = !is_topk && t->op == GGML_OP_GET_ROWS && layer_node(t->name, "ffn_moe_weights-", 16);
    if (!is_topk && !is_weights) return !ask; // not needed / keep computing
    ExpertStore * st = self->store_ptr_.load(std::memory_order_acquire);
    const bool prof = self->profiler_.enabled();
    if (is_weights) {
        // after the routing weights were taken with the real ids: the expert matmuls may read cache slots instead
        if (ask) return st != nullptr;
        if (!st) return true;
        const int il = std::atoi(t->name + 16);
        ggml_tensor * idt = t->src[1]; // weights = get_rows(probs, selected_experts)
        if (!read_ids(idt, self->weights_raw_, self->weights_ids_)) {
            st->on_weights(il, nullptr, 0);
            return true;
        }
        if (st->on_weights(il, self->weights_ids_.data(), (int64_t) self->weights_ids_.size())) {
            const int64_t k = idt->ne[0], n_tok = idt->ne[1];
            for (int64_t j = 0; j < n_tok; j++) {
                for (int64_t i = 0; i < k; i++) memcpy(self->weights_raw_.data() + (size_t) i * idt->nb[0] + (size_t) j * idt->nb[1], &self->weights_ids_[(size_t) (j * k + i)], sizeof(int32_t));
            }
            ggml_backend_tensor_set(idt, self->weights_raw_.data(), 0, self->weights_raw_.size());
        }
        return true;
    }
    if (ask) return prof || st != nullptr;
    if ((!prof && !st) || t->type != GGML_TYPE_I32) return true;

    const int     il    = std::atoi(t->name + 13);
    const int64_t k     = t->ne[0];
    const int64_t n_tok = t->ne[1];
    if (k <= 0 || n_tok <= 0) return true;
    // the tensor is a strided view (top-k rows of an argsort): copy its byte span, then compact it
    const size_t nb = ggml_nbytes(t);
    self->route_bytes_.resize(nb);
    ggml_backend_tensor_get(t, self->route_bytes_.data(), 0, nb);
    self->route_ids_.resize((size_t) (k * n_tok));
    for (int64_t j = 0; j < n_tok; j++) {
        for (int64_t i = 0; i < k; i++) {
            int32_t e;
            memcpy(&e, self->route_bytes_.data() + (size_t) i * t->nb[0] + (size_t) j * t->nb[1], sizeof(e));
            self->route_ids_[(size_t) (j * k + i)] = e;
        }
    }
    if (prof) self->profiler_.on_route(il, self->route_ids_.data(), (int) k, n_tok);
    if (!st) return true;

    // for the gate predictor: this layer's router (once) and, for one token, the hidden state
    //   logits = gate_inp x cur,  cur = rms_norm(x) * w   (x = residual stream entering the FFN)
    const float * hidden = nullptr;
    int64_t hidden_rows = 0;
    bool normed = false;
    if (ggml_tensor * mm = find_router_mm(t, 0)) {
        ggml_tensor * cur = mm->src[1];
        ggml_tensor * x = nullptr, * w = nullptr;
        if (cur->op == GGML_OP_MUL && cur->src[0] && cur->src[1]) {
            if (cur->src[0]->op == GGML_OP_RMS_NORM)      { x = cur->src[0]->src[0]; w = cur->src[1]; }
            else if (cur->src[1]->op == GGML_OP_RMS_NORM) { x = cur->src[1]->src[0]; w = cur->src[0]; }
        }
        const int64_t n_embd = mm->src[0]->ne[0];
        if (w && (ggml_nelements(w) != n_embd || w->ne[0] != n_embd)) w = nullptr;
        if (st->wants_router(il)) {
            const ggml_tensor * g = mm->src[0];
            const bool have_norm = w && tensor_to_floats(w, n_embd, self->route_norm_);
            if (g->ne[2] == 1 && g->ne[3] == 1 && tensor_to_floats(g, ggml_nelements(g), self->route_gate_)) {
                st->set_router(il, self->route_gate_.data(), n_embd, g->ne[1], have_norm ? self->route_norm_.data() : nullptr);
            }
        }
        // the token (generation), or every token of a prompt batch for the read-ahead of the next layer
        const int64_t rows = n_tok == 1 ? 1 : (st->wants_batch_hidden() ? n_tok : 0);
        if (rows > 0 && n_embd == st->n_embd()) {
            if (x && x->ne[0] == n_embd && x->ne[1] >= rows && tensor_to_floats(x, n_embd * rows, self->route_hidden_)) {
                hidden = self->route_hidden_.data();
                hidden_rows = rows;
            } else if (cur->ne[0] == n_embd && cur->ne[1] >= rows && tensor_to_floats(cur, n_embd * rows, self->route_hidden_)) {
                hidden = self->route_hidden_.data();
                hidden_rows = rows;
                normed = true;
            }
        }
    }
    // cache-aware routing needs the router's weight of every expert for this token
    const float * probs = nullptr;
    if (n_tok == 1 && st->wants_probs()) {
        ggml_tensor * pt = find_probs(t);
        const int64_t n_exp = pt ? pt->ne[0] : 0;
        if (pt && n_exp >= k && tensor_to_floats(pt, n_exp, self->route_probs_)) {
            const float * pr = self->route_probs_.data();
            bool sane = true;
            for (int64_t i = 0; i < k && sane; i++) {
                const int32_t e = self->route_ids_[(size_t) i];
                sane = e >= 0 && e < n_exp && std::isfinite(pr[e]) && (i == 0 || pr[e] <= pr[self->route_ids_[(size_t) i - 1]] + 1e-6f);
            }
            if (sane) probs = pr;
            else if (!self->warned_probs_) {
                RWARN("cache-aware routing: the router weights of layer %d do not match its choice, not replacing experts there", il);
                self->warned_probs_ = true;
            }
        }
    }
    if (st->on_route(il, self->route_ids_.data(), (int) k, n_tok, hidden, hidden_rows, normed, probs)) {
        // experts were replaced: the routing weights and the expert matmuls must see the new ids
        for (int64_t i = 0; i < k; i++) memcpy(self->route_bytes_.data() + (size_t) i * t->nb[0], &self->route_ids_[(size_t) i], sizeof(int32_t));
        ggml_backend_tensor_set(t, self->route_bytes_.data(), 0, nb);
    }
    return true;
}

// -------------------------------------------------------------------------------------------------
// expert store (stream mode)
// -------------------------------------------------------------------------------------------------
static std::vector<uint64_t> fingerprint_counts(const ExpertProfiler::Snapshot & snap) {
    std::vector<uint64_t> c = snap.counts_gen;
    if (snap.tokens_gen < 50) {
        for (size_t i = 0; i < c.size() && i < snap.counts_prefill.size(); i++) c[i] += snap.counts_prefill[i];
    }
    return c;
}

void Engine::configure_store(const PlanOptions & opt) {
    ExpertStore * st = store_ptr_.load();
    if (!st) return;
    StreamConfig c;
    c.budget_bytes = opt.stream_budget_bytes ? opt.stream_budget_bytes : st->total_bytes() / 5;
    c.policy       = opt.stream_policy;
    c.prefetch     = opt.stream_prefetch;
    c.swap         = opt.stream_swap;
    st->configure(c, fingerprint_counts(profiler_.snapshot()), profiler_.trace_path());
    st->preload(false);
    const StreamConfig got = st->config();
    RLOG("expert store: RAM cache %s (%.0f%% of the experts), policy %s, prefetch %s, cache-aware routing %s",
         human_bytes(got.budget_bytes).c_str(), 100.0 * (double) got.budget_bytes / (double) std::max<uint64_t>(1, st->total_bytes()),
         got.policy.c_str(), got.prefetch.c_str(), got.swap.c_str());
}

uint64_t Engine::stream_budget_limit() {
    ExpertStore * st = store_ptr_.load();
    if (!st) return 0;
    const nlohmann::json s = st->status();
    uint64_t avail, total;
    {
        std::lock_guard<std::mutex> lk(hws_.mu);
        refresh_ram(hws_.hw);
        avail = hws_.hw.ram_avail;
        total = hws_.hw.ram_total;
    }
    const uint64_t pool    = s.value("pool_bytes", (uint64_t) 0);
    const uint64_t stg_now = s.value("staging_bytes", (uint64_t) 0);
    uint64_t stg_plan, other;
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        stg_plan = plan_.stream_staging;
        other    = plan_.cpu_weights + plan_.cpu_kv + plan_.cpu_compute + plan_.cpu_snapshots + plan_.stream_staging;
    }
    const uint64_t reserve = (1536ull << 20) + (stg_plan > stg_now ? stg_plan - stg_now : 0); // margin + prompt buffers
    const uint64_t extra   = avail > reserve ? avail - reserve : 0;
    uint64_t limit = std::min(st->total_bytes(), pool + extra);
    // "available" also counts file cache and cache pages that Windows already moved out (when the cache could
    // not be locked), so it can promise more than the machine has: the total RAM caps it
    if (total > 0) {
        const uint64_t used_elsewhere = os_ram_reserve(total) + other;
        limit = std::min(limit, total > used_elsewhere ? total - used_elsewhere : 0);
    }
    return limit;
}

bool Engine::stream_configure(const StreamConfig & c_in, std::string & err) {
    std::unique_lock<std::mutex> lk;
    if (!lock_for_chat(lk)) { err = "busy"; return false; }
    ExpertStore * st = store_ptr_.load();
    if (!st) { err = "not_streaming"; return false; }
    // what is left out (0, "", a negative share) stays as it is
    StreamConfig c = c_in;
    const StreamConfig cur = st->config();
    if (c.budget_bytes == 0) c.budget_bytes = cur.budget_bytes;
    if (c.policy.empty())    c.policy       = cur.policy;
    if (c.prefetch.empty())  c.prefetch     = cur.prefetch;
    if (c.swap.empty())      c.swap         = cur.swap;
    if (c.pin_frac < 0)      c.pin_frac     = cur.pin_frac;
    c.freq_half  = cur.freq_half;
    c.prefetch_n = cur.prefetch_n;
    if (c.budget_bytes > stream_budget_limit() + (64ull << 20)) { err = "budget_too_big"; return false; }
    st->configure(c, fingerprint_counts(profiler_.snapshot()), profiler_.trace_path());
    st->preload(false);
    std::lock_guard<std::mutex> g(model_mu_);
    const StreamConfig got = st->config();
    opt_.stream_budget_bytes = got.budget_bytes;
    opt_.stream_policy       = got.policy;
    opt_.stream_prefetch     = got.prefetch;
    opt_.stream_swap         = got.swap;
    return true;
}

static const char * BENCH_PROMPT =
    "اشرح بالتفصيل كيف تعمل الذاكرة المؤقتة (الكاش) في الحاسوب، ولماذا تجعل البرامج أسرع، "
    "ثم اكتب دالة بلغة بايثون تحسب متوسط قائمة من الأرقام، وفي النهاية لخّص الفكرة في نقاط.";

int32_t Engine::argmax_logits() {
    const float * lg = llama_get_logits_ith(ctx_, -1);
    const int n = llama_vocab_n_tokens(vocab_);
    if (!lg || n <= 0) return -1;
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (lg[i] > lg[best]) best = i;
    return best;
}

bool Engine::stream_bench(const std::vector<StreamBenchConfig> & configs, int tokens, int warmup,
                          const std::function<bool(const nlohmann::json &)> & on_event, std::string & err) {
    std::unique_lock<std::mutex> lk;
    if (!lock_for_chat(lk))        { err = "busy"; return false; }
    if (state() != READY || !ctx_) { err = "no_model"; return false; }
    ExpertStore * st = store_ptr_.load();
    if (!st)                       { err = "not_streaming"; return false; }
    const uint64_t limit = stream_budget_limit() + (64ull << 20);
    for (const auto & c : configs) {
        if (c.budget_bytes > limit) { err = "budget_too_big"; return false; }
    }
    BusyGuard busy(busy_);
    stop_flag_.store(false);
    tokens = std::max(8, std::min(tokens, 512));
    warmup = std::max(0, std::min(warmup, 256));

    // the benchmark text must not end up in your fingerprint
    const bool prof_was = profiler_.enabled();
    profiler_.set_enabled(false);
    const StreamConfig original = st->config();
    const std::vector<uint64_t> counts = fingerprint_counts(profiler_.snapshot());
    const std::string trace = profiler_.trace_path();

    std::string prompt;
    build_prompt({ { "user", BENCH_PROMPT } }, prompt);
    const std::vector<int32_t> toks = tokenize(prompt, true);
    if (toks.empty() || (int) toks.size() + warmup + tokens + 8 >= n_ctx_) {
        profiler_.set_enabled(prof_was);
        err = "ctx_too_small";
        return false;
    }
    // every run generates the same text (fixed seed, end-of-text disabled), so every run routes to the same experts
    std::vector<llama_logit_bias> bias;
    const int n_vocab = llama_vocab_n_tokens(vocab_);
    for (int32_t t = 0; t < n_vocab; t++) {
        if (llama_vocab_is_eog(vocab_, t)) bias.push_back({ t, -INFINITY });
    }
    llama_memory_t mem = llama_get_memory(ctx_);
    bool ok = true;
    std::vector<int32_t> ref_tok, ref_arg; // the text of the first exact run, and its most likely token at each step

    for (size_t ci = 0; ci < configs.size() && !stop_flag_.load(); ci++) {
        const StreamBenchConfig & bc = configs[ci];
        StreamConfig c = original;
        c.budget_bytes = bc.budget_bytes;
        c.policy       = bc.policy;
        c.prefetch     = bc.prefetch;
        c.swap         = bc.swap;
        st->configure(c, counts, trace);
        const StreamConfig used = st->config();
        on_event({ { "type", "start" }, { "index", ci }, { "budget", used.budget_bytes }, { "policy", used.policy },
                   { "prefetch", used.prefetch }, { "swap", used.swap } });

        st->reset_cache();
        const double preload_sec = st->preload(true);
        llama_memory_clear(mem, true);
        cache_tokens_.clear();
        ctx_used_.store(0);
        logits_at_ = 0;

        double pp_sec = 0;
        {
            const double t0 = now_sec();
            for (size_t i = 0; i < toks.size() && ok;) {
                const int n = (int) std::min<size_t>((size_t) n_batch_, toks.size() - i);
                if (decode(const_cast<int32_t *>(toks.data()) + i, n) != 0) ok = false;
                i += (size_t) n;
            }
            pp_sec = now_sec() - t0;
            st->prompt_done();
        }
        if (!ok) { err = "decode failed while reading the prompt"; break; }

        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(n_vocab, (int32_t) bias.size(), bias.data()));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

        // the first exact run writes the text; every later run reads exactly that text (teacher forcing), so all
        // runs route the same way, and the most likely next token of each step is compared with the exact run's
        const bool exact  = used.swap == "off";
        const bool forced = !ref_tok.empty();
        std::vector<int32_t> rec_tok, rec_arg;
        int agree = 0, compared = 0;
        StreamStats s0 = st->stats();
        double t0 = now_sec();
        int32_t tok = forced ? ref_tok[0] : llama_sampler_sample(smpl, ctx_, -1);
        int done = 0;
        for (int i = 0; i < warmup + tokens; i++) {
            if (i == warmup) { s0 = st->stats(); t0 = now_sec(); }
            if (decode(&tok, 1) != 0) { ok = false; break; }
            const int32_t best = argmax_logits();
            if (forced) {
                if (i >= warmup && i < (int) ref_arg.size()) { compared++; agree += best == ref_arg[(size_t) i]; }
                tok = i + 1 < (int) ref_tok.size() ? ref_tok[(size_t) i + 1] : llama_sampler_sample(smpl, ctx_, -1);
            } else {
                rec_tok.push_back(tok);
                rec_arg.push_back(best);
                tok = llama_sampler_sample(smpl, ctx_, -1);
            }
            if (i >= warmup) done++;
            if (stop_flag_.load()) break;
        }
        const double dt = now_sec() - t0;
        const StreamStats d = st->stats().delta(s0);
        llama_sampler_free(smpl);
        if (!ok) { err = "decode failed during generation"; break; }
        if (!forced && exact && done == tokens) { ref_tok.swap(rec_tok); ref_arg.swap(rec_arg); }

        nlohmann::json r = d.to_json();
        r["type"]        = "result";
        r["index"]       = ci;
        r["budget"]      = used.budget_bytes;
        r["budget_frac"] = (double) used.budget_bytes / (double) std::max<uint64_t>(1, st->total_bytes());
        r["policy"]      = used.policy;
        r["prefetch"]    = used.prefetch;
        r["swap"]        = used.swap;
        r["agreement"]   = compared > 0 ? nlohmann::json((double) agree / compared) : nlohmann::json(nullptr);
        r["swaps_per_token"] = done ? (double) d.swapped / done : 0.0;
        r["tokens"]      = done;
        r["seconds"]     = dt;
        r["tps"]         = dt > 0 ? done / dt : 0.0;
        r["mb_per_token"] = done ? (double) d.bytes_read / done / 1e6 : 0.0;
        r["wait_share"]  = dt > 0 ? d.wait_sec / dt : 0.0;
        r["preload_sec"] = preload_sec;
        r["prompt_sec"]  = pp_sec;
        r["prompt_tokens"] = (int) toks.size();
        if (!on_event(r)) break; // client went away
    }

    // back to the user's own settings
    st->configure(original, counts, trace);
    st->preload(false);
    llama_memory_clear(mem, true);
    cache_tokens_.clear();
    ctx_used_.store(0);
    logits_at_ = 0;
    profiler_.set_enabled(prof_was);
    return ok;
}

// -------------------------------------------------------------------------------------------------
// fidelity test: the answers the model wrote with exact routing, read back token by token under each routing
// -------------------------------------------------------------------------------------------------
namespace {
// the answer tokens' share of the expert store's numbers (the prompts in between are read exactly in every mode)
struct FidStream {
    uint64_t accesses = 0, hits = 0, bytes_read = 0, swap_candidates = 0, swapped = 0;
    double   wait_sec = 0;
    void add(const StreamStats & d) {
        accesses += d.accesses; hits += d.hits; bytes_read += d.bytes_read;
        swap_candidates += d.swap_candidates; swapped += d.swapped; wait_sec += d.wait_sec;
    }
    nlohmann::json json() const {
        return { { "accesses", accesses }, { "hits", hits }, { "bytes_read", bytes_read }, { "swap_candidates", swap_candidates },
                 { "swapped", swapped }, { "wait_sec", wait_sec } };
    }
};
// running sums of one mode, one conversation or one turn
struct FidSums {
    int64_t tokens = 0, agree_off = 0, agree_ref = 0;
    double  nll = 0, kl = 0, kl_max = 0, sec = 0;
    int64_t kl_big = 0; // steps with KL > 0.1 (the routed model really thought otherwise)
    void add(const FidSums & o) {
        tokens += o.tokens; agree_off += o.agree_off; agree_ref += o.agree_ref; nll += o.nll; kl += o.kl;
        kl_max = std::max(kl_max, o.kl_max); kl_big += o.kl_big; sec += o.sec;
    }
    nlohmann::json json() const {
        return { { "tokens", tokens }, { "agree_off", agree_off }, { "agree_ref", agree_ref }, { "nll", nll }, { "kl", kl },
                 { "kl_max", kl_max }, { "kl_big", kl_big }, { "sec", sec } };
    }
};
}

bool Engine::fidelity(const std::vector<FidelityConv> & convs, const std::vector<std::string> & modes_in, int top_k,
                      const std::function<bool(const nlohmann::json &)> & on_event, std::string & err) {
    std::unique_lock<std::mutex> lk;
    if (!lock_for_chat(lk))        { err = "busy"; return false; }
    if (state() != READY || !ctx_) { err = "no_model"; return false; }
    ExpertStore * st = store_ptr_.load();
    if (!st)                       { err = "not_streaming"; return false; }
    if (convs.empty())             { err = "no_conversations"; return false; }
    std::vector<std::string> modes = { "off" }; // the reference comes first
    for (const std::string & m0 : modes_in) {
        const std::string m = normalize_swap_mode(m0);
        if (std::find(modes.begin(), modes.end(), m) == modes.end()) modes.push_back(m);
    }
    const int K = std::max(1, std::min(top_k, 64));
    BusyGuard busy(busy_);
    stop_flag_.store(false);

    // the tokens, the same for every mode: per turn the part read in batches (template, question; exactly as a
    // prompt is read) and the answer, read one token at a time (as it is written, so the routing applies)
    struct Part { std::vector<int32_t> prompt, answer; bool restart = false; };
    std::vector<std::vector<Part>> parts(convs.size());
    std::vector<bool> skip(convs.size(), false);
    int64_t total_answer = 0, total_prompt = 0;
    nlohmann::json plan_convs = nlohmann::json::array();
    for (size_t ci = 0; ci < convs.size(); ci++) {
        const FidelityConv & C = convs[ci];
        std::vector<ChatMessage> msgs;
        if (!C.system.empty()) msgs.push_back({ "system", C.system });
        std::string prev;
        int64_t used = 0, longest = 0, n_ans = 0, n_pr = 0;
        for (const FidelityTurn & t : C.turns) {
            msgs.push_back({ "user", t.user });
            std::string p;
            if (!build_prompt(msgs, p, false)) { err = "template"; return false; }
            Part part;
            if (!prev.empty() && p.compare(0, prev.size(), prev) == 0) {
                part.prompt = tokenize(p.substr(prev.size()), false);
            } else {
                part.prompt  = tokenize(p, true);
                part.restart = !prev.empty(); // the template wrote the earlier turns otherwise: read it all again
                if (part.restart) used = 0;
            }
            part.answer = tokenize(t.answer, false);
            prev = p + t.answer;
            msgs.push_back({ "assistant", t.answer });
            used += (int64_t) (part.prompt.size() + part.answer.size());
            longest = std::max(longest, used);
            n_ans += (int64_t) part.answer.size();
            n_pr  += (int64_t) part.prompt.size();
            parts[ci].push_back(std::move(part));
        }
        skip[ci] = longest + 8 >= n_ctx_;
        if (!skip[ci]) { total_answer += n_ans; total_prompt += n_pr; }
        plan_convs.push_back({ { "id", C.id }, { "turns", C.turns.size() }, { "answer_tokens", n_ans }, { "prompt_tokens", n_pr },
                               { "context", longest }, { "skipped", skip[ci] } });
    }
    if (total_answer == 0) { err = "no_answers"; return false; }
    if (!on_event({ { "type", "plan" }, { "modes", modes }, { "convs", plan_convs }, { "answer_tokens", total_answer },
                    { "prompt_tokens", total_prompt }, { "top_k", K }, { "n_ctx", n_ctx_ } })) return true;

    const bool prof_was = profiler_.enabled();
    profiler_.set_enabled(false);
    const StreamConfig original = st->config();
    const std::vector<uint64_t> counts = fingerprint_counts(profiler_.snapshot());
    const std::string trace = profiler_.trace_path();
    llama_memory_t mem = llama_get_memory(ctx_);
    const int n_vocab = llama_vocab_n_tokens(vocab_);

    // the reference (exact routing), per answer token: its likeliest token and its K likeliest with their log-probs
    std::vector<int32_t> ref_arg, ref_ids;
    std::vector<float>   ref_lp;
    std::vector<std::pair<float, int32_t>> heap;
    heap.reserve((size_t) K + 1);
    bool ok = true, stopped = false;

    for (size_t mi = 0; mi < modes.size() && ok && !stopped; mi++) {
        const std::string & mode = modes[mi];
        const bool is_ref = mi == 0;
        StreamConfig c = original;
        c.swap = mode;
        st->configure(c, counts, trace);
        st->reset_cache();
        const double preload_sec = st->preload(true);
        if (!on_event({ { "type", "mode" }, { "mode", mode }, { "index", mi }, { "preload_sec", preload_sec } })) { stopped = true; break; }

        FidSums M;
        FidStream MS;
        double prompt_sec = 0;
        size_t pos = 0; // answer token index across all conversations (the same in every mode)
        for (size_t ci = 0; ci < convs.size() && ok && !stopped; ci++) {
            if (skip[ci]) continue;
            llama_memory_clear(mem, true);
            cache_tokens_.clear();
            logits_at_ = 0;
            for (size_t ti = 0; ti < parts[ci].size() && ok && !stopped; ti++) {
                Part & P = parts[ci][ti];
                if (P.restart) llama_memory_clear(mem, true);
                const double tp = now_sec();
                for (size_t i = 0; i < P.prompt.size() && ok;) {
                    const int n = (int) std::min<size_t>((size_t) n_batch_, P.prompt.size() - i);
                    if (decode(P.prompt.data() + i, n) != 0) ok = false;
                    i += (size_t) n;
                }
                st->prompt_done();
                prompt_sec += now_sec() - tp;
                if (!ok) { err = "decode failed while reading a prompt"; break; }

                FidSums T;
                const StreamStats s0 = st->stats();
                const double t0 = now_sec();
                for (size_t j = 0; j < P.answer.size(); j++) {
                    const float * lg = llama_get_logits_ith(ctx_, -1);
                    if (!lg) { ok = false; err = "no logits"; break; }
                    const int32_t want = P.answer[j];
                    float mx = lg[0];
                    int32_t best = 0;
                    for (int32_t v = 1; v < n_vocab; v++) if (lg[v] > mx) { mx = lg[v]; best = v; }
                    double se = 0;
                    for (int32_t v = 0; v < n_vocab; v++) se += std::exp((double) (lg[v] - mx));
                    const double lse = (double) mx + std::log(se);
                    auto lp = [&](int32_t v) { return (double) lg[v] - lse; };
                    T.tokens++;
                    T.nll -= lp(want);
                    T.agree_ref += best == want;
                    if (is_ref) {
                        T.agree_off++;
                        ref_arg.push_back(best);
                        heap.clear();
                        for (int32_t v = 0; v < n_vocab; v++) {
                            if ((int) heap.size() < K) { heap.push_back({ lg[v], v }); std::push_heap(heap.begin(), heap.end(), std::greater<>()); }
                            else if (lg[v] > heap.front().first) {
                                std::pop_heap(heap.begin(), heap.end(), std::greater<>());
                                heap.back() = { lg[v], v };
                                std::push_heap(heap.begin(), heap.end(), std::greater<>());
                            }
                        }
                        for (int k = 0; k < K; k++) {
                            const bool have = k < (int) heap.size();
                            ref_ids.push_back(have ? heap[(size_t) k].second : -1);
                            ref_lp.push_back(have ? (float) lp(heap[(size_t) k].second) : -INFINITY);
                        }
                    } else if (pos < ref_arg.size()) {
                        T.agree_off += best == ref_arg[pos];
                        // KL(reference || routed) over the reference's K likeliest tokens and the rest as one more
                        // outcome: a lower bound of the full KL, close to it since those K hold nearly all the mass
                        double kl = 0, p_in = 0, q_in = 0;
                        for (int k = 0; k < K; k++) {
                            const int32_t id = ref_ids[pos * (size_t) K + (size_t) k];
                            if (id < 0) continue;
                            const double lpp = ref_lp[pos * (size_t) K + (size_t) k], lq = lp(id), p = std::exp(lpp);
                            kl += p * (lpp - lq);
                            p_in += p;
                            q_in += std::exp(lq);
                        }
                        const double pr = 1.0 - p_in, qr = 1.0 - q_in;
                        if (pr > 1e-9 && qr > 1e-12) kl += pr * std::log(pr / qr);
                        kl = std::max(0.0, kl);
                        T.kl += kl;
                        T.kl_max = std::max(T.kl_max, kl);
                        T.kl_big += kl > 0.1;
                    }
                    pos++;
                    int32_t tok = want;
                    if (decode(&tok, 1) != 0) { ok = false; err = "decode failed while reading an answer"; break; }
                    if (stop_flag_.load()) { stopped = true; break; }
                }
                T.sec = now_sec() - t0;
                MS.add(st->stats().delta(s0));
                M.add(T);
                if (!ok) break;
                nlohmann::json ev = T.json();
                ev["type"] = "turn"; ev["mode"] = mode; ev["conv"] = convs[ci].id; ev["turn"] = ti;
                ev["done"] = (int64_t) pos; ev["total"] = total_answer;
                if (!on_event(ev)) { stopped = true; break; }
            }
        }
        if (!ok) break;
        nlohmann::json r = M.json();
        r["type"] = "mode_done"; r["mode"] = mode; r["stopped"] = stopped;
        r["tps"] = M.sec > 0 ? (double) M.tokens / M.sec : 0.0;
        r["prompt_sec"] = prompt_sec;
        r["preload_sec"] = preload_sec;
        r["stream"] = MS.json();
        r["budget"] = st->config().budget_bytes;
        if (!on_event(r)) stopped = true;
    }

    // back to the user's own settings
    st->configure(original, counts, trace);
    st->preload(false);
    llama_memory_clear(mem, true);
    cache_tokens_.clear();
    ctx_used_.store(0);
    logits_at_ = 0;
    profiler_.set_enabled(prof_was);
    return ok;
}

nlohmann::json Engine::status() {
    nlohmann::json j;
    const int s = state();
    j["state"]    = state_name(s);
    j["progress"] = progress_.load();
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        j["phase"] = load_phase_;
    }
    j["busy"]     = busy();
    {
        std::lock_guard<std::mutex> lk(model_mu_);
        j["error"]     = load_error_;
        j["load_path"] = load_path_;
        j["attempts"]  = attempts_;
        if (s == READY && model_) {
            j["model"]           = model_to_json(mi_);
            j["plan"]            = plan_to_json(plan_);
            j["desc"]            = model_desc_;
            j["n_ctx"]           = n_ctx_;
            j["n_threads"]       = n_threads_;
            j["n_threads_batch"] = n_threads_batch_;
            j["load_sec"]        = load_sec_.load();
            j["has_template"]    = !chat_template_.empty();
            j["think_switch"]    = think_switch_;
            j["recurrent"]       = recurrent_;
        }
    }
    if (ExpertStore * st = store_ptr_.load()) {
        j["stream"] = st->status();
        j["stream"]["max_budget"] = stream_budget_limit();
    }
    j["ctx_used"]    = ctx_used_.load();
    j["checkpoints"] = { { "count", ckpt_count_.load() }, { "bytes", ckpt_bytes_.load() }, { "restored", ckpt_restored_.load() } };
    j["prefill"]     = { { "runs", prefill_runs_.load() }, { "tokens", prefill_tokens_.load() }, { "used", prefill_used_.load() } };
    j["last_tps"]    = last_tps_.load();
    j["last_pp_tps"] = last_pp_tps_.load();
    j["profiler"]    = profiler_.summary();
    return j;
}

} // namespace rafid
