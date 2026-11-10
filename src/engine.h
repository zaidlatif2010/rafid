// رافد - the engine: loads a model according to the plan and runs chat generation
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "llama.h"

#include "expert_store.h"
#include "gguf_info.h"
#include "hw_probe.h"
#include "planner.h"
#include "profiler.h"

namespace rafid {

struct ChatMessage {
    std::string role;    // system | user | assistant
    std::string content;
};

struct GenParams {
    float    temperature    = 0.7f;
    float    top_p          = 0.95f;
    float    min_p          = 0.05f;
    int      top_k          = 40;
    float    repeat_penalty = 1.05f;
    int      max_tokens     = 2048;
    uint32_t seed           = 0xFFFFFFFFu;
    bool     think          = true;  // false: a model whose template has an enable_thinking switch answers directly
    bool     ignore_eos     = false; // write max_tokens tokens whatever the model wants (speed measurements)
};

struct GenStats {
    int    prompt_tokens = 0;    // tokens in the whole conversation prompt
    int    cached_tokens = 0;    // tokens reused from the previous turn
    int    shared_tokens = 0;    // tokens the prompt shares with the cache (before going back to a snapshot)
    int    gen_tokens    = 0;
    int    prefilled_tokens = 0; // tokens of this prompt read while the question was being typed (Engine::prefill)
    double prefilled_sec    = 0; // the time that took, hidden behind the typing
    double prompt_sec    = 0;
    double gen_sec       = 0;
    int    dropped_messages = 0; // old messages left out to fit the context
    std::string stop_reason;     // eos | length | stopped | ctx_full | error
    nlohmann::json stream;       // expert store numbers for the generated tokens (stream mode only)
    nlohmann::json stream_prompt;// same while the prompt was read
    nlohmann::json first_logits; // RAFID_DEBUG_LOGITS=1: the 8 largest logits after the prompt (tests compare paths)
};

// one run of the real-speed benchmark of the expert store
struct StreamBenchConfig {
    uint64_t    budget_bytes = 0;
    std::string policy = "freq";
    std::string prefetch = "gate";   // off | history | gate
    std::string swap = "off";        // cache-aware routing
};

// fidelity test: conversations whose answers (written by the model with exact routing) are read back token by
// token under each routing; every token, the routed model's choice is compared with the exact one
struct FidelityTurn { std::string user, answer; };
struct FidelityConv { std::string id, system; std::vector<FidelityTurn> turns; };

// shared hardware state (benchmarks run in the background)
struct HwState {
    std::mutex mu;
    HwInfo     hw;
    HwInfo     baseline; // memory as it was before this process loaded a model
    HwInfo get()          { std::lock_guard<std::mutex> lk(mu); return hw; }
    HwInfo get_baseline() { std::lock_guard<std::mutex> lk(mu); return baseline; }
};

class Engine {
public:
    enum State { EMPTY = 0, LOADING = 1, READY = 2, FAILED = 3 };

    Engine(HwState & hw, std::string profiles_dir);
    ~Engine();

    // planning without loading (reads only the GGUF header)
    bool preview(const std::string & path, const PlanOptions & opt, ModelInfo & mi, Plan & plan, std::string & err);

    bool start_load(const std::string & path, const PlanOptions & opt, std::string & err); // async
    void unload();
    State state() const { return (State) state_.load(); }
    bool  busy()  const { return busy_.load(); }

    bool chat(const std::vector<ChatMessage> & msgs, const GenParams & gp,
              const std::function<bool(const std::string &)> & on_text, GenStats & st, std::string & err);
    // Reads a question while it is typed. msgs ends with the user's draft; what the prompt will surely start with
    // (the conversation and the draft up to its last complete word) goes into the cache now, so that sending it
    // only leaves its last words to read. speculate: the draft is complete as it is (a pause in the typing), so
    // the whole prompt is read as the chat will build it, up to the logits of the answer's first token; more
    // typing goes back to a snapshot before that part. Gives way at its next step when a chat arrives.
    bool prefill(const std::vector<ChatMessage> & msgs, const GenParams & gp, bool speculate,
                 nlohmann::json & out, std::string & err);
    void stop() { stop_flag_.store(true); }
    void reset_conversation();

    nlohmann::json bench_threads(std::string & err);
    nlohmann::json status();

    // expert store (stream mode)
    bool stream_active() const { return store_ptr_.load() != nullptr; }
    bool stream_configure(const StreamConfig & cfg, std::string & err);
    bool stream_bench(const std::vector<StreamBenchConfig> & configs, int tokens, int warmup,
                      const std::function<bool(const nlohmann::json &)> & on_event, std::string & err);

    // the fidelity test (see FidelityConv). "off" runs first, as the reference; events: plan, mode, turn, progress,
    // mode_done. top_k: how many of the reference's likeliest tokens are kept to compare distributions
    bool fidelity(const std::vector<FidelityConv> & convs, const std::vector<std::string> & modes, int top_k,
                  const std::function<bool(const nlohmann::json &)> & on_event, std::string & err);

    // the biggest expert cache that fits now: the current one plus the RAM still free (0 when not streaming)
    uint64_t stream_budget_limit();

    ExpertProfiler & profiler() { return profiler_; }
    bool model_info(ModelInfo & mi);

private:
    void load_worker(std::string path, PlanOptions opt);
    bool try_load(const ModelInfo & mi, const Plan & plan, std::string & err);
    void free_model();
    std::vector<int32_t> tokenize(const std::string & text, bool add_special);
    std::string token_piece(int32_t tok);
    // stable: length of the prompt without the "answer directly" suffix (where the next turn leaves this one)
    bool build_prompt(const std::vector<ChatMessage> & msgs, std::string & out, bool think = true, size_t * stable = nullptr);
    // recurrent/hybrid models (Qwen3.5, Mamba...) cannot rewind their state: snapshots of it at the end of each
    // prompt let the next turn continue from there instead of reading the whole conversation again
    int32_t argmax_logits();  // the most likely next token after the last decode
    // kinds of snapshots, in the order they give way when there is no room: one read on speculation (the newest
    // replaces the others), one at the end of a prompt (asking again), one where a question starts (typing over it)
    enum { CKPT_TURN = 0, CKPT_QUESTION = 1, CKPT_SPEC = 2 };
    void save_checkpoint(int kind = CKPT_TURN);
    bool restore_checkpoint(const std::vector<int32_t> & toks, size_t & n_keep);
    void drop_checkpoints(const std::vector<int32_t> * keep_prefix_of);
    // how much of the cache a new prompt keeps (may rebuild toks from the cached tokens, see there)
    size_t match_cache(std::vector<int32_t> & toks);
    // drops the cache after n_keep (back to a snapshot for a recurrent state; n_keep may move back, to 0 at worst)
    void rewind_cache(const std::vector<int32_t> & toks, size_t & n_keep);
    // takes gen_mu_ for a chat: a prefill holding it stops at its next step, anything else means busy
    bool lock_for_chat(std::unique_lock<std::mutex> & lk);
    void configure_store(const PlanOptions & opt);
    int  decode(int32_t * tokens, int n); // llama_decode + recompute when the expert store's fault handler fired
    static bool route_cb(struct ggml_tensor * t, bool ask, void * ud);

    HwState &   hws_;
    std::string profiles_dir_;

    std::mutex model_mu_;        // guards model_/ctx_ lifetime and the fields below
    llama_model *   model_ = nullptr;
    llama_context * ctx_   = nullptr;
    const llama_vocab * vocab_ = nullptr;
    ModelInfo   mi_;
    Plan        plan_;
    PlanOptions opt_;
    std::string chat_template_;
    bool think_switch_ = false; // the template has an enable_thinking switch (Qwen3, Qwen3.5...)
    std::vector<int32_t> think_toks_; // <think> and </think> when each is one token: barred from direct answers
    bool recurrent_    = false; // the model keeps a recurrent state (pure or hybrid)
    struct Checkpoint {
        std::vector<int32_t> tokens; // the tokens the state has seen
        std::vector<uint8_t> data;   // llama_state_seq_get_data_ext(..., PARTIAL_ONLY)
        int  kind = 0;               // CKPT_*
    };
    std::vector<Checkpoint> ckpts_;  // guarded by gen_mu_
    int max_ckpts_ = 2;
    std::atomic<int>      ckpt_count_{0};
    std::atomic<uint64_t> ckpt_bytes_{0};
    std::atomic<uint64_t> ckpt_restored_{0};
    std::string model_desc_;
    int n_ctx_ = 0, n_batch_ = 0, n_ubatch_ = 0, n_threads_ = 0, n_threads_batch_ = 0;
    int attempts_ = 0;

    // kept alive while the model loads
    std::vector<std::string> ot_patterns_;
    std::vector<llama_model_tensor_buft_override> ot_;
    std::vector<ggml_backend_dev_t> devices_;

    std::vector<int32_t> cache_tokens_; // tokens currently in the KV cache (sequence 0)
    std::atomic<int>     ctx_used_{0};
    size_t logits_at_ = 0; // the context holds the logits after this many cached tokens (0: none usable)

    // reading the question while it is typed (guarded by gen_mu_ except the atomics)
    std::atomic<bool> prefilling_{false};
    std::atomic<int>  waiters_{0};          // chats (or a reset) waiting for a prefill to give way
    long long prefill_from_ = -1;           // where the tokens read during the typing start (-1: none)
    double    prefill_sec_  = 0;            // time spent on them
    std::atomic<uint64_t> prefill_runs_{0}, prefill_tokens_{0}, prefill_used_{0};

    std::atomic<int>   state_{EMPTY};
    std::atomic<float> progress_{0.0f};
    std::atomic<bool>  cancel_load_{false};
    std::atomic<bool>  stop_flag_{false};
    std::atomic<bool>  busy_{false};
    std::string        load_error_;
    std::string        load_phase_;      // "slim" while the slim copy of a model is written
    std::string        load_path_;
    std::thread        loader_;
    std::mutex         gen_mu_;
    std::atomic<double> load_sec_{0.0};

    // last generation numbers for the status bar
    std::atomic<double> last_tps_{0.0}, last_pp_tps_{0.0};

    ExpertProfiler profiler_;

    // expert store (stream mode); the raw pointer is read by the graph callback
    std::unique_ptr<ExpertStore> store_;
    std::atomic<ExpertStore *>   store_ptr_{nullptr};
    std::vector<uint8_t>         route_bytes_;  // scratch for the router tensor
    std::vector<int32_t>         route_ids_;
    std::vector<float>           route_hidden_; // hidden state of one token for the gate predictor
    std::vector<float>           route_gate_, route_norm_;
    std::vector<float>           route_probs_;  // router weights of every expert for one token (cache-aware routing)
    bool                         warned_probs_ = false;
    std::vector<uint8_t>         weights_raw_;  // ids at the routing-weights step (rewritten to cache slots)
    std::vector<int32_t>         weights_ids_;

    static bool progress_cb(float p, void * ud);
};

} // namespace rafid
