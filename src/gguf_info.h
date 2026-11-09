// رافد - reads a GGUF model's metadata and sorts every tensor into groups
// (attention, router, shared experts, routed experts, ...) without loading weights.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace rafid {

struct LayerBytes {
    uint64_t attn    = 0;  // attention weights
    uint64_t norm    = 0;
    uint64_t router  = 0;  // expert gate + selection bias
    uint64_t shared  = 0;  // shared experts (read for every token)
    uint64_t dense   = 0;  // dense FFN (non-MoE layers)
    uint64_t experts = 0;  // routed experts (only k of n are read per token)
    uint64_t other   = 0;
    uint64_t nonexp() const { return attn + norm + router + shared + dense + other; }
    uint64_t total()  const { return nonexp() + experts; }
};

// where a tensor's data lives on disk
struct TensorLoc {
    int      file   = 0;   // index into ModelInfo::files
    uint64_t offset = 0;   // absolute byte offset of the data in that file
    uint64_t nbytes = 0;
    int64_t  ne[4]  = { 1, 1, 1, 1 };
    int      type   = 0;   // ggml_type
};

struct ModelInfo {
    std::string path;                 // first file (split 1)
    std::vector<std::string> files;
    uint64_t    file_bytes = 0;

    std::string arch, name, size_label, quant;
    int n_layer = 0;                  // block_count (includes MTP/nextn layers)
    int n_nextn = 0;                  // MTP layers that are not used for normal decoding
    int n_embd = 0, n_head = 0, n_ctx_train = 0, n_vocab = 0;
    int head_dim_k = 0, head_dim_v = 0, kv_lora_rank = 0, n_rot = 0, sliding_window = 0;
    int full_attn_interval = 0;       // hybrid models (Qwen3-Next / 3.5 / 3.6): only every Nth layer keeps a KV cache
    int ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0, ssm_n_group = 0; // linear-attention layers of the hybrids
    std::vector<int> n_head_kv;       // per layer
    int n_expert = 0, n_expert_used = 0, n_expert_shared = 0;
    std::vector<int> moe_layers;      // layers that hold routed experts
    std::vector<LayerBytes> layers;   // [n_layer]
    std::map<std::string, TensorLoc> exps_loc; // routed expert tensors, by tensor name

    uint64_t bytes_total   = 0;       // everything in the file(s)
    uint64_t bytes_unused  = 0;       // MTP layers that normal decoding never loads
    uint64_t bytes_embed   = 0;       // token embeddings (stay in RAM, 1 row read per token)
    uint64_t bytes_output  = 0;       // output head (fully read per token)
    uint64_t bytes_output_dup = 0;    // extra copy of the embeddings used as output head (tied weights)
    uint64_t bytes_global  = 0;       // other non-layer tensors
    uint64_t bytes_experts = 0;
    bool     output_tied   = false;   // output head shares the embedding matrix
    double   params_total  = 0;       // parameter count
    double   params_active = 0;       // parameters used per token (routed experts scaled by k/n)

    bool is_moe() const { return n_expert > 1 && !moe_layers.empty(); }
    bool layer_has_kv(int l) const { return full_attn_interval <= 1 || (l + 1) % full_attn_interval == 0; }
    int  n_layer_used() const { return n_layer - n_nextn; }
    uint64_t kv_bytes_per_token(int elem_bytes = 2) const;  // all used layers
    uint64_t recurrent_state_bytes() const;                  // state of the linear-attention layers (one sequence, f32)
    uint64_t active_bytes_per_token() const;                 // weights read for one generated token
    double   expert_bytes(int layer) const;                  // bytes of ONE expert in a layer
};

bool read_model_info(const std::string & path, ModelInfo & mi, std::string & err);
nlohmann::json model_to_json(const ModelInfo & mi);
std::string profile_key(const ModelInfo & mi); // stable file-name key for the expert fingerprint

} // namespace rafid
