#include "gguf_info.h"
#include "util.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <map>

namespace rafid {

enum class Kind { Embed, Output, Attn, Norm, Router, Shared, Experts, Dense, LayerOther, GlobalOther };

static Kind classify(const std::string & name, int & layer) {
    layer = -1;
    if (name.rfind("blk.", 0) == 0) {
        const size_t dot = name.find('.', 4);
        if (dot == std::string::npos) return Kind::GlobalOther;
        layer = std::atoi(name.substr(4, dot - 4).c_str());
        const std::string rest = name.substr(dot + 1);
        if (rest.find("_exps") != std::string::npos)  return Kind::Experts;   // ffn_{gate,up,down,gate_up}_exps
        if (rest.find("_shexp") != std::string::npos) return Kind::Shared;    // ffn_*_shexp, ffn_gate_inp_shexp
        if (rest.rfind("ffn_gate_inp", 0) == 0 || rest.rfind("exp_probs_b", 0) == 0) return Kind::Router;
        if (rest.find("norm") != std::string::npos)   return Kind::Norm;
        if (rest.rfind("attn", 0) == 0)               return Kind::Attn;
        if (rest.rfind("ffn_", 0) == 0)               return Kind::Dense;
        return Kind::LayerOther;                                              // ssm, conv, etc.
    }
    if (name.rfind("token_embd.", 0) == 0) return Kind::Embed;
    if (name.rfind("output", 0) == 0)      return Kind::Output;              // output.weight, output_norm.*
    return Kind::GlobalOther;
}

static int64_t scalar_at(enum gguf_type t, const void * d, size_t i) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return ((const uint8_t  *) d)[i];
        case GGUF_TYPE_INT8:    return ((const int8_t   *) d)[i];
        case GGUF_TYPE_UINT16:  return ((const uint16_t *) d)[i];
        case GGUF_TYPE_INT16:   return ((const int16_t  *) d)[i];
        case GGUF_TYPE_UINT32:  return ((const uint32_t *) d)[i];
        case GGUF_TYPE_INT32:   return ((const int32_t  *) d)[i];
        case GGUF_TYPE_UINT64:  return (int64_t) ((const uint64_t *) d)[i];
        case GGUF_TYPE_INT64:   return ((const int64_t  *) d)[i];
        case GGUF_TYPE_FLOAT32: return (int64_t) ((const float  *) d)[i];
        case GGUF_TYPE_FLOAT64: return (int64_t) ((const double *) d)[i];
        case GGUF_TYPE_BOOL:    return ((const int8_t   *) d)[i];
        default:                return 0;
    }
}

static bool get_ints(gguf_context * g, const std::string & key, std::vector<int64_t> & out) {
    out.clear();
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return false;
    const enum gguf_type t = gguf_get_kv_type(g, id);
    if (t == GGUF_TYPE_ARRAY) {
        const enum gguf_type at = gguf_get_arr_type(g, id);
        if (at == GGUF_TYPE_STRING || at == GGUF_TYPE_ARRAY) return false;
        const size_t n = gguf_get_arr_n(g, id);
        const void * d = gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++) out.push_back(scalar_at(at, d, i));
        return !out.empty();
    }
    if (t == GGUF_TYPE_STRING) return false;
    out.push_back(scalar_at(t, gguf_get_val_data(g, id), 0));
    return true;
}

static int64_t get_int(gguf_context * g, const std::string & key, int64_t def) {
    std::vector<int64_t> v;
    if (!get_ints(g, key, v)) return def;
    return *std::max_element(v.begin(), v.end());
}

static std::string get_str(gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return "";
    return gguf_get_val_str(g, id);
}

bool read_model_info(const std::string & path_in, ModelInfo & mi, std::string & err) {
    mi = ModelInfo();

    std::string prefix;
    int no = 0, count = 0;
    if (parse_split_name(path_in, prefix, no, count)) {
        for (int i = 1; i <= count; i++) mi.files.push_back(make_split_name(prefix, i, count));
    } else {
        mi.files.push_back(path_in);
    }
    for (const auto & f : mi.files) {
        if (!file_exists(f)) { err = "missing file: " + f; return false; }
    }
    mi.path = mi.files[0];

    std::map<int, uint64_t> type_bytes;
    bool have_output_weight = false;
    uint64_t embed_weight_bytes = 0;
    double params_experts = 0, params_embed = 0;

    for (size_t fi = 0; fi < mi.files.size(); fi++) {
        gguf_init_params gp;
        gp.no_alloc = true;
        gp.ctx      = nullptr;
        gguf_context * g = gguf_init_from_file(mi.files[fi].c_str(), gp);
        if (!g) { err = "not a valid GGUF file: " + mi.files[fi]; return false; }
        mi.file_bytes += file_size(mi.files[fi]);

        if (fi == 0) {
            mi.arch = get_str(g, "general.architecture");
            if (mi.arch.empty()) { gguf_free(g); err = "GGUF has no general.architecture"; return false; }
            const std::string a = mi.arch + ".";
            mi.name        = get_str(g, "general.name");
            mi.size_label  = get_str(g, "general.size_label");
            mi.n_layer     = (int) get_int(g, a + "block_count", 0);
            mi.n_nextn     = (int) get_int(g, a + "nextn_predict_layers", 0);
            mi.n_embd      = (int) get_int(g, a + "embedding_length", 0);
            mi.n_head      = (int) get_int(g, a + "attention.head_count", 0);
            mi.n_ctx_train = (int) get_int(g, a + "context_length", 0);
            mi.n_vocab     = (int) get_int(g, a + "vocab_size", 0);
            if (mi.n_vocab == 0) {
                const int64_t id = gguf_find_key(g, "tokenizer.ggml.tokens");
                if (id >= 0) mi.n_vocab = (int) gguf_get_arr_n(g, id);
            }
            mi.head_dim_k     = (int) get_int(g, a + "attention.key_length", mi.n_head > 0 ? mi.n_embd / mi.n_head : 0);
            mi.head_dim_v     = (int) get_int(g, a + "attention.value_length", mi.head_dim_k);
            mi.kv_lora_rank   = (int) get_int(g, a + "attention.kv_lora_rank", 0);
            mi.n_rot          = (int) get_int(g, a + "rope.dimension_count", mi.head_dim_k);
            mi.sliding_window = (int) get_int(g, a + "attention.sliding_window", 0);
            // linear-attention hybrids keep a KV cache only in every Nth layer (llama.cpp defaults to 4)
            const bool hybrid = mi.arch == "qwen3next" || mi.arch == "qwen35" || mi.arch == "qwen35moe";
            mi.full_attn_interval = (int) get_int(g, a + "full_attention_interval", hybrid ? 4 : 0);
            mi.ssm_d_conv  = (int) get_int(g, a + "ssm.conv_kernel", 0);
            mi.ssm_d_inner = (int) get_int(g, a + "ssm.inner_size", 0);
            mi.ssm_d_state = (int) get_int(g, a + "ssm.state_size", 0);
            mi.ssm_n_group = (int) get_int(g, a + "ssm.group_count", 0);
            mi.n_expert        = (int) get_int(g, a + "expert_count", 0);
            mi.n_expert_used   = (int) get_int(g, a + "expert_used_count", 0);
            mi.n_expert_shared = (int) get_int(g, a + "expert_shared_count", 0);

            std::vector<int64_t> kvh;
            if (!get_ints(g, a + "attention.head_count_kv", kvh)) kvh.push_back(mi.n_head);
            mi.n_head_kv.assign((size_t) std::max(mi.n_layer, 0), 0);
            for (int l = 0; l < mi.n_layer; l++) {
                mi.n_head_kv[(size_t) l] = (int) (kvh.size() == 1 ? kvh[0] : (l < (int) kvh.size() ? kvh[(size_t) l] : 0));
            }
            mi.layers.assign((size_t) std::max(mi.n_layer, 0), LayerBytes());
        }

        const uint64_t data_off = gguf_get_data_offset(g);
        const int64_t n_t = gguf_get_n_tensors(g);
        for (int64_t ti = 0; ti < n_t; ti++) {
            const std::string name = gguf_get_tensor_name(g, ti);
            const uint64_t size    = gguf_get_tensor_size(g, ti);
            const int      type    = (int) gguf_get_tensor_type(g, ti);
            const int64_t * ne     = gguf_get_tensor_ne(g, ti);
            double n_el = 1;
            for (int d = 0; d < GGML_MAX_DIMS; d++) n_el *= (double) ne[d];

            mi.bytes_total  += size;
            mi.params_total += n_el;

            int layer = -1;
            const Kind k = classify(name, layer);
            if (layer >= (int) mi.layers.size()) mi.layers.resize((size_t) layer + 1);
            LayerBytes * lb = layer >= 0 ? &mi.layers[(size_t) layer] : nullptr;

            switch (k) {
                case Kind::Embed:
                    mi.bytes_embed += size;
                    if (name == "token_embd.weight") { embed_weight_bytes = size; params_embed = n_el; }
                    break;
                case Kind::Output:
                    mi.bytes_output += size;
                    if (name == "output.weight") have_output_weight = true;
                    break;
                case Kind::Attn:       lb->attn   += size; type_bytes[type] += size; break;
                case Kind::Norm:       lb->norm   += size; break;
                case Kind::Router:     lb->router += size; break;
                case Kind::Shared:     lb->shared += size; type_bytes[type] += size; break;
                case Kind::Dense:      lb->dense  += size; type_bytes[type] += size; break;
                case Kind::LayerOther: lb->other  += size; break;
                case Kind::Experts: {
                    lb->experts      += size;
                    mi.bytes_experts += size;
                    params_experts   += n_el;
                    type_bytes[type] += size;
                    TensorLoc loc;
                    loc.file   = (int) fi;
                    loc.offset = data_off + gguf_get_tensor_offset(g, ti);
                    loc.nbytes = size;
                    loc.type   = type;
                    for (int d = 0; d < 4; d++) loc.ne[d] = ne[d];
                    mi.exps_loc[name] = loc;
                    if (std::find(mi.moe_layers.begin(), mi.moe_layers.end(), layer) == mi.moe_layers.end()) {
                        mi.moe_layers.push_back(layer);
                    }
                    break;
                }
                case Kind::GlobalOther: mi.bytes_global += size; break;
            }
        }
        gguf_free(g);
    }

    if (mi.n_layer == 0) mi.n_layer = (int) mi.layers.size();
    if (mi.n_nextn < 0 || mi.n_nextn >= mi.n_layer) mi.n_nextn = 0;
    std::sort(mi.moe_layers.begin(), mi.moe_layers.end());

    // MTP / nextn layers are not loaded for normal decoding
    const int n_used = mi.n_layer_used();
    mi.moe_layers.erase(std::remove_if(mi.moe_layers.begin(), mi.moe_layers.end(), [&](int l) { return l >= n_used; }), mi.moe_layers.end());
    for (int l = n_used; l < (int) mi.layers.size(); l++) mi.bytes_unused += mi.layers[(size_t) l].total();

    if (!have_output_weight && embed_weight_bytes > 0) {
        mi.output_tied      = true;
        mi.bytes_output    += embed_weight_bytes;
        mi.bytes_output_dup = embed_weight_bytes;
    }
    if (mi.n_expert <= 1) mi.moe_layers.clear();
    if (mi.n_expert_used <= 0 && mi.n_expert > 1) mi.n_expert_used = 1;

    uint64_t best = 0;
    for (const auto & kv : type_bytes) {
        if (kv.second > best) { best = kv.second; mi.quant = ggml_type_name((ggml_type) kv.first); }
    }
    if (mi.name.empty()) mi.name = file_name(mi.path);

    const double used_frac = mi.is_moe() ? (double) mi.n_expert_used / (double) mi.n_expert : 1.0;
    mi.params_active = mi.params_total - params_embed - params_experts * (1.0 - used_frac);
    return true;
}

uint64_t ModelInfo::kv_bytes_per_token(int elem_bytes) const {
    uint64_t total = 0;
    for (int l = 0; l < n_layer_used(); l++) {
        if (!layer_has_kv(l)) continue;
        if (kv_lora_rank > 0) {
            total += (uint64_t) (kv_lora_rank + n_rot) * (uint64_t) elem_bytes;
        } else {
            const int kvh = l < (int) n_head_kv.size() ? n_head_kv[(size_t) l] : 0;
            total += (uint64_t) kvh * (uint64_t) (head_dim_k + head_dim_v) * (uint64_t) elem_bytes;
        }
    }
    return total;
}

uint64_t ModelInfo::recurrent_state_bytes() const {
    if (full_attn_interval <= 1 || ssm_d_inner <= 0 || ssm_d_state <= 0) return 0;
    const uint64_t s = (uint64_t) ssm_d_state * (uint64_t) ssm_d_inner;                                  // delta-net state
    const uint64_t r = (uint64_t) std::max(0, ssm_d_conv - 1) * (uint64_t) (ssm_d_inner + 2 * ssm_n_group * ssm_d_state); // conv
    uint64_t n = 0;
    for (int l = 0; l < n_layer_used(); l++) if (!layer_has_kv(l)) n++;
    return n * (s + r) * 4;
}

double ModelInfo::expert_bytes(int layer) const {
    if (n_expert <= 0 || layer < 0 || layer >= (int) layers.size()) return 0;
    return (double) layers[(size_t) layer].experts / (double) n_expert;
}

uint64_t ModelInfo::active_bytes_per_token() const {
    double b = (double) bytes_output;
    for (int l = 0; l < n_layer_used() && l < (int) layers.size(); l++) {
        b += (double) layers[(size_t) l].nonexp();
        if (is_moe()) b += (double) layers[(size_t) l].experts * (double) n_expert_used / (double) n_expert;
        else          b += (double) layers[(size_t) l].experts;
    }
    return (uint64_t) b;
}

std::string profile_key(const ModelInfo & mi) {
    std::string stem = file_name(mi.path);
    std::string prefix;
    int no = 0, count = 0;
    if (parse_split_name(stem, prefix, no, count)) stem = prefix;
    if (stem.size() > 5 && (stem.substr(stem.size() - 5) == ".gguf" || stem.substr(stem.size() - 5) == ".GGUF")) stem.resize(stem.size() - 5);
    return sanitize_name(stem) + fmt("_L%dE%dK%d", mi.n_layer, mi.n_expert, mi.n_expert_used);
}

nlohmann::json model_to_json(const ModelInfo & mi) {
    uint64_t attn = 0, norm = 0, router = 0, shared = 0, dense = 0, other = 0;
    for (int l = 0; l < mi.n_layer_used() && l < (int) mi.layers.size(); l++) {
        const auto & lb = mi.layers[(size_t) l];
        attn += lb.attn; norm += lb.norm; router += lb.router; shared += lb.shared; dense += lb.dense; other += lb.other;
    }
    int kvh_max = 0;
    for (int v : mi.n_head_kv) kvh_max = std::max(kvh_max, v);
    return {
        { "path", mi.path }, { "files", mi.files.size() }, { "file_bytes", mi.file_bytes },
        { "name", mi.name }, { "arch", mi.arch }, { "size_label", mi.size_label }, { "quant", mi.quant },
        { "n_layer", mi.n_layer }, { "n_layer_used", mi.n_layer_used() }, { "n_embd", mi.n_embd },
        { "n_head", mi.n_head }, { "n_head_kv", kvh_max }, { "n_ctx_train", mi.n_ctx_train }, { "n_vocab", mi.n_vocab },
        { "is_moe", mi.is_moe() }, { "n_expert", mi.n_expert }, { "n_expert_used", mi.n_expert_used },
        { "n_expert_shared", mi.n_expert_shared }, { "n_moe_layers", mi.moe_layers.size() },
        { "params_total", mi.params_total }, { "params_active", mi.params_active },
        { "bytes", {
            { "total", mi.bytes_total }, { "unused", mi.bytes_unused }, { "embed", mi.bytes_embed }, { "output", mi.bytes_output },
            { "attn", attn }, { "norm", norm }, { "router", router }, { "shared", shared }, { "dense", dense },
            { "experts", mi.bytes_experts }, { "other", other + mi.bytes_global },
        } },
        { "kv_bytes_per_token", mi.kv_bytes_per_token(2) }, { "recurrent_state_bytes", mi.recurrent_state_bytes() },
        { "active_bytes_per_token", mi.active_bytes_per_token() },
        { "profile_key", profile_key(mi) },
    };
}

} // namespace rafid
