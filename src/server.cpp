#include "server.h"

#include "analyzer.h"
#include "util.h"
#include "web_index.h"

#include "httplib.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace rafid {

using json = nlohmann::json;

namespace {

constexpr uint64_t MiB = 1ull << 20;

template <class T>
T jget(const json & j, const char * key, T def) {
    if (!j.is_object()) return def;
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    try {
        return it->get<T>();
    } catch (...) {
        return def;
    }
}

void send_json(httplib::Response & res, const json & j, int status = 200) {
    res.status = status;
    res.set_header("Cache-Control", "no-store");
    res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
}

void send_error(httplib::Response & res, const std::string & code, int status = 400, const std::string & detail = "") {
    send_json(res, { { "ok", false }, { "error", code }, { "detail", detail } }, status);
}

bool parse_body(const httplib::Request & req, httplib::Response & res, json & out) {
    if (req.body.empty()) { out = json::object(); return true; }
    out = json::parse(req.body, nullptr, false);
    if (out.is_discarded()) { send_error(res, "bad_json"); return false; }
    if (!out.is_object()) out = json::object();
    return true;
}

// prefetch mode from a request: "off" | "history" | "gate", older clients send true / false
std::string prefetch_mode(const json & b, const char * key) {
    if (!b.is_object()) return "gate";
    auto it = b.find(key);
    if (it == b.end() || it->is_null()) return "gate";
    if (it->is_boolean()) return it->get<bool>() ? "gate" : "off";
    if (it->is_string()) return normalize_prefetch_mode(it->get<std::string>());
    return "gate";
}

PlanOptions plan_options_from(const json & b) {
    PlanOptions o;
    o.n_ctx           = jget<int>(b, "n_ctx", 8192);
    o.gpu_slot        = jget<int>(b, "gpu_slot", -1);
    o.n_cpu_moe       = jget<int>(b, "n_cpu_moe", -1);
    o.n_gpu_layers    = jget<int>(b, "n_gpu_layers", -1);
    o.moe_cache       = jget<bool>(b, "moe_cache", false);
    o.moe_cache_bytes = (uint64_t) std::max(0, jget<int>(b, "moe_cache_mb", 0)) * MiB;
    o.cache_hit_hint  = jget<double>(b, "cache_hit_hint", 0.5);
    o.n_threads       = jget<int>(b, "n_threads", -1);
    o.n_threads_batch = jget<int>(b, "n_threads_batch", -1);
    o.n_ctx = std::max(256, std::min(o.n_ctx, 1 << 20));
    o.stream_experts      = jget<bool>(b, "stream", false);
    o.stream_budget_bytes = (uint64_t) (std::max(0.0, jget<double>(b, "stream_budget_mb", 0.0)) * (double) MiB);
    o.stream_policy       = normalize_policy(jget<std::string>(b, "stream_policy", "freq"));
    o.stream_prefetch     = prefetch_mode(b, "stream_prefetch");
    o.stream_swap         = normalize_swap_mode(jget<std::string>(b, "stream_swap", "off"));
    o.stream_hit_hint     = jget<double>(b, "stream_hit_hint", -1.0);
    return o;
}

// the conversation and generation settings of a chat request (the same for /api/prefill)
void chat_request(const json & b, std::vector<ChatMessage> & msgs, GenParams & gp) {
    if (b.contains("messages") && b["messages"].is_array()) {
        for (const auto & m : b["messages"]) {
            ChatMessage cm;
            cm.role    = jget<std::string>(m, "role", "user");
            cm.content = jget<std::string>(m, "content", "");
            if (cm.role != "system" && cm.role != "user" && cm.role != "assistant") cm.role = "user";
            msgs.push_back(cm);
        }
    }
    const json p = b.contains("params") ? b["params"] : json::object();
    gp.temperature    = jget<float>(p, "temperature", gp.temperature);
    gp.top_p          = jget<float>(p, "top_p", gp.top_p);
    gp.min_p          = jget<float>(p, "min_p", gp.min_p);
    gp.top_k          = jget<int>(p, "top_k", gp.top_k);
    gp.repeat_penalty = jget<float>(p, "repeat_penalty", gp.repeat_penalty);
    gp.max_tokens     = std::max(1, std::min(jget<int>(p, "max_tokens", gp.max_tokens), 1 << 20));
    gp.think          = jget<bool>(p, "think", gp.think);
    gp.ignore_eos     = jget<bool>(p, "ignore_eos", false);
    const int seed    = jget<int>(p, "seed", -1);
    if (seed >= 0) gp.seed = (uint32_t) seed;
}

json backend_names() {
    json out = json::array();
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) out.push_back(ggml_backend_reg_name(ggml_backend_reg_get(i)));
    return out;
}

void open_browser(const std::string & url) {
#ifdef _WIN32
    ShellExecuteW(nullptr, L"open", utf8_to_wide(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    const std::string cmd = "open '" + url + "' >/dev/null 2>&1 &";
    if (system(cmd.c_str()) != 0) {}
#else
    if (getenv("DISPLAY") || getenv("WAYLAND_DISPLAY")) {
        const std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
        if (system(cmd.c_str()) != 0) {}
    }
#endif
}

json analysis_for(Engine & engine, HwState & hws, const std::string & path, const std::string & profiles_dir, std::string & err) {
    ModelInfo mi;
    const HwInfo hw = hws.get();
    ModelInfo loaded;
    const bool have_loaded = engine.model_info(loaded);
    if (path.empty() || (have_loaded && loaded.path == path)) {
        if (!have_loaded) { err = "no_model"; return nullptr; }
        return analyze_profile(engine.profiler().snapshot(), loaded, hw);
    }
    // a model that is not loaded: read its saved fingerprint
    if (!read_model_info(path, mi, err)) return nullptr;
    ExpertProfiler tmp;
    tmp.configure(mi, profiles_dir);
    json j = analyze_profile(tmp.snapshot(), mi, hw);
    return j;
}

} // namespace

int run_server(const ServerConfig & cfg, HwState & hws, Engine & engine, std::atomic<bool> & quit) {
    httplib::Server svr;
    std::mutex dir_mu;
    std::string models_dir = cfg.models_dir;

    svr.set_payload_max_length(64 * MiB);
    svr.set_read_timeout(30, 0);
    svr.set_write_timeout(30, 0);
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string what = "unknown";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            what = e.what();
        } catch (...) {}
        RERR("request failed: %s", what.c_str());
        send_error(res, "internal", 500, what);
    });

    // Only the page served by rafid itself may drive the API: other websites open in the same
    // browser must not be able to load models or start generations (CSRF / DNS rebinding).
    const bool loopback = cfg.host == "127.0.0.1" || cfg.host == "localhost" || cfg.host == "::1";
    const std::vector<std::string> allowed_hosts = {
        fmt("127.0.0.1:%d", cfg.port), fmt("localhost:%d", cfg.port), fmt("[::1]:%d", cfg.port),
    };
    svr.set_pre_routing_handler([loopback, allowed_hosts](const httplib::Request & req, httplib::Response & res) {
        if (loopback) {
            const std::string host = req.get_header_value("Host");
            if (std::find(allowed_hosts.begin(), allowed_hosts.end(), host) == allowed_hosts.end()) {
                send_error(res, "forbidden_host", 403);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        if (req.method == "POST") {
            const std::string origin = req.get_header_value("Origin");
            if (!origin.empty()) {
                const std::string o = origin.rfind("http://", 0) == 0 ? origin.substr(7) : origin;
                if (loopback && std::find(allowed_hosts.begin(), allowed_hosts.end(), o) == allowed_hosts.end()) {
                    send_error(res, "forbidden_origin", 403);
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            // the body is not read yet at this point: look at the declared length instead
            const std::string ct  = req.get_header_value("Content-Type");
            const std::string len = req.get_header_value("Content-Length");
            const bool has_body   = req.has_header("Transfer-Encoding") || (!len.empty() && len != "0");
            if (has_body && ct.rfind("application/json", 0) != 0) {
                send_error(res, "json_required", 415);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // ------------------------------------------------------------------ UI
    svr.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        std::string html;
        if (!cfg.web_dir.empty() && read_text_file(path_join(cfg.web_dir, "index.html"), html)) {
            res.set_content(html, "text/html; charset=utf-8");
        } else {
            res.set_content(reinterpret_cast<const char *>(RAFID_INDEX_HTML), RAFID_INDEX_HTML_LEN, "text/html; charset=utf-8");
        }
        res.set_header("Cache-Control", "no-store");
    });

    // ------------------------------------------------------------------ status
    svr.Get("/api/status", [&](const httplib::Request &, httplib::Response & res) {
        std::string dir;
        {
            std::lock_guard<std::mutex> lk(dir_mu);
            dir = models_dir;
        }
        send_json(res, {
            { "app", { { "version", RAFID_VERSION }, { "llama_tag", RAFID_LLAMA_TAG }, { "models_dir", dir },
                       { "profiles_dir", cfg.profiles_dir }, { "backends", backend_names() } } },
            { "hw", hw_to_json(hws.get()) },
            { "engine", engine.status() },
        });
    });

    svr.Get("/api/logs", [&](const httplib::Request &, httplib::Response & res) {
        send_json(res, { { "lines", logs().tail(300) } });
    });

    // ------------------------------------------------------------------ models
    svr.Get("/api/models", [&](const httplib::Request & req, httplib::Response & res) {
        std::string dir;
        {
            std::lock_guard<std::mutex> lk(dir_mu);
            if (req.has_param("dir") && !req.get_param_value("dir").empty()) models_dir = expand_user(req.get_param_value("dir"));
            dir = models_dir;
        }
        json list = json::array();
        for (const auto & e : list_gguf_models(dir)) {
            list.push_back({ { "path", e.path }, { "name", e.name }, { "bytes", e.bytes }, { "n_files", e.n_files } });
        }
        send_json(res, { { "dir", dir }, { "exists", is_directory(dir) }, { "models", list }, { "lmstudio_dir", lmstudio_models_dir() } });
    });

    svr.Post("/api/plan", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        const std::string path = jget<std::string>(b, "path", "");
        if (path.empty()) { send_error(res, "missing_path"); return; }
        ModelInfo mi;
        Plan plan;
        std::string err;
        if (!engine.preview(path, plan_options_from(b), mi, plan, err)) { send_error(res, "bad_model", 400, err); return; }
        send_json(res, { { "ok", true }, { "model", model_to_json(mi) }, { "plan", plan_to_json(plan) } });
    });

    svr.Post("/api/load", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        const std::string path = jget<std::string>(b, "path", "");
        if (path.empty()) { send_error(res, "missing_path"); return; }
        std::string err;
        if (!engine.start_load(path, plan_options_from(b), err)) { send_error(res, err, 409); return; }
        send_json(res, { { "ok", true } });
    });

    svr.Post("/api/unload", [&](const httplib::Request &, httplib::Response & res) {
        engine.unload();
        send_json(res, { { "ok", true } });
    });

    // ------------------------------------------------------------------ chat
    svr.Post("/api/chat", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        if (engine.state() != Engine::READY) { send_error(res, "no_model", 409); return; }
        if (engine.busy())                   { send_error(res, "busy", 409); return; }

        std::vector<ChatMessage> msgs;
        GenParams gp;
        chat_request(b, msgs, gp);
        if (msgs.empty()) { send_error(res, "no_messages"); return; }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [&engine, msgs, gp](size_t, httplib::DataSink & sink) {
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                return sink.write(s.data(), s.size());
            };
            GenStats st;
            std::string err;
            const double t0 = now_sec();
            const bool ok = engine.chat(msgs, gp, [&](const std::string & text) { return send({ { "type", "token" }, { "text", text } }); }, st, err);
            if (!ok) send({ { "type", "error" }, { "message", err } });
            send({ { "type", "done" }, { "first_logits", st.first_logits }, { "stats", {
                { "prompt_tokens", st.prompt_tokens }, { "cached_tokens", st.cached_tokens }, { "shared_tokens", st.shared_tokens },
                { "gen_tokens", st.gen_tokens },
                { "prefilled_tokens", st.prefilled_tokens }, { "prefilled_sec", st.prefilled_sec },
                { "prompt_sec", st.prompt_sec }, { "gen_sec", st.gen_sec }, { "total_sec", now_sec() - t0 },
                { "tps", st.gen_sec > 0 ? st.gen_tokens / st.gen_sec : 0.0 },
                { "pp_tps", st.prompt_sec > 0 ? (st.prompt_tokens - st.cached_tokens) / st.prompt_sec : 0.0 },
                { "dropped_messages", st.dropped_messages }, { "stop_reason", st.stop_reason },
                { "stream", st.stream }, { "stream_prompt", st.stream_prompt },
            } } });
            sink.done();
            return true;
        });
    });

    // reading a question while it is typed: the page sends the draft as it grows (speculate: after a pause, read
    // it to the end as if sent). Expected refusals (busy, no model...) answer 200 with ok=false: the page just
    // tries again later.
    svr.Post("/api/prefill", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        if (engine.state() != Engine::READY) { send_json(res, { { "ok", false }, { "error", "no_model" } }); return; }
        if (engine.busy())                   { send_json(res, { { "ok", false }, { "error", "busy" } }); return; }
        std::vector<ChatMessage> msgs;
        GenParams gp;
        chat_request(b, msgs, gp);
        json out;
        std::string err;
        if (!engine.prefill(msgs, gp, jget<bool>(b, "speculate", false), out, err)) {
            send_json(res, { { "ok", false }, { "error", err } });
            return;
        }
        send_json(res, out);
    });

    // ---- tests: conversations the page plays to the model and grades. Suites are <profiles>/tests/*.json,
    // results go to <profiles>/tests/results/ after every answer (so a run can be continued and read later) ----
    const std::string tests_dir   = path_join(cfg.profiles_dir, "tests");
    const std::string results_dir = path_join(tests_dir, "results");
    auto json_name = [](const std::string & f) {
        return !f.empty() && f.size() < 160 && f.find_first_of("/\\:*?\"<>|") == std::string::npos && f.find("..") == std::string::npos &&
               f.size() > 5 && f.compare(f.size() - 5, 5, ".json") == 0;
    };
    svr.Get("/api/tests", [&](const httplib::Request &, httplib::Response & res) {
        json suites = json::array(), results = json::array();
        for (const std::string & f : list_files(tests_dir, ".json")) {
            std::string text;
            if (!read_text_file(path_join(tests_dir, f), text)) continue;
            try {
                const json j = json::parse(text);
                int turns = 0;
                for (const auto & c : j.value("conversations", json::array())) turns += (int) c.value("turns", json::array()).size();
                suites.push_back({ { "file", f }, { "title", j.value("title", f) }, { "about", j.value("about", "") },
                                   { "author", j.value("author", "") }, { "sha256", sha256_hex(text) }, { "bundled", f == "suite-ar.json" },
                                   { "conversations", j.value("conversations", json::array()).size() }, { "turns", turns } });
            } catch (const std::exception & e) {
                suites.push_back({ { "file", f }, { "error", e.what() } });
            }
        }
        for (const std::string & f : list_files(results_dir, ".json")) {
            std::string text;
            if (!read_text_file(path_join(results_dir, f), text)) continue;
            try {
                const json j = json::parse(text);
                results.push_back({ { "file", f }, { "suite", j.value("suite", "") }, { "updated", j.value("updated", "") },
                                    { "finished", j.value("finished", false) }, { "summary", j.value("summary", json::object()) },
                                    { "kind", j.value("kind", "conversations") }, { "reference", j.value("reference", "") },
                                    { "options", j.value("options", json::object()) } });
            } catch (...) {
            }
        }
        send_json(res, { { "dir", tests_dir }, { "suites", suites }, { "results", results } });
    });
    auto send_file_json = [&](const std::string & path, httplib::Response & res) {
        std::string text;
        if (!read_text_file(path, text)) { send_error(res, "not_found", 404); return; }
        try {
            send_json(res, json::parse(text));
        } catch (const std::exception & e) {
            send_error(res, std::string("bad_json: ") + e.what(), 422);
        }
    };
    // the file as stored, byte for byte (a download keeps its hash)
    auto send_file_raw = [&](const std::string & path, httplib::Response & res) {
        std::string text;
        if (!read_text_file(path, text)) { send_error(res, "not_found", 404); return; }
        res.set_content(text, "application/json; charset=utf-8");
    };
    svr.Get("/api/tests/suite", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string f = req.get_param_value("file");
        if (!json_name(f)) { send_error(res, "bad_name"); return; }
        if (req.has_param("raw")) { send_file_raw(path_join(tests_dir, f), res); return; }
        send_file_json(path_join(tests_dir, f), res);
    });
    // a suite written in the page or brought from a file: saved as given (the bytes of an imported file stay
    // the same, so its hash matches the one its author computed). A suite is conversations of turns, each
    // turn a "user" text and its "checks"
    auto suite_problem = [](const json & j) -> std::string {
        if (!j.is_object()) return "not an object";
        if (!j.contains("conversations") || !j["conversations"].is_array() || j["conversations"].empty()) return "no conversations";
        int n = 0;
        for (const auto & c : j["conversations"]) {
            n++;
            if (!c.is_object() || !c.contains("turns") || !c["turns"].is_array() || c["turns"].empty()) return fmt("conversation %d has no turns", n);
            int t = 0;
            for (const auto & u : c["turns"]) {
                t++;
                if (!u.is_object() || !u.contains("user") || !u["user"].is_string() || u["user"].get<std::string>().empty()) return fmt("conversation %d, turn %d has no \"user\" text", n, t);
                if (u.contains("checks") && !u["checks"].is_array()) return fmt("conversation %d, turn %d: \"checks\" is not a list", n, t);
            }
        }
        return "";
    };
    svr.Post("/api/tests/suite", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        const std::string f = jget<std::string>(b, "file", "");
        if (!json_name(f) || f.rfind("results", 0) == 0) { send_error(res, "bad_name"); return; }
        if (f == "suite-ar.json")                       { send_error(res, "bundled_suite", 409); return; }
        std::string text;
        if (b.contains("text") && b["text"].is_string()) text = b["text"].get<std::string>();
        else if (b.contains("data"))                     text = b["data"].dump(1, ' ', false, json::error_handler_t::replace);
        else { send_error(res, "no_suite"); return; }
        json j;
        try { j = json::parse(text); } catch (const std::exception & e) { send_error(res, std::string("bad_json: ") + e.what(), 422); return; }
        const std::string why = suite_problem(j);
        if (!why.empty()) { send_error(res, "bad_suite: " + why, 422); return; }
        const std::string path = path_join(tests_dir, f);
        if (file_exists(path) && !jget<bool>(b, "overwrite", false)) { send_error(res, "exists", 409); return; }
        make_dirs(tests_dir);
        if (!write_text_file(path, text)) { send_error(res, "write_failed", 500); return; }
        send_json(res, { { "ok", true }, { "file", f }, { "sha256", sha256_hex(text) } });
    });
    svr.Post("/api/tests/suite/delete", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        const std::string f = jget<std::string>(b, "file", "");
        if (!json_name(f)) { send_error(res, "bad_name"); return; }
        if (f == "suite-ar.json") { send_error(res, "bundled_suite", 409); return; }
        std::error_code ec;
        if (!std::filesystem::remove(std::filesystem::u8path(path_join(tests_dir, f)), ec)) { send_error(res, "not_found", 404); return; }
        send_json(res, { { "ok", true } });
    });
    svr.Get("/api/tests/result", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string f = req.get_param_value("file");
        if (!json_name(f)) { send_error(res, "bad_name"); return; }
        if (req.has_param("raw")) { send_file_raw(path_join(results_dir, f), res); return; }
        send_file_json(path_join(results_dir, f), res);
    });
    svr.Post("/api/tests/result", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        const std::string f = jget<std::string>(b, "file", "");
        if (!json_name(f) || !b.contains("data")) { send_error(res, "bad_name"); return; }
        make_dirs(results_dir);
        if (!write_text_file(path_join(results_dir, f), b["data"].dump(1))) { send_error(res, "write_failed", 500); return; }
        send_json(res, { { "ok", true }, { "path", path_join(results_dir, f) } });
    });

    svr.Post("/api/stop", [&](const httplib::Request &, httplib::Response & res) {
        engine.stop();
        send_json(res, { { "ok", true } });
    });

    svr.Post("/api/reset", [&](const httplib::Request &, httplib::Response & res) {
        engine.reset_conversation();
        send_json(res, { { "ok", true } });
    });

    // ------------------------------------------------------------------ expert store (stream mode)
    svr.Post("/api/stream/config", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        // fields left out keep their current values (the tests page changes only the routing)
        StreamConfig c;
        c.budget_bytes = b.contains("budget_mb") ? (uint64_t) (std::max(0.0, jget<double>(b, "budget_mb", 2048.0)) * (double) MiB) : 0;
        c.policy       = b.contains("policy") ? jget<std::string>(b, "policy", "freq") : "";
        c.prefetch     = b.contains("prefetch") ? prefetch_mode(b, "prefetch") : "";
        c.pin_frac     = jget<double>(b, "pin_frac", -1.0);
        c.swap         = b.contains("swap") ? jget<std::string>(b, "swap", "off") : "";
        std::string err;
        if (!engine.stream_configure(c, err)) { send_error(res, err, 409); return; }
        send_json(res, { { "ok", true } });
    });

    svr.Post("/api/stream/bench", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        if (engine.state() != Engine::READY) { send_error(res, "no_model", 409); return; }
        if (!engine.stream_active())         { send_error(res, "not_streaming", 409); return; }
        if (engine.busy())                   { send_error(res, "busy", 409); return; }
        std::vector<StreamBenchConfig> configs;
        if (b.contains("configs") && b["configs"].is_array()) {
            for (const auto & c : b["configs"]) {
                StreamBenchConfig bc;
                bc.budget_bytes = (uint64_t) (std::max(0.0, jget<double>(c, "budget_mb", 2048.0)) * (double) MiB);
                bc.policy       = jget<std::string>(c, "policy", "freq");
                bc.prefetch     = prefetch_mode(c, "prefetch");
                bc.swap         = jget<std::string>(c, "swap", "off");
                configs.push_back(bc);
            }
        }
        if (configs.empty() || configs.size() > 36) { send_error(res, "bad_configs"); return; }
        const int tokens = jget<int>(b, "tokens", 48);
        const int warmup = jget<int>(b, "warmup", 16);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [&engine, configs, tokens, warmup](size_t, httplib::DataSink & sink) {
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                return sink.write(s.data(), s.size());
            };
            std::string err;
            const bool ok = engine.stream_bench(configs, tokens, warmup, [&](const json & ev) { return send(ev); }, err);
            if (!ok && !err.empty()) send({ { "type", "error" }, { "message", err } });
            send({ { "type", "done" } });
            sink.done();
            return true;
        });
    });

    // the fidelity test: answers written with exact routing read back under each routing (see Engine::fidelity)
    svr.Post("/api/tests/fidelity", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        if (engine.state() != Engine::READY) { send_error(res, "no_model", 409); return; }
        if (!engine.stream_active())         { send_error(res, "not_streaming", 409); return; }
        if (engine.busy())                   { send_error(res, "busy", 409); return; }
        std::vector<FidelityConv> convs;
        if (b.contains("convs") && b["convs"].is_array()) {
            for (const auto & c : b["convs"]) {
                FidelityConv fc;
                fc.id     = jget<std::string>(c, "id", "");
                fc.system = jget<std::string>(c, "system", "");
                if (c.contains("turns") && c["turns"].is_array()) {
                    for (const auto & t : c["turns"]) fc.turns.push_back({ jget<std::string>(t, "user", ""), jget<std::string>(t, "answer", "") });
                }
                if (!fc.turns.empty()) convs.push_back(std::move(fc));
            }
        }
        std::vector<std::string> modes;
        if (b.contains("modes") && b["modes"].is_array()) {
            for (const auto & m : b["modes"]) if (m.is_string()) modes.push_back(m.get<std::string>());
        }
        if (convs.empty() || convs.size() > 200) { send_error(res, "bad_convs"); return; }
        const int top_k = jget<int>(b, "top_k", 32);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [&engine, convs, modes, top_k](size_t, httplib::DataSink & sink) {
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                return sink.write(s.data(), s.size());
            };
            std::string err;
            const bool ok = engine.fidelity(convs, modes, top_k, [&](const json & ev) { return send(ev); }, err);
            if (!ok && !err.empty()) send({ { "type", "error" }, { "message", err } });
            send({ { "type", "done" } });
            sink.done();
            return true;
        });
    });

    // ------------------------------------------------------------------ expert fingerprint
    svr.Post("/api/profile", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        if (b.contains("enabled")) engine.profiler().set_enabled(jget<bool>(b, "enabled", true));
        send_json(res, engine.profiler().summary());
    });

    svr.Post("/api/profile/save", [&](const httplib::Request &, httplib::Response & res) {
        std::string err;
        const bool ok = engine.profiler().save(err);
        send_json(res, { { "ok", ok }, { "error", err } }, ok ? 200 : 400);
    });

    svr.Post("/api/profile/reset", [&](const httplib::Request &, httplib::Response & res) {
        engine.profiler().reset();
        send_json(res, { { "ok", true } });
    });

    svr.Get("/api/profile/analysis", [&](const httplib::Request & req, httplib::Response & res) {
        std::string err;
        const std::string path = req.has_param("path") ? req.get_param_value("path") : "";
        const double t0 = now_sec();
        json j = analysis_for(engine, hws, path, cfg.profiles_dir, err);
        if (j.is_null()) { send_error(res, err.empty() ? "no_data" : err, 400); return; }
        j["analysis_sec"] = now_sec() - t0;
        send_json(res, j);
    });

    // ------------------------------------------------------------------ benchmarks
    svr.Post("/api/bench/disk", [&](const httplib::Request & req, httplib::Response & res) {
        json b;
        if (!parse_body(req, res, b)) return;
        const std::string path = jget<std::string>(b, "path", "");
        if (path.empty() || !file_exists(path)) { send_error(res, "missing_path"); return; }
        std::string err;
        const double gbps = measure_disk_read(path, 2ull << 30, err);
        if (gbps <= 0) { send_error(res, "disk_bench_failed", 400, err); return; }
        // the expert store's pattern: scattered 256 KB reads, alone, in parallel, and in parallel while the file is mapped
        std::string e2;
        const double r1  = measure_disk_random(path, 256 << 10, 1, 0.8, false, e2);
        const double r16 = measure_disk_random(path, 256 << 10, 16, 0.8, false, e2);
        const double r16m = measure_disk_random(path, 256 << 10, 16, 0.8, true, e2);
        RLOG("disk: sequential %.2f GB/s, scattered 256 KB: 1 thread %.2f, 16 threads %.2f, 16 threads + mapped %.2f GB/s",
             gbps, r1, r16, r16m);
        {
            std::lock_guard<std::mutex> lk(hws.mu);
            for (HwInfo * h : { &hws.hw, &hws.baseline }) {
                h->disk_gbps = gbps;
                h->disk_path = path;
                h->disk_rand1 = r1;
                h->disk_rand16 = r16;
                h->disk_rand16_mapped = r16m;
            }
        }
        save_disk_measurement(path_join(cfg.profiles_dir, "hardware.json"), hws.get());
        send_json(res, { { "ok", true }, { "gbps", gbps }, { "rand1", r1 }, { "rand16", r16 }, { "rand16_mapped", r16m } });
    });

    svr.Post("/api/bench/memory", [&](const httplib::Request &, httplib::Response & res) {
        if (engine.state() == Engine::READY || engine.state() == Engine::LOADING) { send_error(res, "unload_first", 409); return; }
        HwInfo hw = hws.get();
        run_benchmarks(hw);
        {
            std::lock_guard<std::mutex> lk(hws.mu);
            const HwInfo prev = hws.hw;
            hws.hw = hw;
            hws.hw.disk_gbps = prev.disk_gbps;
            hws.hw.disk_path = prev.disk_path;
            hws.hw.disk_rand1 = prev.disk_rand1;
            hws.hw.disk_rand16 = prev.disk_rand16;
            hws.hw.disk_rand16_mapped = prev.disk_rand16_mapped;
            hws.baseline = hws.hw;
        }
        send_json(res, { { "ok", true }, { "hw", hw_to_json(hws.get()) } });
    });

    svr.Post("/api/bench/threads", [&](const httplib::Request &, httplib::Response & res) {
        std::string err;
        json j = engine.bench_threads(err);
        if (j.is_null()) { send_error(res, err, 409); return; }
        send_json(res, j);
    });

    // ------------------------------------------------------------------ run
    if (!svr.bind_to_port(cfg.host, cfg.port)) {
        RERR("cannot listen on %s:%d (is another copy of rafid running? try --port)", cfg.host.c_str(), cfg.port);
        return 1;
    }
    const std::string url = fmt("http://%s:%d/", cfg.host == "0.0.0.0" ? "127.0.0.1" : cfg.host.c_str(), cfg.port);
    RLOG("rafid is ready: %s", url.c_str());
    if (cfg.open_browser) std::thread([url] { open_browser(url); }).detach();

    std::thread watcher([&] {
        while (!quit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.stop();
        svr.stop();
    });
    svr.listen_after_bind();
    quit.store(true);
    watcher.join();
    return 0;
}

} // namespace rafid
