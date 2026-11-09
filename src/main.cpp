// رافد (Rafid) - home AI platform: runs large MoE models across CPU + GPU and learns
// which experts you actually use.
//
//   rafid                      start the local server and open the web UI
//   rafid hw                   print the hardware report (with bandwidth benchmarks)
//   rafid plan  <model.gguf>   show where every part of the model would live + speed estimate
//   rafid analyze <model.gguf> analyse the saved expert fingerprint of a model

#include "analyzer.h"
#include "engine.h"
#include "gguf_info.h"
#include "hw_probe.h"
#include "planner.h"
#include "server.h"
#include "util.h"

#include "llama.h"
#include "ggml-backend.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using namespace rafid;
using json = nlohmann::json;

static std::atomic<bool> g_quit{false};

static void on_signal(int) {
    g_quit.store(true);
}

static void llama_log_cb(enum ggml_log_level level, const char * text, void *) {
    logs().add((int) level, text);
}

static std::vector<std::string> utf8_args(int argc, char ** argv) {
#ifdef _WIN32
    (void) argc; (void) argv;
    std::vector<std::string> out;
    int n = 0;
    LPWSTR * w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (w) {
        for (int i = 0; i < n; i++) out.push_back(wide_to_utf8(w[i]));
        LocalFree(w);
    }
    return out;
#else
    return std::vector<std::string>(argv, argv + argc);
#endif
}

static void usage() {
    printf(
        "Rafid %s (llama.cpp %s) - home AI platform for CPU + GPU\n\n"
        "usage:\n"
        "  rafid [options]              start the server and open the web interface\n"
        "  rafid hw                     hardware report with memory bandwidth benchmarks\n"
        "  rafid plan <model.gguf>      placement plan and speed estimate\n"
        "  rafid analyze <model.gguf>   analyse the saved expert fingerprint\n\n"
        "options:\n"
        "  --models-dir DIR    folder with .gguf models (default: <exe>/models)\n"
        "  --profiles-dir DIR  folder for expert fingerprints (default: <exe>/profiles)\n"
        "  --model PATH        load this model at start\n"
        "  --ctx N             context size (default 8192)\n"
        "  --cpu-moe N         keep the experts of the first N MoE layers in RAM\n"
        "  --ngl N             number of layers on the GPU (manual)\n"
        "  --moe-cache         experts in RAM + GPU cache of hot experts (experimental)\n"
        "  --threads N         generation threads\n"
        "  --host H --port P   server address (default 127.0.0.1:8080)\n"
        "  --web DIR           serve index.html from DIR (UI development)\n"
        "  --no-browser        do not open the browser\n"
        "  --no-bench          skip the bandwidth benchmarks\n"
        "  --verbose           print all llama.cpp logs\n",
        RAFID_VERSION, RAFID_LLAMA_TAG);
}

static void print_hw(const HwInfo & hw) {
    std::string isa;
    if (hw.avx2)     isa += " AVX2";
    if (hw.avx_vnni) isa += " AVX-VNNI";
    if (hw.avx512)   isa += " AVX-512";
    if (hw.fma)      isa += " FMA";
    printf("CPU : %s - %d cores", hw.cpu_name.c_str(), hw.physical_cores);
    if (hw.e_cores > 0) printf(" (%dP + %dE)", hw.p_cores, hw.e_cores);
    printf(", %d threads,%s\n", hw.logical_cpus, isa.c_str());
    printf("RAM : %s (free %s)", human_bytes(hw.ram_total).c_str(), human_bytes(hw.ram_avail).c_str());
    if (hw.ram_bw_gbps > 0) printf(", measured %.0f GB/s", hw.ram_bw_gbps);
    printf("\n");
    if (hw.gpus.empty()) printf("GPU : none (CPU only build or no GPU found)\n");
    for (const auto & g : hw.gpus) {
        printf("GPU : %s [%s]%s - %s (free %s)", g.description.c_str(), g.name.c_str(), g.integrated ? " integrated" : "",
               human_bytes(g.vram_total).c_str(), human_bytes(g.vram_free).c_str());
        if (g.bw_gbps > 0)  printf(", %.0f GB/s", g.bw_gbps);
        if (g.h2d_gbps > 0) printf(", upload %.1f GB/s", g.h2d_gbps);
        printf("\n");
    }
}

static void print_plan(const ModelInfo & mi, const Plan & p) {
    printf("\nmodel : %s [%s, %s] %s in %zu file(s)\n", mi.name.c_str(), mi.arch.c_str(), mi.quant.c_str(),
           human_bytes(mi.file_bytes).c_str(), mi.files.size());
    printf("        %.1fB params, %.1fB active per token, %d layers", mi.params_total / 1e9, mi.params_active / 1e9, mi.n_layer_used());
    if (mi.is_moe()) printf(", %d experts (%d used) in %zu MoE layers", mi.n_expert, mi.n_expert_used, mi.moe_layers.size());
    printf("\nplan  : %s, ctx %d, threads %d/%d\n", p.mode.c_str(), p.n_ctx, p.n_threads, p.n_threads_batch);
    printf("VRAM  : weights %s + KV %s + compute %s + cache %s  (budget %s of %s free)\n",
           human_bytes(p.gpu_weights).c_str(), human_bytes(p.gpu_kv).c_str(), human_bytes(p.gpu_compute).c_str(),
           human_bytes(p.gpu_cache).c_str(), human_bytes(p.gpu_budget).c_str(), human_bytes(p.vram_free).c_str());
    printf("RAM   : weights %s + KV %s  (budget %s)%s\n", human_bytes(p.cpu_weights).c_str(), human_bytes(p.cpu_kv).c_str(),
           human_bytes(p.ram_budget).c_str(), p.disk_overflow ? (" -> " + human_bytes(p.disk_overflow) + " paged from disk").c_str() : "");
    if (mi.is_moe()) printf("experts in RAM: %d of %d MoE layers\n", p.n_cpu_moe, p.n_moe_layers);
    if (!p.override_pattern.empty()) printf("override: %s=CPU\n", p.override_pattern.c_str());
    printf("speed : about %.1f - %.1f tokens/s\n", p.est_tps_low, p.est_tps_high);
    for (const auto & w : p.warnings) printf("note  : %s\n", w.c_str());
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const std::vector<std::string> args = utf8_args(argc, argv);

    std::string cmd = "serve", target;
    std::string models_dir, profiles_dir, preload, web_dir, host = "127.0.0.1";
    int port = 8080;
    bool open_browser = true, bench = true, verbose = false;
    PlanOptions popt;

    for (size_t i = 1; i < args.size(); i++) {
        const std::string & a = args[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= args.size()) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return args[++i];
        };
        auto next_int = [&](const char * name) -> int {
            const std::string v = next(name);
            try { return std::stoi(v); } catch (...) { fprintf(stderr, "bad number for %s: %s\n", name, v.c_str()); exit(1); }
        };
        if (a == "-h" || a == "--help")      { usage(); return 0; }
        else if (a == "--version")           { printf("rafid %s (llama.cpp %s)\n", RAFID_VERSION, RAFID_LLAMA_TAG); return 0; }
        else if (a == "--models-dir")        models_dir = expand_user(next("--models-dir"));
        else if (a == "--profiles-dir")      profiles_dir = expand_user(next("--profiles-dir"));
        else if (a == "--model" || a == "-m") preload = expand_user(next("--model"));
        else if (a == "--ctx" || a == "-c")  popt.n_ctx = next_int("--ctx");
        else if (a == "--cpu-moe")           popt.n_cpu_moe = next_int("--cpu-moe");
        else if (a == "--ngl")               popt.n_gpu_layers = next_int("--ngl");
        else if (a == "--moe-cache")         popt.moe_cache = true;
        else if (a == "--threads" || a == "-t") popt.n_threads = next_int("--threads");
        else if (a == "--host")              host = next("--host");
        else if (a == "--port")              port = next_int("--port");
        else if (a == "--web")               web_dir = next("--web");
        else if (a == "--no-browser")        open_browser = false;
        else if (a == "--no-bench")          bench = false;
        else if (a == "--verbose" || a == "-v") verbose = true;
        else if (a.size() > 1 && a[0] == '-') { fprintf(stderr, "unknown option: %s\n\n", a.c_str()); usage(); return 1; }
        else if (cmd == "serve" && (a == "hw" || a == "plan" || a == "analyze" || a == "serve")) cmd = a;
        else if (target.empty())             target = a;
        else { fprintf(stderr, "unexpected argument: %s\n", a.c_str()); return 1; }
    }

    // the Windows package keeps rafid.exe next to the llama.cpp DLLs in bin\ ; the data folders live one level up
    std::string base = exe_dir();
    if (file_name(base) == "bin") base = parent_dir(base);
    if (models_dir.empty())   models_dir   = path_join(base, "models");
    if (profiles_dir.empty()) profiles_dir = path_join(base, "profiles");

    logs().set_verbose(verbose);
    llama_log_set(llama_log_cb, nullptr);
    llama_backend_init(); // also loads the dynamic backends (ggml-cuda.dll, ggml-cpu-*.dll) next to the executable
    if (ggml_backend_reg_count() == 0) ggml_backend_load_all();

    printf("Rafid %s (llama.cpp %s)\n", RAFID_VERSION, RAFID_LLAMA_TAG);
    {
        std::string names;
        bool have_cuda = false;
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            const std::string n = ggml_backend_reg_name(ggml_backend_reg_get(i));
            names += (names.empty() ? "" : ", ") + n;
            if (n == "CUDA") have_cuda = true;
        }
        printf("backends : %s\n", names.empty() ? "none" : names.c_str());
        if (names.empty()) {
            fprintf(stderr, "error: no compute backend was found. Keep rafid next to the ggml-*.dll files.\n");
            return 1;
        }
#ifdef _WIN32
        if (!have_cuda && file_exists(path_join(exe_dir(), "ggml-cuda.dll"))) {
            RWARN("ggml-cuda.dll is present but CUDA did not start: update the NVIDIA driver (580 or newer) "
                  "and check that cublas64_13.dll and cublasLt64_13.dll are next to rafid.exe");
        }
#else
        (void) have_cuda;
#endif
    }
    HwState hws;
    hws.hw = probe_hardware();
    if (bench) {
        printf("measuring memory speed (RAM, VRAM, PCIe)...\n");
        fflush(stdout);
        run_benchmarks(hws.hw);
    }
    if (load_disk_measurement(path_join(profiles_dir, "hardware.json"), hws.hw)) {
        RLOG("disk speed from the last measurement: %.1f GB/s", hws.hw.disk_gbps);
    }
    hws.baseline = hws.hw;
    print_hw(hws.hw);

    if (cmd == "hw") {
        printf("\n%s\n", hw_to_json(hws.hw).dump(2).c_str());
        llama_backend_free();
        return 0;
    }

    if (cmd == "plan" || cmd == "analyze") {
        if (target.empty()) { fprintf(stderr, "usage: rafid %s <model.gguf>\n", cmd.c_str()); return 1; }
        ModelInfo mi;
        std::string err;
        if (!read_model_info(target, mi, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        if (cmd == "plan") {
            print_plan(mi, make_plan(hws.hw, mi, popt));
        } else {
            ExpertProfiler prof;
            prof.configure(mi, profiles_dir);
            const json j = analyze_profile(prof.snapshot(), mi, hws.hw);
            printf("%s\n", j.dump(2, ' ', false, json::error_handler_t::replace).c_str());
        }
        llama_backend_free();
        return 0;
    }

    make_dirs(models_dir);
    make_dirs(profiles_dir);
    printf("models   : %s\nprofiles : %s\n", models_dir.c_str(), profiles_dir.c_str());

    int rc = 0;
    {
        Engine engine(hws, profiles_dir);
        if (!preload.empty()) {
            std::string err;
            if (!engine.start_load(preload, popt, err)) fprintf(stderr, "cannot load %s: %s\n", preload.c_str(), err.c_str());
        }

        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        ServerConfig cfg;
        cfg.host         = host;
        cfg.port         = port;
        cfg.models_dir   = models_dir;
        cfg.profiles_dir = profiles_dir;
        cfg.web_dir      = web_dir;
        cfg.open_browser = open_browser;
        rc = run_server(cfg, hws, engine, g_quit);

        printf("shutting down...\n");
        engine.unload();
    }
    llama_backend_free();
    return rc;
}
