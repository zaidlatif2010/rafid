#include "hw_probe.h"
#include "util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <set>
#include <thread>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define RAFID_X86 1
#else
#define RAFID_X86 0
#endif

#if defined(_WIN32)
#include <windows.h>
#if RAFID_X86
#include <intrin.h>
#include <immintrin.h>
#endif
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if RAFID_X86
#include <cpuid.h>
#endif
#endif

namespace rafid {

int HwInfo::best_gpu() const {
    for (size_t i = 0; i < gpus.size(); i++) {
        if (!gpus[i].integrated) return (int) i;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// CPU identification
// -----------------------------------------------------------------------------
#if RAFID_X86
static void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
#ifdef _MSC_VER
    int t[4];
    __cpuidex(t, (int) leaf, (int) sub);
    for (int i = 0; i < 4; i++) r[i] = (uint32_t) t[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static uint64_t xgetbv0() {
#ifdef _MSC_VER
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t) edx << 32) | eax;
#endif
}

static void probe_cpuid(HwInfo & hw) {
    uint32_t r[4];
    cpuid(0x80000000u, 0, r);
    if (r[0] >= 0x80000004u) {
        char brand[49] = {0};
        for (uint32_t i = 0; i < 3; i++) {
            cpuid(0x80000002u + i, 0, r);
            memcpy(brand + i * 16, r, 16);
        }
        std::string b = brand;
        b.erase(0, b.find_first_not_of(' '));
        while (!b.empty() && (b.back() == ' ' || b.back() == '\0')) b.pop_back();
        hw.cpu_name = b;
    }

    cpuid(0, 0, r);
    const uint32_t max_leaf = r[0];
    cpuid(1, 0, r);
    const uint32_t ecx = r[2];
    const bool osxsave = (ecx >> 27) & 1;
    const uint64_t xcr0 = osxsave ? xgetbv0() : 0;
    const bool os_avx    = (xcr0 & 0x6) == 0x6;
    const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;

    hw.fma  = ((ecx >> 12) & 1) && os_avx;
    hw.f16c = ((ecx >> 29) & 1) && os_avx;
    if (max_leaf >= 7) {
        cpuid(7, 0, r);
        hw.avx2   = ((r[1] >> 5) & 1) && os_avx;
        hw.avx512 = ((r[1] >> 16) & 1) && os_avx512;
        cpuid(7, 1, r);
        hw.avx_vnni = ((r[0] >> 4) & 1) && os_avx;
    }
}
#endif

#if defined(_WIN32)
static void probe_cores_win(HwInfo & hw) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) return;

    std::vector<int> classes;
    int logical = 0;
    for (DWORD off = 0; off < len;) {
        auto * info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (info->Relationship == RelationProcessorCore) {
            // EfficiencyClass is the byte right after Flags (older MinGW headers call it Reserved[0])
            const BYTE * flags = &info->Processor.Flags;
            classes.push_back(flags[1]);
            for (WORD g = 0; g < info->Processor.GroupCount; g++) {
                KAFFINITY m = info->Processor.GroupMask[g].Mask;
                while (m) { logical += (int) (m & 1); m >>= 1; }
            }
        }
        off += info->Size;
    }
    if (classes.empty()) return;
    hw.physical_cores = (int) classes.size();
    hw.logical_cpus   = logical;
    const int maxc = *std::max_element(classes.begin(), classes.end());
    if (maxc > 0) {
        hw.p_cores = (int) std::count(classes.begin(), classes.end(), maxc);
        hw.e_cores = hw.physical_cores - hw.p_cores;
    } else {
        hw.p_cores = hw.physical_cores;
        hw.e_cores = 0;
    }
}
#else
static std::string read_line(const std::string & path) {
    std::string s;
    if (!read_text_file(path, s)) return "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

static std::vector<int> parse_cpu_list(const std::string & s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string part = s.substr(i, j - i);
        const size_t dash = part.find('-');
        try {
            if (dash == std::string::npos) {
                if (!part.empty()) out.push_back(std::stoi(part));
            } else {
                const int a = std::stoi(part.substr(0, dash)), b = std::stoi(part.substr(dash + 1));
                for (int k = a; k <= b; k++) out.push_back(k);
            }
        } catch (...) {}
        i = j + 1;
    }
    return out;
}

static void probe_cores_linux(HwInfo & hw) {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    hw.logical_cpus = n > 0 ? (int) n : (int) std::thread::hardware_concurrency();

    std::map<int, std::pair<std::string, std::string>> core_of; // cpu -> (package, core)
    std::set<std::pair<std::string, std::string>> cores;
    for (int c = 0; c < hw.logical_cpus; c++) {
        const std::string base = fmt("/sys/devices/system/cpu/cpu%d/topology/", c);
        const std::string pkg  = read_line(base + "physical_package_id");
        const std::string core = read_line(base + "core_id");
        if (core.empty()) continue;
        core_of[c] = { pkg, core };
        cores.insert({ pkg, core });
    }
    hw.physical_cores = cores.empty() ? hw.logical_cpus : (int) cores.size();

    auto count_cores = [&](const std::string & list_path) {
        std::set<std::pair<std::string, std::string>> s;
        for (int c : parse_cpu_list(read_line(list_path))) {
            auto it = core_of.find(c);
            if (it != core_of.end()) s.insert(it->second);
        }
        return (int) s.size();
    };
    hw.p_cores = count_cores("/sys/devices/cpu_core/cpus");
    hw.e_cores = count_cores("/sys/devices/cpu_atom/cpus");
    if (hw.p_cores == 0 && hw.e_cores == 0) hw.p_cores = hw.physical_cores;

    if (hw.cpu_name.empty()) {
        std::string info;
        if (read_text_file("/proc/cpuinfo", info)) {
            for (const char * key : { "model name", "Model", "Hardware" }) {
                const size_t p = info.find(key);
                if (p == std::string::npos) continue;
                const size_t colon = info.find(':', p);
                const size_t eol   = info.find('\n', p);
                if (colon != std::string::npos && colon < eol) {
                    std::string v = info.substr(colon + 1, eol - colon - 1);
                    v.erase(0, v.find_first_not_of(' '));
                    hw.cpu_name = v;
                    break;
                }
            }
        }
    }
}
#endif

static void read_ram(HwInfo & hw) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        hw.ram_total = ms.ullTotalPhys;
        hw.ram_avail = ms.ullAvailPhys;
    }
#else
    std::string info;
    if (read_text_file("/proc/meminfo", info)) {
        auto get = [&](const char * key) -> uint64_t {
            const size_t p = info.find(key);
            if (p == std::string::npos) return 0;
            return (uint64_t) std::strtoull(info.c_str() + p + strlen(key), nullptr, 10) * 1024ull;
        };
        hw.ram_total = get("MemTotal:");
        hw.ram_avail = get("MemAvailable:");
    }
#endif
}

static void read_gpus(HwInfo & hw) {
    std::vector<GpuInfo> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        GpuInfo g;
        g.dev_index   = (int) i;
        g.name        = ggml_backend_dev_name(dev);
        g.description = ggml_backend_dev_description(dev);
        g.integrated  = type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        g.vram_free  = free_b;
        g.vram_total = total_b;
        // keep earlier measurements when refreshing
        for (const auto & old : hw.gpus) {
            if (old.dev_index == g.dev_index) { g.bw_gbps = old.bw_gbps; g.h2d_gbps = old.h2d_gbps; }
        }
        gpus.push_back(g);
    }
    std::stable_sort(gpus.begin(), gpus.end(), [](const GpuInfo & a, const GpuInfo & b) {
        if (a.integrated != b.integrated) return !a.integrated;
        return a.vram_total > b.vram_total;
    });
    hw.gpus = gpus;
}

HwInfo probe_hardware() {
    HwInfo hw;
#if defined(_WIN32)
    hw.os = "Windows";
#elif defined(__APPLE__)
    hw.os = "macOS";
#else
    hw.os = "Linux";
#endif
#if RAFID_X86
    probe_cpuid(hw);
#endif
#if defined(_WIN32)
    probe_cores_win(hw);
#else
    probe_cores_linux(hw);
#endif
    if (hw.logical_cpus <= 0)   hw.logical_cpus   = (int) std::max(1u, std::thread::hardware_concurrency());
    if (hw.physical_cores <= 0) hw.physical_cores = hw.logical_cpus;
    if (hw.p_cores <= 0 && hw.e_cores <= 0) hw.p_cores = hw.physical_cores;
    if (hw.cpu_name.empty()) hw.cpu_name = "Unknown CPU";
    read_ram(hw);
    read_gpus(hw);
    return hw;
}

void refresh_memory(HwInfo & hw) {
    read_ram(hw);
    read_gpus(hw);
}

void refresh_ram(HwInfo & hw) {
    read_ram(hw);
}

// -----------------------------------------------------------------------------
// Benchmarks
// -----------------------------------------------------------------------------
static double bench_ram(int nthreads, size_t bytes) {
    const size_t n = bytes / sizeof(uint64_t);
    std::vector<uint64_t> buf;
    try {
        buf.assign(n, 1);
    } catch (...) {
        return 0.0;
    }
    std::atomic<uint64_t> sink{0};
    double best = 0.0;
    for (int rep = 0; rep < 6; rep++) {
        std::atomic<int>  ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> th;
        th.reserve((size_t) nthreads);
        for (int t = 0; t < nthreads; t++) {
            th.emplace_back([&, t] {
                const size_t b = n * (size_t) t / (size_t) nthreads;
                const size_t e = n * (size_t) (t + 1) / (size_t) nthreads;
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {}
                const uint64_t * p = buf.data();
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                size_t i = b;
                for (; i + 8 <= e; i += 8) {
                    s0 += p[i + 0] + p[i + 4];
                    s1 += p[i + 1] + p[i + 5];
                    s2 += p[i + 2] + p[i + 6];
                    s3 += p[i + 3] + p[i + 7];
                }
                for (; i < e; i++) s0 += p[i];
                sink.fetch_add(s0 + s1 + s2 + s3, std::memory_order_relaxed);
            });
        }
        while (ready.load() < nthreads) std::this_thread::yield();
        const double t0 = now_sec();
        go.store(true, std::memory_order_release);
        for (auto & x : th) x.join();
        const double dt = now_sec() - t0;
        if (dt > 0) best = std::max(best, (double) (n * sizeof(uint64_t)) / dt / 1e9);
    }
    if (sink.load() == 42) fprintf(stderr, " "); // keep the reads alive
    return best;
}

static void bench_gpu(GpuInfo & g) {
    ggml_backend_dev_t dev = ggml_backend_dev_get((size_t) g.dev_index);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) return;

    int64_t n = 64ll * 1024 * 1024; // 256 MB per tensor
    if (g.vram_free > 0 && g.vram_free < (uint64_t) (n * 4 * 2) + (768ull << 20)) n = 16ll * 1024 * 1024;

    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_cpy(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf) {
        ggml_backend_buffer_clear(buf, 0);
        if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
            double best = 0;
            for (int i = 0; i < 8; i++) {
                const double t0 = now_sec();
                ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                const double dt = now_sec() - t0;
                if (dt > 0) best = std::max(best, 2.0 * (double) n * 4.0 / dt / 1e9);
            }
            g.bw_gbps = best;
        }

        const size_t hb = (size_t) n * 4;
        std::vector<uint8_t> host(hb, 1);
        double best = 0;
        for (int i = 0; i < 4; i++) {
            const double t0 = now_sec();
            ggml_backend_tensor_set(a, host.data(), 0, hb);
            ggml_backend_synchronize(be);
            const double dt = now_sec() - t0;
            if (dt > 0) best = std::max(best, (double) hb / dt / 1e9);
        }
        g.h2d_gbps = best;
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(be);
}

void run_benchmarks(HwInfo & hw) {
    refresh_memory(hw);
    uint64_t bytes = 1ull << 30;
    if (hw.ram_avail > 0) bytes = std::min<uint64_t>(bytes, hw.ram_avail / 4);
    bytes = std::max<uint64_t>(bytes, 128ull << 20);
    hw.ram_bw_gbps = bench_ram(std::max(1, hw.logical_cpus), (size_t) bytes);
    for (auto & g : hw.gpus) {
        if (!g.integrated) bench_gpu(g);
    }
    refresh_memory(hw);
    hw.benchmarked = true;
}

// sequential read of a big file, bypassing the OS cache when possible
double measure_disk_random(const std::string & path, size_t block, int threads, double seconds, bool mapped, std::string & err) {
    const uint64_t size = file_size(path);
    block = std::max<size_t>(4096, block - block % 4096);
    if (size < (uint64_t) block * 64) { err = "file too small"; return 0; }
    const uint64_t slots = (size - block) / block;
    std::atomic<uint64_t> bytes{0};
    std::atomic<bool> failed{false};
    threads = std::max(1, std::min(threads, 64));
#if defined(_WIN32)
    const std::wstring wpath = utf8_to_wide(path);
    HANDLE mh = INVALID_HANDLE_VALUE, map = nullptr;
    void * view = nullptr;
    if (mapped) {
        mh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (mh != INVALID_HANDLE_VALUE) map = CreateFileMappingW(mh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (map) view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
        if (view) { volatile const uint8_t * v = (const uint8_t *) view; (void) v[0]; } // one resident page, like a real mapping
    }
    std::vector<HANDLE> handles;
    for (int t = 0; t < threads; t++) {
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (h == INVALID_HANDLE_VALUE) { failed = true; break; }
        handles.push_back(h);
    }
    const double t0 = now_sec();
    std::vector<std::thread> th;
    for (int t = 0; t < (int) handles.size() && !failed; t++) {
        th.emplace_back([&, t] {
            void * mem = VirtualAlloc(nullptr, block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!mem) { failed = true; return; }
            uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t) (t + 1);
            while (!failed && now_sec() - t0 < seconds) {
                x = x * 6364136223846793005ull + 1442695040888963407ull;
                const uint64_t pos = ((x >> 17) % slots) * block;
                OVERLAPPED ov = {};
                ov.Offset     = (DWORD) (pos & 0xFFFFFFFFu);
                ov.OffsetHigh = (DWORD) (pos >> 32);
                DWORD got = 0;
                if (!ReadFile(handles[(size_t) t], mem, (DWORD) block, &got, &ov) || got == 0) { failed = true; break; }
                bytes += got;
            }
            VirtualFree(mem, 0, MEM_RELEASE);
        });
    }
    for (auto & x : th) x.join();
    const double dt = now_sec() - t0;
    for (HANDLE h : handles) CloseHandle(h);
    if (view) UnmapViewOfFile(view);
    if (map) CloseHandle(map);
    if (mh != INVALID_HANDLE_VALUE) CloseHandle(mh);
#else
    int fd = open(path.c_str(), O_RDONLY
#ifdef O_DIRECT
                  | O_DIRECT
#endif
    );
    if (fd < 0) fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "cannot open file"; return 0; }
    void * view = nullptr;
    if (mapped) {
        int mfd = open(path.c_str(), O_RDONLY);
        if (mfd >= 0) {
            view = mmap(nullptr, size, PROT_READ, MAP_SHARED, mfd, 0);
            close(mfd);
            if (view == MAP_FAILED) view = nullptr;
            if (view) { volatile const uint8_t * v = (const uint8_t *) view; (void) v[0]; }
        }
    }
    const double t0 = now_sec();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++) {
        th.emplace_back([&, t] {
            void * mem = nullptr;
            if (posix_memalign(&mem, 4096, block) != 0) { failed = true; return; }
            uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t) (t + 1);
            while (!failed && now_sec() - t0 < seconds) {
                x = x * 6364136223846793005ull + 1442695040888963407ull;
                const uint64_t pos = ((x >> 17) % slots) * block;
                const ssize_t got = pread(fd, mem, block, (off_t) pos);
                if (got <= 0) { failed = true; break; }
                bytes += (uint64_t) got;
            }
            free(mem);
        });
    }
    for (auto & x : th) x.join();
    const double dt = now_sec() - t0;
    if (view) munmap(view, size);
    close(fd);
#endif
    if (failed) { err = "read failed"; return 0; }
    return dt > 0 ? (double) bytes.load() / dt / 1e9 : 0;
}

double measure_disk_read(const std::string & path, uint64_t max_bytes, std::string & err) {
    const size_t chunk    = 8u << 20;
    const int    nthreads = 4;
    const uint64_t size   = file_size(path);
    if (size < (uint64_t) chunk * 8) { err = "file too small"; return 0; }
    uint64_t total = std::min<uint64_t>(max_bytes, size / 2);
    total -= total % ((uint64_t) chunk * nthreads);
    const uint64_t start = (size / 4) - (size / 4) % chunk;
    const uint64_t per_thread = total / nthreads;
    std::atomic<bool> failed{false};

#if defined(_WIN32)
    const std::wstring wpath = utf8_to_wide(path);
    // one handle per thread: Windows serializes reads issued on a single synchronous handle
    std::vector<HANDLE> handles;
    for (int t = 0; t < nthreads; t++) {
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                               OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            for (HANDLE o : handles) CloseHandle(o);
            err = "cannot open file";
            return 0;
        }
        handles.push_back(h);
    }
    const double t0 = now_sec();
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; t++) {
        th.emplace_back([&, t] {
            void * mem = VirtualAlloc(nullptr, chunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!mem) { failed = true; return; }
            for (uint64_t off = 0; off < per_thread && !failed; off += chunk) {
                const uint64_t pos = start + (uint64_t) t * per_thread + off;
                OVERLAPPED ov = {};
                ov.Offset     = (DWORD) (pos & 0xFFFFFFFFu);
                ov.OffsetHigh = (DWORD) (pos >> 32);
                DWORD got = 0;
                if (!ReadFile(handles[(size_t) t], mem, (DWORD) chunk, &got, &ov) || got == 0) { failed = true; break; }
            }
            VirtualFree(mem, 0, MEM_RELEASE);
        });
    }
    for (auto & x : th) x.join();
    const double dt = now_sec() - t0;
    for (HANDLE h : handles) CloseHandle(h);
#else
    int fd = open(path.c_str(), O_RDONLY
#ifdef O_DIRECT
                  | O_DIRECT
#endif
    );
    bool direct = true;
    if (fd < 0) {
        direct = false;
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open file"; return 0; }
    }
#ifdef POSIX_FADV_DONTNEED
    if (!direct) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    const double t0 = now_sec();
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; t++) {
        th.emplace_back([&, t] {
            void * mem = nullptr;
            if (posix_memalign(&mem, 4096, chunk) != 0) { failed = true; return; }
            for (uint64_t off = 0; off < per_thread && !failed; off += chunk) {
                const uint64_t pos = start + (uint64_t) t * per_thread + off;
                const ssize_t got = pread(fd, mem, chunk, (off_t) pos);
                if (got <= 0) { failed = true; break; }
            }
            free(mem);
        });
    }
    for (auto & x : th) x.join();
    const double dt = now_sec() - t0;
    close(fd);
#endif
    if (failed) { err = "read failed"; return 0; }
    return dt > 0 ? (double) total / dt / 1e9 : 0;
}

ggml_backend_dev_t gpu_device(const HwInfo & hw, int gpu_slot) {
    if (gpu_slot < 0 || gpu_slot >= (int) hw.gpus.size()) return nullptr;
    return ggml_backend_dev_get((size_t) hw.gpus[(size_t) gpu_slot].dev_index);
}

void save_disk_measurement(const std::string & file, const HwInfo & hw) {
    if (hw.disk_gbps <= 0) return;
    const nlohmann::json j = {
        { "cpu", hw.cpu_name }, { "disk_path", hw.disk_path }, { "disk_gbps", hw.disk_gbps },
        { "disk_rand1", hw.disk_rand1 }, { "disk_rand16", hw.disk_rand16 }, { "disk_rand16_mapped", hw.disk_rand16_mapped },
    };
    FILE * f = fopen_utf8(file, "wb");
    if (!f) return;
    const std::string s = j.dump(2);
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
}

bool load_disk_measurement(const std::string & file, HwInfo & hw) {
    std::string text;
    if (!read_text_file(file, text)) return false;
    try {
        const nlohmann::json j = nlohmann::json::parse(text);
        if (j.value("cpu", std::string()) != hw.cpu_name) return false; // another machine
        hw.disk_gbps          = j.value("disk_gbps", 0.0);
        hw.disk_path          = j.value("disk_path", std::string());
        hw.disk_rand1         = j.value("disk_rand1", 0.0);
        hw.disk_rand16        = j.value("disk_rand16", 0.0);
        hw.disk_rand16_mapped = j.value("disk_rand16_mapped", 0.0);
        return hw.disk_gbps > 0;
    } catch (...) {
        return false;
    }
}

nlohmann::json hw_to_json(const HwInfo & hw) {
    nlohmann::json gpus = nlohmann::json::array();
    for (const auto & g : hw.gpus) {
        gpus.push_back({
            { "name", g.name }, { "description", g.description }, { "integrated", g.integrated },
            { "vram_total", g.vram_total }, { "vram_free", g.vram_free },
            { "bw_gbps", g.bw_gbps }, { "h2d_gbps", g.h2d_gbps },
        });
    }
    return {
        { "os", hw.os }, { "cpu_name", hw.cpu_name },
        { "logical_cpus", hw.logical_cpus }, { "physical_cores", hw.physical_cores },
        { "p_cores", hw.p_cores }, { "e_cores", hw.e_cores },
        { "avx2", hw.avx2 }, { "avx512", hw.avx512 }, { "avx_vnni", hw.avx_vnni }, { "fma", hw.fma }, { "f16c", hw.f16c },
        { "ram_total", hw.ram_total }, { "ram_avail", hw.ram_avail }, { "ram_bw_gbps", hw.ram_bw_gbps },
        { "disk_gbps", hw.disk_gbps }, { "disk_path", hw.disk_path },
        { "disk_rand1", hw.disk_rand1 }, { "disk_rand16", hw.disk_rand16 }, { "disk_rand16_mapped", hw.disk_rand16_mapped },
        { "gpus", gpus }, { "best_gpu", hw.best_gpu() }, { "benchmarked", hw.benchmarked },
    };
}

} // namespace rafid
