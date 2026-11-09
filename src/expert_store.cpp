#include "expert_store.h"
#include "util.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RAFID_X86 1
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace rafid {

namespace {

constexpr uint64_t PAGE = 4096; // also the alignment used for unbuffered reads

inline uint64_t floor_to(uint64_t v, uint64_t a) { return v - v % a; }
inline uint64_t ceil_to(uint64_t v, uint64_t a)  { return (v + a - 1) / a * a; }

// ---- virtual memory ---------------------------------------------------------------------------
uint8_t * vm_reserve(size_t size) {
#ifdef _WIN32
    return (uint8_t *) VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
#else
    void * p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : (uint8_t *) p;
#endif
}

// committing pages that are already committed keeps their contents (both platforms)
bool vm_commit(uint8_t * p, size_t n) {
    if (n == 0) return true;
#ifdef _WIN32
    return VirtualAlloc(p, n, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    return mprotect(p, n, PROT_READ | PROT_WRITE) == 0;
#endif
}

void vm_release(uint8_t * p, size_t n) {
#ifdef _WIN32
    (void) n;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, n);
#endif
}

// ---- resident blocks for the cache slots and the staging buffers ---------------------------------
// allocated once and kept: on Windows every newly touched page costs a fault that the OS handles one at a
// time per process (~1 GB/s), so the cache must not ask for fresh memory while the model generates
void touch_pages(uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; i += PAGE) p[i] = 0;
}

#ifdef _WIN32
// the pages a process may lock are limited by its minimum working set: raise it by delta (may be negative)
bool adjust_working_set(int64_t delta) {
    HANDLE proc = GetCurrentProcess();
    SIZE_T mn = 0, mx = 0;
    DWORD fl = 0;
    if (!GetProcessWorkingSetSizeEx(proc, &mn, &mx, &fl)) return false;
    const int64_t want_min = std::max<int64_t>((int64_t) mn + delta, 16ll << 20);
    const SIZE_T new_min = (SIZE_T) want_min;
    const SIZE_T new_max = std::max<SIZE_T>(mx, new_min + ((SIZE_T) 512 << 20));
    return SetProcessWorkingSetSizeEx(proc, new_min, new_max, fl) != 0;
}
#endif

uint8_t * mem_alloc(size_t n, bool & locked) {
    locked = false;
    if (n == 0) return nullptr;
#ifdef _WIN32
    uint8_t * p = (uint8_t *) VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) return nullptr;
    // locking keeps the cache in RAM and brings every page in at once (much faster than touching them)
    if (adjust_working_set((int64_t) n + (64ll << 20))) {
        if (VirtualLock(p, n)) locked = true;
        else adjust_working_set(-((int64_t) n + (64ll << 20)));
    }
    if (!locked) touch_pages(p, n);
    return p;
#else
    void * v = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (v == MAP_FAILED) return nullptr;
#ifdef MADV_HUGEPAGE
    madvise(v, n, MADV_HUGEPAGE);
#endif
    touch_pages((uint8_t *) v, n);
    return (uint8_t *) v;
#endif
}

void mem_free(uint8_t * p, size_t n, bool locked) {
    if (!p) return;
#ifdef _WIN32
    if (locked) VirtualUnlock(p, n);
    VirtualFree(p, 0, MEM_RELEASE);
    if (locked) adjust_working_set(-((int64_t) n + (64ll << 20)));
#else
    (void) locked;
    munmap(p, n);
#endif
}

// ---- unbuffered file reads (the OS cache must not hide the real disk speed) -------------------
class RawFile {
public:
    ~RawFile() { close(); }

    bool open(const std::string & path, bool want_direct) {
        close();
#ifdef _WIN32
        const std::wstring w = utf8_to_wide(path);
        const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
        h_ = CreateFileW(w.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING,
                         want_direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_RANDOM_ACCESS, nullptr);
        direct_ = want_direct;
        if (h_ == INVALID_HANDLE_VALUE && want_direct) {
            h_ = CreateFileW(w.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
            direct_ = false;
        }
        return h_ != INVALID_HANDLE_VALUE;
#else
        fd_ = -1;
        direct_ = false;
#ifdef O_DIRECT
        if (want_direct) {
            fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            direct_ = fd_ >= 0;
        }
#endif
        if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY);
        path_ = path;
        return fd_ >= 0;
#endif
    }

    // reads len bytes at off into buf (page aligned, len a multiple of PAGE); returns the bytes read (short at EOF) or -1
    int64_t read_at(uint64_t off, uint8_t * buf, size_t len) {
#ifdef _WIN32
        if (h_ == INVALID_HANDLE_VALUE) return -1;
        size_t done = 0;
        while (done < len) {
            OVERLAPPED ov = {};
            const uint64_t pos = off + done;
            ov.Offset     = (DWORD) (pos & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD) (pos >> 32);
            DWORD got = 0;
            const DWORD want = (DWORD) std::min<size_t>(len - done, 64u << 20);
            if (!ReadFile(h_, buf + done, want, &got, &ov)) {
                if (GetLastError() == ERROR_HANDLE_EOF) break;
                return -1;
            }
            if (got == 0) break;
            done += got;
            if (got < want) break;
        }
        return (int64_t) done;
#else
        if (fd_ < 0) return -1;
        size_t done = 0;
        while (done < len) {
            const ssize_t r = pread(fd_, buf + done, len - done, (off_t) (off + done));
            if (r < 0) {
                if (errno == EINTR) continue;
                if (errno == EINVAL && direct_) {
                    // the file system refuses O_DIRECT: fall back to cached reads + drop the cache
                    ::close(fd_);
                    fd_ = ::open(path_.c_str(), O_RDONLY);
                    direct_ = false;
                    if (fd_ < 0) return -1;
                    continue;
                }
                return -1;
            }
            if (r == 0) break;
            done += (size_t) r;
        }
#ifdef POSIX_FADV_DONTNEED
        if (!direct_) posix_fadvise(fd_, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
#endif
        return (int64_t) done;
#endif
    }

    bool direct() const { return direct_; }

    void close() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
        h_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }

private:
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
    std::string path_;
#endif
    bool direct_ = false;
};

// ---- fault handler registry (Windows only) -----------------------------------------------------
#ifdef _WIN32
std::mutex                  g_reg_mu;
std::vector<ExpertStore *>  g_stores;
PVOID                       g_veh = nullptr;

LONG CALLBACK rafid_veh(PEXCEPTION_POINTERS ep) {
    const EXCEPTION_RECORD * er = ep->ExceptionRecord;
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const uintptr_t addr = (uintptr_t) er->ExceptionInformation[1];
    ExpertStore * target = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_reg_mu);
        for (ExpertStore * s : g_stores) {
            if (s->contains(addr)) { target = s; break; }
        }
    }
    if (target && target->handle_fault(addr)) return EXCEPTION_CONTINUE_EXECUTION;
    return EXCEPTION_CONTINUE_SEARCH;
}

void register_store(ExpertStore * s) {
    const char * off = getenv("RAFID_NO_FAULT_HANDLER");
    if (off && *off && *off != '0') return;
    std::lock_guard<std::mutex> lk(g_reg_mu);
    g_stores.push_back(s);
    if (!g_veh) g_veh = AddVectoredExceptionHandler(1, rafid_veh);
}

void unregister_store(ExpertStore * s) {
    std::lock_guard<std::mutex> lk(g_reg_mu);
    g_stores.erase(std::remove(g_stores.begin(), g_stores.end(), s), g_stores.end());
    if (g_stores.empty() && g_veh) {
        RemoveVectoredExceptionHandler(g_veh);
        g_veh = nullptr;
    }
}
#else
void register_store(ExpertStore *) {}
void unregister_store(ExpertStore *) {}
#endif

int layer_of_name(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    return std::atoi(name.c_str() + 4);
}

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? std::atoi(v) : def;
}

bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v && *v && *v != '0';
}

bool page_committed(uintptr_t addr) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery((LPCVOID) addr, &mbi, sizeof(mbi)) != sizeof(mbi)) return false;
    return mbi.State == MEM_COMMIT;
#else
    (void) addr;
    return false;
#endif
}

// ---- fp16 storage for the router copies ---------------------------------------------------------
uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007FFFFFu;
    int32_t  exp  = (int32_t) ((x >> 23) & 0xFF);
    if (exp == 0xFF) return (uint16_t) (sign | 0x7C00u | (mant ? 0x200u : 0u));
    exp = exp - 127 + 15;
    if (exp >= 0x1F) return (uint16_t) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        mant |= 0x00800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (h & 1u))) h++;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t) (sign | h);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            x = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

void matvec_f16_scalar(const uint16_t * G, const float * v, int64_t n, int64_t rows, float * out) {
    for (int64_t r = 0; r < rows; r++) {
        const uint16_t * g = G + r * n;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int64_t j = 0;
        for (; j + 4 <= n; j += 4) {
            a0 += f16_to_f32(g[j]) * v[j];
            a1 += f16_to_f32(g[j + 1]) * v[j + 1];
            a2 += f16_to_f32(g[j + 2]) * v[j + 2];
            a3 += f16_to_f32(g[j + 3]) * v[j + 3];
        }
        for (; j < n; j++) a0 += f16_to_f32(g[j]) * v[j];
        out[r] = (a0 + a1) + (a2 + a3);
    }
}

#ifdef RAFID_X86
__attribute__((target("avx2,fma,f16c")))
void matvec_f16_simd(const uint16_t * G, const float * v, int64_t n, int64_t rows, float * out) {
    for (int64_t r = 0; r < rows; r++) {
        const uint16_t * g = G + r * n;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 16 <= n; j += 16) {
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (g + j))), _mm256_loadu_ps(v + j), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (g + j + 8))), _mm256_loadu_ps(v + j + 8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float acc = _mm_cvtss_f32(s);
        for (; j < n; j++) acc += f16_to_f32(g[j]) * v[j];
        out[r] = acc;
    }
}
#endif

} // namespace

static std::string lower(const std::string & in) {
    std::string s;
    for (char c : in) s += (char) std::tolower((unsigned char) c);
    return s;
}

std::string normalize_policy(const std::string & in) {
    const std::string s = lower(in);
    if (s == "fingerprint" || s == "pin") return "fingerprint";
    if (s == "freq" || s == "lfu" || s == "frequency") return "freq";
    return "lru";
}

std::string normalize_swap_mode(const std::string & in) {
    const std::string s = lower(in);
    if (s == "light" || s == "medium" || s == "strong") return s;
    return "off";
}

void swap_params(const std::string & mode, int k, int & from_rank, double & ratio) {
    // the lowest 30% / 50% / 70% of the ranks may be replaced, by an expert with 70% / 50% / 30% of the weight
    double share = 0;
    ratio = 1;
    if (mode == "light")       { share = 0.3; ratio = 0.7; }
    else if (mode == "medium") { share = 0.5; ratio = 0.5; }
    else if (mode == "strong") { share = 0.7; ratio = 0.3; }
    const int n = share > 0 ? std::max(1, (int) std::ceil(share * k - 1e-9)) : 0;
    from_rank = n > 0 && k > 1 ? std::max(2, k - n + 1) : 0; // the expert with the highest weight always stays
}

std::string normalize_prefetch_mode(const std::string & in) {
    std::string s;
    for (char c : in) s += (char) std::tolower((unsigned char) c);
    if (s == "off" || s == "none" || s == "false" || s == "0" || s == "no") return "off";
    if (s == "history" || s == "fingerprint" || s == "trace") return "history";
    return "gate";
}

// -------------------------------------------------------------------------------------------------
// stats
// -------------------------------------------------------------------------------------------------
StreamStats StreamStats::delta(const StreamStats & b) const {
    StreamStats d;
    d.routed = routed - b.routed;
    d.accesses = accesses - b.accesses;
    d.hits = hits - b.hits;
    d.routes_pool = routes_pool - b.routes_pool;
    d.routes_staged = routes_staged - b.routes_staged;
    d.bytes_copied = bytes_copied - b.bytes_copied;
    d.inflight = inflight - b.inflight;
    d.misses = misses - b.misses;
    d.prefetch_issued = prefetch_issued - b.prefetch_issued;
    d.prefetch_used = prefetch_used - b.prefetch_used;
    d.prefetch_cancelled = prefetch_cancelled - b.prefetch_cancelled;
    d.pred_total = pred_total - b.pred_total;
    d.pred_hits = pred_hits - b.pred_hits;
    d.pred2_total = pred2_total - b.pred2_total;
    d.pred2_hits = pred2_hits - b.pred2_hits;
    d.pred_late = pred_late - b.pred_late;
    d.bulk_issued = bulk_issued - b.bulk_issued;
    d.bytes_read = bytes_read - b.bytes_read;
    d.reads = reads - b.reads;
    d.evictions = evictions - b.evictions;
    d.over_budget = over_budget - b.over_budget;
    d.fault_loads = fault_loads - b.fault_loads;
    d.io_errors = io_errors - b.io_errors;
    d.swap_candidates = swap_candidates - b.swap_candidates;
    d.ahead_issued = ahead_issued - b.ahead_issued;
    d.ahead_used = ahead_used - b.ahead_used;
    d.swapped = swapped - b.swapped;
    d.wait_sec = wait_sec - b.wait_sec;
    d.io_busy_sec = io_busy_sec - b.io_busy_sec;
    return d;
}

nlohmann::json StreamStats::to_json() const {
    const double acc = (double) std::max<uint64_t>(1, accesses);
    auto ratio = [](uint64_t a, uint64_t b) -> nlohmann::json { return b ? nlohmann::json((double) a / (double) b) : nlohmann::json(nullptr); };
    return {
        { "routed", routed }, { "accesses", accesses }, { "hits", hits }, { "inflight", inflight }, { "misses", misses },
        { "routes_pool", routes_pool }, { "routes_staged", routes_staged }, { "bytes_copied", bytes_copied },
        { "hit_rate", accesses ? (double) hits / acc : 0.0 },
        { "ready_rate", accesses ? (double) (hits + inflight) / acc : 0.0 },
        { "prefetch_issued", prefetch_issued }, { "prefetch_used", prefetch_used }, { "prefetch_cancelled", prefetch_cancelled },
        { "prefetch_accuracy", prefetch_issued ? (double) prefetch_used / (double) prefetch_issued : 0.0 },
        { "pred_total", pred_total }, { "pred_hits", pred_hits }, { "pred_accuracy", ratio(pred_hits, pred_total) },
        { "pred2_total", pred2_total }, { "pred2_hits", pred2_hits }, { "pred2_accuracy", ratio(pred2_hits, pred2_total) },
        { "pred_late", pred_late }, { "bulk_issued", bulk_issued },
        { "bytes_read", bytes_read }, { "reads", reads }, { "evictions", evictions },
        { "over_budget", over_budget }, { "fault_loads", fault_loads }, { "io_errors", io_errors },
        { "wait_sec", wait_sec }, { "io_busy_sec", io_busy_sec },
        { "swap_candidates", swap_candidates }, { "swapped", swapped },
        { "ahead_issued", ahead_issued }, { "ahead_used", ahead_used },
        { "disk_gbps", io_busy_sec > 0 ? (double) bytes_read / io_busy_sec / 1e9 : 0.0 },
    };
}

// -------------------------------------------------------------------------------------------------
// construction
// -------------------------------------------------------------------------------------------------
ExpertStore::ExpertStore(const ModelInfo & mi, bool simd) : mi_(mi), simd_(simd) {
    buft_.iface.get_name         = cb_buft_name;
    buft_.iface.alloc_buffer     = cb_buft_alloc;
    buft_.iface.alloc_buffer_n   = nullptr;
    buft_.iface.get_alignment    = cb_buft_align;
    buft_.iface.get_max_size     = nullptr;
    buft_.iface.get_alloc_size   = cb_buft_alloc_size;
    buft_.iface.get_alloc_size_n = nullptr;
    buft_.iface.is_host          = cb_buft_is_host;
    buft_.device  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    buft_.context = this;

    S_ = (int) mi_.moe_layers.size();
    E_ = mi_.n_expert;
    K_ = mi_.n_expert_used;
    n_embd_ = mi_.n_embd;
    slot_of_layer_.assign((size_t) std::max(mi_.n_layer, 0) + 1, -1);
    for (int s = 0; s < S_; s++) {
        const int l = mi_.moe_layers[(size_t) s];
        if (l >= 0 && l < (int) slot_of_layer_.size()) slot_of_layer_[(size_t) l] = s;
    }
    direct_io_     = !env_flag("RAFID_STREAM_BUFFERED");
    debug_no_hook_ = env_flag("RAFID_STREAM_DEBUG_NO_HOOK");
    debug_layer_ms_ = env_int("RAFID_STREAM_DEBUG_LAYER_MS", 0); // test switch: slow prompt layers down
    n_io_          = std::max(1, std::min(env_int("RAFID_STREAM_IO_THREADS", 16), 64));
    const int ckb  = env_int("RAFID_STREAM_CHUNK_KB", 256);
    chunk_small_   = ceil_to((uint64_t) std::max(4, std::min(ckb, 16384)) << 10, PAGE);
    chunk_big_     = std::max<uint64_t>(chunk_small_, 1 << 20);
#ifndef RAFID_X86
    simd_ = false;
#endif
    register_store(this);
}

ExpertStore::~ExpertStore() {
    shutdown_io();
    unregister_store(this);
    free_pool();
    free_staging();
    for (Region * r : regions_) {
        if (r->base) vm_release(r->base, r->size);
        delete r;
    }
}

// -------------------------------------------------------------------------------------------------
// ggml buffer type (the "home" of the expert tensors: reserved address space only)
// -------------------------------------------------------------------------------------------------
const char * ExpertStore::cb_buft_name(ggml_backend_buffer_type_t) { return "Rafid_ExpertStore"; }
size_t ExpertStore::cb_buft_align(ggml_backend_buffer_type_t) { return PAGE; }
bool ExpertStore::cb_buft_is_host(ggml_backend_buffer_type_t) { return true; }

// two spare pages per tensor: room to shift it to its in-page position from the file
size_t ExpertStore::cb_buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    return (size_t) ceil_to(ggml_nbytes(t), PAGE) + 2 * PAGE;
}

ggml_backend_buffer_t ExpertStore::cb_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * self = static_cast<ExpertStore *>(buft->context);
    Region * r = new Region();
    r->store = self;
    r->size  = (size_t) ceil_to(std::max<size_t>(size, PAGE), PAGE);
    r->base  = vm_reserve(r->size);
    if (!r->base) {
        RERR("expert store: cannot reserve %s of address space", human_bytes(r->size).c_str());
        delete r;
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(self->mu_);
        self->regions_.push_back(r);
    }
    ggml_backend_buffer_i iface = {};
    iface.free_buffer   = cb_free;
    iface.get_base      = cb_base;
    iface.init_tensor   = cb_init_tensor;
    iface.memset_tensor = cb_memset;
    iface.set_tensor    = cb_set;
    iface.get_tensor    = cb_get;
    iface.set_tensor_2d = nullptr;
    iface.get_tensor_2d = nullptr;
    iface.cpy_tensor    = cb_cpy;
    iface.clear         = cb_clear;
    iface.reset         = nullptr;
    iface.set_tensor_n  = nullptr;
    iface.get_tensor_n  = nullptr;
    iface.cpy_tensor_n  = nullptr;
    return ggml_backend_buffer_init(buft, iface, r, size);
}

void ExpertStore::cb_free(ggml_backend_buffer_t buf) {
    Region * r = static_cast<Region *>(buf->context);
    ExpertStore * self = r->store;
    self->shutdown_io(); // no reader may write any more
    std::lock_guard<std::mutex> lk(self->mu_);
    if (r->base) vm_release(r->base, r->size);
    r->base = nullptr;
    r->size = 0;
    self->finalized_ = false;
}

void * ExpertStore::cb_base(ggml_backend_buffer_t buf) {
    return static_cast<Region *>(buf->context)->base;
}

enum ggml_status ExpertStore::cb_init_tensor(ggml_backend_buffer_t buf, ggml_tensor * t) {
    Region * r = static_cast<Region *>(buf->context);
    ExpertStore * self = r->store;
    if (t->view_src != nullptr) return GGML_STATUS_SUCCESS;
    MTensor m;
    m.name   = ggml_get_name(t);
    m.t      = t;
    m.layer  = layer_of_name(m.name);
    m.nbytes = ggml_nbytes(t);
    // the allocator gave this tensor ceil(nbytes) + 2 pages starting at a page boundary
    uint8_t * base = (uint8_t *) ceil_to((uint64_t) (uintptr_t) t->data, PAGE);
    auto it = self->mi_.exps_loc.find(m.name);
    const int slot = (m.layer >= 0 && m.layer < (int) self->slot_of_layer_.size()) ? self->slot_of_layer_[(size_t) m.layer] : -1;
    if (it != self->mi_.exps_loc.end() && slot >= 0 && t->ne[2] == self->E_ && it->second.nbytes == m.nbytes &&
        ggml_is_contiguous(t) && m.nbytes % (uint64_t) t->ne[2] == 0) {
        m.managed = true;
        m.slot    = slot;
        m.n_exp   = t->ne[2];
        m.nb2     = m.nbytes / (uint64_t) t->ne[2];
        m.file    = it->second.file;
        m.foff    = it->second.offset;
        // same position inside a page as in the file, so the fault path can read straight into it
        t->data = base + m.foff % PAGE;
        m.home  = (uint8_t *) t->data;
        const size_t dot = m.name.find('.', 4);
        const std::string role = dot == std::string::npos ? m.name : m.name.substr(dot + 1);
        std::lock_guard<std::mutex> lk(self->mu_);
        auto rit = std::find(self->roles_.begin(), self->roles_.end(), role);
        m.part = (int) (rit - self->roles_.begin());
        if (rit == self->roles_.end()) self->roles_.push_back(role);
        self->tensors_.push_back(m);
        return GGML_STATUS_SUCCESS;
    }
    // anything else that lands here is kept in RAM like a normal CPU tensor
    m.home = (uint8_t *) t->data;
    if (!vm_commit((uint8_t *) floor_to((uint64_t) (uintptr_t) m.home, PAGE), (size_t) ceil_to(m.nbytes, PAGE) + PAGE)) return GGML_STATUS_ALLOC_FAILED;
    RWARN("expert store: tensor %s is not a routed expert tensor, keeping it in RAM", m.name.c_str());
    std::lock_guard<std::mutex> lk(self->mu_);
    self->tensors_.push_back(m);
    return GGML_STATUS_SUCCESS;
}

ExpertStore::MTensor * ExpertStore::tensor_of(const ggml_tensor * t) {
    for (auto & m : tensors_) if (m.t == t) return &m;
    return nullptr;
}

ExpertStore::MTensor * ExpertStore::tensor_at_home(uintptr_t addr) {
    for (auto & m : tensors_) {
        const uintptr_t lo = (uintptr_t) floor_to((uint64_t) (uintptr_t) m.home, PAGE);
        const uintptr_t hi = (uintptr_t) ceil_to((uint64_t) (uintptr_t) (m.home + m.nbytes), PAGE);
        if (addr >= lo && addr < hi) return &m;
    }
    return nullptr;
}

void ExpertStore::cb_memset(ggml_backend_buffer_t buf, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    ExpertStore * self = static_cast<Region *>(buf->context)->store;
    MTensor * m = self->tensor_of(t);
    if (m && !m->managed) memset((uint8_t *) t->data + offset, value, size);
}

void ExpertStore::cb_set(ggml_backend_buffer_t buf, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    ExpertStore * self = static_cast<Region *>(buf->context)->store;
    MTensor * m = self->tensor_of(t);
    if (m && m->managed) return; // expert data stays on disk; it is read on demand
    memcpy((uint8_t *) t->data + offset, data, size);
}

void ExpertStore::cb_get(ggml_backend_buffer_t buf, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    ExpertStore * self = static_cast<Region *>(buf->context)->store;
    MTensor * m = self->tensor_of(t);
    if (!m || !m->managed) {
        memcpy(data, (const uint8_t *) t->data + offset, size);
        return;
    }
    // read straight from the model file
    FILE * f = fopen_utf8(self->mi_.files[(size_t) m->file], "rb");
    bool ok = false;
    if (f) {
#ifdef _WIN32
        ok = _fseeki64(f, (long long) (m->foff + offset), SEEK_SET) == 0;
#else
        ok = fseeko(f, (off_t) (m->foff + offset), SEEK_SET) == 0;
#endif
        ok = ok && fread(data, 1, size, f) == size;
        fclose(f);
    }
    if (!ok) memset(data, 0, size);
}

bool ExpertStore::cb_cpy(ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) { return false; }
void ExpertStore::cb_clear(ggml_backend_buffer_t, uint8_t) {}

// -------------------------------------------------------------------------------------------------
// finalize: index the tensors, start the readers (the slots are allocated by configure)
// -------------------------------------------------------------------------------------------------
bool ExpertStore::finalize(std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (S_ <= 0 || E_ <= 1 || K_ <= 0) { err = "the model has no routed experts"; return false; }
    P_ = (int) roles_.size();
    if (P_ <= 0) { err = "no expert tensors were placed in the expert store"; return false; }
    slot_part_.assign((size_t) S_ * P_, -1);
    for (size_t i = 0; i < tensors_.size(); i++) {
        const MTensor & m = tensors_[i];
        if (m.managed) slot_part_[(size_t) m.slot * P_ + m.part] = (int) i;
    }
    // every expert tensor of the used layers must be here
    int missing = 0;
    for (const auto & kv : mi_.exps_loc) {
        const int l = layer_of_name(kv.first);
        if (l < 0 || l >= mi_.n_layer_used()) continue;
        bool found = false;
        for (const auto & m : tensors_) if (m.managed && m.name == kv.first) { found = true; break; }
        if (!found) missing++;
    }
    if (missing > 0) { err = fmt("%d expert tensors were not placed in the expert store", missing); return false; }
    // other tensors that landed here (e.g. per-expert biases) are indexed by the same expert ids: slot numbers
    // would point them at the wrong rows, so such models always compute from the staging buffers
    int others = 0;
    for (const auto & m : tensors_) if (!m.managed) others++;
    remap_safe_ = others == 0;
    if (!remap_safe_) RWARN("expert store: %d other tensors use the expert ids (expert biases?): the slots are copied to a staging buffer for every layer", others);
    key_bytes_.assign((size_t) S_, 0);
    total_bytes_ = 0;
    max_key_bytes_ = 0;
    for (int s = 0; s < S_; s++) {
        int parts = 0;
        for (int p = 0; p < P_; p++) {
            const MTensor * m = part_tensor(s, p);
            if (!m) continue;
            parts++;
            key_bytes_[(size_t) s] += m->nb2;
        }
        if (parts == 0) { err = fmt("MoE layer %d has no expert tensors", mi_.moe_layers[(size_t) s]); return false; }
        total_bytes_ += key_bytes_[(size_t) s] * (uint64_t) E_;
        max_key_bytes_ = std::max(max_key_bytes_, key_bytes_[(size_t) s]);
    }
    const size_t n_keys = (size_t) S_ * E_;
    where_.assign(n_keys, -1);
    remap_.assign(n_keys, 0);
    pop_.assign(n_keys, 0.0);
    mode_.assign((size_t) S_, MODE_NONE);
    remap_ok_.assign((size_t) S_, 0);
    last_ids_.assign((size_t) S_, {});
    pf_entries_.assign((size_t) S_, {});
    lru_head_.assign((size_t) S_, -1);
    lru_tail_.assign((size_t) S_, -1);
    pool_mem_.assign((size_t) S_ * P_, nullptr);
    pred_seq_.assign((size_t) S_ + 2, 0);
    pred2_seq_.assign((size_t) S_ + 2, 0);
    pred_set_.assign((size_t) S_ + 2, {});
    pred2_set_.assign((size_t) S_ + 2, {});
    // a batch of n tokens touches 1 - (1 - K/E)^n of the experts; from 90% on, gather the next layer whole
    bulk_tok_ = E_ > K_ ? (int) std::ceil(std::log(0.1) / std::log(1.0 - (double) K_ / (double) E_)) : 1;
    bulk_tok_ = std::max(16, bulk_tok_);
    {
        std::lock_guard<std::mutex> g(router_mu_);
        routers_.assign((size_t) S_, Router());
        routers_ok_.store(0);
    }
    finalized_ = true;
    stop_ = false;
    for (int i = 0; i < n_io_; i++) workers_.emplace_back(&ExpertStore::worker_main, this, i);
    pred_thread_ = std::thread(&ExpertStore::predictor_main, this);
    RLOG("expert store: %d layers x %d experts (%d tensors each), %s on disk, largest expert %s, %d reader threads, %s reads",
         S_, E_, P_, human_bytes(total_bytes_).c_str(), human_bytes(max_key_bytes_).c_str(), n_io_, direct_io_ ? "unbuffered" : "buffered");
    return true;
}

uint64_t ExpertStore::min_budget() const {
    // every layer needs the experts of one token plus two spare slots
    return max_key_bytes_ * (uint64_t) (K_ + 2) * (uint64_t) std::max(1, S_);
}

bool ExpertStore::idle_locked() const {
    return dq_.empty() && pq_.empty() && jobs_active_ == 0;
}

void ExpertStore::shutdown_io() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (workers_.empty() && !pred_thread_.joinable()) return;
        stop_ = true;
    }
    cv_work_.notify_all();
    cv_pred_.notify_all();
    cv_done_.notify_all();
    for (auto & t : workers_) t.join();
    workers_.clear();
    if (pred_thread_.joinable()) pred_thread_.join();
    std::lock_guard<std::mutex> lk(mu_);
    dq_.clear();
    pq_.clear();
    runs_.clear();
    free_runs_.clear();
    for (auto & x : ent_) {
        if (x.state == LOADING) { x.state = FREE; x.gen++; }
        x.pins = 0;
    }
    for (auto & g : stg_) { g.gen++; g.slot = -1; g.active = 0; }
    // rebuild the expert -> slot index from the entries that are still valid
    std::fill(where_.begin(), where_.end(), -1);
    for (int s = 0; s < S_ && !ent0_.empty(); s++) {
        for (int q = ent0_[(size_t) s]; q < ent0_[(size_t) s + 1]; q++) {
            if (ent_[(size_t) q].state == READY) where_[(size_t) key_of(s, ent_[(size_t) q].expert)] = q;
            else ent_[(size_t) q].expert = -1;
        }
    }
    restore_home_pointers();
    cv_done_.notify_all();
}

// -------------------------------------------------------------------------------------------------
// memory: the slots of every layer in one resident block, staging buffers for prompt batches
// -------------------------------------------------------------------------------------------------
uint8_t * ExpertStore::slot_mem(int q, int p) const {
    const int s = (int) (std::upper_bound(ent0_.begin(), ent0_.end(), q) - ent0_.begin()) - 1;
    const MTensor * m = part_tensor(s, p);
    return pool_mem_[(size_t) s * P_ + p] + (uint64_t) (q - ent0_[(size_t) s]) * m->nb2;
}

// staging buffers are shared by all layers; the layer's in-page position from the file is kept, so reads land
// directly in place
uint8_t * ExpertStore::stage_ptr(int b, int slot, int p) const {
    const MTensor * m = part_tensor(slot, p);
    return stg_[b].base[(size_t) p] + m->foff % PAGE;
}

bool ExpertStore::alloc_pool(std::string & err) {
    const double frac = total_bytes_ ? (double) cfg_.budget_bytes / (double) total_bytes_ : 0.0;
    ent0_.assign((size_t) S_ + 1, 0);
    std::vector<int> cap((size_t) S_);
    size_t total = 0;
    for (int s = 0; s < S_; s++) {
        // every layer keeps the same share of its experts (on real traces this is within ~1% of a global LRU)
        cap[(size_t) s] = std::max(K_ + 2, std::min(E_, (int) std::floor(frac * E_ + 1e-9)));
        ent0_[(size_t) s + 1] = ent0_[(size_t) s] + cap[(size_t) s];
        for (int p = 0; p < P_; p++) {
            const MTensor * m = part_tensor(s, p);
            if (m) total += (size_t) ceil_to((uint64_t) cap[(size_t) s] * m->nb2, PAGE);
        }
    }
    pool_block_ = mem_alloc(total, pool_locked_);
    if (!pool_block_) {
        err = fmt("cannot allocate %s for the expert cache", human_bytes(total).c_str());
        ent0_.clear();
        ent_.clear();
        return false;
    }
    pool_size_ = total;
    size_t off = 0;
    for (int s = 0; s < S_; s++) {
        for (int p = 0; p < P_; p++) {
            const MTensor * m = part_tensor(s, p);
            if (!m) continue;
            pool_mem_[(size_t) s * P_ + p] = pool_block_ + off;
            off += (size_t) ceil_to((uint64_t) cap[(size_t) s] * m->nb2, PAGE);
        }
    }
    ent_.assign((size_t) ent0_[(size_t) S_], Entry());
    std::fill(where_.begin(), where_.end(), -1);
    std::fill(lru_head_.begin(), lru_head_.end(), -1);
    std::fill(lru_tail_.begin(), lru_tail_.end(), -1);
    RLOG("expert store: %s of cache slots (%d per layer)%s", human_bytes(total).c_str(), cap.empty() ? 0 : cap[0],
         pool_locked_ ? ", locked in RAM" : "");
    return true;
}

void ExpertStore::free_pool() {
    mem_free(pool_block_, pool_size_, pool_locked_);
    pool_block_ = nullptr;
    pool_size_ = 0;
    pool_locked_ = false;
    std::fill(pool_mem_.begin(), pool_mem_.end(), nullptr);
    ent_.clear();
    ent0_.clear();
    std::fill(where_.begin(), where_.end(), -1);
}

void ExpertStore::free_staging() {
    if (n_stg_ > 0) mem_free(stg_[0].block, stg_bytes_ * (size_t) n_stg_, stg_locked_);
    for (auto & g : stg_) {
        g.block = nullptr;
        g.base.clear();
        g.slot = -1;
        g.gen++;
    }
    n_stg_ = 0;
    stg_locked_ = false;
}

bool ExpertStore::ensure_staging(std::unique_lock<std::mutex> & lk) {
    if (n_stg_ > 0) return true;
    // one buffer holds a full layer of experts for every part, in the file's in-page position
    std::vector<size_t> part_bytes((size_t) P_, 0);
    for (int p = 0; p < P_; p++) {
        for (int s = 0; s < S_; s++) {
            const MTensor * m = part_tensor(s, p);
            if (m) part_bytes[(size_t) p] = std::max(part_bytes[(size_t) p], (size_t) ceil_to(m->nbytes, PAGE) + 2 * PAGE);
        }
    }
    size_t one = 0;
    for (size_t b : part_bytes) one += b;
    // two buffers (one layer computes while the next is gathered) unless that is a lot of memory
    const int want = 2 * one <= std::max<size_t>(2ull << 30, cfg_.budget_bytes / 2) ? 2 : 1;
    lk.unlock();
    bool locked = false;
    uint8_t * block = nullptr;
    int got = 0;
    for (int n = want; n >= 1 && !block; n--) {
        block = mem_alloc(one * (size_t) n, locked);
        if (block) got = n;
    }
    lk.lock();
    if (!block) {
        RERR("expert store: cannot allocate %s for reading prompts", human_bytes(one).c_str());
        return false;
    }
    stg_bytes_  = one;
    stg_locked_ = locked;
    n_stg_      = got;
    for (int b = 0; b < n_stg_; b++) {
        Staging & g = stg_[b];
        g.block = block + one * (size_t) b;
        g.size  = one;
        g.base.assign((size_t) P_, nullptr);
        size_t off = 0;
        for (int p = 0; p < P_; p++) { g.base[(size_t) p] = g.block + off; off += part_bytes[(size_t) p]; }
        g.slot = -1;
        g.gen++;
        g.state.assign((size_t) E_, 0);
        g.left.assign((size_t) E_, 0);
        g.err.assign((size_t) E_, 0);
    }
    RLOG("expert store: %d prompt buffer%s of %s%s", n_stg_, n_stg_ > 1 ? "s" : "", human_bytes(one).c_str(), locked ? ", locked in RAM" : "");
    return true;
}

void ExpertStore::prompt_done() {
    std::unique_lock<std::mutex> lk(mu_);
    if (n_stg_ == 0) return;
    // small buffers stay for the next prompt; big ones (huge models) go back to the OS
    if (stg_bytes_ * (size_t) n_stg_ <= ((size_t) 3 << 29)) return;
    for (int b = 0; b < n_stg_; b++) { stg_[b].gen++; stg_[b].slot = -1; }
    cv_done_.wait(lk, [&] { return stop_ || idle_locked(); });
    free_staging();
}

// -------------------------------------------------------------------------------------------------
// LRU of each layer (READY, unpinned entries); head = most recently used
// -------------------------------------------------------------------------------------------------
static inline int slot_of_entry(const std::vector<int32_t> & ent0, int q) {
    return (int) (std::upper_bound(ent0.begin(), ent0.end(), q) - ent0.begin()) - 1;
}

void ExpertStore::lru_remove(int q) {
    Entry & x = ent_[(size_t) q];
    if (!x.in_lru) return;
    const int s = slot_of_entry(ent0_, q);
    if (x.prev >= 0) ent_[(size_t) x.prev].next = x.next; else lru_head_[(size_t) s] = x.next;
    if (x.next >= 0) ent_[(size_t) x.next].prev = x.prev; else lru_tail_[(size_t) s] = x.prev;
    x.prev = x.next = -1;
    x.in_lru = 0;
}

void ExpertStore::lru_push_front(int q) {
    lru_remove(q);
    const int s = slot_of_entry(ent0_, q);
    Entry & x = ent_[(size_t) q];
    x.prev = -1;
    x.next = lru_head_[(size_t) s];
    if (x.next >= 0) ent_[(size_t) x.next].prev = q;
    lru_head_[(size_t) s] = q;
    if (lru_tail_[(size_t) s] < 0) lru_tail_[(size_t) s] = q;
    x.in_lru = 1;
}

void ExpertStore::lru_push_back(int q) {
    lru_remove(q);
    const int s = slot_of_entry(ent0_, q);
    Entry & x = ent_[(size_t) q];
    x.next = -1;
    x.prev = lru_tail_[(size_t) s];
    if (x.prev >= 0) ent_[(size_t) x.prev].next = q;
    lru_tail_[(size_t) s] = q;
    if (lru_head_[(size_t) s] < 0) lru_head_[(size_t) s] = q;
    x.in_lru = 1;
}

// a free slot, else the least recently used one that may be replaced: a demand read may take anything the
// current batch does not use; a guess only what the last token did not use
int ExpertStore::pick_victim(int slot, bool prefetch, const std::vector<uint8_t> * needed) {
    for (int q = ent0_[(size_t) slot]; q < ent0_[(size_t) slot + 1]; q++) {
        if (ent_[(size_t) q].state == FREE) return q;
    }
    const uint32_t recent = epoch_ > (uint32_t) S_ ? epoch_ - (uint32_t) S_ : 0;
    if (freq_) {
        // the lowest use count with decay goes first (never used: guesses that were wrong)
        int best = -1;
        double best_key = INFINITY;
        for (int q = ent0_[(size_t) slot]; q < ent0_[(size_t) slot + 1]; q++) {
            const Entry & x = ent_[(size_t) q];
            if (x.state != READY || x.pinned || x.pins) continue;
            if (needed && x.expert >= 0 && (*needed)[(size_t) x.expert]) continue;
            if (prefetch ? x.use_epoch > recent : x.use_epoch == epoch_) continue;
            if (best < 0 || x.fkey < best_key) { best = q; best_key = x.fkey; }
        }
        return best;
    }
    for (int q = lru_tail_[(size_t) slot]; q >= 0; q = ent_[(size_t) q].prev) {
        const Entry & x = ent_[(size_t) q];
        if (x.state != READY || x.pinned || x.pins) continue;
        if (needed && x.expert >= 0 && (*needed)[(size_t) x.expert]) continue;
        if (prefetch ? x.use_epoch > recent : x.use_epoch == epoch_) continue;
        return q;
    }
    return -1;
}

void ExpertStore::assign_entry(int q, int slot, int e) {
    Entry & x = ent_[(size_t) q];
    if (x.expert >= 0) {
        where_[(size_t) key_of(slot, x.expert)] = -1;
        if (x.state == READY) stats_.evictions++;
    }
    lru_remove(q);
    int parts = 0;
    for (int p = 0; p < P_; p++) if (part_tensor(slot, p)) parts++;
    x.expert = e;
    x.state = LOADING;
    x.gen++;
    x.parts_left = (uint8_t) parts;
    x.prefetched = x.demote = x.started = x.err = 0;
    x.pinned = 0;
    x.use_epoch = 0;
    x.fkey = -INFINITY;
    where_[(size_t) key_of(slot, e)] = q;
}

// a guess that turned out wrong before any of it was read
void ExpertStore::cancel_entry(int q) {
    Entry & x = ent_[(size_t) q];
    const int s = slot_of_entry(ent0_, q);
    if (x.expert >= 0) where_[(size_t) key_of(s, x.expert)] = -1;
    x.gen++;
    x.expert = -1;
    x.state = FREE;
    x.prefetched = x.demote = x.started = 0;
    stats_.prefetch_cancelled++;
}

void ExpertStore::entry_loaded(int q) {
    Entry & x = ent_[(size_t) q];
    x.state = READY;
    x.started = 0;
    if (x.err) { stats_.io_errors++; x.err = 0; }
    if (!x.pinned) {
        if (x.demote) lru_push_back(q); else lru_push_front(q);
    }
    x.demote = 0;
    cv_done_.notify_all();
}

// -------------------------------------------------------------------------------------------------
// work items: a "run" is one contiguous file range of one tensor part; it is split into jobs that the
// reader threads take in parallel
// -------------------------------------------------------------------------------------------------
int32_t ExpertStore::new_run() {
    int32_t rid;
    if (!free_runs_.empty()) { rid = free_runs_.back(); free_runs_.pop_back(); }
    else { rid = (int32_t) runs_.size(); runs_.emplace_back(); }
    runs_[(size_t) rid] = Run();
    return rid;
}

void ExpertStore::push_jobs(int32_t rid, uint64_t f0, uint64_t f1, int prio) {
    // prio: 0 = demand (queue end), 1 = urgent (queue front), 2 = background
    std::vector<Job> jobs;
    if (f1 > f0) {
        const uint64_t ch = f1 - f0 > 4 * chunk_big_ ? chunk_big_ : chunk_small_;
        for (uint64_t a = f0; a < f1; a += ch) jobs.push_back({ rid, a, std::min(a + ch, f1) });
    } else {
        jobs.push_back({ rid, 0, 0 });
    }
    runs_[(size_t) rid].jobs_left = (int32_t) jobs.size();
    if (prio == 1) for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) dq_.push_front(*it);
    else for (const Job & j : jobs) (prio == 0 ? dq_ : pq_).push_back(j);
}

void ExpertStore::load_entry(int q, int prio) {
    const int s = slot_of_entry(ent0_, q);
    const Entry & x = ent_[(size_t) q];
    const int e = x.expert;
    for (int p = 0; p < P_; p++) {
        const MTensor * m = part_tensor(s, p);
        if (!m) continue;
        const int32_t rid = new_run();
        Run & r = runs_[(size_t) rid];
        r.kind = J_POOL; r.slot = s; r.part = p; r.e0 = e; r.e1 = e + 1; r.entry = q; r.gen = x.gen;
        const uint64_t b0 = (uint64_t) e * m->nb2, b1 = std::min<uint64_t>(b0 + m->nb2, m->nbytes);
        push_jobs(rid, floor_to(m->foff + b0, PAGE), ceil_to(m->foff + b1, PAGE), prio);
    }
}

void ExpertStore::claim_staging(int b, int slot, std::unique_lock<std::mutex> & lk) {
    Staging & g = stg_[b];
    for (;;) {
        if (stop_) return;
        // the predictor thread may be emptying it for this very layer: wait, then use what it prepared
        if (g.claiming) { cv_done_.wait(lk, [&] { return stop_ || !g.claiming; }); continue; }
        if (g.slot == slot) return;
        g.claiming = true;
        g.slot = -1;
        g.gen++;   // work still queued for the previous layer is dropped
        // reads of the previous layer that are already running must finish before this buffer is reused
        cv_done_.wait(lk, [&] { return stop_ || g.active == 0; });
        std::fill(g.state.begin(), g.state.end(), 0);
        std::fill(g.left.begin(), g.left.end(), 0);
        std::fill(g.err.begin(), g.err.end(), 0);
        g.slot = g.block ? slot : -1; // freed meanwhile (end of a prompt): holds nothing
        g.claiming = false;
        cv_done_.notify_all();
        return;
    }
}

// experts ex (sorted) of a layer into staging buffer b: copied from the cache slots when they are there,
// otherwise read straight from the disk (neighbours as one long read)
void ExpertStore::stage_experts(int b, int slot, const std::vector<int> & ex, int prio, bool copy_from_pool) {
    Staging & g = stg_[b];
    int parts = 0;
    for (int p = 0; p < P_; p++) if (part_tensor(slot, p)) parts++;
    std::vector<int> rd;
    for (int e : ex) {
        if (g.state[(size_t) e] != 0) continue; // ready or on its way
        g.state[(size_t) e] = LOADING;
        g.left[(size_t) e]  = (uint8_t) parts;
        g.err[(size_t) e]   = 0;
        const int q = where_[(size_t) key_of(slot, e)];
        if (copy_from_pool && q >= 0 && ent_[(size_t) q].state == READY) {
            for (int p = 0; p < P_; p++) {
                if (!part_tensor(slot, p)) continue;
                const int32_t rid = new_run();
                Run & r = runs_[(size_t) rid];
                r.kind = J_STAGE_COPY; r.slot = slot; r.part = p; r.e0 = e; r.e1 = e + 1; r.entry = q; r.buf = b; r.gen = g.gen;
                ent_[(size_t) q].pins++;
                push_jobs(rid, 0, 0, prio);
            }
        } else {
            rd.push_back(e);
        }
    }
    for (size_t i = 0; i < rd.size();) {
        size_t j = i + 1;
        while (j < rd.size() && rd[j] == rd[j - 1] + 1) j++;
        const int e0 = rd[i], e1 = rd[j - 1] + 1;
        for (int p = 0; p < P_; p++) {
            const MTensor * m = part_tensor(slot, p);
            if (!m) continue;
            const int32_t rid = new_run();
            Run & r = runs_[(size_t) rid];
            r.kind = J_STAGE_READ; r.slot = slot; r.part = p; r.e0 = e0; r.e1 = e1; r.buf = b; r.gen = g.gen;
            const uint64_t b0 = (uint64_t) e0 * m->nb2, b1 = std::min<uint64_t>((uint64_t) e1 * m->nb2, m->nbytes);
            push_jobs(rid, floor_to(m->foff + b0, PAGE), ceil_to(m->foff + b1, PAGE), prio);
        }
        i = j;
    }
}

// queued background work for this layer becomes urgent: reads of experts it needs, and its staging buffer
void ExpertStore::promote_slot(int slot, const std::vector<uint8_t> & needed) {
    if (pq_.empty()) return;
    std::deque<Job> keep;
    std::vector<Job> up;
    for (const Job & j : pq_) {
        const Run & r = runs_[(size_t) j.run];
        bool want = false;
        if (r.slot == slot) {
            if (r.kind == J_POOL) want = needed[(size_t) r.e0] != 0;
            else want = r.buf >= 0 && stg_[r.buf].slot == slot;
        }
        if (want) up.push_back(j); else keep.push_back(j);
    }
    if (up.empty()) return;
    pq_.swap(keep);
    for (auto it = up.rbegin(); it != up.rend(); ++it) dq_.push_front(*it);
}

bool ExpertStore::run_stale(const Run & r) const {
    if (r.kind == J_POOL) return r.entry < 0 || r.entry >= (int) ent_.size() || ent_[(size_t) r.entry].gen != r.gen;
    return r.buf < 0 || stg_[r.buf].gen != r.gen;
}

void ExpertStore::finish_job(const Job & j, bool ran, bool ok) {
    Run & r = runs_[(size_t) j.run];
    if (ran && !ok) r.error = 1;
    if (--r.jobs_left > 0) return;
    const bool stale = run_stale(r);
    if (r.kind == J_POOL) {
        if (!stale) {
            Entry & x = ent_[(size_t) r.entry];
            if (r.error) x.err = 1;
            if (x.parts_left > 0 && --x.parts_left == 0) entry_loaded(r.entry);
        }
    } else {
        if (r.kind == J_STAGE_COPY && r.entry >= 0 && r.entry < (int) ent_.size() && ent_[(size_t) r.entry].pins > 0) ent_[(size_t) r.entry].pins--;
        if (!stale) {
            Staging & g = stg_[r.buf];
            for (int e = r.e0; e < r.e1; e++) {
                if (r.error) g.err[(size_t) e] = 1;
                if (g.left[(size_t) e] > 0 && --g.left[(size_t) e] == 0) {
                    g.state[(size_t) e] = READY;
                    if (g.err[(size_t) e]) stats_.io_errors++;
                }
            }
            cv_done_.notify_all();
        }
    }
    free_runs_.push_back(j.run);
}

void ExpertStore::io_begin() {
    if (io_active_++ == 0) io_busy_start_ = now_sec();
}

void ExpertStore::io_end() {
    if (--io_active_ == 0) stats_.io_busy_sec += now_sec() - io_busy_start_;
}

void ExpertStore::worker_main(int id) {
    (void) id;
    std::vector<RawFile> files(mi_.files.size());
    std::vector<uint64_t> fsize(mi_.files.size(), 0);
    for (size_t i = 0; i < mi_.files.size(); i++) {
        if (!files[i].open(mi_.files[i], direct_io_)) RERR("expert store: cannot open %s", mi_.files[i].c_str());
        fsize[i] = file_size(mi_.files[i]);
    }
    // aligned scratch for reads into a cache slot (the slot itself is not page aligned)
    bool bounce_locked = false;
    uint8_t * bounce = mem_alloc((size_t) chunk_big_, bounce_locked);
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_work_.wait(lk, [&] { return stop_ || !dq_.empty() || !pq_.empty(); });
        if (stop_) break;
        Job j;
        if (!dq_.empty()) { j = dq_.front(); dq_.pop_front(); }
        else { j = pq_.front(); pq_.pop_front(); }
        const Run r = runs_[(size_t) j.run]; // a copy: runs_ may grow while the lock is released
        if (run_stale(r)) {
            finish_job(j, false, true);
            if (idle_locked()) cv_done_.notify_all();
            continue;
        }
        const MTensor * m = part_tensor(r.slot, r.part);
        uint8_t * dst = nullptr;
        const uint8_t * src = nullptr;
        size_t len = 0;
        uint64_t x0 = 0, x1 = 0; // J_POOL: the part of this chunk that belongs to the expert (file offsets)
        if (r.kind == J_POOL) {
            ent_[(size_t) r.entry].started = 1;
            const uint64_t b0 = (uint64_t) r.e0 * m->nb2, b1 = std::min<uint64_t>(b0 + m->nb2, m->nbytes);
            x0 = std::max(j.f0, m->foff + b0);
            x1 = std::min(j.f1, m->foff + b1);
            dst = slot_mem(r.entry, r.part) + (x0 - (m->foff + b0));
            len = (size_t) (j.f1 - j.f0);
        } else if (r.kind == J_STAGE_READ) {
            dst = stage_ptr(r.buf, r.slot, r.part) + ((int64_t) j.f0 - (int64_t) m->foff);
            len = (size_t) (j.f1 - j.f0);
        } else {
            const uint64_t b0 = (uint64_t) r.e0 * m->nb2;
            src = slot_mem(r.entry, r.part);
            dst = stage_ptr(r.buf, r.slot, r.part) + b0;
            len = (size_t) (std::min<uint64_t>(b0 + m->nb2, m->nbytes) - b0);
        }
        const bool is_read = r.kind != J_STAGE_COPY;
        const int fi = m->file;
        jobs_active_++;
        if (r.kind != J_POOL) stg_[r.buf].active++;
        if (is_read) io_begin();
        lk.unlock();

        bool ok = true;
        if (is_read) {
            RawFile & f = files[(size_t) fi];
            uint8_t * target = r.kind == J_POOL ? bounce : dst;
            int64_t got = target ? f.read_at(j.f0, target, len) : -1;
            if (got < 0 && target && f.open(mi_.files[(size_t) fi], direct_io_)) got = f.read_at(j.f0, target, len);
            const uint64_t need = std::min<uint64_t>(len, fsize[(size_t) fi] > j.f0 ? fsize[(size_t) fi] - j.f0 : 0); // short only at EOF
            ok = got >= 0 && (uint64_t) got >= need;
            if (ok && r.kind == J_POOL && x1 > x0) memcpy(dst, bounce + (x0 - j.f0), (size_t) (x1 - x0));
        } else {
            memcpy(dst, src, len);
        }

        lk.lock();
        jobs_active_--;
        if (r.kind != J_POOL && --stg_[r.buf].active == 0) cv_done_.notify_all();
        if (is_read) {
            io_end();
            stats_.bytes_read += len;
            stats_.reads++;
        } else {
            stats_.bytes_copied += len;
        }
        finish_job(j, true, ok);
        if (idle_locked()) cv_done_.notify_all();
    }
    lk.unlock();
    mem_free(bounce, (size_t) chunk_big_, bounce_locked);
}

// safety net: an expert read into its home memory (stays there; this should not happen in normal use)
bool ExpertStore::read_home_sync(int slot, int e) {
    std::lock_guard<std::mutex> fl(fault_mu_);
    bool ok = true;
    for (int p = 0; p < P_; p++) {
        const MTensor * m = part_tensor(slot, p);
        if (!m) continue;
        const uint64_t b0 = (uint64_t) e * m->nb2, b1 = std::min<uint64_t>(b0 + m->nb2, m->nbytes);
        const uint64_t f0 = floor_to(m->foff + b0, PAGE), f1 = ceil_to(m->foff + b1, PAGE);
        uint8_t * dst = m->home + ((int64_t) f0 - (int64_t) m->foff);
        if (!vm_commit(dst, (size_t) (f1 - f0))) { ok = false; continue; }
        RawFile f;
        if (!f.open(mi_.files[(size_t) m->file], direct_io_)) { ok = false; continue; }
        const int64_t got = f.read_at(f0, dst, (size_t) (f1 - f0));
        if (got < (int64_t) (m->foff + b1 - f0)) ok = false;
    }
    return ok;
}

// -------------------------------------------------------------------------------------------------
// which memory the layer's expert tensors read in this step
// -------------------------------------------------------------------------------------------------
void ExpertStore::set_pointers(int slot, uint8_t mode, int b) {
    for (int p = 0; p < P_; p++) {
        const MTensor * m = part_tensor(slot, p);
        if (!m) continue;
        m->t->data = mode == MODE_POOL ? pool_mem_[(size_t) slot * P_ + p] : mode == MODE_STAGE ? stage_ptr(b, slot, p) : m->home;
    }
}

void ExpertStore::restore_home_pointers() {
    for (auto & m : tensors_) if (m.managed && m.t) m.t->data = m.home;
}

// cancel what is queued, wait for what runs (caller holds mu_ through lk)
void ExpertStore::quiesce(std::unique_lock<std::mutex> & lk) {
    for (size_t q = 0; q < ent_.size(); q++) {
        Entry & x = ent_[q];
        if (x.state != LOADING) continue;
        const int s = slot_of_entry(ent0_, (int) q);
        if (x.expert >= 0) where_[(size_t) key_of(s, x.expert)] = -1;
        x.gen++;
        x.expert = -1;
        x.state = FREE;
        x.prefetched = x.demote = x.started = 0;
    }
    for (auto & g : stg_) { g.gen++; g.slot = -1; }
    for (auto & v : pf_entries_) v.clear();
    std::fill(mode_.begin(), mode_.end(), MODE_NONE);
    pred_has_ = false;
    cv_work_.notify_all();
    cv_done_.wait(lk, [&] { return stop_ || idle_locked(); });
}

// -------------------------------------------------------------------------------------------------
// configuration from the fingerprint
// -------------------------------------------------------------------------------------------------
void ExpertStore::build_transitions(const std::string & path) {
    trans_.clear();
    trans_src_.clear();
    if (path.empty() || S_ < 2) return;
    const size_t n = (size_t) (S_ - 1) * E_ * E_;
    if (n > (size_t) 64 * 1024 * 1024) return; // too big to keep, fall back to popularity
    FILE * f = fopen_utf8(path, "rb");
    if (!f) return;
    char magic[8];
    uint32_t h[4] = { 0, 0, 0, 0 };
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "RFTRACE1", 8) != 0 || fread(h, 4, 4, f) != 4 ||
        h[1] != (uint32_t) S_ || h[2] != (uint32_t) E_ || h[3] != (uint32_t) K_) {
        fclose(f);
        return;
    }
    const uint64_t rec = (uint64_t) S_ * K_ * 2;
    const uint64_t size = file_size(path);
    const uint64_t T = size > 24 ? (size - 24) / rec : 0;
    const uint64_t take = std::min<uint64_t>(T, 20000);
    if (take < 16) { fclose(f); return; }
#ifdef _WIN32
    _fseeki64(f, (long long) (24 + (T - take) * rec), SEEK_SET);
#else
    fseeko(f, (off_t) (24 + (T - take) * rec), SEEK_SET);
#endif
    trans_.assign(n, 0);
    std::vector<uint16_t> r((size_t) S_ * K_);
    for (uint64_t t = 0; t < take; t++) {
        if (fread(r.data(), 2, r.size(), f) != r.size()) break;
        for (int s = 0; s + 1 < S_; s++) {
            for (int i = 0; i < K_; i++) {
                const int a = r[(size_t) s * K_ + i];
                if (a >= E_) continue;
                uint32_t * row = &trans_[((size_t) s * E_ + a) * E_];
                for (int j = 0; j < K_; j++) {
                    const int b = r[(size_t) (s + 1) * K_ + j];
                    if (b < E_) row[b]++;
                }
            }
        }
    }
    fclose(f);
    trans_src_ = path;
}

void ExpertStore::configure(const StreamConfig & cfg_in, const std::vector<uint64_t> & counts, const std::string & trace_path) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!finalized_) return;
    StreamConfig c = cfg_in;
    c.budget_bytes = std::max<uint64_t>(c.budget_bytes, min_budget());
    c.pin_frac     = std::max(0.0, std::min(0.95, c.pin_frac));
    c.policy       = normalize_policy(c.policy);
    c.freq_half    = std::max(1.0, std::min(4096.0, c.freq_half));
    c.prefetch     = normalize_prefetch_mode(c.prefetch);
    c.swap         = normalize_swap_mode(c.swap);
    if (c.prefetch_n <= 0) c.prefetch_n = K_;
    c.prefetch_n = std::min(c.prefetch_n, E_);
    const bool resize = pool_block_ == nullptr || c.budget_bytes != cfg_.budget_bytes;
    const bool policy_changed = c.policy != cfg_.policy;
    cfg_ = c;
    freq_        = cfg_.policy == "freq";
    freq_lambda_ = std::log(2.0) / cfg_.freq_half;
    {
        int from = 0;
        double ratio = 1;
        swap_params(cfg_.swap, K_, from, ratio);
        swap_ratio_ = ratio;
        swap_from_.store(from);
    }
    gate_on_.store(cfg_.prefetch == "gate");
    if (policy_changed) {
        // the frequency keys start from the recency order of what is in RAM now
        for (int s = 0; s < S_ && !ent_.empty(); s++) {
            int i = 0;
            for (int q = lru_tail_[(size_t) s]; q >= 0; q = ent_[(size_t) q].prev) ent_[(size_t) q].fkey = freq_lambda_ * (double) tok_seq_ - 3.0 + 1e-3 * i++;
        }
    }

    const size_t n_keys = (size_t) S_ * E_;
    for (size_t k = 0; k < n_keys; k++) pop_[k] = counts.size() == n_keys ? (double) counts[k] : 0.0;
    if (trace_path != trans_src_ || trans_.empty()) build_transitions(trace_path);

    if (resize) {
        quiesce(lk);
        restore_home_pointers();
        free_pool();
        std::string err;
        if (!alloc_pool(err)) RERR("expert store: %s", err.c_str());
    }
    // pinning ("fingerprint" policy): the most used experts of each layer keep a share of its slots
    for (size_t q = 0; q < ent_.size(); q++) {
        Entry & x = ent_[q];
        if (!x.pinned) continue;
        x.pinned = 0;
        if (x.state == READY) lru_push_front((int) q);
    }
    if (cfg_.policy == "fingerprint" && !ent_.empty()) {
        for (int s = 0; s < S_; s++) {
            const int C = ent0_[(size_t) s + 1] - ent0_[(size_t) s];
            const int n_pin = (int) ((double) C * cfg_.pin_frac);
            std::vector<int> order((size_t) E_);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return pop_[(size_t) key_of(s, a)] > pop_[(size_t) key_of(s, b)]; });
            for (int i = 0; i < n_pin && i < E_; i++) {
                const int e = order[(size_t) i];
                if (pop_[(size_t) key_of(s, e)] <= 0) break;
                const int q = where_[(size_t) key_of(s, e)];
                if (q >= 0) { ent_[(size_t) q].pinned = 1; lru_remove(q); }
            }
        }
    }
    cv_work_.notify_all();
}

void ExpertStore::reset_cache() {
    std::unique_lock<std::mutex> lk(mu_);
    if (!finalized_) return;
    quiesce(lk);
    epoch_++;
    for (size_t q = 0; q < ent_.size(); q++) {
        Entry & x = ent_[q];
        if (x.state != READY || x.pinned) continue;
        lru_remove((int) q);
        const int s = slot_of_entry(ent0_, (int) q);
        where_[(size_t) key_of(s, x.expert)] = -1;
        x.gen++;
        x.expert = -1;
        x.state = FREE;
    }
    std::fill(pred_seq_.begin(), pred_seq_.end(), 0);
    std::fill(pred2_seq_.begin(), pred2_seq_.end(), 0);
}

double ExpertStore::preload(bool wait) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!finalized_ || ent_.empty()) return 0;
    // the free slots of every layer get its most used experts (pinned ones first, see configure)
    std::vector<int> loaded;
    std::vector<int> order((size_t) E_);
    for (int s = 0; s < S_ && !debug_no_hook_; s++) {
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return pop_[(size_t) key_of(s, a)] > pop_[(size_t) key_of(s, b)]; });
        const bool pin_layer = cfg_.policy == "fingerprint";
        const int C = ent0_[(size_t) s + 1] - ent0_[(size_t) s];
        const int n_pin = pin_layer ? (int) ((double) C * cfg_.pin_frac) : 0;
        int rank = 0;
        for (int e : order) {
            if (pop_[(size_t) key_of(s, e)] <= 0) break;
            const bool pin = rank++ < n_pin;
            if (where_[(size_t) key_of(s, e)] >= 0) continue;
            int q = -1;
            for (int i = ent0_[(size_t) s]; i < ent0_[(size_t) s + 1]; i++) if (ent_[(size_t) i].state == FREE) { q = i; break; }
            if (q < 0) break;
            assign_entry(q, s, e);
            ent_[(size_t) q].pinned = pin ? 1 : 0;
            // "freq": as if used a while ago, the most used experts a little more
            ent_[(size_t) q].fkey = freq_lambda_ * (double) tok_seq_ - 3.0 - 1e-3 * rank;
            load_entry(q, wait ? 0 : 2);
            loaded.push_back(q);
        }
    }
    cv_work_.notify_all();
    if (!wait || loaded.empty()) return 0;
    const double t0 = now_sec();
    cv_done_.wait(lk, [&] {
        if (stop_) return true;
        for (int q : loaded) if (ent_[(size_t) q].state == LOADING) return false;
        return true;
    });
    // most used experts at the head of each layer's LRU list
    for (auto it = loaded.rbegin(); it != loaded.rend(); ++it) {
        if (ent_[(size_t) *it].state == READY && !ent_[(size_t) *it].pinned) lru_push_front(*it);
    }
    return now_sec() - t0;
}

// -------------------------------------------------------------------------------------------------
// predictions for the next layer
// -------------------------------------------------------------------------------------------------
bool ExpertStore::router_ready(int s) {
    std::lock_guard<std::mutex> g(router_mu_);
    return s >= 0 && s < (int) routers_.size() && routers_[(size_t) s].ok;
}

bool ExpertStore::wants_router(int layer) {
    if (layer < 0 || layer >= (int) slot_of_layer_.size()) return false;
    const int s = slot_of_layer_[(size_t) layer];
    if (s < 0 || routers_ok_.load() >= S_) return false;
    std::lock_guard<std::mutex> g(router_mu_);
    return s < (int) routers_.size() && !routers_[(size_t) s].ok;
}

void ExpertStore::set_router(int layer, const float * gate, int64_t n_embd, int64_t n_exp, const float * norm_w) {
    if (layer < 0 || layer >= (int) slot_of_layer_.size() || !gate) return;
    const int s = slot_of_layer_[(size_t) layer];
    if (s < 0 || n_exp != E_ || n_embd <= 0 || (n_embd_ > 0 && n_embd != n_embd_)) return;
    Router R;
    R.gate.resize((size_t) (n_exp * n_embd));
    for (size_t i = 0; i < R.gate.size(); i++) R.gate[i] = f32_to_f16(gate[i]);
    if (norm_w) R.norm.assign(norm_w, norm_w + n_embd);
    R.ok = true;
    std::lock_guard<std::mutex> g(router_mu_);
    if (s >= (int) routers_.size() || routers_[(size_t) s].ok) return;
    n_embd_ = n_embd;
    routers_[(size_t) s] = std::move(R);
    routers_ok_++;
}

// "gate": the next layer's router applied to this layer's hidden state. Its real input is
// norm(x_next) * w_next with x_next = x + (this layer's FFN) + (next attention); the residual stream changes
// slowly, so norm(x) * w_next is a close guess, and the RMS scale does not change which experts win.
bool ExpertStore::predict_gate(int sn, const float * h, bool normed, int sfrom, int n, std::vector<int> & out) {
    out.clear();
    std::lock_guard<std::mutex> g(router_mu_);
    if (sn < 0 || sn >= S_ || sfrom < 0 || sfrom >= S_ || n_embd_ <= 0) return false;
    const Router & R = routers_[(size_t) sn];
    const Router & F = routers_[(size_t) sfrom];
    if (!R.ok) return false;
    static thread_local std::vector<float> v, logit;
    static thread_local std::vector<int> idx;
    v.resize((size_t) n_embd_);
    for (int64_t j = 0; j < n_embd_; j++) {
        float x = h[j];
        if (normed && !F.norm.empty()) { const float w = F.norm[(size_t) j]; x = std::fabs(w) > 1e-6f ? x / w : 0.0f; }
        if (!R.norm.empty()) x *= R.norm[(size_t) j];
        v[(size_t) j] = x;
    }
    logit.resize((size_t) E_);
#ifdef RAFID_X86
    if (simd_) matvec_f16_simd(R.gate.data(), v.data(), n_embd_, E_, logit.data());
    else
#endif
    matvec_f16_scalar(R.gate.data(), v.data(), n_embd_, E_, logit.data());
    idx.resize((size_t) E_);
    std::iota(idx.begin(), idx.end(), 0);
    const int top = std::min(n, E_);
    std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int a, int b) { return logit[(size_t) a] > logit[(size_t) b]; });
    out.assign(idx.begin(), idx.begin() + top);
    return true;
}

// "history": which experts followed the chosen ones in your fingerprint trace (caller holds mu_)
void ExpertStore::predict_history(int slot_next, const std::vector<int> & chosen, int n, std::vector<int> & out) {
    out.clear();
    std::vector<double> score((size_t) E_, 0.0);
    double popmax = 0;
    for (int b = 0; b < E_; b++) popmax = std::max(popmax, pop_[(size_t) key_of(slot_next, b)]);
    const bool have_trans = !trans_.empty() && slot_next >= 1;
    if (!have_trans && popmax <= 0) return;
    if (have_trans) {
        for (int a : chosen) {
            const uint32_t * row = &trans_[((size_t) (slot_next - 1) * E_ + a) * E_];
            for (int b = 0; b < E_; b++) score[(size_t) b] += row[b];
        }
    }
    if (popmax > 0) {
        for (int b = 0; b < E_; b++) score[(size_t) b] += 1e-3 * pop_[(size_t) key_of(slot_next, b)] / popmax;
    }
    std::vector<int> idx((size_t) E_);
    std::iota(idx.begin(), idx.end(), 0);
    const int top = std::min(n, E_);
    std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int x, int y) { return score[(size_t) x] > score[(size_t) y]; });
    for (int i = 0; i < top; i++) if (score[(size_t) idx[(size_t) i]] > 0) out.push_back(idx[(size_t) i]);
}

// a prompt batch (called without mu_, returns with it): the gate's top experts of every sampled token, as
// one set, gathered into the next layer's slots (when they fit) or its staging buffer
void ExpertStore::run_batch_guess(const PredReq & rq, int sn, std::unique_lock<std::mutex> & lk) {
    std::vector<uint8_t> want((size_t) E_, 0);
    std::vector<int> top;
    bool any = false;
    for (int r = 0; r < rq.rows; r++) {
        if (!predict_gate(sn, rq.hidden.data() + (size_t) r * (size_t) n_embd_, rq.normed, rq.slot, K_, top)) break;
        for (int e : top) want[(size_t) e] = 1;
        any = true;
    }
    lk.lock();
    if (stop_ || !any || ent_.empty()) return;
    if (rq.seq != tok_seq_ || last_slot_ >= sn) { stats_.pred_late++; return; } // the layer is already there
    std::vector<int> list;
    for (int e = 0; e < E_; e++) if (want[(size_t) e]) list.push_back(e);
    const int C = ent0_[(size_t) sn + 1] - ent0_[(size_t) sn];
    if ((int) list.size() + 2 <= C) {
        // fits in the layer's slots: load the missing ones there (the batch will likely compute from the slots)
        for (int e : list) {
            if (where_[(size_t) key_of(sn, e)] >= 0) continue;
            const int q = pick_victim(sn, true, nullptr);
            if (q < 0) break;
            assign_entry(q, sn, e);
            load_entry(q, 2);
            ent_[(size_t) q].prefetched = 1;
            if (freq_) freq_touch(q);
            pf_entries_[(size_t) sn].push_back(q);
            stats_.ahead_issued++;
        }
    } else if (n_stg_ >= 2 && ensure_staging(lk)) {
        const int b2 = sn % n_stg_;
        claim_staging(b2, sn, lk);
        // the wait inside may have let the prompt end (buffers freed) or another batch start
        if (stop_ || n_stg_ < 2 || !stg_[b2].block || rq.seq != tok_seq_ || last_slot_ >= sn || stg_[b2].slot != sn) return;
        size_t n0 = 0;
        for (int e : list) if (stg_[b2].state[(size_t) e] == 0) n0++;
        stats_.ahead_issued += n0;
        stage_experts(b2, sn, list, 2, true);
    }
    cv_work_.notify_all();
}

// runs on its own thread while the current layer waits for the disk and computes
void ExpertStore::predictor_main() {
    PredReq rq;
    std::vector<int> top1, top2;
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_pred_.wait(lk, [&] { return stop_ || pred_has_; });
        if (stop_) return;
        std::swap(rq, pred_req_);
        pred_has_ = false;
        const std::string mode = cfg_.prefetch;
        const int P = cfg_.prefetch_n;
        if (mode == "off") continue;
        const int sn = rq.slot + 1;
        if (sn >= S_) continue;
        if (rq.prompt) {
            lk.unlock();
            run_batch_guess(rq, sn, lk);
            if (stop_) return;
            continue;
        }
        lk.unlock();
        bool gate_ok = false;
        top2.clear();
        if (mode == "gate" && rq.hidden_ok) {
            gate_ok = predict_gate(sn, rq.hidden.data(), rq.normed, rq.slot, std::max(P, K_), top1);
            if (gate_ok && sn + 1 < S_) predict_gate(sn + 1, rq.hidden.data(), rq.normed, rq.slot, K_, top2);
        }
        lk.lock();
        if (stop_) return;
        if (!gate_ok) predict_history(sn, rq.chosen, std::max(P, K_), top1);
        if (rq.seq != tok_seq_ || last_slot_ >= sn) { stats_.pred_late++; continue; } // the layer is already there
        if (top1.empty() || ent_.empty()) continue;
        pred_set_[(size_t) sn].assign(top1.begin(), top1.begin() + std::min<size_t>((size_t) K_, top1.size()));
        pred_seq_[(size_t) sn] = rq.seq;
        if (!top2.empty()) {
            pred2_set_[(size_t) sn + 1].assign(top2.begin(), top2.begin() + std::min<size_t>((size_t) K_, top2.size()));
            pred2_seq_[(size_t) sn + 1] = rq.seq;
        }
        bool issued = false;
        for (int i = 0; i < P && i < (int) top1.size(); i++) {
            const int e = top1[(size_t) i];
            if (where_[(size_t) key_of(sn, e)] >= 0) continue; // there or on its way
            const int q = pick_victim(sn, true, nullptr);
            if (q < 0) break;
            assign_entry(q, sn, e);
            load_entry(q, 2);
            ent_[(size_t) q].prefetched = 1;
            // "freq": a guess counts as a use until its layer is routed (else the next guess would replace it)
            if (freq_) freq_touch(q);
            pf_entries_[(size_t) sn].push_back(q);
            stats_.prefetch_issued++;
            issued = true;
        }
        if (issued) cv_work_.notify_all();
    }
}

// -------------------------------------------------------------------------------------------------
// the hot path: router output of one layer
// -------------------------------------------------------------------------------------------------
bool ExpertStore::on_route(int layer, int32_t * ids, int k, int64_t n_tok, const float * hidden, int64_t hidden_rows,
                           bool hidden_normed, const float * probs) {
    if (debug_no_hook_) return false; // test switch: leave every load to the fault handler
    if (layer < 0 || layer >= (int) slot_of_layer_.size()) return false;
    const int slot = slot_of_layer_[(size_t) layer];
    if (slot < 0) return false;

    // unique experts of this batch
    std::vector<int> & uniq = scratch_ids_;
    std::vector<uint8_t> & seen = scratch_seen_;
    uniq.clear();
    seen.assign((size_t) E_, 0);
    for (int64_t t = 0; t < n_tok; t++) {
        for (int i = 0; i < k; i++) {
            const int e = ids[t * k + i];
            if (e >= 0 && e < E_ && !seen[(size_t) e]) { seen[(size_t) e] = 1; uniq.push_back(e); }
        }
    }

    std::unique_lock<std::mutex> lk(mu_);
    if (!finalized_ || stop_) return false;
    stats_.routed++;
    epoch_++;
    if (slot <= last_slot_) tok_seq_++; // the next token (or batch) starts again at the first MoE layer
    last_slot_ = slot;
    mode_[(size_t) slot] = MODE_NONE;

    // how good were the guesses for this layer
    if (n_tok == 1) {
        if (pred_seq_[(size_t) slot] == tok_seq_) {
            for (int e : pred_set_[(size_t) slot]) stats_.pred_hits += seen[(size_t) e];
            stats_.pred_total += pred_set_[(size_t) slot].size();
        }
        if (pred2_seq_[(size_t) slot] == tok_seq_) {
            for (int e : pred2_set_[(size_t) slot]) stats_.pred2_hits += seen[(size_t) e];
            stats_.pred2_total += pred2_set_[(size_t) slot].size();
        }
    }
    // wrong guesses for this layer: drop the ones not read yet, send the rest to the end of the LRU list
    for (int q : pf_entries_[(size_t) slot]) {
        Entry & x = ent_[(size_t) q];
        if (!x.prefetched || (x.expert >= 0 && seen[(size_t) x.expert])) continue;
        x.prefetched = 0;
        x.fkey = -INFINITY; // "freq": a wrong guess goes first
        if (x.state == LOADING) {
            if (!x.started) cancel_entry(q); else x.demote = 1;
        } else if (x.state == READY && !x.pinned) {
            lru_push_back(q);
        }
    }
    pf_entries_[(size_t) slot].clear();

    const int C = ent_.empty() ? 0 : ent0_[(size_t) slot + 1] - ent0_[(size_t) slot];

    // cache-aware routing: a low-ranked expert that is neither in RAM nor on its way is replaced by the expert
    // in RAM that the router weighs most, if that one gets at least swap_ratio_ of the weight of the first.
    // The routing weights are gathered after this from the new ids, so the replacement gets its own weight.
    bool changed = false;
    const int swap_from = swap_from_.load();
    if (swap_from > 0 && n_tok == 1 && probs && C > 0 && k <= C) {
        for (int i = swap_from - 1; i < k; i++) {
            const int e = ids[i];
            if (e < 0 || e >= E_ || where_[(size_t) key_of(slot, e)] >= 0) continue;
            stats_.swap_candidates++;
            int best = -1;
            float best_p = (float) swap_ratio_ * probs[e];
            for (int q = ent0_[(size_t) slot]; q < ent0_[(size_t) slot + 1]; q++) {
                const Entry & x = ent_[(size_t) q];
                if (x.state != READY || x.expert < 0 || seen[(size_t) x.expert]) continue;
                const float pq = probs[x.expert];
                if (pq >= best_p) { best_p = pq; best = x.expert; }
            }
            if (best < 0) continue;
            ids[i] = best;
            seen[(size_t) e] = 0;
            seen[(size_t) best] = 1;
            stats_.swapped++;
            changed = true;
        }
        if (changed) uniq.assign(ids, ids + k);
    }
    last_ids_[(size_t) slot].assign(ids, ids + n_tok * k);

    // a prompt batch: the predictor thread guesses which experts the next layer will want for these tokens
    // (the gate applied to a sample of them) and gathers them while this layer waits and computes
    if (n_tok > 1 && cfg_.prefetch == "gate" && hidden && hidden_rows > 1 && n_embd_ > 0 && slot + 1 < S_) {
        const int want_rows = (int) std::min<int64_t>(hidden_rows, 128);
        pred_req_.slot = slot;
        pred_req_.seq  = tok_seq_;
        pred_req_.prompt = true;
        pred_req_.rows = want_rows;
        pred_req_.normed = hidden_normed;
        pred_req_.hidden_ok = true;
        pred_req_.chosen.clear();
        pred_req_.hidden.resize((size_t) want_rows * (size_t) n_embd_);
        for (int r = 0; r < want_rows; r++) {
            const int64_t src = want_rows > 1 ? (int64_t) r * (hidden_rows - 1) / (want_rows - 1) : 0; // evenly over the batch
            std::copy(hidden + src * n_embd_, hidden + (src + 1) * n_embd_, pred_req_.hidden.begin() + (size_t) r * (size_t) n_embd_);
        }
        pred_has_ = true;
        cv_pred_.notify_one();
        if (debug_layer_ms_ > 0) {
            lk.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(debug_layer_ms_));
            lk.lock();
        }
    }

    bool to_stage = C == 0 || (int) uniq.size() > C;
    // the read-ahead of a prompt already gathered this layer into its staging buffer: use it (the guess was
    // bigger than the slots, the batch turned out to fit; the staging path costs only copies of the hits)
    if (!to_stage && n_tok > 1 && n_stg_ > 0) {
        const Staging & g = stg_[(size_t) (slot % n_stg_)];
        if (g.slot == slot && !g.claiming) {
            int staged = 0;
            for (int e : uniq) staged += g.state[(size_t) e] != 0;
            if (staged * 2 >= (int) uniq.size()) to_stage = true;
        }
    }
    bool waiting = false;

    if (!to_stage) {
        // ---- decode and small batches: the experts live in the layer's slots
        promote_slot(slot, seen);
        for (int e : uniq) {
            const int q = where_[(size_t) key_of(slot, e)];
            if (q >= 0) ent_[(size_t) q].use_epoch = epoch_;
        }
        std::vector<int> & miss = scratch_miss_;
        miss.clear();
        for (int e : uniq) {
            const int q = where_[(size_t) key_of(slot, e)];
            stats_.accesses++;
            if (q < 0) { miss.push_back(e); continue; }
            Entry & x = ent_[(size_t) q];
            if (x.prefetched) {
                stats_.prefetch_used++;
                if (n_tok > 1) stats_.ahead_used++; // gathered ahead for this prompt batch
                x.prefetched = 0;
                x.fkey = -INFINITY; // the protection of the guess was not a real use
            }
            if (freq_) freq_touch(q);
            if (x.state == READY) {
                stats_.hits++;
                if (!x.pinned) lru_push_front(q);
            } else {
                stats_.inflight++;
                x.demote = 0;
                waiting = true;
            }
        }
        for (int e : miss) {
            stats_.misses++;
            const int q = pick_victim(slot, false, &seen);
            if (q < 0) { stats_.over_budget++; to_stage = true; continue; } // every slot is busy: read it in a staging buffer
            assign_entry(q, slot, e);
            ent_[(size_t) q].use_epoch = epoch_;
            if (freq_) freq_touch(q);
            load_entry(q, 0);
            waiting = true;
        }
        if (waiting) cv_work_.notify_all();
        if (n_tok == 1 && cfg_.prefetch != "off" && slot + 1 < S_) {
            // the guess for the next layer is made on the predictor thread while this layer waits and computes
            pred_req_.slot = slot;
            pred_req_.seq  = tok_seq_;
            pred_req_.prompt = false;
            pred_req_.rows = 1;
            pred_req_.chosen.assign(uniq.begin(), uniq.end());
            pred_req_.hidden_ok = hidden != nullptr && n_embd_ > 0;
            pred_req_.normed = hidden_normed;
            if (pred_req_.hidden_ok) pred_req_.hidden.assign(hidden, hidden + n_embd_);
            pred_has_ = true;
            cv_pred_.notify_one();
        }
        if (waiting && !to_stage) {
            const double t0 = now_sec();
            cv_done_.wait(lk, [&] {
                if (stop_) return true;
                for (int e : uniq) {
                    const int q = where_[(size_t) key_of(slot, e)];
                    if (q < 0 || ent_[(size_t) q].state != READY) return false;
                }
                return true;
            });
            stats_.wait_sec += now_sec() - t0;
        }
        if (stop_) return changed;
        if (!to_stage && remap_ok_[(size_t) slot]) {
            // compute straight from the slots; on_weights rewrites the ids to slot numbers
            for (int e : uniq) remap_[(size_t) key_of(slot, e)] = (int16_t) (where_[(size_t) key_of(slot, e)] - ent0_[(size_t) slot]);
            mode_[(size_t) slot] = MODE_POOL;
            set_pointers(slot, MODE_POOL, -1);
            stats_.routes_pool++;
            return changed;
        }
        // not confirmed yet that the ids can be rewritten for this layer (first step after loading): this one
        // step is computed from a staging buffer with the original ids
    } else {
        // ---- prompt batches: more experts than the layer has slots
        for (int e : uniq) {
            const int q = where_[(size_t) key_of(slot, e)];
            stats_.accesses++;
            if (q >= 0 && ent_[(size_t) q].state == READY) { stats_.hits++; ent_[(size_t) q].use_epoch = epoch_; }
            else stats_.misses++;
        }
    }

    // ---- staging buffer (identity ids)
    if (!ensure_staging(lk)) {
        // no memory for a staging buffer: read what is needed into the home memory (never released)
        lk.unlock();
        for (int e : uniq) read_home_sync(slot, e);
        lk.lock();
        set_pointers(slot, MODE_NONE, -1);
        return changed;
    }
    const int b = slot % n_stg_;
    claim_staging(b, slot, lk);
    std::vector<int> & list = scratch_list_;
    list.assign(uniq.begin(), uniq.end());
    std::sort(list.begin(), list.end());
    // reads queued for this buffer (gathered ahead while the previous layer computed) become urgent
    promote_slot(slot, seen);
    for (int e : list) if (stg_[b].state[(size_t) e] != 0) stats_.ahead_used++;
    stage_experts(b, slot, list, 0, true);
    // without the gate's guesses (other modes, or its router not seen yet in this session): a big batch needs
    // nearly every expert, so gather the whole next layer while this one computes (the guesses do better: the
    // tokens of one prompt share most of their experts)
    if (n_tok >= bulk_tok_ && n_stg_ >= 2 && slot + 1 < S_ && (cfg_.prefetch != "gate" || !router_ready(slot + 1))) {
        const int b2 = (slot + 1) % n_stg_;
        claim_staging(b2, slot + 1, lk);
        std::vector<int> all;
        for (int e = 0; e < E_; e++) if (stg_[b2].state[(size_t) e] == 0) all.push_back(e);
        stats_.bulk_issued += all.size();
        stage_experts(b2, slot + 1, all, 2, true);
    }
    cv_work_.notify_all();
    const double t0 = now_sec();
    cv_done_.wait(lk, [&] {
        if (stop_) return true;
        for (int e : uniq) if (stg_[b].state[(size_t) e] != READY) return false;
        return true;
    });
    stats_.wait_sec += now_sec() - t0;
    if (stop_) return changed;
    mode_[(size_t) slot] = MODE_STAGE;
    set_pointers(slot, MODE_STAGE, b);
    stats_.routes_staged++;
    return changed;
}

// after the routing weights were taken with the real ids: slot numbers for the expert matmuls
bool ExpertStore::on_weights(int layer, int32_t * ids, int64_t n) {
    if (debug_no_hook_) return false;
    if (layer < 0 || layer >= (int) slot_of_layer_.size()) return false;
    const int slot = slot_of_layer_[(size_t) layer];
    if (slot < 0) return false;
    std::unique_lock<std::mutex> lk(mu_);
    if (!finalized_ || stop_) return false;
    const std::vector<int32_t> & seen = last_ids_[(size_t) slot];
    const bool same = ids && n > 0 && (size_t) n == seen.size() && std::equal(ids, ids + n, seen.begin());
    const uint8_t mode = mode_[(size_t) slot];
    if (!same) {
        if (!warned_remap_) {
            RWARN("expert store: the routing ids of layer %d look different at the weights step; that layer computes from a staging buffer", layer);
            warned_remap_ = true;
        }
        remap_ok_[(size_t) slot] = 0;
        if (mode == MODE_POOL && ids && n > 0 && ensure_staging(lk)) {
            // cannot apply slot numbers to these ids: gather what they need in a staging buffer right now
            std::vector<uint8_t> need((size_t) E_, 0);
            std::vector<int> list;
            for (int64_t i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < E_ && !need[(size_t) ids[i]]) { need[(size_t) ids[i]] = 1; list.push_back(ids[i]); }
            std::sort(list.begin(), list.end());
            const int b = slot % n_stg_;
            claim_staging(b, slot, lk);
            stage_experts(b, slot, list, 1, true);
            cv_work_.notify_all();
            cv_done_.wait(lk, [&] {
                if (stop_) return true;
                for (int e : list) if (stg_[b].state[(size_t) e] != READY) return false;
                return true;
            });
            mode_[(size_t) slot] = MODE_STAGE;
            set_pointers(slot, MODE_STAGE, b);
        }
        return false;
    }
    if (mode == MODE_POOL) {
        for (int64_t i = 0; i < n; i++) {
            if (ids[i] >= 0 && ids[i] < E_) ids[i] = remap_[(size_t) key_of(slot, ids[i])];
        }
        return true;
    }
    if (remap_safe_) remap_ok_[(size_t) slot] = 1; // the ids can be rewritten here: next steps compute straight from the slots
    return false;
}

// -------------------------------------------------------------------------------------------------
// safety net for accesses to the home memory (Windows fault handler)
// -------------------------------------------------------------------------------------------------
bool ExpertStore::contains(uintptr_t addr) const {
    for (const Region * r : regions_) {
        if (r->base && addr >= (uintptr_t) r->base && addr < (uintptr_t) r->base + r->size) return true;
    }
    return false;
}

bool ExpertStore::handle_fault(uintptr_t addr) {
    MTensor * m = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!finalized_) return false;
        m = tensor_at_home(addr);
        if (!m || !m->managed) {
            // padding or an unknown page of the region: give it a zero page
            if (!contains(addr)) return false;
            return vm_commit((uint8_t *) floor_to((uint64_t) addr, PAGE), PAGE);
        }
        stats_.fault_loads++;
    }
    fault_count_++;
    const int64_t off = (int64_t) addr - (int64_t) (uintptr_t) m->home;
    const int e = off < 0 ? 0 : (int) std::min<int64_t>(off / (int64_t) m->nb2, m->n_exp - 1);
    if (page_committed(addr)) return true; // another thread loaded it meanwhile
    if (!read_home_sync(m->slot, e)) RWARN("expert store: safety-net read of expert %d of layer %d failed", e, m->layer);
    return true;
}

// -------------------------------------------------------------------------------------------------
// reporting
// -------------------------------------------------------------------------------------------------
StreamStats ExpertStore::stats() {
    std::lock_guard<std::mutex> lk(mu_);
    StreamStats s = stats_;
    if (io_active_ > 0) s.io_busy_sec += now_sec() - io_busy_start_;
    return s;
}

StreamConfig ExpertStore::config() {
    std::lock_guard<std::mutex> lk(mu_);
    return cfg_;
}

nlohmann::json ExpertStore::status() {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t pinned_bytes = 0, pinned_ready = 0, n_res = 0, res_bytes = 0;
    size_t n_pinned = 0;
    for (size_t q = 0; q < ent_.size(); q++) {
        const Entry & x = ent_[q];
        const int s = slot_of_entry(ent0_, (int) q);
        if (x.state == READY) { n_res++; res_bytes += key_bytes_[(size_t) s]; }
        if (x.pinned) {
            n_pinned++;
            pinned_bytes += key_bytes_[(size_t) s];
            if (x.state == READY) pinned_ready += key_bytes_[(size_t) s];
        }
    }
    int remap_ready = 0;
    for (uint8_t v : remap_ok_) remap_ready += v;
    const int slots = ent0_.size() > 1 ? ent0_[1] - ent0_[0] : 0;
    StreamStats st = stats_;
    if (io_active_ > 0) st.io_busy_sec += now_sec() - io_busy_start_;
    return {
        { "active", finalized_ }, { "budget", cfg_.budget_bytes }, { "min_budget", min_budget() },
        { "policy", cfg_.policy }, { "pin_frac", cfg_.pin_frac }, { "prefetch", cfg_.prefetch }, { "prefetch_n", cfg_.prefetch_n },
        { "freq_half", cfg_.freq_half }, { "swap", cfg_.swap }, { "swap_from_rank", swap_from_.load() }, { "swap_ratio", swap_ratio_ },
        { "total_bytes", total_bytes_ }, { "resident_bytes", res_bytes }, { "resident_experts", n_res },
        { "pool_bytes", pool_size_ }, { "pool_locked", pool_locked_ }, { "slots_per_layer", slots },
        { "staging_bytes", stg_bytes_ * (size_t) n_stg_ }, { "staging_buffers", n_stg_ }, { "staging_locked", stg_locked_ },
        { "pinned_experts", n_pinned }, { "pinned_bytes", pinned_bytes }, { "pinned_ready", pinned_ready },
        { "n_moe_layers", S_ }, { "n_expert", E_ }, { "n_expert_used", K_ },
        { "io_threads", n_io_ }, { "direct_io", direct_io_ }, { "chunk_bytes", chunk_small_ },
        { "has_transitions", !trans_.empty() }, { "routers_ready", routers_ok_.load() }, { "simd", simd_ },
        { "remap_ready", remap_ready },
        { "stats", st.to_json() },
    };
}

} // namespace rafid
