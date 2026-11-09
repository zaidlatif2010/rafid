// رافد - hardware probe: what the machine has and how fast its memories really are
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml-backend.h"

namespace rafid {

struct GpuInfo {
    int         dev_index = -1;   // index in ggml_backend_dev_get()
    std::string name;             // backend device name, e.g. CUDA0
    std::string description;      // e.g. NVIDIA GeForce RTX 5060 Laptop GPU
    bool        integrated = false;
    uint64_t    vram_total = 0;
    uint64_t    vram_free  = 0;
    double      bw_gbps    = 0;   // measured device memory bandwidth (read+write copy)
    double      h2d_gbps   = 0;   // measured host -> device copy bandwidth (PCIe)
};

struct HwInfo {
    std::string os;
    std::string cpu_name;
    int  logical_cpus   = 0;
    int  physical_cores = 0;
    int  p_cores        = 0;      // performance cores (hybrid CPUs)
    int  e_cores        = 0;      // efficiency cores (hybrid CPUs)
    bool avx2 = false, avx512 = false, avx_vnni = false, fma = false, f16c = false;
    uint64_t ram_total = 0;
    uint64_t ram_avail = 0;
    double   ram_bw_gbps = 0;     // measured multi-threaded read bandwidth
    double   disk_gbps   = 0;     // measured sequential read of a model file (0 = not measured)
    std::string disk_path;
    // scattered 256 KB unbuffered reads (what the expert store does): one thread, 16 threads, and 16 threads
    // while the file is also memory-mapped (Windows/NTFS runs those one at a time)
    double   disk_rand1 = 0, disk_rand16 = 0, disk_rand16_mapped = 0;
    std::vector<GpuInfo> gpus;    // discrete GPUs first
    bool benchmarked = false;

    int best_gpu() const;         // index into gpus, -1 if none
};

HwInfo probe_hardware();                  // fast, no benchmarks
void   refresh_memory(HwInfo & hw);       // re-read free RAM / VRAM
void   refresh_ram(HwInfo & hw);          // re-read free RAM only (cheap)
// RAM that Windows / Linux and the GPU driver keep for themselves, whatever "available" says (part of
// "available" is file cache, and pages of a cache that could not be locked show up there too)
inline uint64_t os_ram_reserve(uint64_t ram_total) { return std::max<uint64_t>(3ull << 30, ram_total / 8); }
void   run_benchmarks(HwInfo & hw);       // RAM + GPU bandwidth, a few seconds
double measure_disk_read(const std::string & path, uint64_t max_bytes, std::string & err);
// GB/s of random unbuffered reads of `block` bytes by `threads` threads for about `seconds`;
// mapped = keep a read-only mapping of the whole file open meanwhile
double measure_disk_random(const std::string & path, size_t block, int threads, double seconds, bool mapped, std::string & err);

ggml_backend_dev_t gpu_device(const HwInfo & hw, int gpu_slot);
nlohmann::json     hw_to_json(const HwInfo & hw);
// the disk measurement survives restarts (it needs a model file and takes a few seconds)
void save_disk_measurement(const std::string & file, const HwInfo & hw);
bool load_disk_measurement(const std::string & file, HwInfo & hw);

// default bandwidths used when a measurement is missing (GB/s)
constexpr double DEFAULT_RAM_BW  = 50.0;
constexpr double DEFAULT_GPU_BW  = 300.0;
constexpr double DEFAULT_PCIE_BW = 20.0;
constexpr double DEFAULT_DISK_BW = 2.5;

} // namespace rafid
