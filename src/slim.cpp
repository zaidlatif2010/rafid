#include "slim.h"
#include "util.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <winioctl.h>
#endif

namespace rafid {

#ifdef _WIN32
namespace {

constexpr uint64_t PAGE  = 4096;
constexpr DWORD    CHUNK = 8u << 20;

struct Range { uint64_t a, b; };

bool env_on(const char * name) {
    const char * v = getenv(name);
    return v && *v && *v != '0';
}

uint64_t mtime_of(const std::wstring & w) {
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExW(w.c_str(), GetFileExInfoStandard, &d)) return 0;
    return ((uint64_t) d.ftLastWriteTime.dwHighDateTime << 32) | d.ftLastWriteTime.dwLowDateTime;
}

bool volume_supports_sparse(const std::wstring & dir) {
    wchar_t root[MAX_PATH + 1] = { 0 };
    if (!GetVolumePathNameW(dir.c_str(), root, MAX_PATH)) return false;
    DWORD flags = 0;
    if (!GetVolumeInformationW(root, nullptr, 0, nullptr, nullptr, &flags, nullptr, 0)) return false;
    return (flags & FILE_SUPPORTS_SPARSE_FILES) != 0;
}

uint64_t free_space(const std::wstring & dir) {
    ULARGE_INTEGER avail;
    if (!GetDiskFreeSpaceExW(dir.c_str(), &avail, nullptr, nullptr)) return 0;
    return avail.QuadPart;
}

bool read_at(HANDLE h, uint64_t off, uint8_t * buf, DWORD len, DWORD & got) {
    OVERLAPPED ov = {};
    ov.Offset     = (DWORD) (off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) (off >> 32);
    got = 0;
    if (!ReadFile(h, buf, len, &got, &ov)) return GetLastError() == ERROR_HANDLE_EOF;
    return true;
}

bool write_at(HANDLE h, uint64_t off, const uint8_t * buf, DWORD len) {
    OVERLAPPED ov = {};
    ov.Offset     = (DWORD) (off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) (off >> 32);
    DWORD put = 0;
    return WriteFile(h, buf, len, &put, &ov) && put == len;
}

nlohmann::json meta_of(const std::string & src, uint64_t size, uint64_t mtime, size_t n_holes, uint64_t hole_bytes) {
    return { { "version", 1 }, { "source", src }, { "size", size }, { "mtime", mtime }, { "holes", n_holes }, { "hole_bytes", hole_bytes } };
}

} // namespace
#endif

std::string prepare_slim_model(const ModelInfo & mi, const std::string & dir,
                               const std::function<bool(double)> & progress, std::string & err) {
#ifndef _WIN32
    (void) mi; (void) dir; (void) progress;
    err = "not needed on this system";
    return "";
#else
    if (env_on("RAFID_NO_SLIM")) { err = "disabled by RAFID_NO_SLIM"; return ""; }
    if (mi.files.empty() || mi.exps_loc.empty()) { err = "the model has no expert tensors"; return ""; }
    if (!make_dirs(dir)) { err = "cannot create " + dir; return ""; }
    const std::wstring wdir = utf8_to_wide(dir);
    const bool force = env_on("RAFID_SLIM_FORCE");
    if (!force && !volume_supports_sparse(wdir)) { err = "the drive of " + dir + " has no sparse files (not NTFS?)"; return ""; }

    // holes: the routed experts of the layers that decode, shrunk to whole pages
    const size_t nf = mi.files.size();
    std::vector<std::vector<Range>> holes(nf);
    for (const auto & kv : mi.exps_loc) {
        const std::string & name = kv.first;
        const int layer = name.rfind("blk.", 0) == 0 ? std::atoi(name.c_str() + 4) : -1;
        if (layer < 0 || layer >= mi.n_layer_used()) continue; // MTP layers keep their data
        const TensorLoc & t = kv.second;
        if (t.file < 0 || (size_t) t.file >= nf) continue;
        const uint64_t a = (t.offset + PAGE - 1) / PAGE * PAGE;
        const uint64_t b = (t.offset + t.nbytes) / PAGE * PAGE;
        if (b > a) holes[(size_t) t.file].push_back({ a, b });
    }
    std::vector<uint64_t> sizes(nf), mtimes(nf), hole_bytes(nf, 0);
    std::vector<std::string> outs(nf);
    std::vector<bool> valid(nf, false);
    uint64_t to_copy = 0;
    for (size_t i = 0; i < nf; i++) {
        auto & h = holes[i];
        std::sort(h.begin(), h.end(), [](const Range & x, const Range & y) { return x.a < y.a; });
        std::vector<Range> merged;
        for (const Range & r : h) {
            if (!merged.empty() && r.a <= merged.back().b) merged.back().b = std::max(merged.back().b, r.b);
            else merged.push_back(r);
        }
        h.swap(merged);
        for (const Range & r : h) hole_bytes[i] += r.b - r.a;
        sizes[i]  = file_size(mi.files[i]);
        mtimes[i] = mtime_of(utf8_to_wide(mi.files[i]));
        outs[i]   = path_join(dir, file_name(mi.files[i]));
        std::string meta_text;
        nlohmann::json meta;
        if (file_exists(outs[i]) && file_size(outs[i]) == sizes[i] && read_text_file(outs[i] + ".rafid.json", meta_text)) {
            try { meta = nlohmann::json::parse(meta_text); } catch (...) { meta = nullptr; }
            valid[i] = meta == meta_of(mi.files[i], sizes[i], mtimes[i], h.size(), hole_bytes[i]);
        }
        if (!valid[i]) to_copy += sizes[i] - hole_bytes[i];
    }
    if (to_copy == 0) return outs[0];

    if (free_space(wdir) < to_copy + (1ull << 30)) {
        err = fmt("not enough free space in %s for the slim copy (%s)", dir.c_str(), human_bytes(to_copy).c_str());
        return "";
    }
    RLOG("expert store: writing a slim copy of the model without the routed experts (%s, once) to %s",
         human_bytes(to_copy).c_str(), dir.c_str());
    uint8_t * buf = (uint8_t *) VirtualAlloc(nullptr, CHUNK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!buf) { err = "out of memory"; return ""; }
    uint64_t done = 0;
    bool ok = true;
    for (size_t i = 0; i < nf && ok; i++) {
        if (valid[i]) continue;
        const std::wstring wsrc = utf8_to_wide(mi.files[i]);
        const std::wstring wtmp = utf8_to_wide(outs[i] + ".part");
        // the original is read unbuffered so that it does not land in the file cache either
        HANDLE src = CreateFileW(wsrc.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                 OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        HANDLE dst = CreateFileW(wtmp.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (src == INVALID_HANDLE_VALUE || dst == INVALID_HANDLE_VALUE) {
            err = "cannot open the model file or create the copy";
            ok = false;
        }
        if (ok) {
            DWORD br = 0;
            if (!DeviceIoControl(dst, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &br, nullptr) && !force) {
                err = "cannot make the copy sparse";
                ok = false;
            }
        }
        if (ok) {
            LARGE_INTEGER li;
            li.QuadPart = (LONGLONG) sizes[i];
            ok = SetFilePointerEx(dst, li, nullptr, FILE_BEGIN) && SetEndOfFile(dst);
            if (!ok) err = "cannot size the copy";
        }
        // copy everything between the holes
        std::vector<Range> keep;
        uint64_t pos = 0;
        for (const Range & h : holes[i]) {
            if (h.a > pos) keep.push_back({ pos, h.a });
            pos = std::max(pos, h.b);
        }
        if (pos < sizes[i]) keep.push_back({ pos, sizes[i] });
        for (const Range & r : keep) {
            for (uint64_t x = r.a; x < r.b && ok; x += CHUNK) {
                const uint64_t want = std::min<uint64_t>(CHUNK, r.b - x);
                const DWORD len = (DWORD) ((want + PAGE - 1) / PAGE * PAGE); // unbuffered reads take whole pages
                DWORD got = 0;
                if (!read_at(src, x, buf, len, got) || got < std::min<uint64_t>(want, sizes[i] - x)) {
                    err = "read error while copying the model";
                    ok = false;
                    break;
                }
                if (!write_at(dst, x, buf, (DWORD) std::min<uint64_t>(want, got))) {
                    err = "write error while copying the model (disk full?)";
                    ok = false;
                    break;
                }
                done += want;
                if (progress && !progress((double) done / (double) to_copy)) {
                    err = "cancelled";
                    ok = false;
                }
            }
            if (!ok) break;
        }
        if (src != INVALID_HANDLE_VALUE) CloseHandle(src);
        if (dst != INVALID_HANDLE_VALUE) {
            if (ok) FlushFileBuffers(dst);
            CloseHandle(dst);
        }
        if (ok) {
            ok = MoveFileExW(wtmp.c_str(), utf8_to_wide(outs[i]).c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
            if (!ok) err = "cannot rename the copy";
        }
        if (ok) {
            FILE * f = fopen_utf8(outs[i] + ".rafid.json", "wb");
            if (f) {
                const std::string s = meta_of(mi.files[i], sizes[i], mtimes[i], holes[i].size(), hole_bytes[i]).dump(2);
                fwrite(s.data(), 1, s.size(), f);
                fclose(f);
            }
        } else {
            DeleteFileW(wtmp.c_str());
        }
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    if (!ok) return "";
    return outs[0];
#endif
}

} // namespace rafid
