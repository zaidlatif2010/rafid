#include "util.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <regex>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <climits>
#endif

namespace fs = std::filesystem;

namespace rafid {

void keep_awake(bool on) {
#ifdef _WIN32
    SetThreadExecutionState(on ? (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) : ES_CONTINUOUS);
#else
    (void) on;
#endif
}

double now_sec() {
    using clk = std::chrono::steady_clock;
    static const clk::time_point t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

std::string fmt(const char * f, ...) {
    va_list ap;
    va_start(ap, f);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    std::string out;
    if (n > 0) {
        out.resize((size_t) n + 1);
        vsnprintf(&out[0], out.size(), f, ap2);
        out.resize((size_t) n);
    }
    va_end(ap2);
    return out;
}

std::string human_bytes(uint64_t b) {
    const double d = (double) b;
    if (d >= 1024.0 * 1024 * 1024) return fmt("%.1f GB", d / (1024.0 * 1024 * 1024));
    if (d >= 1024.0 * 1024)        return fmt("%.0f MB", d / (1024.0 * 1024));
    if (d >= 1024.0)               return fmt("%.0f KB", d / 1024.0);
    return fmt("%llu B", (unsigned long long) b);
}

size_t utf8_complete_prefix(const std::string & s) {
    const size_t n = s.size();
    if (n == 0) return 0;
    // walk back over at most 3 continuation bytes to the lead byte of the last character
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char) s[i - 1];
        if ((c & 0xC0) != 0x80) break;
        i--; back++;
    }
    if (i == 0) return n; // only continuation bytes: invalid, let the JSON layer replace them
    const unsigned char lead = (unsigned char) s[i - 1];
    size_t need = 1;
    if      ((lead & 0x80) == 0x00) need = 1;
    else if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    else return n; // invalid lead byte
    const size_t have = n - (i - 1);
    return have >= need ? n : i - 1;
}

std::string sanitize_name(const std::string & s) {
    std::string out;
    for (char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
        out += ok ? c : '_';
    }
    if (out.size() > 120) out.resize(120);
    if (out.empty()) out = "model";
    return out;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), &w[0], n);
    return w;
}
std::string wide_to_utf8(const std::wstring & w) {
    if (w.empty()) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), &s[0], n, nullptr, nullptr);
    return s;
}
#endif

static fs::path to_path(const std::string & s) {
    return fs::u8path(s);
}

static std::string from_path(const fs::path & p) {
    return p.u8string();
}

std::string exe_dir() {
#ifdef _WIN32
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, &buf[0], (DWORD) buf.size());
    buf.resize(n);
    return from_path(fs::path(buf).parent_path());
#else
    char buf[PATH_MAX + 1] = {0};
    const ssize_t n = readlink("/proc/self/exe", buf, PATH_MAX);
    if (n > 0) {
        buf[n] = 0;
        return from_path(fs::path(buf).parent_path());
    }
    std::error_code ec;
    return from_path(fs::current_path(ec));
#endif
}

std::string path_join(const std::string & a, const std::string & b) {
    return from_path(to_path(a) / to_path(b));
}

std::string file_name(const std::string & p) {
    return from_path(to_path(p).filename());
}

std::string parent_dir(const std::string & p) {
    return from_path(to_path(p).parent_path());
}

std::string home_dir() {
#ifdef _WIN32
    const wchar_t * h = _wgetenv(L"USERPROFILE");
    return h ? wide_to_utf8(h) : std::string();
#else
    const char * h = getenv("HOME");
    return h ? std::string(h) : std::string();
#endif
}

std::string expand_user(const std::string & p) {
    if (p.empty() || p[0] != '~') return p;
    if (p.size() > 1 && p[1] != '/' && p[1] != '\\') return p;
    const std::string h = home_dir();
    if (h.empty()) return p;
    return p.size() <= 2 ? h : path_join(h, p.substr(2));
}

std::string lmstudio_models_dir() {
    const std::string h = home_dir();
    if (h.empty()) return "";
    const std::string d = path_join(path_join(h, ".lmstudio"), "models");
    return is_directory(d) ? d : "";
}

bool file_exists(const std::string & p) {
    std::error_code ec;
    return fs::is_regular_file(to_path(p), ec);
}

bool is_directory(const std::string & p) {
    std::error_code ec;
    return fs::is_directory(to_path(p), ec);
}

uint64_t file_size(const std::string & p) {
    std::error_code ec;
    const auto s = fs::file_size(to_path(p), ec);
    return ec ? 0 : (uint64_t) s;
}

bool make_dirs(const std::string & p) {
    std::error_code ec;
    fs::create_directories(to_path(p), ec);
    return fs::is_directory(to_path(p), ec);
}

FILE * fopen_utf8(const std::string & path, const char * mode) {
#ifdef _WIN32
    return _wfopen(utf8_to_wide(path).c_str(), utf8_to_wide(mode).c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

bool read_text_file(const std::string & path, std::string & out) {
    FILE * f = fopen_utf8(path, "rb");
    if (!f) return false;
    out.clear();
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return true;
}

bool write_text_file(const std::string & path, const std::string & data) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen_utf8(tmp, "wb");
    if (!f) return false;
    const bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    if (fclose(f) != 0 || !ok) return false;
#ifdef _WIN32
    return MoveFileExW(utf8_to_wide(tmp).c_str(), utf8_to_wide(path).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
}

static bool ends_with_icase(const std::string & s, const std::string & suf);

std::vector<std::string> list_files(const std::string & dir, const std::string & ext) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::is_directory(to_path(dir), ec)) return out;
    for (fs::directory_iterator it(to_path(dir), fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code ec2;
        if (!it->is_regular_file(ec2)) continue;
        const std::string name = from_path(it->path().filename());
        if (!ext.empty() && !ends_with_icase(name, ext)) continue;
        out.push_back(name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool parse_split_name(const std::string & path, std::string & prefix, int & no, int & count) {
    static const std::regex re("^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(path, m, re)) return false;
    prefix = m[1].str();
    no     = std::stoi(m[2].str());
    count  = std::stoi(m[3].str());
    return no >= 1 && count >= 1 && no <= count;
}

std::string make_split_name(const std::string & prefix, int no, int count) {
    return prefix + fmt("-%05d-of-%05d.gguf", no, count);
}

static bool ends_with_icase(const std::string & s, const std::string & suf) {
    if (s.size() < suf.size()) return false;
    for (size_t i = 0; i < suf.size(); i++) {
        if (tolower((unsigned char) s[s.size() - suf.size() + i]) != tolower((unsigned char) suf[i])) return false;
    }
    return true;
}

std::vector<GgufFileEntry> list_gguf_models(const std::string & dir) {
    std::vector<GgufFileEntry> out;
    std::error_code ec;
    if (!fs::is_directory(to_path(dir), ec)) return out;

    fs::recursive_directory_iterator it(to_path(dir), fs::directory_options::skip_permission_denied, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        if (it.depth() > 4) { it.disable_recursion_pending(); continue; }
        std::error_code ec2;
        if (!it->is_regular_file(ec2)) continue;
        const std::string p = from_path(it->path());
        if (!ends_with_icase(p, ".gguf")) continue;
        std::string lname = from_path(it->path().filename());
        std::transform(lname.begin(), lname.end(), lname.begin(), [](unsigned char c) { return (char) tolower(c); });
        if (lname.find("mmproj") != std::string::npos) continue; // vision projectors are not chat models

        GgufFileEntry e;
        std::string prefix;
        int no = 0, count = 0;
        if (parse_split_name(p, prefix, no, count)) {
            if (no != 1) continue;
            e.path = p;
            e.n_files = count;
            for (int i = 1; i <= count; i++) e.bytes += file_size(make_split_name(prefix, i, count));
            e.name = from_path(fs::u8path(prefix).filename()) + fmt(" (%d parts)", count);
        } else {
            e.path  = p;
            e.bytes = file_size(p);
            e.name  = from_path(it->path().filename());
        }
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(), [](const GgufFileEntry & a, const GgufFileEntry & b) { return a.name < b.name; });
    return out;
}

// ---- logs --------------------------------------------------------------------

LogRing & logs() {
    static LogRing ring;
    return ring;
}

void LogRing::add(int level, const char * text) {
    if (!text) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (level != 5 /* GGML_LOG_LEVEL_CONT */) partial_level_ = level;
    partial_ += text;
    size_t pos;
    while ((pos = partial_.find('\n')) != std::string::npos) {
        std::string line = partial_.substr(0, pos);
        partial_.erase(0, pos + 1);
        if (verbose_ || partial_level_ >= 3) {
            fprintf(stderr, "%s\n", line.c_str());
        }
        const char * tag = partial_level_ >= 4 ? "E " : partial_level_ == 3 ? "W " : "  ";
        lines_.push_back(tag + line);
        while (lines_.size() > 600) lines_.pop_front();
    }
}

void LogRing::addf(int level, const char * f, ...) {
    va_list ap;
    va_start(ap, f);
    char buf[2048];
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    std::string s = buf;
    while (!s.empty() && s.back() == '\n') s.pop_back();

    // rafid's own messages always reach the console
    fprintf(stderr, "[rafid] %s\n", s.c_str());

    std::lock_guard<std::mutex> lk(mu_);
    const char * tag = level >= 4 ? "E " : level == 3 ? "W " : "  ";
    lines_.push_back(std::string(tag) + "[rafid] " + s);
    while (lines_.size() > 600) lines_.pop_front();
}

std::vector<std::string> LogRing::tail(size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    const size_t start = lines_.size() > n ? lines_.size() - n : 0;
    for (size_t i = start; i < lines_.size(); i++) out.push_back(lines_[i]);
    return out;
}

} // namespace rafid

// ------------------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4): test suites are identified by the hash of their file, so whoever wrote one can check
// that the questions that were run are exactly theirs
namespace rafid {

std::string sha256_hex(const std::string & data) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    std::string m = data;
    const uint64_t bits = (uint64_t) data.size() * 8;
    m.push_back((char) 0x80);
    while (m.size() % 64 != 56) m.push_back('\0');
    for (int i = 7; i >= 0; i--) m.push_back((char) ((bits >> (i * 8)) & 0xff));
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            const unsigned char * p = (const unsigned char *) m.data() + off + i * 4;
            w[i] = ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) | ((uint32_t) p[2] << 8) | (uint32_t) p[3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h) for (int i = 28; i >= 0; i -= 4) out.push_back(hex[(v >> i) & 15]);
    return out;
}

} // namespace rafid
