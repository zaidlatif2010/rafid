// رافد - Rafid: small shared helpers (paths, time, UTF-8, logging)
#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace rafid {

// ---- time -------------------------------------------------------------------
double now_sec(); // monotonic clock, seconds
// while on, this thread keeps the computer from going to sleep when idle (Windows; closing the lid still does)
void keep_awake(bool on);

// ---- text -------------------------------------------------------------------
std::string human_bytes(uint64_t bytes);           // "12.4 GB"
std::string fmt(const char * f, ...);              // printf into std::string
size_t      utf8_complete_prefix(const std::string & s); // length of the prefix made of complete UTF-8 characters
std::string sanitize_name(const std::string & s);  // safe file name

// ---- files (all paths are UTF-8, also on Windows) ----------------------------
std::string exe_dir();
std::string path_join(const std::string & a, const std::string & b);
std::string file_name(const std::string & p);      // "c:/x/model.gguf" -> "model.gguf"
std::string parent_dir(const std::string & p);     // "c:/x/bin" -> "c:/x"
std::string home_dir();                            // %USERPROFILE% on Windows, $HOME elsewhere
std::string expand_user(const std::string & p);    // "~/x" -> "<home>/x"
std::string lmstudio_models_dir();                 // LM Studio's default models folder, "" if missing
bool        file_exists(const std::string & p);
bool        is_directory(const std::string & p);
uint64_t    file_size(const std::string & p);
bool        make_dirs(const std::string & p);
FILE *      fopen_utf8(const std::string & path, const char * mode);
bool        read_text_file(const std::string & path, std::string & out);
// writes through a temporary file renamed over the old one, so a reader never sees half a file
bool        write_text_file(const std::string & path, const std::string & data);
// SHA-256 of bytes, as 64 lowercase hex digits (the same as sha256sum or certutil -hashfile X SHA256)
std::string sha256_hex(const std::string & data);
// names (not paths) of the regular files in dir ending with ext ("" for all), sorted
std::vector<std::string> list_files(const std::string & dir, const std::string & ext);

struct GgufFileEntry {
    std::string path;      // first split
    std::string name;      // display name
    uint64_t    bytes = 0; // all splits together
    int         n_files = 1;
};
std::vector<GgufFileEntry> list_gguf_models(const std::string & dir);

// "<prefix>-00001-of-00005.gguf" -> true, prefix, 1, 5
bool parse_split_name(const std::string & path, std::string & prefix, int & no, int & count);
std::string make_split_name(const std::string & prefix, int no, int count);

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string & s);
std::string  wide_to_utf8(const std::wstring & w);
#endif

// ---- log ring: keeps the last lines of llama.cpp / rafid logs for the UI ----
class LogRing {
public:
    void add(int level, const char * text);   // accepts partial lines
    void addf(int level, const char * f, ...);
    std::vector<std::string> tail(size_t n);
    void set_verbose(bool v) { verbose_ = v; }
private:
    std::mutex              mu_;
    std::deque<std::string> lines_;
    std::string             partial_;
    int                     partial_level_ = 2;
    bool                    verbose_ = false;
};

LogRing & logs();

} // namespace rafid

#define RLOG(...)  rafid::logs().addf(2, __VA_ARGS__)
#define RWARN(...) rafid::logs().addf(3, __VA_ARGS__)
#define RERR(...)  rafid::logs().addf(4, __VA_ARGS__)
