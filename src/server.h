// رافد - local HTTP server: JSON API + streaming chat (SSE) + the embedded Arabic web UI
#pragma once

#include <atomic>
#include <string>

#include "engine.h"

namespace rafid {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int         port = 8080;
    std::string models_dir;
    std::string profiles_dir;
    std::string web_dir;      // serve web/index.html from here instead of the embedded copy
    bool        open_browser = true;
};

int run_server(const ServerConfig & cfg, HwState & hws, Engine & engine, std::atomic<bool> & quit);

} // namespace rafid
