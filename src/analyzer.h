// رافد - analyzer: turns the expert fingerprint into answers for the three questions
//   1) concentration: do a few experts carry most of the work?
//   2) caching: what hit rate would a RAM / VRAM cache of hot experts reach?
//   3) prediction: can layer L tell us which experts layer L+1 will need?
// Every cache and predictor is trained on the first half of the trace and scored on the second half.
#pragma once

#include <nlohmann/json.hpp>

#include "gguf_info.h"
#include "hw_probe.h"
#include "profiler.h"

namespace rafid {

struct AnalyzeOptions {
    int max_trace_tokens   = 12000; // newest generated tokens used for simulations
    int max_predict_tokens = 2000;  // test tokens used for the predictor score
};

nlohmann::json analyze_profile(const ExpertProfiler::Snapshot & snap, const ModelInfo & mi, const HwInfo & hw,
                               const AnalyzeOptions & opt = AnalyzeOptions());

} // namespace rafid
