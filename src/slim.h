// رافد - "slim" copy of a model for the expert store (Windows).
//
// llama.cpp memory-maps the model file. On Windows (NTFS) unbuffered reads of a file that is mapped do not run
// in parallel: every read waits for the previous one, so the expert store got ~1 GB/s out of a disk that does
// 5+ GB/s. In stream mode Rafid therefore gives llama.cpp a sparse copy of the model: same size and layout,
// with the routed experts left as holes (they take no disk space and read as zeros). llama.cpp maps the copy;
// the experts are read from the original file, which nothing maps.
#pragma once

#include <functional>
#include <string>

#include "gguf_info.h"

namespace rafid {

// Creates (or reuses) the slim copy of every file of the model in dir. progress(fraction) returns false to
// cancel. Returns the path to load (the copy of the first file), or "" when no copy is used (err says why).
std::string prepare_slim_model(const ModelInfo & mi, const std::string & dir,
                               const std::function<bool(double)> & progress, std::string & err);

} // namespace rafid
