# Third-party software

Rafid's own code is under the MIT licence (see `LICENSE`). It builds on, and the Windows release ships, the following.

| Component | Used for | Licence | Where |
|---|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), build b12167 | the compute engine: CPU and CUDA kernels, GGUF loading | MIT, © 2023-2026 The ggml authors | downloaded at build time; `llama.dll`, `ggml*.dll` in the release (`bin/LICENSE-llama.cpp.txt`) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | the local HTTP server | MIT, © 2017 yhirose | vendored inside llama.cpp, compiled into `rafid.exe` |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON | MIT, © 2013-2025 Niels Lohmann | vendored inside llama.cpp, compiled into `rafid.exe` |
| LLVM OpenMP runtime | CPU threads in the ggml CPU backend | Apache-2.0 with LLVM exception | `libomp.dll` in the release (`bin/LICENSE-LLVM-OpenMP`) |
| NVIDIA CUDA runtime and cuBLAS | GPU kernels | NVIDIA CUDA Toolkit EULA (redistributable files) | `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` in the release, as shipped in llama.cpp's official CUDA build |
| IBM Plex Sans Arabic, Noto Kufi Arabic | the interface's fonts | SIL Open Font License 1.1 | loaded from Google Fonts by the page; not bundled |
| `llama-bench` (llama.cpp) | `baseline_llama.bat` | MIT | `llama-bench.exe`, `llama-bench-impl.dll`, `llama-common.dll` in the release |

No model weights are included. Models you run with Rafid come under their own licences.
