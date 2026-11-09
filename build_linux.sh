#!/usr/bin/env bash
# Rafid - Linux build. Needs: a C++17 compiler, CMake 3.21+, Git, and the CUDA toolkit for NVIDIA GPUs.
set -euo pipefail
cd "$(dirname "$0")"

flags=()
if command -v nvcc >/dev/null 2>&1; then
    flags+=(-DGGML_CUDA=ON)
    echo "[+] CUDA found: building with GPU support"
else
    echo "[!] nvcc not found: building a CPU-only version"
fi
gen=()
if command -v ninja >/dev/null 2>&1; then gen=(-G Ninja); fi

cmake -S . -B build "${gen[@]}" -DCMAKE_BUILD_TYPE=Release "${flags[@]}"
cmake --build build -j "$(nproc)"
echo "[+] Done: build/rafid   (start it with: ./build/rafid --profiles-dir ./profiles)"
