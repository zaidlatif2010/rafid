#!/usr/bin/env bash
# Builds bin/rafid.exe for Windows on Linux with MinGW-w64, linked against the official
# prebuilt llama.cpp DLLs, so Windows needs neither Visual Studio nor the CUDA Toolkit.
#
#   tools/build_windows_mingw.sh <llama.cpp-source-dir> <dir-with-llama.dll> <out-dir>
#
# The llama.cpp sources must be the same release tag as the DLLs (headers only are used).
# Needs: x86_64-w64-mingw32-g++-posix (Ubuntu: apt install g++-mingw-w64-x86-64-posix), cmake.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
L="${1:?llama.cpp source dir}"
D="${2:?dir with llama.dll ggml.dll ggml-base.dll}"
OUT="${3:?output dir}"
TAG="${RAFID_LLAMA_TAG:-b12167}"
CXX="${CXX:-x86_64-w64-mingw32-g++-posix}"
mkdir -p "$OUT/gen" "$OUT/lib"

# embed the web UI exactly like CMakeLists.txt does
cat > "$OUT/gen/embed.cmake" <<'EOF'
file(READ ${SRC} HEX HEX)
string(REGEX REPLACE "(................................................................................................................................)" "\\1\n" HEX "${HEX}")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ARR "${HEX}")
file(WRITE ${OUT}/web_index.h "// generated from web/index.html - do not edit\n#pragma once\n#include <cstddef>\nstatic const unsigned char RAFID_INDEX_HTML[] = {\n${ARR}\n};\nstatic const size_t RAFID_INDEX_HTML_LEN = sizeof(RAFID_INDEX_HTML);\n")
EOF
cmake -DSRC="$here/web/index.html" -DOUT="$OUT/gen" -P "$OUT/gen/embed.cmake"

# import libraries made from the DLL export tables (keeps the rest of the link fully static)
for dll in llama ggml ggml-base; do
    {
        echo "LIBRARY $dll.dll"
        echo "EXPORTS"
        # C symbols only (MSVC-mangled C++ names start with '?' and are not used by rafid)
        x86_64-w64-mingw32-objdump -p "$D/$dll.dll" \
            | awk '/\[Ordinal\/Name Pointer\] Table/{f=1;next} f&&/^[ \t]*\[/{print $NF} f&&/^[ \t]*$/{f=0}' \
            | grep -E '^[A-Za-z_][A-Za-z0-9_]*$'
    } > "$OUT/lib/$dll.def"
    x86_64-w64-mingw32-dlltool -d "$OUT/lib/$dll.def" -l "$OUT/lib/lib$dll.a" -D "$dll.dll"
done

"$CXX" -O2 -std=c++17 -static -s \
    -D_WIN32_WINNT=0x0A00 -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DCPPHTTPLIB_TCP_NODELAY=1 \
    -DGGML_SHARED -DLLAMA_SHARED \
    -DRAFID_VERSION="\"0.8.5\"" -DRAFID_LLAMA_TAG="\"$TAG\"" \
    -I"$here/src" -I"$OUT/gen" -I"$L/include" -I"$L/ggml/include" -I"$L/ggml/src" -I"$L/vendor" -I"$L/vendor/cpp-httplib" \
    "$here"/src/*.cpp "$L/vendor/cpp-httplib/httplib.cpp" \
    -L"$OUT/lib" -lllama -lggml -lggml-base -lws2_32 -lshell32 \
    -o "$OUT/rafid.exe"
echo "built $OUT/rafid.exe"
x86_64-w64-mingw32-objdump -p "$OUT/rafid.exe" | grep "DLL Name" | sed 's/.*DLL Name: /  needs /'
