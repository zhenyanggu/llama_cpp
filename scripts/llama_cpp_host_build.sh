#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  llama_cpp_host_build.sh [--repo-dir <path>] [--build-dir <path>] [--build-type <type>] [--clean]

Example:
  llama_cpp_host_build.sh \
    --repo-dir /home/gugugu/work/llama.cpp-kv260-20260407 \
    --build-dir /home/gugugu/work/llama.cpp-kv260-20260407/build-host \
    --clean
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR=""
BUILD_TYPE="Release"
CLEAN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --clean) CLEAN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR="$REPO_DIR/build-host"
fi

[ -d "$REPO_DIR" ] || { echo "Repo dir not found: $REPO_DIR" >&2; exit 1; }
[ -f "$REPO_DIR/CMakeLists.txt" ] || { echo "Not a llama.cpp source tree: $REPO_DIR" >&2; exit 1; }

CMAKE_BIN="$(command -v cmake)"
NINJA_BIN="$(command -v ninja)"
CC_BIN="$(command -v cc)"
CXX_BIN="$(command -v c++)"

[ -n "$CMAKE_BIN" ] || { echo "cmake not found in PATH" >&2; exit 1; }
[ -n "$NINJA_BIN" ] || { echo "ninja not found in PATH" >&2; exit 1; }
[ -n "$CC_BIN" ] || { echo "cc not found in PATH" >&2; exit 1; }
[ -n "$CXX_BIN" ] || { echo "c++ not found in PATH" >&2; exit 1; }

if [ "$CLEAN" -eq 1 ]; then
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

# Prefer the host toolchain explicitly so a sourced PetaLinux SDK does not turn
# this into a cross-build by accident.
env \
  -u CC -u CXX -u CPP -u LD -u AR -u AS -u NM -u STRIP -u RANLIB \
  -u OBJCOPY -u OBJDUMP -u READELF -u CROSS_COMPILE -u ARCH \
  -u PKG_CONFIG -u PKG_CONFIG_PATH -u PKG_CONFIG_LIBDIR -u PKG_CONFIG_SYSROOT_DIR \
  -u SDKTARGETSYSROOT -u OECORE_TARGET_SYSROOT -u OECORE_NATIVE_SYSROOT \
  "$CMAKE_BIN" -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA_BIN" \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DGGML_BLAS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON

env \
  -u CC -u CXX -u CPP -u LD -u AR -u AS -u NM -u STRIP -u RANLIB \
  -u OBJCOPY -u OBJDUMP -u READELF -u CROSS_COMPILE -u ARCH \
  -u PKG_CONFIG -u PKG_CONFIG_PATH -u PKG_CONFIG_LIBDIR -u PKG_CONFIG_SYSROOT_DIR \
  -u SDKTARGETSYSROOT -u OECORE_TARGET_SYSROOT -u OECORE_NATIVE_SYSROOT \
  "$CMAKE_BIN" --build "$BUILD_DIR" --target llama-server -j"$(nproc)"

echo
echo "Built host-native llama-server:"
echo "  repo   : $REPO_DIR"
echo "  build  : $BUILD_DIR"
echo "  binary : $BUILD_DIR/bin/llama-server"
