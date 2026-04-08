#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_kv260_runtime_libs.sh [--root <aicas-dir>]

Downloads and extracts the aarch64 runtime libraries needed by the KV260
llama-server baseline into:
  third_party/kv260_runtime/lib
EOF
}

ROOT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

DEB_DIR="$ROOT_DIR/third_party/kv260_runtime/debs"
EXTRACT_DIR="$ROOT_DIR/third_party/kv260_runtime/extract"
LIB_DIR="$ROOT_DIR/third_party/kv260_runtime/lib"

mkdir -p "$DEB_DIR" "$EXTRACT_DIR" "$LIB_DIR"
rm -rf "$EXTRACT_DIR"/*
rm -rf "$LIB_DIR"/*

download_deb() {
  local url="$1"
  local dst="$2"
  if [ ! -f "$dst" ]; then
    echo "Downloading $(basename "$dst")"
    curl -fsSL -o "$dst" "$url"
  else
    echo "Reusing $(basename "$dst")"
  fi
}

copy_libs() {
  local src_dir="$1"
  shift
  local name=""
  for name in "$@"; do
    cp -a "$src_dir/$name" "$LIB_DIR/"
  done
}

OPENBLAS_DEB="$DEB_DIR/libopenblas0-pthread_0.3.26+ds-1ubuntu0.1_arm64.deb"
GFORTRAN_DEB="$DEB_DIR/libgfortran5_14.2.0-4ubuntu2~24.04.1_arm64.deb"

download_deb \
  "http://ports.ubuntu.com/ubuntu-ports/pool/universe/o/openblas/$(basename "$OPENBLAS_DEB")" \
  "$OPENBLAS_DEB"
download_deb \
  "http://ports.ubuntu.com/ubuntu-ports/pool/main/g/gcc-14/$(basename "$GFORTRAN_DEB")" \
  "$GFORTRAN_DEB"

mkdir -p "$EXTRACT_DIR/openblas" "$EXTRACT_DIR/gfortran"
dpkg-deb -x "$OPENBLAS_DEB" "$EXTRACT_DIR/openblas"
dpkg-deb -x "$GFORTRAN_DEB" "$EXTRACT_DIR/gfortran"

copy_libs \
  "$EXTRACT_DIR/openblas/usr/lib/aarch64-linux-gnu/openblas-pthread" \
  libopenblas.so.0 libopenblasp-r0.3.26.so
copy_libs \
  "$EXTRACT_DIR/gfortran/usr/lib/aarch64-linux-gnu" \
  libgfortran.so.5 libgfortran.so.5.0.0

echo
echo "Prepared KV260 runtime libraries:"
find "$LIB_DIR" -maxdepth 1 \( -type f -o -type l \) -printf '%f -> %l\n' | sort
