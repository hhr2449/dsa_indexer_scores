#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SOURCE OUTPUT" >&2
    exit 2
fi

source_file=$1
output_file=$2

if [[ ! -f "$source_file" ]]; then
    echo "source file not found: $source_file" >&2
    exit 2
fi

if [[ -n "${NVCC:-}" ]]; then
    nvcc=$NVCC
elif [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    nvcc=/usr/local/cuda-12.4/bin/nvcc
else
    nvcc=$(command -v nvcc || true)
fi

if [[ -z "$nvcc" || ! -x "$nvcc" ]]; then
    echo "nvcc not found; set NVCC to a CUDA 12 compiler" >&2
    exit 1
fi

if ! "$nvcc" -arch=sm_90a --dryrun "$source_file" >/dev/null 2>&1; then
    echo "$nvcc does not support the required sm_90a target" >&2
    exit 1
fi

mkdir -p "$(dirname "$output_file")"
"$nvcc" \
    -O3 \
    -std=c++17 \
    -shared \
    -Xcompiler -fPIC \
    -lineinfo \
    -arch=sm_90a \
    "$source_file" \
    -o "$output_file"
