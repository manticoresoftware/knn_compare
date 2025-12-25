# KNN Compare: Manticore vs FAISS vs HNSWlib

## Requirements

- macOS (Homebrew) or Linux
- Manticore running locally on `127.0.0.1:9306`
- Manticore `t` table containing columns:
  - `id` (integer)
  - `vec` (vector as string, e.g. `[0.1, 0.2, ...]`)
- Build tools: C++ compiler + CMake 3.24+ (for FAISS submodule)
- Dependencies: MySQL client headers, OpenMP (macOS), BLAS/LAPACK (Linux)
- Submodules: `faiss`, `hnswlib`

## Install dependencies

### macOS (Homebrew)

```sh
brew install mysql-client libomp
git submodule update --init --recursive
```

### Linux

Ubuntu/Debian:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake libmysqlclient-dev libopenblas-dev liblapack-dev
git submodule update --init --recursive
```

RHEL:

```sh
sudo dnf install cmake gcc-c++ mysql-devel openblas-devel lapack-devel
git submodule update --init --recursive
```

Optional: use a system FAISS package instead of the submodule:

```sh
make FAISS_PREFIX=/usr/local FAISS_DIR=
```

## Build

```sh
make
```

The first build compiles the FAISS submodule into `faiss/build`.

On macOS, BLAS/LAPACK comes from Accelerate and OpenMP comes from `libomp`.

FAISS builds in `Release` mode and auto-selects SIMD (`avx512`, `avx2`, or
`generic`) based on your CPU. Override if needed:

```sh
make FAISS_OPT_LEVEL=generic
```

Verbose FAISS build output:

```sh
make FAISS_VERBOSE=1
```

If you see linker errors like `undefined reference to __kmpc_fork_call`, rebuild
FAISS with the same compiler as the demo (e.g., `make CXX=g++` or
`make CXX=clang++`), and delete `faiss/build` before retrying.

Clean build artifacts and cached indexes:

```sh
make clean
```

Clear cached indexes before running:

```sh
./knn_compare --clear
```

## Run

```sh
./knn_compare
```

Optional:

```sh
./knn_compare -k 10
```

```sh
./knn_compare -k 10 -ef 64 -efc 200
```

```sh
./knn_compare -metric l2 -k 10 -ef 64 -efc 200
```

The program caches indexes to `index_hnsw_<metric>_efc<efc>.faiss`,
`index_flat_<metric>_efc<efc>.faiss`, and `index_hnswlib_<metric>_efc<efc>.bin`
in the working directory to avoid rebuilding on subsequent runs.
