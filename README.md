# Manticore -> FAISS HNSW Demo

## Requirements

- macOS with Homebrew
- Manticore running locally on `127.0.0.1:9306`
- Manticore `test` table containing columns:
  - `id` (integer)
  - `vec` (vector as string, e.g. `[0.1, 0.2, ...]`)
- Dependencies (Homebrew):
  - `faiss`
  - `mysql-client`
  - `libomp`
- `hnswlib` submodule checked out

## Install dependencies

```sh
brew install faiss mysql-client libomp
```

```sh
git submodule update --init --recursive
```

## Build

```sh
make
```

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
