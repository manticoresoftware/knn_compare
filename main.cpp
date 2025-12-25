#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include "hnswlib.h"
#include <mysql/mysql.h>

#include <algorithm>
#include <exception>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <limits>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <vector>

static bool parse_vector_string(const char* data, size_t len,
                                std::vector<float>& out) {
  out.clear();
  out.reserve(len / 2);
  size_t i = 0;
  while (i < len) {
    char c = data[i];
    while (i < len &&
           (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' ||
            c == '[' || c == ']' || c == '(' || c == ')')) {
      ++i;
      if (i < len) c = data[i];
    }
    if (i >= len) break;
    char* end = nullptr;
    float v = std::strtof(data + i, &end);
    if (end == data + i) {
      return false;
    }
    out.push_back(v);
    i = static_cast<size_t>(end - data);
  }
  return !out.empty();
}

static void l2_normalize(std::vector<float>& v) {
  double sum = 0.0;
  for (float x : v) sum += static_cast<double>(x) * x;
  if (sum <= std::numeric_limits<double>::min()) return;
  float inv = static_cast<float>(1.0 / std::sqrt(sum));
  for (float& x : v) x *= inv;
}

static bool terminal_supports_color() {
  // Only emit ANSI colors for interactive terminals with a real TERM.
  if (!isatty(fileno(stdout))) {
    return false;
  }
  const char* term = std::getenv("TERM");
  if (!term || std::strcmp(term, "dumb") == 0) {
    return false;
  }
  return true;
}

static void print_cell(std::ostream& os, const std::string& text, int width,
                       const std::string& color, bool use_color) {
  // Apply color to the cell text and keep padding aligned regardless of ANSI.
  if (use_color && !color.empty()) {
    os << color << text << "\033[0m";
  } else {
    os << text;
  }
  int pad = width - static_cast<int>(text.size());
  if (pad > 0) {
    os << std::string(static_cast<size_t>(pad), ' ');
  }
}

static std::string format_float(float value) {
  // Keep numeric formatting consistent across all columns.
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  return oss.str();
}

struct IdDist {
  long long id;
  float dist;
};

static void sort_by_dist_id(std::vector<long long>& ids,
                            std::vector<float>& dists) {
  // Normalize ordering to match SQL: distance asc, then id asc.
  std::vector<IdDist> items;
  items.reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0) {
      continue;
    }
    if (dists[i] != dists[i]) {
      continue;
    }
    items.push_back({ids[i], dists[i]});
  }
  std::sort(items.begin(), items.end(),
            [](const IdDist& a, const IdDist& b) {
              if (a.dist != b.dist) {
                return a.dist < b.dist;
              }
              return a.id < b.id;
            });
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i < items.size()) {
      ids[i] = items[i].id;
      dists[i] = items[i].dist;
    } else {
      ids[i] = -1;
      dists[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }
}

enum class MetricMode { kCosine, kL2 };

static bool parse_metric(const std::string& value, MetricMode& out) {
  // Accept a small, explicit set of metric names.
  if (value == "cosine") {
    out = MetricMode::kCosine;
    return true;
  }
  if (value == "l2") {
    out = MetricMode::kL2;
    return true;
  }
  return false;
}

static const char* metric_name(MetricMode mode) {
  // Used to suffix cache files by metric.
  return mode == MetricMode::kCosine ? "cosine" : "l2";
}

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --clear   Remove cached index files before running\n"
      << "  -k N       Number of neighbors to return (default: 10)\n"
      << "  -ef N      efSearch for FAISS/HNSWlib and ef for Manticore (default: 64)\n"
      << "  -efc N     FAISS HNSW efConstruction (default: 200)\n"
      << "  -metric M  Distance metric: cosine or l2 (default: cosine)\n"
      << "  --help,-h  Show this help\n";
}

static bool file_exists(const std::string& path) {
  // Simple presence check for index cache files.
  std::ifstream f(path);
  return f.good();
}

int main(int argc, char** argv) {
  int k = 10;
  int ef_search = 64;
  int ef_construction = 200;
  MetricMode metric = MetricMode::kCosine;
  bool clear_cache = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--clear") == 0) {
      clear_cache = true;
    } else if (std::strcmp(argv[i], "-k") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for -k\n";
        return 1;
      }
      k = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-ef") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for -ef\n";
        return 1;
      }
      ef_search = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-efc") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for -efc\n";
        return 1;
      }
      ef_construction = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-metric") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for -metric\n";
        return 1;
      }
      MetricMode parsed;
      if (!parse_metric(argv[++i], parsed)) {
        std::cerr << "Invalid metric. Use: cosine or l2\n";
        return 1;
      }
      metric = parsed;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }
  if (k <= 0) {
    std::cerr << "k must be > 0\n";
    return 1;
  }
  if (ef_search <= 0) {
    std::cerr << "ef must be > 0\n";
    return 1;
  }
  if (ef_construction <= 0) {
    std::cerr << "efc must be > 0\n";
    return 1;
  }

  // Cosine mode uses inner product on normalized vectors.
  const bool use_cosine = metric == MetricMode::kCosine;
  const char* host = "127.0.0.1";
  const unsigned int port = 9306;
  const char* user = nullptr;
  const char* pass = nullptr;
  const char* db = nullptr;
  const size_t limit = 1000000;
  const int hnsw_m = 32;

  const std::string metric_tag = metric_name(metric);
  const std::string efc_tag = std::to_string(ef_construction);
  const std::string hnsw_path =
      "index_hnsw_" + metric_tag + "_efc" + efc_tag + ".faiss";
  const std::string flat_path =
      "index_flat_" + metric_tag + "_efc" + efc_tag + ".faiss";
  const std::string hnswlib_path =
      "index_hnswlib_" + metric_tag + "_efc" + efc_tag + ".bin";

  if (clear_cache) {
    // Remove cache files for the requested metric/efc before running.
    std::remove(hnsw_path.c_str());
    std::remove(flat_path.c_str());
    std::remove(hnswlib_path.c_str());
  }

  std::cerr << "Connecting to Manticore...\n";
  MYSQL* conn = mysql_init(nullptr);
  if (!conn) {
    std::cerr << "mysql_init failed\n";
    return 1;
  }

  if (!mysql_real_connect(conn, host, user, pass, db, port, nullptr, 0)) {
    std::cerr << "mysql_real_connect failed: " << mysql_error(conn) << "\n";
    mysql_close(conn);
    return 1;
  }

  size_t dim = 0;
  std::unique_ptr<faiss::IndexIDMap2> index;
  std::unique_ptr<faiss::IndexIDMap2> flat_index;
  faiss::IndexHNSWFlat* hnsw_base = nullptr;
  faiss::IndexFlat* flat_base = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> hnswlib_space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnswlib_index;

  std::vector<float> vec;
  size_t count = 0;

  if (file_exists(hnsw_path) && file_exists(flat_path) &&
      file_exists(hnswlib_path)) {
    // Load cached indexes when all three are available for this metric.
    std::cerr << "Loading indexes from disk...\n";
    std::unique_ptr<faiss::Index> hnsw_any(
        faiss::read_index(hnsw_path.c_str(), faiss::IO_FLAG_MMAP));
    auto* hnsw_map = dynamic_cast<faiss::IndexIDMap2*>(hnsw_any.get());
    if (!hnsw_map) {
      std::cerr << "Loaded HNSW index is not IndexIDMap2\n";
      mysql_close(conn);
      return 1;
    }
    hnsw_map->own_fields = true;
    hnsw_base = dynamic_cast<faiss::IndexHNSWFlat*>(hnsw_map->index);
    if (!hnsw_base) {
      std::cerr << "Loaded HNSW base is not IndexHNSWFlat\n";
      mysql_close(conn);
      return 1;
    }
    // Safety check to avoid mixing cached metrics.
    if (use_cosine && hnsw_base->metric_type != faiss::METRIC_INNER_PRODUCT) {
      std::cerr << "Loaded HNSW metric does not match cosine\n";
      mysql_close(conn);
      return 1;
    }
    if (!use_cosine && hnsw_base->metric_type != faiss::METRIC_L2) {
      std::cerr << "Loaded HNSW metric does not match l2\n";
      mysql_close(conn);
      return 1;
    }
    dim = static_cast<size_t>(hnsw_map->d);
    index.reset(hnsw_map);
    hnsw_any.release();

    std::unique_ptr<faiss::Index> flat_any(
        faiss::read_index(flat_path.c_str(), faiss::IO_FLAG_MMAP));
    auto* flat_map = dynamic_cast<faiss::IndexIDMap2*>(flat_any.get());
    if (!flat_map) {
      std::cerr << "Loaded flat index is not IndexIDMap2\n";
      mysql_close(conn);
      return 1;
    }
    flat_map->own_fields = true;
    flat_base = dynamic_cast<faiss::IndexFlat*>(flat_map->index);
    if (!flat_base) {
      std::cerr << "Loaded flat base is not IndexFlat\n";
      mysql_close(conn);
      return 1;
    }
    // Ensure the flat cache matches the requested metric.
    if (use_cosine && flat_base->metric_type != faiss::METRIC_INNER_PRODUCT) {
      std::cerr << "Loaded flat metric does not match cosine\n";
      mysql_close(conn);
      return 1;
    }
    if (!use_cosine && flat_base->metric_type != faiss::METRIC_L2) {
      std::cerr << "Loaded flat metric does not match l2\n";
      mysql_close(conn);
      return 1;
    }
    if (static_cast<size_t>(flat_map->d) != dim) {
      std::cerr << "Loaded index dims do not match\n";
      mysql_close(conn);
      return 1;
    }
    flat_index.reset(flat_map);
    flat_any.release();
    count = static_cast<size_t>(index->ntotal);

    try {
      // HNSWlib requires a space instance to load the graph.
      if (use_cosine) {
        hnswlib_space = std::make_unique<hnswlib::InnerProductSpace>(dim);
      } else {
        hnswlib_space =
            std::make_unique<hnswlib::L2Space>(static_cast<int>(dim));
      }
      hnswlib_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
          hnswlib_space.get(), hnswlib_path, false, limit);
    } catch (const std::exception& e) {
      std::cerr << "Failed to load HNSWlib index: " << e.what() << "\n";
      mysql_close(conn);
      return 1;
    }
    std::cerr << "Loaded " << count << " vectors from disk\n";
  } else {
    // No cache: stream vectors from Manticore and build all indexes.
    std::string query =
        "select id, vec from t order by id asc limit " + std::to_string(limit) +
        " option max_matches=" + std::to_string(limit);
    std::cerr << "Running query: " << query << "\n";
    if (mysql_query(conn, query.c_str())) {
      std::cerr << "Query failed: " << mysql_error(conn) << "\n";
      mysql_close(conn);
      return 1;
    }

    MYSQL_RES* result = mysql_use_result(conn);
    if (!result) {
      std::cerr << "mysql_use_result failed: " << mysql_error(conn) << "\n";
      mysql_close(conn);
      return 1;
    }

    std::cerr << "Reading vectors and building indices...\n";
    for (;;) {
      MYSQL_ROW row = mysql_fetch_row(result);
      if (!row) break;
      unsigned long* lengths = mysql_fetch_lengths(result);
      if (!lengths || !row[0] || !row[1]) {
        continue;
      }
      long long id = std::strtoll(row[0], nullptr, 10);

      std::string vec_str(row[1], lengths[1]);
      if (!parse_vector_string(vec_str.data(), vec_str.size(), vec)) {
        std::cerr << "Skipping id " << id << " due to vector parse failure\n";
        continue;
      }
      if (dim == 0) {
        // First vector defines dimensionality for all indexes.
        dim = vec.size();
        auto hnsw = std::make_unique<faiss::IndexHNSWFlat>(
            static_cast<int>(dim), hnsw_m,
            use_cosine ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2);
        hnsw->hnsw.efConstruction = ef_construction;
        index = std::make_unique<faiss::IndexIDMap2>(hnsw.release());
        index->own_fields = true;
        hnsw_base = dynamic_cast<faiss::IndexHNSWFlat*>(index->index);

        std::unique_ptr<faiss::IndexFlat> flat;
        if (use_cosine) {
          flat = std::make_unique<faiss::IndexFlatIP>(static_cast<int>(dim));
        } else {
          flat = std::make_unique<faiss::IndexFlatL2>(static_cast<int>(dim));
        }
        flat_index = std::make_unique<faiss::IndexIDMap2>(flat.release());
        flat_index->own_fields = true;
        flat_base = dynamic_cast<faiss::IndexFlat*>(flat_index->index);

        if (use_cosine) {
          hnswlib_space = std::make_unique<hnswlib::InnerProductSpace>(dim);
        } else {
          hnswlib_space =
              std::make_unique<hnswlib::L2Space>(static_cast<int>(dim));
        }
        hnswlib_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            hnswlib_space.get(), limit, hnsw_m, ef_construction);
      }
      if (vec.size() != dim) {
        std::cerr << "Skipping id " << id << " due to dim mismatch: got "
                  << vec.size() << " expected " << dim << "\n";
        continue;
      }

      if (use_cosine) {
        // Normalize only for cosine to convert IP to cosine similarity.
        l2_normalize(vec);
      }
      faiss::idx_t fid = static_cast<faiss::idx_t>(id);
      index->add_with_ids(1, vec.data(), &fid);
      flat_index->add_with_ids(1, vec.data(), &fid);
      hnswlib_index->addPoint(vec.data(), static_cast<size_t>(id));
      ++count;
      if (count % 1000 == 0) {
        std::cerr << "Indexed " << count << " vectors...\n";
      }
    }

    mysql_free_result(result);

    if (count == 0) {
      std::cerr << "No vectors indexed\n";
      mysql_close(conn);
      return 1;
    }

    // Persist indexes so subsequent runs don't rebuild.
    std::cerr << "Saving indexes to disk...\n";
    faiss::write_index(index.get(), hnsw_path.c_str());
    faiss::write_index(flat_index.get(), flat_path.c_str());
    hnswlib_index->saveIndex(hnswlib_path);
  }

  // Match search ef across FAISS and HNSWlib for consistency.
  hnsw_base->hnsw.efSearch = ef_search;
  hnswlib_index->setEf(ef_search);
  std::vector<float> query_vec(dim, 0.5f);
  if (use_cosine) {
    // Use the same normalization as indexed vectors.
    l2_normalize(query_vec);
  }

  std::vector<faiss::idx_t> ids(k);
  std::vector<float> sims(k);
  std::cerr << "Searching HNSW...\n";
  index->search(1, query_vec.data(), k, sims.data(), ids.data());

  std::vector<long long> hnsw_ids(k, -1);
  std::vector<float> hnsw_dists(k);
  for (int i = 0; i < k; ++i) {
    if (ids[i] < 0) {
      hnsw_dists[i] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    hnsw_ids[i] = static_cast<long long>(ids[i]);
    float dist = sims[i];
    if (use_cosine) {
      dist = 1.0f - sims[i];
    }
    hnsw_dists[i] = dist;
  }
  // Enforce deterministic ordering on ties.
  sort_by_dist_id(hnsw_ids, hnsw_dists);

  std::cerr << "Searching flat (exact)...\n";
  std::vector<faiss::idx_t> flat_ids(k);
  std::vector<float> flat_sims(k);
  flat_index->search(1, query_vec.data(), k, flat_sims.data(),
                     flat_ids.data());

  std::vector<long long> exact_ids(k, -1);
  std::vector<float> exact_dists(k);
  for (int i = 0; i < k; ++i) {
    if (flat_ids[i] < 0) {
      exact_dists[i] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    exact_ids[i] = static_cast<long long>(flat_ids[i]);
    float dist = flat_sims[i];
    if (use_cosine) {
      dist = 1.0f - flat_sims[i];
    }
    exact_dists[i] = dist;
  }
  // Enforce deterministic ordering on ties.
  sort_by_dist_id(exact_ids, exact_dists);

  std::cerr << "Searching HNSWlib...\n";
  std::vector<long long> hnswlib_ids(k, -1);
  std::vector<float> hnswlib_dists(
      k, std::numeric_limits<float>::quiet_NaN());
  auto hnswlib_res = hnswlib_index->searchKnn(query_vec.data(), k);
  std::vector<std::pair<size_t, float>> hnswlib_pairs;
  hnswlib_pairs.reserve(k);
  while (!hnswlib_res.empty()) {
    auto item = hnswlib_res.top();
    hnswlib_res.pop();
    hnswlib_pairs.push_back({static_cast<size_t>(item.second), item.first});
  }
  std::reverse(hnswlib_pairs.begin(), hnswlib_pairs.end());
  for (size_t i = 0; i < hnswlib_pairs.size() && i < static_cast<size_t>(k);
       ++i) {
    hnswlib_ids[i] = static_cast<long long>(hnswlib_pairs[i].first);
    hnswlib_dists[i] = hnswlib_pairs[i].second;
  }
  // Enforce deterministic ordering on ties.
  sort_by_dist_id(hnswlib_ids, hnswlib_dists);

  std::ostringstream vec_literal;
  // Use the exact query vector values for Manticore.
  vec_literal.setf(std::ios::fixed);
  vec_literal << std::setprecision(6);
  vec_literal << "(";
  for (size_t i = 0; i < dim; ++i) {
    if (i) vec_literal << ", ";
    vec_literal << query_vec[i];
  }
  vec_literal << ")";

  std::ostringstream knn_query;
  knn_query << "select id, knn_dist() as score from t where knn(vec, " << k
            << ", " << vec_literal.str() << ", { ef=" << ef_search
            << " }) order by score asc, id asc limit " << k;
  std::cerr << "Searching Manticore KNN: " << knn_query.str() << "\n";

  if (mysql_query(conn, knn_query.str().c_str())) {
    std::cerr << "KNN query failed: " << mysql_error(conn) << "\n";
    mysql_close(conn);
    return 1;
  }

  MYSQL_RES* knn_res = mysql_store_result(conn);
  if (!knn_res) {
    std::cerr << "mysql_store_result failed: " << mysql_error(conn) << "\n";
    mysql_close(conn);
    return 1;
  }

  std::vector<std::string> mc_ids(k);
  std::vector<std::string> mc_dists(k);
  for (int i = 0; i < k; ++i) {
    MYSQL_ROW row = mysql_fetch_row(knn_res);
    if (!row) break;
    if (!row[0] || !row[1]) continue;
    mc_ids[i] = row[0];
    mc_dists[i] = row[1];
  }

  mysql_free_result(knn_res);
  mysql_close(conn);

  std::cout << "Top " << k << " comparison (";
  if (use_cosine) {
    std::cout << "distance=1-sim, knn_dist";
  } else {
    std::cout << "distance=l2, knn_dist";
  }
  std::cout << "):\n";
  const int w_id = 14;
  const int w_dist = 14;
  const bool use_color = terminal_supports_color();
  // Highlight the top-5 flat IDs and mirror the color in other columns.
  const char* colors[5] = {"\033[32m", "\033[34m", "\033[33m", "\033[35m",
                           "\033[36m"};
  std::unordered_map<long long, const char*> id_colors;
  for (int i = 0; i < k && i < 5; ++i) {
    if (exact_ids[i] >= 0) {
      id_colors[static_cast<long long>(exact_ids[i])] = colors[i];
    }
  }
  auto color_for_id = [&](long long id) -> const char* {
    auto it = id_colors.find(id);
    if (it == id_colors.end()) {
      return nullptr;
    }
    return it->second;
  };

  // Subtle, low-contrast greys per column group.
  const std::string flat_bg;
  const std::string faiss_bg;
  const std::string hnswlib_bg;
  const std::string manticore_bg;

  print_cell(std::cout, "flat_id", w_id, flat_bg, use_color);
  print_cell(std::cout, "flat_dist", w_dist, flat_bg, use_color);
  print_cell(std::cout, "faiss_id", w_id, faiss_bg, use_color);
  print_cell(std::cout, "faiss_dist", w_dist, faiss_bg, use_color);
  print_cell(std::cout, "hnswlib_id", w_id, hnswlib_bg, use_color);
  print_cell(std::cout, "hnswlib_dist", w_dist, hnswlib_bg, use_color);
  print_cell(std::cout, "manticore_id", w_id, manticore_bg, use_color);
  print_cell(std::cout, "manticore_dist", w_dist, manticore_bg, use_color);
  std::cout << "\n";
  for (int i = 0; i < k; ++i) {
    if (exact_ids[i] >= 0) {
      const auto exact_id = static_cast<long long>(exact_ids[i]);
      std::string color = flat_bg;
      if (const char* fg = color_for_id(exact_id)) {
        color += fg;
      }
      print_cell(std::cout, std::to_string(exact_id), w_id, color, use_color);
      print_cell(std::cout, format_float(exact_dists[i]), w_dist, flat_bg,
                 use_color);
    } else {
      print_cell(std::cout, "-", w_id, flat_bg, use_color);
      print_cell(std::cout, "-", w_dist, flat_bg, use_color);
    }
    if (hnsw_ids[i] >= 0) {
      const auto hnsw_id = static_cast<long long>(hnsw_ids[i]);
      std::string color = faiss_bg;
      if (const char* fg = color_for_id(hnsw_id)) {
        color += fg;
      }
      print_cell(std::cout, std::to_string(hnsw_id), w_id, color, use_color);
      print_cell(std::cout, format_float(hnsw_dists[i]), w_dist, faiss_bg,
                 use_color);
    } else {
      print_cell(std::cout, "-", w_id, faiss_bg, use_color);
      print_cell(std::cout, "-", w_dist, faiss_bg, use_color);
    }
    if (hnswlib_dists[i] == hnswlib_dists[i]) {
      const auto hnswlib_id = static_cast<long long>(hnswlib_ids[i]);
      std::string color = hnswlib_bg;
      if (const char* fg = color_for_id(hnswlib_id)) {
        color += fg;
      }
      print_cell(std::cout, std::to_string(hnswlib_id), w_id, color, use_color);
      print_cell(std::cout, format_float(hnswlib_dists[i]), w_dist, hnswlib_bg,
                 use_color);
    } else {
      print_cell(std::cout, "-", w_id, hnswlib_bg, use_color);
      print_cell(std::cout, "-", w_dist, hnswlib_bg, use_color);
    }
    if (!mc_ids[i].empty()) {
      const char* mc_color = nullptr;
      try {
        long long mc_id = std::stoll(mc_ids[i]);
        mc_color = color_for_id(mc_id);
      } catch (const std::exception&) {
        mc_color = nullptr;
      }
      std::string color = manticore_bg;
      if (mc_color) {
        color += mc_color;
      }
      print_cell(std::cout, mc_ids[i], w_id, color, use_color);
      print_cell(std::cout, mc_dists[i], w_dist, manticore_bg, use_color);
    } else {
      print_cell(std::cout, "-", w_id, manticore_bg, use_color);
      print_cell(std::cout, "-", w_dist, manticore_bg, use_color);
    }
    std::cout << "\n";
  }
  return 0;
}
