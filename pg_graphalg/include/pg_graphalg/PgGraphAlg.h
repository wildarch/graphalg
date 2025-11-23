#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

extern "C" {
#include "GraphBLAS.h"
}

namespace pg_graphalg {

struct ScanState {
  std::size_t row = 0;
  std::size_t col = 0;

  void reset() {
    row = 0;
    col = 0;
  }
};

class PgGraphAlg {
private:
  std::map<std::pair<std::size_t, std::size_t>, std::int64_t> _values;

public:
  PgGraphAlg();

  std::size_t size() { return _values.size(); }
  void addTuple(std::size_t row, std::size_t col, std::int64_t value);
  std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
  scan(ScanState &state);
};

} // namespace pg_graphalg
