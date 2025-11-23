#include <optional>
#include <tuple>

extern "C" {
#include "GraphBLAS.h"
}

#include "pg_graphalg/PgGraphAlg.h"

namespace pg_graphalg {

PgGraphAlg::PgGraphAlg() {
  // TODO: init graphblas
}

void PgGraphAlg::addTuple(std::size_t row, std::size_t col,
                          std::int64_t value) {
  _values[{row, col}] = value;
}

std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
PgGraphAlg::scan(ScanState &state) {
  auto it = _values.lower_bound({state.row, state.col});
  if (it == _values.end()) {
    return std::nullopt;
  }

  std::size_t row = it->first.first;
  std::size_t col = it->first.second;
  std::int64_t val = it->second;

  state.row = row;
  state.col = col + 1;
  return std::make_tuple(row, col, val);
}

} // namespace pg_graphalg
