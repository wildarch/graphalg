#include <pg_graphalg/MatrixTable.h>

namespace pg_graphalg {

MatrixTable::MatrixTable(const MatrixTableDef &def)
    : _name(def.name), _nRows(def.nRows), _nCols(def.nCols) {}

void MatrixTable::clear() { _values.clear(); }

void MatrixTable::setValue(std::size_t row, std::size_t col,
                           std::int64_t value) {
  _values[{row, col}] = value;
}

std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
MatrixTable::scan(MatrixTableScanState &state) {
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
