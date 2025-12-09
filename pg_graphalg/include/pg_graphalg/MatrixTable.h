#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace pg_graphalg {

struct MatrixTableDef {
  std::string name;
  std::size_t nRows;
  std::size_t nCols;
  // TODO: data type.
};

class MatrixTable;

struct MatrixTableScanState {
  MatrixTable *table;
  std::size_t row = 0;
  std::size_t col = 0;

  MatrixTableScanState(MatrixTable *table) : table(table) {}

  void reset() {
    row = 0;
    col = 0;
  }
};

class MatrixTable {
private:
  std::string _name;
  std::size_t _nRows;
  std::size_t _nCols;
  std::map<std::pair<std::size_t, std::size_t>, std::int64_t> _values;

public:
  MatrixTable(const MatrixTableDef &def);

  std::size_t nRows() const { return _nRows; }
  std::size_t nCols() const { return _nCols; }
  const auto &values() const { return _values; }

  void clear();
  void setValue(std::size_t row, std::size_t col, std::int64_t value);
  std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
  scan(MatrixTableScanState &state);

  std::size_t nValues() { return _values.size(); }
};

} // namespace pg_graphalg
