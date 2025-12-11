#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>

namespace pg_graphalg {

enum class MatrixValueType {
  BOOL,
  INT,
  FLOAT,
};

struct MatrixTableDef {
  std::string name;
  std::size_t nRows;
  std::size_t nCols;
  MatrixValueType type;
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
  const MatrixValueType _type;

  using AnyValue = std::variant<bool, std::int64_t, double>;
  std::map<std::pair<std::size_t, std::size_t>, AnyValue> _values;

public:
  MatrixTable(const MatrixTableDef &def)
      : _name(def.name), _nRows(def.nRows), _nCols(def.nCols), _type(def.type) {
  }

  std::size_t nRows() const { return _nRows; }
  std::size_t nCols() const { return _nCols; }
  MatrixValueType getType() const { return _type; }

  std::size_t nValues() { return _values.size(); }

  void clear() { _values.clear(); }

  void setValue(std::size_t row, std::size_t col, AnyValue value) {
    assert(getType() != MatrixValueType::BOOL ||
           std::holds_alternative<bool>(value));
    assert(getType() != MatrixValueType::INT ||
           std::holds_alternative<std::int64_t>(value));
    assert(getType() != MatrixValueType::FLOAT ||
           std::holds_alternative<double>(value));
    _values[{row, col}] = value;
  }

  const auto &values() const { return _values; }

  std::optional<std::tuple<std::size_t, std::size_t, AnyValue>>
  scan(MatrixTableScanState &state) {
    auto it = _values.lower_bound({state.row, state.col});
    if (it == _values.end()) {
      return std::nullopt;
    }

    std::size_t row = it->first.first;
    std::size_t col = it->first.second;
    AnyValue val = it->second;

    state.row = row;
    state.col = col + 1;
    return std::make_tuple(row, col, val);
  }
};

} // namespace pg_graphalg
