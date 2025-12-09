#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

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

public:
  MatrixTable(const MatrixTableDef &def)
      : _name(def.name), _nRows(def.nRows), _nCols(def.nCols), _type(def.type) {
  }

  virtual ~MatrixTable() = default;

  std::size_t nRows() const { return _nRows; }
  std::size_t nCols() const { return _nCols; }
  MatrixValueType getType() const { return _type; }

  virtual std::size_t nValues() = 0;
  virtual void clear() = 0;
};

class MatrixTableInt : public MatrixTable {
private:
  std::map<std::pair<std::size_t, std::size_t>, std::int64_t> _values;

public:
  MatrixTableInt(const MatrixTableDef &def) : MatrixTable(def) {}

  static bool classof(const MatrixTable *t) {
    return t->getType() == MatrixValueType::INT;
  }

  std::size_t nValues() override { return _values.size(); }
  void clear() override { _values.clear(); }

  void setValue(std::size_t row, std::size_t col, std::int64_t value) {
    _values[{row, col}] = value;
  }

  const auto &values() const { return _values; }

  std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
  scan(MatrixTableScanState &state);
};

} // namespace pg_graphalg
