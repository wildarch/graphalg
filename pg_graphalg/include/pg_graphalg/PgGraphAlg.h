#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <graphalg/GraphAlgAttr.h>
#include <graphalg/GraphAlgDialect.h>
#include <graphalg/GraphAlgPasses.h>
#include <graphalg/GraphAlgTypes.h>
#include <graphalg/SemiringTypes.h>
#include <graphalg/evaluate/Evaluator.h>
#include <graphalg/parse/Parser.h>

namespace pg_graphalg {

using TableId = unsigned int;

class MatrixTable;

struct ScanState {
  MatrixTable *table;
  std::size_t row = 0;
  std::size_t col = 0;

  ScanState(MatrixTable *table) : table(table) {}

  void reset() {
    row = 0;
    col = 0;
  }
};

struct MatrixTableDef {
  std::size_t nRows;
  std::size_t nCols;
  // TODO: data type.
};

class MatrixTable {
private:
  std::size_t _nRows;
  std::size_t _nCols;
  std::map<std::pair<std::size_t, std::size_t>, std::int64_t> _values;

public:
  MatrixTable(const MatrixTableDef &def);

  void setValue(std::size_t row, std::size_t col, std::int64_t value);
  std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
  scan(ScanState &state);

  std::size_t nValues() { return _values.size(); }
};

class PgGraphAlg {
private:
  mlir::DialectRegistry _registry;
  mlir::MLIRContext _ctx;
  std::unordered_map<TableId, MatrixTable> _tables;

public:
  PgGraphAlg();

  MatrixTable &getTable(TableId tableId);
  MatrixTable &getOrCreateTable(TableId tableId, const MatrixTableDef &def);
};

} // namespace pg_graphalg
