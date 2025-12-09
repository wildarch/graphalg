#pragma once

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
  std::string name;
  std::size_t nRows;
  std::size_t nCols;
  // TODO: data type.
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
  scan(ScanState &state);

  std::size_t nValues() { return _values.size(); }
};

class PgGraphAlg {
private:
  mlir::DialectRegistry _registry;
  mlir::MLIRContext _ctx;
  std::unordered_map<TableId, MatrixTable> _tables;
  llvm::StringMap<TableId> _nameToId;

public:
  PgGraphAlg(llvm::function_ref<void(mlir::Diagnostic &)> diagHandler);

  MatrixTable &getTable(TableId tableId);
  MatrixTable &getOrCreateTable(TableId tableId, const MatrixTableDef &def);
  MatrixTable *lookupTable(llvm::StringRef tableName);

  bool execute(llvm::StringRef programSource, llvm::StringRef function,
               llvm::ArrayRef<const MatrixTable *> arguments,
               MatrixTable &output);
};

} // namespace pg_graphalg
