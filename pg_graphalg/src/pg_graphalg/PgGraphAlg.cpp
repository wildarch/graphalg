#include <cassert>
#include <iostream>
#include <optional>
#include <tuple>

#include <pg_graphalg/PgGraphAlg.h>

namespace pg_graphalg {

MatrixTable::MatrixTable(const MatrixTableDef &def)
    : _name(def.name), _nRows(def.nRows), _nCols(def.nCols) {}

void MatrixTable::setValue(std::size_t row, std::size_t col,
                           std::int64_t value) {
  _values[{row, col}] = value;
}

std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
MatrixTable::scan(ScanState &state) {
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

static mlir::DialectRegistry createDialectRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<graphalg::GraphAlgDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::func::registerInlinerExtension(registry);
  return registry;
}

PgGraphAlg::PgGraphAlg()
    : _registry(createDialectRegistry()), _ctx(_registry) {}

MatrixTable &PgGraphAlg::getTable(TableId tableId) {
  assert(_tables.count(tableId) && "getTable called before getOrCreateTable");
  return _tables.at(tableId);
}

MatrixTable &PgGraphAlg::getOrCreateTable(TableId tableId,
                                          const MatrixTableDef &def) {
  if (!_tables.count(tableId)) {
    _tables.emplace(tableId, def);
    _nameToId[def.name] = tableId;
    std::cerr << "registered table " << def.name << "\n";
  }

  return getTable(tableId);
}

MatrixTable *PgGraphAlg::lookupTable(llvm::StringRef tableName) {
  if (_nameToId.contains(tableName)) {
    return &getTable(_nameToId[tableName]);
  } else {
    return nullptr;
  }
}

} // namespace pg_graphalg
