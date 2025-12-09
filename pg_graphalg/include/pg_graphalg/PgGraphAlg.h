#pragma once

#include <memory>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include "pg_graphalg/MatrixTable.h"

namespace pg_graphalg {

using TableId = unsigned int;

class PgGraphAlg {
private:
  mlir::DialectRegistry _registry;
  mlir::MLIRContext _ctx;
  llvm::DenseMap<TableId, std::unique_ptr<MatrixTable>> _tables;
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
