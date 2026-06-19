#pragma once

#include <memory>
#include <optional>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include "pg_graphalg/MatrixTable.h"

namespace pg_graphalg {

using TableId = unsigned int;

class PgGraphAlg {
private:
  mlir::DialectRegistry _registry;
  mlir::MLIRContext _ctx;
  llvm::DenseMap<TableId, std::unique_ptr<MatrixTable>> _tables;

public:
  PgGraphAlg(llvm::function_ref<void(mlir::Diagnostic &)> diagHandler);

  std::optional<MatrixTable *> getOrCreateTable(
      TableId tableId,
      llvm::function_ref<std::optional<MatrixTableDef>(TableId id)> createFunc);

  bool execute(llvm::StringRef programSource, llvm::StringRef function,
               llvm::ArrayRef<const MatrixTable *> arguments,
               MatrixTable &output);
};

} // namespace pg_graphalg
