#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

namespace garel {

enum class SQLDialect {
  DUCKDB_PYTHON,
  UMBRA_ITERATE,
};

mlir::LogicalResult translateToSQL(mlir::Operation *op, llvm::raw_ostream &os,
                                   SQLDialect dialect);

} // namespace garel
