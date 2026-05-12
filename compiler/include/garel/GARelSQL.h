#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

namespace garel {

mlir::LogicalResult translateToSQL(mlir::Operation *op, llvm::raw_ostream &os);

} // namespace garel
