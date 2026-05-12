#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace garel {

mlir::LogicalResult translateToSQL(mlir::Operation *op, llvm::raw_ostream &os) {
  return mlir::success();
}

} // namespace garel
