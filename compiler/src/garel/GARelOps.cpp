#include <llvm/ADT/STLExtras.h>

#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"

#define GET_OP_CLASSES
#include "garel/GARelOps.cpp.inc"

namespace garel {

mlir::LogicalResult ExtractOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getColumn().getType());
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  auto columns = getTuple().getType().getColumns();
  if (!llvm::is_contained(columns, getColumn())) {
    return emitOpError("column ")
           << getColumn() << " not included in tuple " << getTuple().getType();
  }

  return mlir::success();
}

} // namespace garel
