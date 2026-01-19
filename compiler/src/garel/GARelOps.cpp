#include <llvm/ADT/STLExtras.h>

#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"

#define GET_OP_CLASSES
#include "garel/GARelOps.cpp.inc"

namespace garel {

mlir::LogicalResult ProjectOp::verifyRegions() {
  if (getProjections().getNumArguments() != 1) {
    return emitOpError("projections block should have exactly one argument");
  }

  auto blockArg = getProjections().getArgument(0);
  auto blockType = llvm::dyn_cast<TupleType>(blockArg.getType());
  if (!blockType) {
    return emitOpError("projections block arg must be of type tuple");
  }

  if (getInput().getType().getColumns() != blockType.getColumns()) {
    return emitOpError("projections block columns do not match input columns");
  }

  auto terminator = getProjections().front().getTerminator();
  if (!terminator) {
    return emitOpError("missing return from projections block");
  }

  auto returnOp = llvm::dyn_cast<ProjectReturnOp>(terminator);
  if (!returnOp) {
    return emitOpError("projections block not terminated by project.return");
  }

  if (returnOp.getProjections().size() != getType().getColumns().size()) {
    return emitOpError("projections block returns a different number of "
                       "values than specified in the projection return type");
  }

  for (const auto &[val, col] :
       llvm::zip_equal(returnOp.getProjections(), getType().getColumns())) {
    if (val.getType() != col.getType()) {
      return emitOpError("projections block return types do not match the "
                         "projection output column types");
    }
  }

  return mlir::success();
}

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
