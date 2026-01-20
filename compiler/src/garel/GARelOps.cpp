#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"
#include "garel/GARelTypes.h"

#define GET_OP_CLASSES
#include "garel/GARelOps.cpp.inc"

namespace garel {

// === ProjectOp ===
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

ProjectReturnOp ProjectOp::getTerminator() {
  return llvm::cast<ProjectReturnOp>(getProjections().front().getTerminator());
}

// === SelectOp ===
void SelectOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value input) {
  auto region = state.addRegion();
  auto &block = region->emplaceBlock();
  auto inputType = llvm::cast<RelationType>(input.getType());
  block.addArgument(builder.getType<TupleType>(inputType.getColumns()),
                    builder.getUnknownLoc());
  state.addTypes(input.getType());
  state.addOperands(input);
}

mlir::LogicalResult SelectOp::verifyRegions() {
  if (getPredicates().getNumArguments() != 1) {
    return emitOpError("predicates block should have exactly one argument");
  }

  auto blockArg = getPredicates().getArgument(0);
  auto blockType = llvm::dyn_cast<TupleType>(blockArg.getType());
  if (!blockType) {
    return emitOpError("predicates block arg must be of type tuple");
  }

  if (getInput().getType().getColumns() != blockType.getColumns()) {
    return emitOpError("predicates block slots do not match child slots");
  }

  auto terminator = getPredicates().front().getTerminator();
  if (!terminator || !llvm::isa<SelectReturnOp>(terminator)) {
    return emitOpError("predicates block not terminated with select.return");
  }

  return mlir::success();
}

SelectReturnOp SelectOp::getTerminator() {
  return llvm::cast<SelectReturnOp>(getPredicates().front().getTerminator());
}

// === JoinOp ===
mlir::LogicalResult JoinOp::verify() {
  // TODO: Inputs must use distinct columns.
  // TODO: Predicates must refer to columns in distinct inputs (and to columns
  // present in the input).
  return emitOpError("TODO: verify JoinOp");
}

mlir::LogicalResult JoinOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  llvm::SmallVector<ColumnAttr> outputColumns;
  for (auto input : adaptor.getInputs()) {
    auto inputColumns = llvm::cast<RelationType>(input.getType()).getColumns();
    outputColumns.append(inputColumns.begin(), inputColumns.end());
  }

  inferredReturnTypes.push_back(RelationType::get(ctx, outputColumns));
  return mlir::success();
}

} // namespace garel
