#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"
#include "garel/GARelTypes.h"
#include "llvm/ADT/ArrayRef.h"

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
    if (val.getType() != col) {
      return emitOpError("projections block return types do not match the "
                         "projection output column types");
    }
  }

  return mlir::success();
}

mlir::Block &ProjectOp::createProjectionsBlock() {
  assert(getProjections().empty() && "Already have a projections block");
  auto &block = getProjections().emplaceBlock();
  // Same columns as the input, but as a tuple.
  block.addArgument(
      TupleType::get(getContext(), getInput().getType().getColumns()),
      getInput().getLoc());
  return block;
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
mlir::OpFoldResult JoinOp::fold(FoldAdaptor adaptor) {
  if (getInputs().size() == 1) {
    assert(getInputs()[0].getType() == getType());
    return getInputs()[0];
  }

  return nullptr;
}

mlir::LogicalResult JoinOp::verify() {
  // TODO: Inputs must use distinct columns.
  // TODO: Predicates must refer to columns in distinct inputs (and to columns
  // present in the input).
  return mlir::success();
}

mlir::LogicalResult JoinOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  llvm::SmallVector<mlir::Type> outputColumns;
  for (auto input : adaptor.getInputs()) {
    auto inputColumns = llvm::cast<RelationType>(input.getType()).getColumns();
    outputColumns.append(inputColumns.begin(), inputColumns.end());
  }

  inferredReturnTypes.push_back(RelationType::get(ctx, outputColumns));
  return mlir::success();
}

// === AggregateOp ===
mlir::LogicalResult AggregateOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  llvm::SmallVector<mlir::Type> outputColumns;

  auto inputType = llvm::cast<RelationType>(adaptor.getInput().getType());
  auto inputColumns = inputType.getColumns();

  // Key columns
  for (auto key : adaptor.getGroupBy()) {
    outputColumns.push_back(inputColumns[key]);
  }

  // Aggregator outputs
  for (auto agg : adaptor.getAggregators()) {
    outputColumns.push_back(agg.getResultType(inputType));
  }

  inferredReturnTypes.push_back(RelationType::get(ctx, outputColumns));
  return mlir::success();
}

// === ForOp ===
static mlir::LogicalResult
verifyResultIdx(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                mlir::ValueRange initArgs, std::uint64_t resultIdx) {
  // resultIdx is within bounds of init args.
  if (initArgs.size() <= resultIdx) {
    return emitError() << "has result_idx=" << resultIdx
                       << ", but there are only " << initArgs.size()
                       << " init args";
  }

  return mlir::success();
}

mlir::LogicalResult ForOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  auto loc = location ? *location : mlir::UnknownLoc::get(ctx);
  if (mlir::failed(verifyResultIdx(
          [&]() {
            return mlir::emitError(loc)
                   << ForOp::getOperationName() << " to build with init args "
                   << adaptor.getInit() << " ";
          },
          adaptor.getInit(), adaptor.getResultIdx()))) {
    return mlir::failure();
  }

  auto resultType = adaptor.getInit()[adaptor.getResultIdx()].getType();
  inferredReturnTypes.emplace_back(resultType);
  return mlir::success();
}

mlir::LogicalResult ForOp::verify() {
  return verifyResultIdx([this]() { return emitOpError(); }, getInit(),
                         getResultIdx());
}

mlir::LogicalResult ForOp::verifyRegions() {
  auto initTypes = getInit().getTypes();

  // Body arg types match init args
  auto argTypes = getBody().front().getArgumentTypes();
  if (initTypes != argTypes) {
    return emitOpError("body arg types do not match the initial value types");
  }

  // Body result types match init args
  auto yieldOp = llvm::cast<ForYieldOp>(getBody().front().getTerminator());
  auto resTypes = yieldOp.getInputs().getTypes();
  if (initTypes != resTypes) {
    auto diag =
        emitOpError("body result types do not match the initial value types");
    diag.attachNote(yieldOp.getLoc()) << "body result is here";
    return diag;
  }

  return mlir::success();
}

// === RangeOp ===
mlir::LogicalResult RangeOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      RelationType::get(ctx, {mlir::IndexType::get(ctx)}));
  return mlir::success();
}

// === RemapOp ===
mlir::LogicalResult RemapOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  llvm::SmallVector<mlir::Type> outputColumns;
  auto inputType = llvm::cast<RelationType>(adaptor.getInput().getType());
  for (auto inputCol : adaptor.getRemap()) {
    outputColumns.push_back(inputType.getColumns()[inputCol]);
  }

  inferredReturnTypes.push_back(RelationType::get(ctx, outputColumns));
  return mlir::success();
}

mlir::LogicalResult RemapOp::verify() {
  auto inputColumns = getInput().getType().getColumns();
  for (auto inputCol : getRemap()) {
    if (inputCol >= inputColumns.size()) {
      return emitOpError("remap refers to input column ")
             << inputCol << ", but input only has " << inputColumns.size()
             << " columns";
    }
  }

  return mlir::success();
}

// Checks for mapping [0, 1, 2, ...]
static bool isIdentityRemap(llvm::ArrayRef<ColumnIdx> indexes) {
  for (auto [i, idx] : llvm::enumerate(indexes)) {
    if (i != idx) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult RemapOp::fold(FoldAdaptor adaptor) {
  if (isIdentityRemap(getRemap())) {
    assert(getInput().getType() == getType());
    return getInput();
  }

  return nullptr;
}

// === ConstantOp ===
mlir::LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  llvm::SmallVector<mlir::Type, 1> outputColumns{adaptor.getValue().getType()};
  inferredReturnTypes.push_back(RelationType::get(ctx, outputColumns));
  return mlir::success();
}

} // namespace garel
