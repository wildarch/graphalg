#include <numeric>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"
#include "garel/GARelTypes.h"
#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgOps.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"

namespace garel {

#define GEN_PASS_DEF_GRAPHALGTOREL
#include "garel/GARelPasses.h.inc"

namespace {

class GraphAlgToRel : public impl::GraphAlgToRelBase<GraphAlgToRel> {
public:
  using impl::GraphAlgToRelBase<GraphAlgToRel>::GraphAlgToRelBase;

  void runOnOperation() final;
};

/** Converts semiring types into their relational equivalents. */
class SemiringTypeConverter : public mlir::TypeConverter {
private:
  static mlir::Type convertSemiringType(graphalg::SemiringTypeInterface type);

public:
  SemiringTypeConverter();
};

/** Converts matrix types into relations. */
class MatrixTypeConverter : public mlir::TypeConverter {
private:
  SemiringTypeConverter _semiringConverter;

  mlir::FunctionType convertFunctionType(mlir::FunctionType type) const;
  RelationType convertMatrixType(graphalg::MatrixType type) const;

public:
  MatrixTypeConverter(mlir::MLIRContext *ctx,
                      const SemiringTypeConverter &semiringConverter);
};

/**
 * Convenient wrapper around a matrix value and its relation equivalent
 * after type conversion.
 *
 * This class is particularly useful for retrieving the relation column for
 * the rows, columns or values of the matrix.
 */
class MatrixAdaptor {
private:
  mlir::TypedValue<graphalg::MatrixType> _matrix;

  RelationType _relType;
  // May be null for outputs, in which case only the relation type is available.
  mlir::TypedValue<RelationType> _relation;

public:
  // For output matrices, where we only have the desired output type.
  MatrixAdaptor(mlir::Value matrix, mlir::Type relType)
      : _matrix(llvm::cast<mlir::TypedValue<graphalg::MatrixType>>(matrix)),
        _relType(llvm::cast<RelationType>(relType)) {}

  // For input matrices, where the OpAdaptor provides the relation value.
  MatrixAdaptor(mlir::Value matrix, mlir::Value relation)
      : MatrixAdaptor(matrix, relation.getType()) {
    this->_relation = llvm::cast<mlir::TypedValue<RelationType>>(relation);
  }

  graphalg::MatrixType matrixType() { return _matrix.getType(); }

  RelationType relType() { return _relType; }

  mlir::TypedValue<RelationType> relation() {
    assert(!!_relation && "No relation value (only type)");
    return _relation;
  }

  auto columns() const { return _relType.getColumns(); }

  bool isScalar() const { return _matrix.getType().isScalar(); }

  bool hasRowColumn() const { return !_matrix.getType().getRows().isOne(); }

  bool hasColColumn() const { return !_matrix.getType().getCols().isOne(); }

  ColumnIdx rowColumn() const {
    assert(hasRowColumn());
    return 0;
  }

  ColumnIdx colColumn() const {
    assert(hasColColumn());
    // Follow row column, if there is one.
    return hasRowColumn() ? 1 : 0;
  }

  ColumnIdx valColumn() const {
    // Last column in the relation.
    return columns().size() - 1;
  }

  graphalg::SemiringTypeInterface semiring() {
    return llvm::cast<graphalg::SemiringTypeInterface>(
        _matrix.getType().getSemiring());
  }
};

template <typename T> class OpConversion : public mlir::OpConversionPattern<T> {
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op,
                  typename mlir::OpConversionPattern<T>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ApplyOpConversion : public mlir::OpConversionPattern<graphalg::ApplyOp> {
private:
  const SemiringTypeConverter &_bodyArgConverter;

  mlir::LogicalResult
  matchAndRewrite(graphalg::ApplyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

public:
  ApplyOpConversion(const SemiringTypeConverter &bodyArgConverter,
                    const MatrixTypeConverter &typeConverter,
                    mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<graphalg::ApplyOp>(typeConverter, ctx),
        _bodyArgConverter(bodyArgConverter) {}
};

struct InputColumnRef {
  unsigned relIdx;
  ColumnIdx colIdx;
  ColumnIdx outIdx;
};

} // namespace

// =============================================================================
// =============================== Class Methods ===============================
// =============================================================================

mlir::Type
SemiringTypeConverter::convertSemiringType(graphalg::SemiringTypeInterface t) {
  auto *ctx = t.getContext();
  // To i1
  if (t == graphalg::SemiringTypes::forBool(ctx)) {
    return mlir::IntegerType::get(ctx, 1);
  }

  // To i64
  if (t == graphalg::SemiringTypes::forInt(ctx) ||
      t == graphalg::SemiringTypes::forTropInt(ctx) ||
      t == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    return mlir::IntegerType::get(ctx, 64);
  }

  // To f64
  if (t == graphalg::SemiringTypes::forReal(ctx) ||
      t == graphalg::SemiringTypes::forTropReal(ctx)) {
    return mlir::Float64Type::get(ctx);
  }

  return nullptr;
}

SemiringTypeConverter::SemiringTypeConverter() {
  addConversion(convertSemiringType);
}

mlir::FunctionType
MatrixTypeConverter::convertFunctionType(mlir::FunctionType type) const {
  llvm::SmallVector<mlir::Type> inputs;
  if (mlir::failed(convertTypes(type.getInputs(), inputs))) {
    return {};
  }

  llvm::SmallVector<mlir::Type> results;
  if (mlir::failed(convertTypes(type.getResults(), results))) {
    return {};
  }

  return mlir::FunctionType::get(type.getContext(), inputs, results);
}

RelationType
MatrixTypeConverter::convertMatrixType(graphalg::MatrixType type) const {
  llvm::SmallVector<mlir::Type> columns;
  auto *ctx = type.getContext();
  if (!type.getRows().isOne()) {
    columns.push_back(mlir::IndexType::get(ctx));
  }

  if (!type.getCols().isOne()) {
    columns.push_back(mlir::IndexType::get(ctx));
  }

  auto valueType = _semiringConverter.convertType(type.getSemiring());
  if (!valueType) {
    return {};
  }

  columns.push_back(valueType);
  return RelationType::get(ctx, columns);
}

MatrixTypeConverter::MatrixTypeConverter(
    mlir::MLIRContext *ctx, const SemiringTypeConverter &semiringConverter)
    : _semiringConverter(semiringConverter) {
  addConversion(
      [this](mlir::FunctionType t) { return convertFunctionType(t); });

  addConversion(
      [this](graphalg::MatrixType t) { return convertMatrixType(t); });
}

// =============================================================================
// ============================== Helper Methods ===============================
// =============================================================================

/**
 * Create a relation with all indices for a matrix dimension.
 *
 * Used to broadcast scalar values to a larger matrix.
 */
static RangeOp createDimRead(mlir::Location loc, graphalg::DimAttr dim,
                             mlir::OpBuilder &builder) {
  return builder.create<RangeOp>(loc, dim.getConcreteDim());
}

static void
buildApplyJoinPredicates(mlir::MLIRContext *ctx,
                         llvm::SmallVectorImpl<JoinPredicateAttr> &predicates,
                         llvm::ArrayRef<InputColumnRef> columnsToJoin) {
  if (columnsToJoin.size() < 2) {
    return;
  }

  auto first = columnsToJoin.front();
  for (auto other : columnsToJoin.drop_front()) {
    predicates.push_back(JoinPredicateAttr::get(ctx, first.relIdx, first.colIdx,
                                                other.relIdx, other.colIdx));
  }
}

static mlir::FailureOr<mlir::TypedAttr> convertConstant(mlir::Operation *op,
                                                        mlir::TypedAttr attr) {
  auto *ctx = attr.getContext();
  auto type = attr.getType();
  if (type == graphalg::SemiringTypes::forBool(ctx)) {
    return attr;
  } else if (type == graphalg::SemiringTypes::forInt(ctx)) {
    // TODO: Need to convert to signed?
    return attr;
  } else if (type == graphalg::SemiringTypes::forReal(ctx)) {
    return attr;
  } else if (type == graphalg::SemiringTypes::forTropInt(ctx)) {
    std::int64_t value;
    if (llvm::isa<graphalg::TropInfAttr>(attr)) {
      // Positive infinity, kind of.
      value = std::numeric_limits<std::int64_t>::max();
    } else {
      auto intAttr = llvm::cast<graphalg::TropIntAttr>(attr);
      value = intAttr.getValue().getValue().getSExtValue();
    }

    return mlir::TypedAttr(
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value));
  } else if (type == graphalg::SemiringTypes::forTropReal(ctx)) {
    double value;
    if (llvm::isa<graphalg::TropInfAttr>(attr)) {
      // Has a proper positive infinity value
      value = std::numeric_limits<double>::infinity();
    } else {
      auto floatAttr = llvm::cast<graphalg::TropFloatAttr>(attr);
      value = floatAttr.getValue().getValueAsDouble();
    }

    return mlir::TypedAttr(
        mlir::FloatAttr::get(mlir::Float64Type::get(ctx), value));
  } else if (type == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    std::int64_t value;
    if (llvm::isa<graphalg::TropInfAttr>(attr)) {
      // Negative infinity, kind of.
      value = std::numeric_limits<std::int64_t>::min();
    } else {
      auto intAttr = llvm::cast<graphalg::TropIntAttr>(attr);
      value = intAttr.getValue().getValue().getSExtValue();
    }

    return mlir::TypedAttr(
        mlir::FloatAttr::get(mlir::Float64Type::get(ctx), value));
  }

  return op->emitOpError("cannot convert constant ") << attr;
}

// =============================================================================
// =============================== Op Conversion ===============================
// =============================================================================

template <>
mlir::LogicalResult OpConversion<mlir::func::FuncOp>::matchAndRewrite(
    mlir::func::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto funcType = llvm::cast_if_present<mlir::FunctionType>(
      typeConverter->convertType(op.getFunctionType()));
  if (!funcType) {
    return op->emitOpError("function type ")
           << op.getFunctionType() << " cannot be converted";
  }

  auto result = mlir::success();
  rewriter.modifyOpInPlace(op, [&]() {
    // Update function type.
    op.setFunctionType(funcType);
  });

  // Convert block args.
  mlir::TypeConverter::SignatureConversion newSig(funcType.getNumInputs());
  if (mlir::failed(
          rewriter.convertRegionTypes(&op.getFunctionBody(), *typeConverter))) {
    return mlir::failure();
  }

  return result;
}

template <>
mlir::LogicalResult OpConversion<mlir::func::ReturnOp>::matchAndRewrite(
    mlir::func::ReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.modifyOpInPlace(op,
                           [&]() { op->setOperands(adaptor.getOperands()); });

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::TransposeOp>::matchAndRewrite(
    graphalg::TransposeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  auto projectOp = rewriter.replaceOpWithNewOp<ProjectOp>(op, output.relType(),
                                                          input.relation());

  auto &body = projectOp.createProjectionsBlock();
  rewriter.setInsertionPointToStart(&body);

  llvm::SmallVector<ColumnIdx, 3> columns(input.columns().size());
  std::iota(columns.begin(), columns.end(), 0);
  assert(columns.size() <= 3);
  // Transpose is a no-op if there are fewer than 3 columns.
  if (columns.size() == 3) {
    // Swap row and column
    std::swap(columns[0], columns[1]);
  }

  // Return the input slots (after row and column have been swapped)
  llvm::SmallVector<mlir::Value, 3> results;
  for (auto col : columns) {
    results.emplace_back(
        rewriter.create<ExtractOp>(op.getLoc(), col, body.getArgument(0)));
  }

  rewriter.create<ProjectReturnOp>(op.getLoc(), results);

  return mlir::success();
}

static constexpr llvm::StringLiteral APPLY_ROW_IDX_ATTR_KEY =
    "garel.apply.row_idx";
static constexpr llvm::StringLiteral APPLY_COL_IDX_ATTR_KEY =
    "garel.apply.col_idx";

mlir::LogicalResult ApplyOpConversion::matchAndRewrite(
    graphalg::ApplyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<MatrixAdaptor> inputs;
  for (auto [matrix, relation] :
       llvm::zip_equal(op.getInputs(), adaptor.getInputs())) {
    auto &input = inputs.emplace_back(matrix, relation);
  }

  llvm::SmallVector<mlir::Value> joinChildren;
  llvm::SmallVector<InputColumnRef> rowColumns;
  llvm::SmallVector<InputColumnRef> colColumns;
  llvm::SmallVector<ColumnIdx> valColumns;
  ColumnIdx nextColumnIdx = 0;
  for (const auto &[idx, input] : llvm::enumerate(inputs)) {
    joinChildren.emplace_back(input.relation());

    if (input.hasRowColumn()) {
      rowColumns.push_back(InputColumnRef{
          .relIdx = static_cast<unsigned int>(idx),
          .colIdx = input.rowColumn(),
          .outIdx = nextColumnIdx + input.rowColumn(),
      });
    }

    if (input.hasColColumn()) {
      colColumns.push_back(InputColumnRef{
          .relIdx = static_cast<unsigned int>(idx),
          .colIdx = input.colColumn(),
          .outIdx = nextColumnIdx + input.colColumn(),
      });
    }

    valColumns.push_back(nextColumnIdx + input.valColumn());
    nextColumnIdx += input.columns().size();
  }

  auto outputType = typeConverter->convertType(op.getType());
  MatrixAdaptor output(op.getResult(), outputType);
  if (rowColumns.empty() && output.hasRowColumn()) {
    // None of the inputs have a row column, but we need it in the output.
    // Broadcast to all rows.
    auto rowsOp =
        createDimRead(op.getLoc(), output.matrixType().getRows(), rewriter);
    joinChildren.emplace_back(rowsOp);
    rowColumns.push_back(InputColumnRef{
        .relIdx = static_cast<unsigned int>(joinChildren.size() - 1),
        .colIdx = 0,
        .outIdx = nextColumnIdx++,
    });
  }

  if (colColumns.empty() && output.hasColColumn()) {
    // None of the inputs have a col column, but we need it in the output.
    // Broadcast to all columns.
    auto colsOp =
        createDimRead(op.getLoc(), output.matrixType().getCols(), rewriter);
    joinChildren.emplace_back(colsOp);
    colColumns.push_back(InputColumnRef{
        .relIdx = static_cast<unsigned int>(joinChildren.size() - 1),
        .colIdx = 0,
        .outIdx = nextColumnIdx++,
    });
  }

  mlir::Value joined;
  if (joinChildren.size() == 1) {
    joined = joinChildren.front();
  } else {
    llvm::SmallVector<JoinPredicateAttr> predicates;
    buildApplyJoinPredicates(rewriter.getContext(), predicates, rowColumns);
    buildApplyJoinPredicates(rewriter.getContext(), predicates, colColumns);
    joined = rewriter.create<JoinOp>(op.getLoc(), joinChildren, predicates);
  }

  auto projectOp = rewriter.create<ProjectOp>(op->getLoc(), outputType, joined);

  // Convert old body
  if (mlir::failed(
          rewriter.convertRegionTypes(&op.getBody(), _bodyArgConverter))) {
    return op->emitOpError("failed to convert body argument types");
  }

  // Read value columns, to be used as arg replacements for the old body.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto &body = projectOp.createProjectionsBlock();
  rewriter.setInsertionPointToStart(&body);

  llvm::SmallVector<mlir::Value> slotReads;
  for (auto col : valColumns) {
    slotReads.emplace_back(
        rewriter.create<ExtractOp>(op->getLoc(), col, body.getArgument(0)));
  }

  // Inline into new body
  rewriter.inlineBlockBefore(&op.getBody().front(), &body, body.end(),
                             slotReads);

  rewriter.replaceOp(op, projectOp);

  // Attach the row and column slot to the return op.
  auto returnOp = llvm::cast<graphalg::ApplyReturnOp>(body.getTerminator());
  if (!rowColumns.empty()) {
    returnOp->setAttr(APPLY_ROW_IDX_ATTR_KEY,
                      rewriter.getI32IntegerAttr(rowColumns[0].outIdx));
  }

  if (!colColumns.empty()) {
    returnOp->setAttr(APPLY_COL_IDX_ATTR_KEY,
                      rewriter.getI32IntegerAttr(colColumns[0].outIdx));
  }

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::ApplyReturnOp>::matchAndRewrite(
    graphalg::ApplyReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<mlir::Value> results;

  // Note: conversion is done top-down, so the ApplyOp is converted to
  // ProjectOp before we reach this op in its body.
  auto inputTuple = op->getBlock()->getArgument(0);

  if (auto idx = op->getAttrOfType<mlir::IntegerAttr>(APPLY_ROW_IDX_ATTR_KEY)) {
    results.emplace_back(
        rewriter.create<ExtractOp>(op->getLoc(), idx, inputTuple));
  }

  if (auto idx = op->getAttrOfType<mlir::IntegerAttr>(APPLY_COL_IDX_ATTR_KEY)) {
    results.emplace_back(
        rewriter.create<ExtractOp>(op->getLoc(), idx, inputTuple));
  }

  // The value slot
  results.emplace_back(adaptor.getValue());

  rewriter.replaceOpWithNewOp<ProjectReturnOp>(op, results);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::BroadcastOp>::matchAndRewrite(
    graphalg::BroadcastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  llvm::SmallVector<mlir::Value> joinChildren;
  if (input.hasRowColumn()) {
    // Already have a row column.
    // TODO: record row column.
  } else if (output.hasRowColumn()) {
    // Broadcast over all rows.
    joinChildren.push_back(
        createDimRead(op.getLoc(), output.matrixType().getRows(), rewriter));
    // TODO: record row column.
  }

  if (input.hasColColumn()) {
    // Already have a col column.
    // TODO: record col column.
  } else if (output.hasColColumn()) {
    // Broadcast over all columns.
    joinChildren.push_back(
        createDimRead(op.getLoc(), output.matrixType().getCols(), rewriter));
    // TODO: record col column.
  }

  joinChildren.push_back(input.relation());
  // TODO: record val column.

  /*
  // Join with a dim read for row/col slots that we want in the output, but do
  // not have on the input.
  llvm::SmallVector<ipr::SlotOffset> renameSlots;
  llvm::SmallVector<mlir::Value> joinChildren;
  if (auto rowSlot = input.rowSlot()) {
      // Already have a row slot.
      renameSlots.emplace_back(rowSlot.getSlot());
  } else if (auto rowSlot = output.rowSlot()) {
      // Broadcast over all rows.
      joinChildren.emplace_back(
              createDimRead(op.getLoc(), rowSlot, rewriter));
      renameSlots.emplace_back(rowSlot.getSlot());
  }

  if (auto colSlot = input.colSlot()) {
      // Already have a col slot.
      renameSlots.emplace_back(colSlot.getSlot());
  } else if (auto colSlot = output.colSlot()) {
      // Broadcast over all columns.
      joinChildren.emplace_back(
              createDimRead(op.getLoc(), colSlot, rewriter));
      renameSlots.emplace_back(colSlot.getSlot());
  }

  joinChildren.emplace_back(input.stream());
  renameSlots.emplace_back(input.valSlot().getSlot());

  auto joinOp = rewriter.create<ipr::JoinOp>(
          op.getLoc(),
          joinChildren);
  {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto& body = joinOp.getPredicates().front();
      rewriter.setInsertionPointToStart(&body);
      // No predicates
      rewriter.create<ipr::JoinReturnOp>(op.getLoc(), std::nullopt);
  }

  // Rename to the desired output slots. This also handles reordering slots.
  // We want (row, col, val) order, but the join output could be e.g.
  // (col, row, val) if the input does not have a col slot.
  rewriter.replaceOpWithNewOp<ipr::RenameOp>(
          op,
          output.streamType(),
          joinOp,
          rewriter.getAttr<ipr::ArrayOfSlotOffsetAttr>(renameSlots));
  return mlir::success();
  */
  return mlir::failure();
}

// =============================================================================
// ============================ Tuple Op Conversion ============================
// =============================================================================

template <>
mlir::LogicalResult OpConversion<graphalg::ConstantOp>::matchAndRewrite(
    graphalg::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto value = convertConstant(op, op.getValue());
  if (mlir::failed(value)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, *value);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::AddOp>::matchAndRewrite(
    graphalg::AddOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto sring = op.getType();
  auto *ctx = rewriter.getContext();
  if (sring == graphalg::SemiringTypes::forBool(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forInt(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forReal(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forTropInt(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::MinSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forTropReal(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::MinimumFOp>(op, adaptor.getLhs(),
                                                         adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::MaxSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
  } else {
    return op->emitOpError("conversion not supported for semiring ") << sring;
  }

  return mlir::success();
}

static bool hasRelationSignature(mlir::func::FuncOp op) {
  // All inputs should be relations
  auto funcType = op.getFunctionType();
  for (auto input : funcType.getInputs()) {
    if (!llvm::isa<RelationType>(input)) {
      return false;
    }
  }

  // There should be exactly one relation result
  return funcType.getNumResults() == 1 &&
         llvm::isa<RelationType>(funcType.getResult(0));
}

static bool hasRelationOperands(mlir::Operation *op) {
  return llvm::all_of(op->getOperandTypes(),
                      [](auto t) { return llvm::isa<RelationType>(t); });
}

void GraphAlgToRel::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  // Eliminate all graphalg ops
  target.addIllegalDialect<graphalg::GraphAlgDialect>();
  // Turn them into relational ops.
  target.addLegalDialect<GARelDialect>();
  // and arith ops for the scalar operations.
  target.addLegalDialect<mlir::arith::ArithDialect>();
  // Keep container module.
  target.addLegalOp<mlir::ModuleOp>();
  // Keep functions, but change their signature.
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(hasRelationSignature);
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>(hasRelationOperands);

  SemiringTypeConverter semiringTypeConverter;
  MatrixTypeConverter matrixTypeConverter(&getContext(), semiringTypeConverter);

  mlir::RewritePatternSet patterns(&getContext());
  patterns
      .add<OpConversion<mlir::func::FuncOp>, OpConversion<mlir::func::ReturnOp>,
           OpConversion<graphalg::TransposeOp>>(matrixTypeConverter,
                                                &getContext());
  patterns.add<ApplyOpConversion>(semiringTypeConverter, matrixTypeConverter,
                                  &getContext());

  // Scalar patterns.
  // patterns.add(convertArithConstant);
  patterns
      .add<OpConversion<graphalg::ApplyReturnOp>,
           OpConversion<graphalg::ConstantOp>, OpConversion<graphalg::AddOp>>(
          semiringTypeConverter, &getContext());

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace garel
