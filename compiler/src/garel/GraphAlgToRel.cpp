#include <numeric>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"
#include "garel/GARelTypes.h"
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
  MatrixAdaptor(mlir::Value matrix, mlir::Type streamType)
      : _matrix(llvm::cast<mlir::TypedValue<graphalg::MatrixType>>(matrix)),
        _relType(llvm::cast<RelationType>(streamType)) {}

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

  auto columns() { return _relType.getColumns(); }

  bool isScalar() { return _matrix.getType().isScalar(); }

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

} // namespace

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
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
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
  // Eliminate all graphalg ops (and the few ops we use from arith) ...
  target.addIllegalDialect<graphalg::GraphAlgDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect>();
  // And turn them into relational ops.
  target.addLegalDialect<GARelDialect>();
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

  // Scalar patterns.

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace garel
