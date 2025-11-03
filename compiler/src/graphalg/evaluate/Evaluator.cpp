#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgOps.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"
#include "graphalg/evaluate/Evaluator.h"

namespace graphalg {

namespace {

class Evaluator {
private:
  llvm::DenseMap<mlir::Value, MatrixAttr> _values;

  mlir::LogicalResult evaluate(TransposeOp op);
  mlir::LogicalResult evaluate(DiagOp op);
  mlir::LogicalResult evaluate(MatMulOp op);
  mlir::LogicalResult evaluate(ReduceOp op);
  mlir::LogicalResult evaluate(BroadcastOp op);
  mlir::LogicalResult evaluate(ConstantMatrixOp op);
  mlir::LogicalResult evaluate(ForConstOp op);
  mlir::LogicalResult evaluate(mlir::Operation *op);

  MatrixAttr value(mlir::Value v) { return _values.at(v); }

public:
  MatrixAttr evaluate(mlir::func::FuncOp funcOp,
                      llvm::ArrayRef<MatrixAttr> args);
};

} // namespace

mlir::LogicalResult Evaluator::evaluate(TransposeOp op) {
  MatrixAttrReader input(_values[op.getInput()]);
  MatrixAttrBuilder result(op.getType());
  for (std::size_t row = 0; row < input.nRows(); row++) {
    for (std::size_t col = 0; col < input.nCols(); col++) {
      result.set(col, row, input.at(row, col));
    }
  }

  _values[op.getResult()] = result.build();
  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(DiagOp op) {
  MatrixAttrReader input(_values[op.getInput()]);
  MatrixAttrBuilder result(op.getType());

  for (std::size_t row = 0; row < input.nRows(); row++) {
    result.set(row, row, input.at(row, 0));
  }

  _values[op.getResult()] = result.build();
  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(MatMulOp op) {
  MatrixAttrReader lhs(_values[op.getLhs()]);
  MatrixAttrReader rhs(_values[op.getRhs()]);
  MatrixAttrBuilder result(op.getType());

  auto ring = result.ring();
  // result[row, col] = SUM{i}(lhs[row, i] * rhs[i, col])
  for (std::size_t row = 0; row < lhs.nRows(); row++) {
    for (std::size_t col = 0; col < rhs.nCols(); col++) {
      auto value = ring.addIdentity();
      for (std::size_t i = 0; i < lhs.nCols(); i++) {
        value = ring.add(value, ring.mul(lhs.at(row, i), rhs.at(i, col)));
      }

      result.set(row, col, value);
    }
  }

  _values[op.getResult()] = result.build();
  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ReduceOp op) {
  MatrixAttrReader input(_values[op.getInput()]);
  MatrixAttrBuilder result(op.getType());

  auto ring = result.ring();
  if (op.getType().isScalar()) {
    // Reduce all to a single value.
    auto value = ring.addIdentity();
    for (std::size_t row = 0; row < input.nRows(); row++) {
      for (std::size_t col = 0; col < input.nCols(); col++) {
        value = ring.add(value, input.at(row, col));
      }
    }

    result.set(0, 0, value);
  } else if (op.getType().isColumnVector()) {
    // Per-row reduce.
    for (std::size_t row = 0; row < input.nRows(); row++) {
      auto value = ring.addIdentity();
      for (std::size_t col = 0; col < input.nCols(); col++) {
        value = ring.add(value, input.at(row, col));
      }

      result.set(row, 0, value);
    }
  } else if (op.getType().isRowVector()) {
    // Per-column reduce.
    for (std::size_t col = 0; col < input.nCols(); col++) {
      auto value = ring.addIdentity();
      for (std::size_t row = 0; row < input.nRows(); row++) {
        value = ring.add(value, input.at(row, col));
      }

      result.set(0, col, value);
    }
  } else {
    // Reduce nothing.
    return op.emitOpError("Not reducing along any dimension");
  }

  _values[op.getResult()] = result.build();
  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(BroadcastOp op) {
  MatrixAttrReader input(_values[op.getInput()]);
  MatrixAttrBuilder result(op.getType());

  for (std::size_t row = 0; row < result.nRows(); row++) {
    for (std::size_t col = 0; col < result.nCols(); col++) {
      auto inRow = input.nRows() == 1 ? 0 : row;
      auto inCol = input.nCols() == 1 ? 0 : col;
      result.set(row, col, input.at(inRow, inCol));
    }
  }

  _values[op.getResult()] = result.build();
  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ConstantMatrixOp op) {
  MatrixAttrBuilder result(op.getType());

  for (std::size_t row = 0; row < result.nRows(); row++) {
    for (std::size_t col = 0; col < result.nCols(); col++) {
      result.set(row, col, op.getValue());
    }
  }

  _values[op.getResult()] = result.build();
  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(ForConstOp op) {
  MatrixAttrReader rangeBeginMat(_values[op.getRangeBegin()]);
  MatrixAttrReader rangeEndMat(_values[op.getRangeEnd()]);
  auto rangeBegin =
      llvm::cast<mlir::IntegerAttr>(rangeBeginMat.at(0, 0)).getInt();
  auto rangeEnd = llvm::cast<mlir::IntegerAttr>(rangeEndMat.at(0, 0)).getInt();

  auto &body = op.getBody().front();
  auto *ctx = op.getContext();

  // Initialize block arguments
  for (auto [init, blockArg] :
       llvm::zip_equal(op.getInitArgs(), body.getArguments().drop_front())) {
    _values[blockArg] = _values[init];
  }

  for (auto i : llvm::seq(rangeBegin, rangeEnd)) {
    // Iteration variable.
    auto iterAttr = mlir::IntegerAttr::get(SemiringTypes::forInt(ctx), i);
    auto iterArg = body.getArgument(0);
    auto iterType = llvm::cast<MatrixType>(iterArg.getType());
    MatrixAttrBuilder iterBuilder(iterType);
    iterBuilder.set(0, 0, iterAttr);
    _values[body.getArgument(0)] = iterBuilder.build();

    for (auto &op : body) {
      if (auto yieldOp = llvm::dyn_cast<YieldOp>(op)) {
        // Update block arguments
        for (auto [value, blockArg] : llvm::zip_equal(
                 yieldOp.getInputs(), body.getArguments().drop_front())) {
          _values[blockArg] = _values[value];
        }
      } else if (mlir::failed(evaluate(&op))) {
        return mlir::failure();
      }
    }

    if (!op.getUntil().empty()) {
      return op->emitOpError("'until' is not yet supported");
    }
  }

  // Set loop results.
  for (auto [value, result] :
       llvm::zip_equal(body.getArguments().drop_front(), op->getResults())) {
    _values[result] = _values[value];
  }

  return mlir::success();
}

mlir::LogicalResult Evaluator::evaluate(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
#define GA_CASE(Op) .Case<Op>([&](Op op) { return evaluate(op); })
      GA_CASE(TransposeOp) GA_CASE(DiagOp) GA_CASE(MatMulOp) GA_CASE(ReduceOp)
          GA_CASE(BroadcastOp) GA_CASE(ConstantMatrixOp) GA_CASE(ForConstOp)
#undef GA_CASE
              .Default([](mlir::Operation *op) {
                return op->emitOpError("unsupported op");
              });
}

MatrixAttr Evaluator::evaluate(mlir::func::FuncOp funcOp,
                               llvm::ArrayRef<MatrixAttr> args) {
  auto &body = funcOp.getFunctionBody().front();
  if (body.getNumArguments() != args.size()) {
    funcOp->emitOpError("function has ")
        << funcOp.getFunctionType().getNumInputs() << " inputs, got "
        << args.size() << "inputs";
    return nullptr;
  }

  for (auto [i, value] : llvm::enumerate(args)) {
    auto arg = body.getArgument(i);
    if (arg.getType() != value.getType()) {
      mlir::emitError(arg.getLoc())
          << "parameter " << i << " has type " << arg.getType()
          << ", but argument value has type " << value.getType();
      return nullptr;
    }

    _values[arg] = value;
  }

  for (auto &op : body) {
    if (auto retOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
      assert(retOp->getNumOperands() == 1);
      return value(retOp->getOperand(0));
    }

    if (mlir::failed(evaluate(&op))) {
      return nullptr;
    }
  }

  funcOp->emitOpError("missing return op");
  return nullptr;
}

MatrixAttr evaluate(mlir::func::FuncOp funcOp,
                    llvm::ArrayRef<MatrixAttr> args) {
  Evaluator evaluator;
  return evaluator.evaluate(funcOp, args);
}

} // namespace graphalg
