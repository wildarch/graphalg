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

#include <graphalg/GraphAlgAttr.h>
#include <graphalg/GraphAlgOps.h>
#include <graphalg/evaluate/Evaluator.h>

namespace graphalg {

namespace {

class Evaluator {
private:
  llvm::DenseMap<mlir::Value, MatrixAttr> _values;

  mlir::LogicalResult evaluate(TransposeOp op);
  mlir::LogicalResult evaluate(DiagOp op);
  mlir::LogicalResult evaluate(MatMulOp op);
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

mlir::LogicalResult Evaluator::evaluate(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
#define GA_CASE(Op) .Case<Op>([&](Op op) { return evaluate(op); })
      GA_CASE(TransposeOp) GA_CASE(DiagOp) GA_CASE(MatMulOp)
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
