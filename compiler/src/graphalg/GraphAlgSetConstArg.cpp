#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include "graphalg/GraphAlgPasses.h"
#include "graphalg/GraphAlgSetConstArg.h"

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGSETCONSTARG
#include "graphalg/GraphAlgPasses.h.inc"

namespace {

/**
 * Propagate integer constant arguments into function bodies.
 *
 * Run canonicalization after this pass to propagate the constant through the
 * program. This pass does not change the function signature (that is, it does
 * not remove the original argument).
 */
class GraphAlgSetConstArg
    : public impl::GraphAlgSetConstArgBase<GraphAlgSetConstArg> {
  using impl::GraphAlgSetConstArgBase<
      GraphAlgSetConstArg>::GraphAlgSetConstArgBase;

  void runOnOperation() final;
};

} // namespace

void GraphAlgSetConstArg::runOnOperation() {
  if (functionName.empty()) {
    getOperation().emitError("missing value for required option 'func'");
    return signalPassFailure();
  }

  if (argumentNumber < 0) {
    getOperation().emitError("missing value for required option 'argNum'");
    return signalPassFailure();
  }

  auto func = llvm::dyn_cast_if_present<mlir::func::FuncOp>(
      getOperation().lookupSymbol(functionName));
  if (!func) {
    getOperation().emitOpError("does not contain a function named '")
        << functionName << "'";
    return signalPassFailure();
  }

  auto &body = func.getBody().front();
  auto numArgs = body.getNumArguments();
  if (argumentNumber >= numArgs) {
    getOperation().emitOpError("argument number ")
        << argumentNumber << " is out of bounds function " << functionName
        << ", which only has " << numArgs << " parameters";
    return signalPassFailure();
  }

  llvm::SmallVector<mlir::TypedAttr> values(numArgs);
  values[argumentNumber] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 64), value);
  if (mlir::failed(setConstantArguments(func, values))) {
    signalPassFailure();
  }
}

mlir::LogicalResult
setConstantArguments(mlir::func::FuncOp op,
                     llvm::ArrayRef<mlir::TypedAttr> values) {
  auto &body = op.getBody().front();
  auto numArgs = body.getNumArguments();
  if (values.size() != numArgs) {
    return op.emitOpError("expected a function with ")
           << values.size() << " parameters, but only has " << numArgs;
  }

  mlir::IRRewriter rewriter(op);
  rewriter.setInsertionPointToStart(&body);

  for (auto i : llvm::seq(numArgs)) {
    auto val = values[i];
    if (!val) {
      // Not constant
      continue;
    }

    auto arg = body.getArgument(i);
    auto type = llvm::dyn_cast<MatrixType>(arg.getType());
    if (!type || !type.isScalar()) {
      return op.emitOpError("argument ") << i << " is not a scalar matrix";
    } else if (type.getSemiring() != val.getType()) {
      return op.emitOpError("cannot inline value of type ")
             << val.getType() << " into argument " << i << " of type " << type;
    }

    auto constOp = rewriter.create<ConstantMatrixOp>(op.getLoc(), type, val);
    rewriter.replaceAllUsesWith(body.getArgument(i), constOp);
  }

  return mlir::success();
}

} // namespace graphalg
