#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "graphalg/GraphAlgOps.h"
#include "graphalg/GraphAlgPasses.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"

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

  mlir::IRRewriter rewriter(func);
  rewriter.setInsertionPointToStart(&body);
  auto constOp = rewriter.create<ConstantMatrixOp>(
      func.getLoc(), MatrixType::scalarOf(SemiringTypes::forInt(&getContext())),
      rewriter.getI64IntegerAttr(value));
  rewriter.replaceAllUsesWith(body.getArgument(argumentNumber), constOp);
}

} // namespace graphalg
