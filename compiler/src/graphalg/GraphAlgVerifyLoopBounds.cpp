#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include "graphalg/GraphAlgOps.h"

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGVERIFYLOOPBOUNDS
#include "graphalg/GraphAlgPasses.h.inc"

namespace {

class GraphAlgVerifyLoopBounds
    : public impl::GraphAlgVerifyLoopBoundsBase<GraphAlgVerifyLoopBounds> {
  using impl::GraphAlgVerifyLoopBoundsBase<
      GraphAlgVerifyLoopBounds>::GraphAlgVerifyLoopBoundsBase;

  void runOnOperation() final;
};

} // namespace

void GraphAlgVerifyLoopBounds::runOnOperation() {
  getOperation()->walk([&](ForOp op) {
    if (op.isDynamicRange()) {
      op.emitOpError("loop bound is not a constant or matrix dimension");
      signalPassFailure();
    }
  });
}

} // namespace graphalg
