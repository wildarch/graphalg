#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>

#include "garel/GARelDialect.h"
#include "graphalg/GraphAlgDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<graphalg::GraphAlgDialect>();
  registry.insert<garel::GARelDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  return failed(MlirLspServerMain(argc, argv, registry));
}
