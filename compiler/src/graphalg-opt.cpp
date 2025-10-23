#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<graphalg::GraphAlgDialect>();
  registry.insert<mlir::func::FuncDialect>();

  graphalg::registerPasses();
  mlir::registerCanonicalizerPass();
  mlir::registerInlinerPass();
  mlir::registerCSEPass();
  mlir::func::registerInlinerExtension(registry);

  // Testing only
  graphalg::registerTestDensePass();

  llvm::cl::HideUnrelatedOptions({});
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "GraphAlg optimizer driver\n", registry));
}
