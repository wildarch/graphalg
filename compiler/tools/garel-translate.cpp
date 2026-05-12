#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include "garel/GARelDialect.h"
#include "garel/GARelSQL.h"
#include "graphalg/GraphAlgDialect.h"

int main(int argc, char *argv[]) {
  mlir::TranslateFromMLIRRegistration exportSQL(
      "export-sql", "export to SQL", garel::translateToSQL,
      [](mlir::DialectRegistry &registry) {
        registry.insert<garel::GARelDialect>();
        registry.insert<graphalg::GraphAlgDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
      });

  return failed(
      mlir::mlirTranslateMain(argc, argv, "garel translation testing tool"));
}
