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

namespace {

using namespace llvm;

cl::opt<garel::SQLDialect> sqlDialect(
    "sql-dialect", cl::desc("The SQL dialect to export"),
    cl::init(garel::SQLDialect::DUCKDB_PYTHON),
    cl::values(clEnumValN(garel::SQLDialect::DUCKDB_PYTHON, "duckdb_python",
                          "DuckDB (with Python driver for control flow)"),
               clEnumValN(garel::SQLDialect::UMBRA_ITERATE, "umbra", "Umbra")));

} // namespace

int main(int argc, char *argv[]) {
  // TODO: Use dialect flag.
  mlir::TranslateFromMLIRRegistration exportSQL(
      "export-sql", "export to SQL",
      [](mlir::Operation *op, llvm::raw_ostream &os) {
        return garel::translateToSQL(op, os, sqlDialect);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<garel::GARelDialect>();
        registry.insert<graphalg::GraphAlgDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
      });

  return failed(
      mlir::mlirTranslateMain(argc, argv, "garel translation testing tool"));
}
