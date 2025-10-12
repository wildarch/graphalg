#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include "graphalg/parse/Parser.h"

int main(int argc, char *argv[]) {
  // llvm::setBugReportMsg(CUSTOM_BUG_REPORT_MSG);
  mlir::TranslateToMLIRRegistration importGraphAlg(
      "import-graphalg", "import .gr",
      [](llvm::SourceMgr &sourceMgr,
         mlir::MLIRContext *context) -> mlir::OwningOpRef<mlir::ModuleOp> {
        const auto *sourceBuf =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        auto filename = sourceBuf->getBufferIdentifier();
        auto loc = mlir::FileLineColLoc::get(context, filename,
                                             /*line=*/1, /*column=*/0);
        mlir::OwningOpRef<mlir::ModuleOp> moduleOp(
            mlir::ModuleOp::create(loc, filename));

        if (mlir::failed(graphalg::parse(sourceBuf->getBuffer(), *moduleOp))) {
          return nullptr;
        }

        return moduleOp;
      });

  return failed(
      mlir::mlirTranslateMain(argc, argv, "Graphalg Translation Testing Tool"));
}
