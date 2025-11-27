#include <cstdint>
#include <cstdlib>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>

#include <graphalg/parse/Parser.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, std::size_t size) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    // Drop the diagnostic here to avoid printing them to stdout, which would
    // slow down the fuzzing.
  });

  llvm::StringRef filename = "<input>";
  auto loc = mlir::FileLineColLoc::get(&ctx, filename,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::ModuleOp::create(loc, filename);
  llvm::StringRef input(reinterpret_cast<const char *>(data), size);
  if (mlir::failed(graphalg::parse(input, *moduleOp))) {
    // OK if we fail to parse, as long as we don't crash.
    return 0;
  }

  if (mlir::failed(
          mlir::verify(moduleOp->getOperation(), /*verifyRecursively=*/true))) {
    // Parser says OK but the op verifiers disagree.
    llvm::errs() << "Parser says OK but the op verifiers disagree\n";
    std::abort();
  }

  return 0;
}
