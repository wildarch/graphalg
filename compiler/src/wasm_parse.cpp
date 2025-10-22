#include <cstring>
#include <iostream>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "graphalg/parse/Parser.h"

extern "C" {

void ga_parse(char *input) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::BuiltinDialect>();

  llvm::StringRef filename = "<input>";
  auto loc = mlir::FileLineColLoc::get(&ctx, filename,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp(
      mlir::ModuleOp::create(loc, filename));

  if (mlir::failed(graphalg::parse(input, *moduleOp))) {
    std::cerr << "failed to parse\n";
  }

  moduleOp->print(llvm::outs());
}
}
