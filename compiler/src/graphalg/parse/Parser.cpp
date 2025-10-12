#include <vector>

#include <llvm/Support/Casting.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/parse/Lexer.h"
#include "graphalg/parse/Parser.h"

namespace graphalg {

mlir::LogicalResult parse(llvm::StringRef program, mlir::ModuleOp moduleOp) {
  llvm::StringRef filename = "<unknown>";
  auto loc = llvm::dyn_cast<mlir::FileLineColLoc>(moduleOp.getLoc());
  if (loc) {
    filename = loc.getFilename();
  }

  std::vector<Token> tokens;
  if (mlir::failed(lex(moduleOp->getContext(), program, filename,
                       /*startLine=*/0, /*startCol=*/0, tokens))) {

    return mlir::failure();
  }

  moduleOp->emitError("parse: not implemented");
  return mlir::failure();
}

} // namespace graphalg
