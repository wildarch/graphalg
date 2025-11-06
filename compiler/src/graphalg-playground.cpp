#include <cstdint>
#include <cstring>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "graphalg/GraphAlgPasses.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/evaluate/Evaluator.h"
#include "graphalg/parse/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Casting.h"

class Playground {
private:
  mlir::MLIRContext _ctx;
  mlir::OwningOpRef<mlir::ModuleOp> _moduleOp;
  llvm::SmallVector<graphalg::CallArgumentDimensions> _argDims;
  mlir::func::FuncOp _funcOp;
  llvm::SmallVector<graphalg::MatrixAttrBuilder> _argBuilders;

  void setArgumentValue(std::size_t argIdx, std::uint64_t row,
                        std::uint64_t col, mlir::TypedAttr value);

public:
  Playground();

  bool parse(llvm::StringRef input);

  bool desugarToCore();

  void addArgument(std::uint64_t rows, std::uint64_t cols);
  bool setDimensions(char *funcName);

  void setArgumentValue(std::size_t argIdx, std::uint64_t row,
                        std::uint64_t col, bool value);
  void setArgumentValue(std::size_t argIdx, std::uint64_t row,
                        std::uint64_t col, std::int64_t value);
  void setArgumentValue(std::size_t argIdx, std::uint64_t row,
                        std::uint64_t col, double value);

  bool evaluate();
};

void Playground::setArgumentValue(std::size_t argIdx, std::uint64_t row,
                                  std::uint64_t col, mlir::TypedAttr value) {
  while (argIdx >= _argBuilders.size()) {
    auto type = _funcOp.getFunctionType().getInput(argIdx);
    _argBuilders.emplace_back(llvm::cast<graphalg::MatrixType>(type));
  }
}

Playground::Playground() { _ctx.getOrLoadDialect<mlir::BuiltinDialect>(); }

bool Playground::parse(llvm::StringRef input) {
  llvm::StringRef filename = "<input>";
  auto loc = mlir::FileLineColLoc::get(&_ctx, filename,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp(
      mlir::ModuleOp::create(loc, filename));

  if (mlir::failed(graphalg::parse(input, *moduleOp))) {
    return false;
  }

  return true;
}

extern "C" {

Playground *ga_new() { return new Playground(); }

bool ga_try_parse(Playground *inst, char *input) { return inst->parse(input); }

void ga_free(Playground *inst) { delete inst; }
}

#ifndef __EMSCRIPTEN__
// Dummy main when building for non-wasm platforms.
int main(int argc, char **argv) { return 0; }
#endif
