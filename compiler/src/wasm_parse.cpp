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

class Instance {
private:
  mlir::MLIRContext _ctx;
  std::string _printed;

public:
  Instance();

  bool tryParse(llvm::StringRef input);

  const char *getPrinted();
};

Instance::Instance() { _ctx.getOrLoadDialect<mlir::BuiltinDialect>(); }

bool Instance::tryParse(llvm::StringRef input) {
  llvm::StringRef filename = "<input>";
  auto loc = mlir::FileLineColLoc::get(&_ctx, filename,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp(
      mlir::ModuleOp::create(loc, filename));

  if (mlir::failed(graphalg::parse(input, *moduleOp))) {
    std::cerr << "failed to parse\n";
    return false;
  }

  _printed.clear();
  llvm::raw_string_ostream os(_printed);
  moduleOp->print(os);
  return true;
}

const char *Instance::getPrinted() { return _printed.c_str(); }

extern "C" {

Instance *ga_new() { return new Instance(); }

bool ga_try_parse(Instance *inst, char *input) { return inst->tryParse(input); }

const char *ga_get_printed(Instance *inst) { return inst->getPrinted(); }

void ga_free(Instance *inst) { delete inst; }
}

#ifndef __EMSCRIPTEN__
// Dummy main when building for non-wasm platforms.
int main(int argc, char **argv) { return 0; }
#endif
