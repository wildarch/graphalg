#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
#include <cstdlib>
#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <graphalg/GraphAlgPasses.h>
#include <graphalg/parse/Parser.h>

namespace {

using namespace llvm;

cl::opt<std::string> input(cl::Positional, cl::Required,
                           cl::desc("<source file>"),
                           cl::value_desc("source file"));
cl::opt<std::string> func(cl::Positional, cl::Required,
                          cl::desc("<function name>"),
                          cl::value_desc("function to execute"));
cl::list<std::string> args(cl::Positional, cl::desc("<arguments...>"),
                           cl::value_desc("input files"));

} // namespace

static llvm::Error readFileToString(llvm::Twine path, std::string &out) {
  int inputFd;
  auto errCode = llvm::sys::fs::openFileForRead(path, inputFd);
  if (errCode) {
    return llvm::errorCodeToError(errCode);
  }

  llvm::SmallVector<char> buf;
  auto err = llvm::sys::fs::readNativeFileToEOF(inputFd, buf);
  if (err) {
    return std::move(err);
  }

  // Buffer to string.
  out.append(buf.data(), buf.size());

  errCode = llvm::sys::fs::closeFile(inputFd);
  return llvm::errorCodeToError(errCode);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Execute GraphAlg program\n");
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::MLIRContext ctx(registry);

  std::string inputString;
  llvm::handleAllErrors(
      readFileToString(input, inputString), [](const llvm::ECError &e) {
        llvm::errs() << "failed to read source file: " << e.message() << "\n";
        std::exit(1);
      });

  // Parse source file.
  auto loc = mlir::FileLineColLoc::get(&ctx, input,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp(
      mlir::ModuleOp::create(loc, input));
  if (mlir::failed(graphalg::parse(inputString, *moduleOp))) {
    return 1;
  }

  // Convert source file to Core.
  mlir::PassManager pm(moduleOp.get()->getName());
  pm.addPass(graphalg::createGraphAlgPrepareInline());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      graphalg::createGraphAlgScalarizeApply());
  pm.addNestedPass<mlir::func::FuncOp>(graphalg::createGraphAlgToCore());
  pm.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*moduleOp))) {
    llvm::errs() << "failed to convert to Core\n";
    return 1;
  }

  // Find function to execute.
  auto funcOp = llvm::dyn_cast_if_present<mlir::func::FuncOp>(
      moduleOp->lookupSymbol(func));
  if (!funcOp) {
    llvm::errs() << "no such function '" << func << " in Core IR:'\n";
    moduleOp->print(llvm::errs());
  }

  // TODO: Load arguments

  // TODO: Execute

  // TODO: Write out the result

  return 0;
}
