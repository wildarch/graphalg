#include <cstdlib>
#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <graphalg/GraphAlgPasses.h>
#include <graphalg/parse/Parser.h>

namespace cmd {

using namespace llvm;

cl::opt<std::string> input(cl::Positional, cl::Required,
                           cl::desc("<source file>"),
                           cl::value_desc("source file"));
cl::opt<std::string> func(cl::Positional, cl::Required,
                          cl::desc("<function name>"),
                          cl::value_desc("function to execute"));
cl::list<std::string> args(cl::Positional, cl::desc("<arguments...>"),
                           cl::value_desc("input files"));

} // namespace cmd

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Execute GraphAlg program\n");
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::MLIRContext ctx(registry);

  auto inputBuffer =
      llvm::MemoryBuffer::getFileOrSTDIN(cmd::input, /*IsText=*/true);
  if (auto ec = inputBuffer.getError()) {
    llvm::WithColor::error() << "could not open source file '" << cmd::input
                             << "': " << ec.message();
    return 1;
  }

  // Parse source file.
  auto loc = mlir::FileLineColLoc::get(&ctx, cmd::input,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp(
      mlir::ModuleOp::create(loc, cmd::input));
  if (mlir::failed(
          graphalg::parse(inputBuffer->get()->getBuffer(), *moduleOp))) {
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
      moduleOp->lookupSymbol(cmd::func));
  if (!funcOp) {
    llvm::WithColor::error()
        << "no such function '" << cmd::func << "' in Core IR\n";
    moduleOp->print(llvm::errs());
  }

  // TODO: Load arguments
  for (const auto &arg : cmd::args) {
    auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(arg, /*IsText=*/true);
    if (auto ec = buffer.getError()) {
      llvm::WithColor::error() << "could not open argument file '" << arg
                               << "': " << ec.message() << "\n";
      return 1;
    }
  }

  // TODO: Execute

  // TODO: Write out the result

  return 0;
}
