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
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <graphalg/GraphAlgDialect.h>

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
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<graphalg::GraphAlgDialect>();

  mlir::ParserConfig parserConfig(&ctx);
  auto moduleOp =
      mlir::parseSourceFile<mlir::ModuleOp>(cmd::input, parserConfig);
  if (!moduleOp) {
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
