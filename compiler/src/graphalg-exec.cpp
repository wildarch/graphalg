#include "graphalg/GraphAlgTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
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

#include <graphalg/GraphAlgAttr.h>
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

static graphalg::MatrixAttr parseMatrix(llvm::Twine filename,
                                        graphalg::MatrixType type) {
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(filename, true);
  if (auto ec = buffer.getError()) {
    llvm::WithColor::error() << "could not open argument file '" << filename
                             << "': " << ec.message() << "\n";
    return nullptr;
  }

  assert(type.getRows().isConcrete() && type.getCols().isConcrete());
  auto rows = type.getRows().getConcreteDim();
  auto cols = type.getCols().getConcreteDim();
  llvm::SmallVector<mlir::Attribute> elems(rows * cols);

  auto data = buffer->get()->getBuffer();
  for (std::size_t i = 0; i < data.size();) {
    auto lineEnd = data.find_first_of('\n', i);
    if (lineEnd == std::string::npos) {
      lineEnd = data.size();
    }

    auto line = data.substr(i, lineEnd);
    llvm::SmallVector<llvm::StringRef, 3> parts;
    llvm::SplitString(line, parts);
    // TODO: use parts.
    i = lineEnd + 1;
  }
}

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
  llvm::SmallVector<graphalg::MatrixAttr> args;
  // TODO: Different number of arguments.
  for (const auto &[i, filename] : llvm::enumerate(cmd::args)) {
    auto type = funcOp.getFunctionType().getInput(i);
    /*
    auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(filename, true);
    if (auto ec = buffer.getError()) {
      llvm::WithColor::error() << "could not open argument file '" << arg
                               << "': " << ec.message() << "\n";
      return 1;
    }
    */

    auto matType = llvm::dyn_cast<graphalg::MatrixType>(type);
    if (!matType) {
      funcOp->emitOpError("parameter ")
          << i << " is not of type " << graphalg::MatrixType::getMnemonic();
      return 1;
    }

    auto parsed = parseMatrix(filename, matType);
    if (!parsed) {
      return 1;
    }

    args.push_back(parsed);
  }

  // TODO: Execute

  // TODO: Write out the result

  return 0;
}
