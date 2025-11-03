#include "graphalg/SemiringTypes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include <cstdint>
#include <cstdlib>
#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <graphalg/GraphAlgAttr.h>
#include <graphalg/GraphAlgCast.h>
#include <graphalg/GraphAlgDialect.h>
#include <graphalg/GraphAlgTypes.h>
#include <graphalg/evaluate/Evaluator.h>

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

  auto *ctx = type.getContext();
  graphalg::MatrixAttrBuilder result(type);

  assert(type.getRows().isConcrete() && type.getCols().isConcrete());
  auto data = buffer->get()->getBuffer();
  for (auto line : llvm::split(data, '\n')) {
    if (line.empty()) {
      continue;
    }

    llvm::SmallVector<llvm::StringRef, 3> parts;
    llvm::SplitString(line, parts);
    if (parts.size() < 2) {
      llvm::WithColor::error()
          << "expected at least 2 parts, got " << parts.size() << "\n";
      llvm::errs() << "line: '" << line << "'\n";
      return nullptr;
    }

    std::size_t rowIdx;
    if (!llvm::to_integer(parts[0], rowIdx, /*Base=*/10)) {
      llvm::WithColor::error() << "invalid row index\n";
      return nullptr;
    } else if (rowIdx >= type.getRows().getConcreteDim()) {
      llvm::WithColor::error() << "row index " << rowIdx << " out of bounds\n";
      return nullptr;
    }

    std::size_t colIdx;
    if (!llvm::to_integer(parts[1], colIdx, /*Base=*/10)) {
      llvm::WithColor::error() << "invalid column index\n";
      return nullptr;
    } else if (colIdx >= type.getCols().getConcreteDim()) {
      llvm::WithColor::error() << "col index " << colIdx << " out of bounds\n";
      return nullptr;
    }

    mlir::TypedAttr valueAttr;
    if (type.getSemiring() == graphalg::SemiringTypes::forInt(ctx)) {
      std::int64_t value;
      if (parts.size() != 3) {
        llvm::WithColor::error()
            << "expected 3 parts, got " << parts.size() << "\n";
        llvm::errs() << "line: '" << line << "'\n";
        return nullptr;
      }

      if (!llvm::to_integer(parts[2], value, /*Base=*/10)) {
        llvm::WithColor::error() << "invalid value\n";
        return nullptr;
      }

      valueAttr = mlir::IntegerAttr::get(type.getSemiring(), value);
    } else if (type.getSemiring() == graphalg::SemiringTypes::forBool(ctx)) {
      if (parts.size() != 2) {
        llvm::WithColor::error()
            << "expected 2 parts, got " << parts.size() << "\n";
        llvm::errs() << "line: '" << line << "'\n";
        return nullptr;
      }

      valueAttr = mlir::BoolAttr::get(ctx, true);
    } else {
      llvm::WithColor::error()
          << "unsupported semiring: " << type.getSemiring() << "\n";
      return nullptr;
    }

    result.set(rowIdx, colIdx, valueAttr);
  }

  return result.build();
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Execute GraphAlg program\n");
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<graphalg::GraphAlgDialect>();

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &ctx);
  std::string inputIncluded;
  auto inputId =
      sourceMgr.AddIncludeFile(cmd::input, llvm::SMLoc(), inputIncluded);
  if (!inputId) {
    return 1;
  }

  mlir::ParserConfig parserConfig(&ctx);
  auto moduleOp =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);
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

  // Number of arguments must match function parameters.
  if (cmd::args.size() != funcOp.getFunctionType().getNumInputs()) {
    funcOp->emitOpError("expected ") << funcOp.getFunctionType().getNumInputs()
                                     << "arguments, got " << cmd::args.size();
  }

  llvm::SmallVector<graphalg::MatrixAttr> args;
  for (const auto &[i, filename] : llvm::enumerate(cmd::args)) {
    auto type = funcOp.getFunctionType().getInput(i);

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

  auto result = graphalg::evaluate(funcOp, args);
  if (!result) {
    return 1;
  }

  graphalg::MatrixAttrReader resultReader(result);
  for (std::size_t row = 0; row < resultReader.nRows(); row++) {
    for (std::size_t col = 0; col < resultReader.nCols(); col++) {
      auto val = resultReader.at(row, col);
      if (val != resultReader.ring().addIdentity()) {
        llvm::outs() << row << " " << col << " " << val << "\n";
      }
    }
  }

  return 0;
}
