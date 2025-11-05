/**
 * Executes GraphAlg Core IR.
 *
 * It is used in regression tests for \c graphalg::evaluate.
 */
#include <cstdint>
#include <cstdlib>
#include <string>

#include <llvm/ADT/STLExtras.h>
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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgCast.h"
#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"
#include "graphalg/evaluate/Evaluator.h"

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

static graphalg::MatrixAttr parseMatrix(llvm::StringRef filename,
                                        const llvm::MemoryBuffer *buffer,
                                        graphalg::MatrixType type) {
  auto *ctx = type.getContext();
  graphalg::MatrixAttrBuilder result(type);

  assert(type.getRows().isConcrete() && type.getCols().isConcrete());

  std::size_t lineIdx = 0;
  auto emitError = [&]() {
    return mlir::emitError(
        mlir::FileLineColLoc::get(type.getContext(), filename, lineIdx, 0));
  };

  auto data = buffer->getBuffer();
  for (auto line : llvm::split(data, '\n')) {
    lineIdx++;

    if (line.empty()) {
      continue;
    }

    llvm::SmallVector<llvm::StringRef, 3> parts;
    llvm::SplitString(line, parts);
    if (parts.size() < 2) {
      emitError() << "expected at least 2 parts, got " << parts.size();
      return nullptr;
    }

    std::size_t rowIdx;
    if (!llvm::to_integer(parts[0], rowIdx, /*Base=*/10)) {
      emitError() << "invalid row index";
      return nullptr;
    } else if (rowIdx >= type.getRows().getConcreteDim()) {
      emitError() << "row index " << rowIdx << " out of bounds";
      return nullptr;
    }

    std::size_t colIdx;
    if (!llvm::to_integer(parts[1], colIdx, /*Base=*/10)) {
      emitError() << "invalid column index";
      return nullptr;
    } else if (colIdx >= type.getCols().getConcreteDim()) {
      emitError() << "col index " << colIdx << " out of bounds";
      return nullptr;
    }

    mlir::TypedAttr valueAttr;
    if (type.getSemiring() == graphalg::SemiringTypes::forInt(ctx)) {
      std::int64_t value;
      if (parts.size() != 3) {
        emitError() << "expected 3 parts, got " << parts.size();
        return nullptr;
      }

      if (!llvm::to_integer(parts[2], value, /*Base=*/10)) {
        emitError() << "invalid integer value";
        return nullptr;
      }

      valueAttr = mlir::IntegerAttr::get(type.getSemiring(), value);
    } else if (type.getSemiring() == graphalg::SemiringTypes::forBool(ctx)) {
      if (parts.size() != 2) {
        emitError() << "expected 2 parts, got " << parts.size();
        return nullptr;
      }

      valueAttr = mlir::BoolAttr::get(ctx, true);
    } else if (type.getSemiring() == graphalg::SemiringTypes::forReal(ctx)) {
      double value;
      if (parts.size() != 3) {
        emitError() << "expected 3 parts, got " << parts.size();
        return nullptr;
      }

      if (!llvm::to_float(parts[2], value)) {
        emitError() << "invalid float value";
        return nullptr;
      }

      valueAttr = mlir::FloatAttr::get(type.getSemiring(), value);
    } else {
      emitError() << "unsupported semiring: " << type.getSemiring();
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
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &ctx);
  std::string inputIncluded;
  auto inputId =
      sourceMgr.AddIncludeFile(cmd::input, llvm::SMLoc(), inputIncluded);
  if (!inputId) {
    llvm::WithColor::error()
        << "could not find input file '" << cmd::input << "'\n";
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
    return 1;
  }

  // Number of arguments must match function parameters.
  if (cmd::args.size() != funcOp.getFunctionType().getNumInputs()) {
    funcOp->emitOpError("expected ") << funcOp.getFunctionType().getNumInputs()
                                     << "arguments, got " << cmd::args.size();
    return 1;
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

    std::string argIncluded;
    auto id = sourceMgr.AddIncludeFile(filename, llvm::SMLoc(), argIncluded);
    if (!id) {
      llvm::WithColor::error()
          << "could not find argument file '" << filename << "'\n";
      return 1;
    }

    auto parsed = parseMatrix(filename, sourceMgr.getMemoryBuffer(id), matType);
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
  for (auto row : llvm::seq(resultReader.nRows())) {
    for (auto col : llvm::seq(resultReader.nCols())) {
      auto val = resultReader.at(row, col);
      if (val != resultReader.ring().addIdentity()) {
        llvm::outs() << row << " " << col << " " << val << "\n";
      }
    }
  }

  return 0;
}
