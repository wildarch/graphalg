#include <cstdint>
#include <cstring>
#include <optional>

#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgPasses.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"
#include "graphalg/evaluate/Evaluator.h"
#include "graphalg/parse/Parser.h"

class Playground {
private:
  mlir::DialectRegistry _registry;
  mlir::MLIRContext _ctx;
  mlir::OwningOpRef<mlir::ModuleOp> _moduleOp;
  llvm::SmallVector<graphalg::CallArgumentDimensions> _argDims;
  mlir::func::FuncOp _funcOp;
  llvm::SmallVector<graphalg::MatrixAttrBuilder> _argBuilders;
  graphalg::MatrixAttr _result;
  std::optional<graphalg::MatrixAttrReader> _resultReader;

  void setArgumentValue(std::size_t argIdx, std::size_t row, std::size_t col,
                        mlir::TypedAttr value);

public:
  Playground();

  bool parse(llvm::StringRef input);

  bool desugarToCore();

  void addArgument(std::size_t rows, std::size_t cols);
  bool setDimensions(llvm::StringRef func);

  void setArgumentValue(std::size_t argIdx, std::size_t row, std::size_t col,
                        bool value) {
    setArgumentValue(argIdx, row, col, mlir::BoolAttr::get(&_ctx, value));
  }
  void setArgumentValue(std::size_t argIdx, std::size_t row, std::size_t col,
                        std::int64_t value) {
    setArgumentValue(
        argIdx, row, col,
        mlir::IntegerAttr::get(graphalg::SemiringTypes::forInt(&_ctx), value));
  }
  void setArgumentValue(std::size_t argIdx, std::size_t row, std::size_t col,
                        double value) {

    setArgumentValue(
        argIdx, row, col,
        mlir::FloatAttr::get(graphalg::SemiringTypes::forReal(&_ctx), value));
  }

  bool evaluate();

  bool getResultInfinity(std::size_t row, std::size_t col);
  std::int64_t getResultInt(std::size_t row, std::size_t col);
  double getResultReal(std::size_t row, std::size_t col);
};

void Playground::setArgumentValue(std::size_t argIdx, std::size_t row,
                                  std::size_t col, mlir::TypedAttr value) {
  // Create arg builder if it does not exist yet.
  while (argIdx >= _argBuilders.size()) {
    auto type = _funcOp.getFunctionType().getInput(argIdx);
    _argBuilders.emplace_back(llvm::cast<graphalg::MatrixType>(type));
  }

  auto &arg = _argBuilders[argIdx];
  if (arg.ring() == graphalg::SemiringTypes::forTropInt(&_ctx) ||
      arg.ring() == graphalg::SemiringTypes::forTropMaxInt(&_ctx)) {
    value = graphalg::TropIntAttr::get(&_ctx, arg.ring(),
                                       llvm::cast<mlir::IntegerAttr>(value));
  } else if (arg.ring() == graphalg::SemiringTypes::forTropReal(&_ctx)) {
    value = graphalg::TropFloatAttr::get(&_ctx, arg.ring(),
                                         llvm::cast<mlir::FloatAttr>(value));
  }

  arg.set(row, col, value);
}

static mlir::DialectRegistry createDialectRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<graphalg::GraphAlgDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::func::registerInlinerExtension(registry);
  return registry;
}

Playground::Playground()
    : _registry(createDialectRegistry()), _ctx(_registry) {}

bool Playground::parse(llvm::StringRef input) {
  llvm::StringRef filename = "<input>";
  auto loc = mlir::FileLineColLoc::get(&_ctx, filename,
                                       /*line=*/1, /*column=*/1);
  _moduleOp = mlir::ModuleOp::create(loc, filename);
  if (mlir::failed(graphalg::parse(input, *_moduleOp))) {
    return false;
  }

  return true;
}

void Playground::addArgument(std::size_t rows, std::size_t cols) {
  _argDims.push_back(graphalg::CallArgumentDimensions{rows, cols});
}

bool Playground::desugarToCore() {
  mlir::PassManager pm(&_ctx);
  graphalg::GraphAlgToCorePipelineOptions toCoreOptions;
  graphalg::buildGraphAlgToCorePipeline(pm, toCoreOptions);
  if (mlir::failed(pm.run(*_moduleOp))) {
    return false;
  } else {
    return true;
  }
}

bool Playground::setDimensions(llvm::StringRef func) {
  graphalg::GraphAlgSetDimensionsOptions options{
      .functionName = func.str(),
      .argDims = std::move(_argDims),
  };

  mlir::PassManager pm(&_ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      graphalg::createGraphAlgVerifyDimensions());
  pm.addPass(graphalg::createGraphAlgSetDimensions(options));
  pm.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*_moduleOp))) {
    return false;
  }

  // Since the pass was successful, no need to check if the function exists.
  _funcOp = llvm::cast<mlir::func::FuncOp>(_moduleOp->lookupSymbol(func));
  return true;
}

bool Playground::evaluate() {
  llvm::SmallVector<graphalg::MatrixAttr> args;
  for (auto &builder : _argBuilders) {
    args.push_back(builder.build());
  }

  _argBuilders.clear();
  _result = graphalg::evaluate(_funcOp, args);
  if (!_result) {
    return false;
  }

  _resultReader.emplace(_result);
  return true;
}

bool Playground::getResultInfinity(std::size_t row, std::size_t col) {
  auto value = _resultReader->at(row, col);
  return llvm::isa<graphalg::TropInfAttr>(value);
}

std::int64_t Playground::getResultInt(std::size_t row, std::size_t col) {
  auto value = _resultReader->at(row, col);
  return llvm::cast<mlir::IntegerAttr>(value).getInt();
}

double Playground::getResultReal(std::size_t row, std::size_t col) {
  auto value = _resultReader->at(row, col);
  return llvm::cast<mlir::FloatAttr>(value).getValueAsDouble();
}

extern "C" {

Playground *ga_new() { return new Playground(); }
void ga_free(Playground *pg) { delete pg; }

bool ga_parse(Playground *inst, const char *input) {
  return inst->parse(input);
}

bool ga_desugar(Playground *pg) { return pg->desugarToCore(); }

void ga_add_arg(Playground *pg, std::size_t rows, std::size_t cols) {
  pg->addArgument(rows, cols);
}

bool ga_set_dims(Playground *inst, const char *func) {
  return inst->setDimensions(func);
}

void ga_set_arg_bool(Playground *pg, std::size_t argIdx, std::size_t row,
                     std::size_t col, bool v) {
  pg->setArgumentValue(argIdx, row, col, v);
}

void ga_set_arg_int(Playground *pg, std::size_t argIdx, std::size_t row,
                    std::size_t col, std::int64_t v) {
  pg->setArgumentValue(argIdx, row, col, v);
}

void ga_set_arg_real(Playground *pg, std::size_t argIdx, std::size_t row,
                     std::size_t col, double v) {
  pg->setArgumentValue(argIdx, row, col, v);
}

bool ga_evaluate(Playground *pg) { return pg->evaluate(); }

bool ga_get_res_inf(Playground *pg, std::size_t row, std::size_t col) {
  return pg->getResultInfinity(row, col);
}

std::int64_t ga_get_res_int(Playground *pg, std::size_t row, std::size_t col) {
  return pg->getResultInt(row, col);
}

std::int64_t ga_get_res_real(Playground *pg, std::size_t row, std::size_t col) {
  return pg->getResultReal(row, col);
}
}

#ifndef __EMSCRIPTEN__
// Dummy main when building for non-wasm platforms.
int main(int argc, char **argv) {
  auto pg = ga_new();
  auto program = R"(
    func
    MatMul(lhs : Matrix<s, s, int>, rhs : Matrix<s, s, int>)
        -> Matrix<s, s, int> {
      return lhs * rhs;
    }
  )";
  if (!ga_parse(pg, program)) {
    return 1;
  }

  if (!ga_desugar(pg)) {
    return 1;
  }

  ga_add_arg(pg, 2, 2); // lhs
  ga_add_arg(pg, 2, 2); // rhs
  ga_set_dims(pg, "MatMul");

  ga_set_arg_int(pg, 0, 0, 0, 3);
  ga_set_arg_int(pg, 0, 0, 1, 5);
  ga_set_arg_int(pg, 0, 1, 0, 7);
  ga_set_arg_int(pg, 0, 1, 1, 11);

  ga_set_arg_int(pg, 1, 0, 0, 13);
  ga_set_arg_int(pg, 1, 0, 1, 17);
  ga_set_arg_int(pg, 1, 1, 0, 19);
  ga_set_arg_int(pg, 1, 1, 1, 23);

  if (!ga_evaluate(pg)) {
    return 1;
  }

  for (auto row : llvm::seq(2)) {
    for (auto col : llvm::seq(2)) {
      llvm::outs() << row << " " << col << " " << ga_get_res_int(pg, row, col)
                   << "\n";
    }
  }

  ga_free(pg);
}
#endif
