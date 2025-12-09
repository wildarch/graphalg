#include "graphalg/GraphAlgTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <cassert>
#include <iostream>
#include <optional>
#include <tuple>

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Transforms/Passes.h>

#include <graphalg/GraphAlgAttr.h>
#include <graphalg/GraphAlgPasses.h>

#include <pg_graphalg/PgGraphAlg.h>

namespace pg_graphalg {

MatrixTable::MatrixTable(const MatrixTableDef &def)
    : _name(def.name), _nRows(def.nRows), _nCols(def.nCols) {}

void MatrixTable::clear() { _values.clear(); }

void MatrixTable::setValue(std::size_t row, std::size_t col,
                           std::int64_t value) {
  _values[{row, col}] = value;
}

std::optional<std::tuple<std::size_t, std::size_t, std::int64_t>>
MatrixTable::scan(ScanState &state) {
  auto it = _values.lower_bound({state.row, state.col});
  if (it == _values.end()) {
    return std::nullopt;
  }

  std::size_t row = it->first.first;
  std::size_t col = it->first.second;
  std::int64_t val = it->second;

  state.row = row;
  state.col = col + 1;
  return std::make_tuple(row, col, val);
}

static mlir::DialectRegistry createDialectRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<graphalg::GraphAlgDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::func::registerInlinerExtension(registry);
  return registry;
}

PgGraphAlg::PgGraphAlg(llvm::function_ref<void(mlir::Diagnostic &)> diagHandler)
    : _registry(createDialectRegistry()), _ctx(_registry) {
  auto &engine = _ctx.getDiagEngine();
  engine.registerHandler(diagHandler);
}

MatrixTable &PgGraphAlg::getTable(TableId tableId) {
  assert(_tables.count(tableId) && "getTable called before getOrCreateTable");
  return _tables.at(tableId);
}

MatrixTable &PgGraphAlg::getOrCreateTable(TableId tableId,
                                          const MatrixTableDef &def) {
  if (!_tables.count(tableId)) {
    _tables.emplace(tableId, def);
    _nameToId[def.name] = tableId;
    std::cerr << "registered table " << def.name << "\n";
  }

  return getTable(tableId);
}

MatrixTable *PgGraphAlg::lookupTable(llvm::StringRef tableName) {
  if (_nameToId.contains(tableName)) {
    return &getTable(_nameToId[tableName]);
  } else {
    return nullptr;
  }
}

bool PgGraphAlg::execute(llvm::StringRef programSource,
                         llvm::StringRef function,
                         llvm::ArrayRef<const MatrixTable *> arguments,
                         MatrixTable &output) {
  // Parse
  llvm::StringRef filename = "<input>";
  auto loc = mlir::FileLineColLoc::get(&_ctx, filename,
                                       /*line=*/1, /*column=*/1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::ModuleOp::create(loc, filename);
  if (mlir::failed(graphalg::parse(programSource, *moduleOp))) {
    return false;
  }

  // Desugar
  {
    mlir::PassManager pm(&_ctx);
    graphalg::GraphAlgToCorePipelineOptions toCoreOptions;
    graphalg::buildGraphAlgToCorePipeline(pm, toCoreOptions);
    if (mlir::failed(pm.run(*moduleOp))) {
      return false;
    }
  }

  // Set dimensions
  {
    llvm::SmallVector<graphalg::CallArgumentDimensions> argDims;
    for (const auto *arg : arguments) {
      argDims.push_back(graphalg::CallArgumentDimensions{
          .rows = arg->nRows(),
          .cols = arg->nCols(),
      });
    }

    graphalg::GraphAlgSetDimensionsOptions options{
        .functionName = function.str(),
        .argDims = std::move(argDims),
    };

    mlir::PassManager pm(&_ctx);
    pm.addNestedPass<mlir::func::FuncOp>(
        graphalg::createGraphAlgVerifyDimensions());
    pm.addPass(graphalg::createGraphAlgSetDimensions(options));
    pm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(*moduleOp))) {
      return false;
    }
  }

  auto funcOp =
      llvm::cast<mlir::func::FuncOp>(moduleOp->lookupSymbol(function));

  // TODO: Check semiring and value type are compatible

  // Build arguments
  llvm::SmallVector<graphalg::MatrixAttr> argAttrs;
  for (const auto &[arg, type] :
       llvm::zip_equal(arguments, funcOp.getFunctionType().getInputs())) {
    auto matType = llvm::cast<graphalg::MatrixType>(type);
    graphalg::MatrixAttrBuilder builder(matType);
    for (auto [pos, val] : arg->values()) {
      auto [row, col] = pos;
      // TODO: support more value types
      auto valAttr = mlir::IntegerAttr::get(matType.getSemiring(), val);
      builder.set(row, col, valAttr);
    }

    argAttrs.push_back(builder.build());
  }

  auto result = graphalg::evaluate(funcOp, argAttrs);
  if (!result) {
    return false;
  }

  graphalg::MatrixAttrReader resultReader(result);
  // TODO: Check rows/cols match.
  // TODO: Check semiring is compatible with value type.
  output.clear();
  auto defaultValue = resultReader.ring().addIdentity();
  for (auto r : llvm::seq(resultReader.nRows())) {
    for (auto c : llvm::seq(resultReader.nCols())) {
      auto v = resultReader.at(r, c);
      if (v != defaultValue) {
        // TODO: Support bool/real value types
        auto vInt = llvm::cast<mlir::IntegerAttr>(v).getInt();
        output.setValue(r, c, vInt);
      }
    }
  }

  return true;
}

} // namespace pg_graphalg
