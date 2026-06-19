#include <cassert>
#include <memory>
#include <optional>
#include <variant>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Transforms/Passes.h>

#include <graphalg/GraphAlgAttr.h>
#include <graphalg/GraphAlgPasses.h>
#include <graphalg/GraphAlgTypes.h>
#include <graphalg/SemiringTypes.h>
#include <graphalg/evaluate/Evaluator.h>
#include <graphalg/parse/Parser.h>

#include <pg_graphalg/MatrixTable.h>
#include <pg_graphalg/PgGraphAlg.h>

namespace pg_graphalg {

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

std::optional<MatrixTable *> PgGraphAlg::getOrCreateTable(
    TableId tableId,
    llvm::function_ref<std::optional<MatrixTableDef>(TableId id)> createFunc) {
  if (!_tables.contains(tableId)) {
    auto def = createFunc(tableId);
    if (!def) {
      return std::nullopt;
    }

    _tables[tableId] = std::make_unique<MatrixTable>(*def);
  }
  return _tables[tableId].get();
}

static mlir::TypedAttr
matrixValueToAttr(mlir::Type t, std::variant<bool, std::int64_t, double> v) {
  auto *ctx = t.getContext();
  auto intType = graphalg::SemiringTypes::forInt(ctx);
  auto realType = graphalg::SemiringTypes::forReal(ctx);
  auto tropIntType = graphalg::SemiringTypes::forTropInt(ctx);
  auto tropRealType = graphalg::SemiringTypes::forTropReal(ctx);
  auto tropMaxIntType = graphalg::SemiringTypes::forTropMaxInt(ctx);
  if (t == graphalg::SemiringTypes::forBool(ctx)) {
    assert(std::holds_alternative<bool>(v));
    return mlir::BoolAttr::get(ctx, std::get<bool>(v));
  } else if (t == intType) {
    assert(std::holds_alternative<std::int64_t>(v));
    return mlir::IntegerAttr::get(intType, std::get<std::int64_t>(v));
  } else if (t == realType) {
    assert(std::holds_alternative<double>(v));
    return mlir::FloatAttr::get(realType, std::get<double>(v));
  } else if (t == tropIntType) {
    assert(std::holds_alternative<std::int64_t>(v));
    return graphalg::TropIntAttr::get(
        ctx, tropIntType,
        mlir::IntegerAttr::get(intType, std::get<std::int64_t>(v)));
  } else if (t == tropRealType) {
    assert(std::holds_alternative<double>(v));
    return graphalg::TropFloatAttr::get(
        ctx, tropRealType, mlir::FloatAttr::get(realType, std::get<double>(v)));
  } else if (t == tropMaxIntType) {
    assert(std::holds_alternative<std::int64_t>(v));
    return graphalg::TropIntAttr::get(
        ctx, tropMaxIntType,
        mlir::IntegerAttr::get(intType, std::get<std::int64_t>(v)));
  } else {
    mlir::emitError(mlir::UnknownLoc::get(ctx))
        << "invalid target type for matrix value: " << t;
    return nullptr;
  }
}

static std::variant<bool, std::int64_t, double>
attrToMatrixValue(mlir::TypedAttr attr) {
  if (auto b = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
    return b.getValue();
  } else if (auto i = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    return i.getInt();
  } else if (auto f = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    return f.getValueAsDouble();
  } else if (auto i = llvm::dyn_cast<graphalg::TropIntAttr>(attr)) {
    return i.getValue().getInt();
  } else if (auto f = llvm::dyn_cast<graphalg::TropFloatAttr>(attr)) {
    return f.getValue().getValueAsDouble();
  } else {
    mlir::emitError(mlir::UnknownLoc::get(attr.getContext()))
        << "attribute cannot be converted to matrix value: " << attr;
    std::abort();
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
    const auto &values = arg->values();
    for (auto [pos, val] : values) {
      auto [row, col] = pos;
      auto valAttr = matrixValueToAttr(matType.getSemiring(), val);
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
        output.setValue(r, c, attrToMatrixValue(v));
      }
    }
  }

  return true;
}

} // namespace pg_graphalg
