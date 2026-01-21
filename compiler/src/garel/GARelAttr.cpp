#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"

#include "garel/GARelEnumAttr.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "garel/GARelAttr.cpp.inc"

namespace garel {

ColumnAttr ColumnAttr::newOfType(mlir::Type type) {
  auto *ctx = type.getContext();
  auto colId = mlir::DistinctAttr::create(mlir::UnitAttr::get(ctx));
  return ColumnAttr::get(ctx, colId, type);
}

mlir::Type AggregatorAttr::getResultType() {
  switch (getFunc()) {
  case AggregateFunc::SUM:
  case AggregateFunc::MIN:
  case AggregateFunc::MAX:
  case AggregateFunc::LOR:
  case AggregateFunc::ARGMIN:
    // NOTE: argmin(arg, val) also uses first input as output type.
    return getInputs()[0].getType();
  }
}

mlir::LogicalResult
AggregatorAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       AggregateFunc func, llvm::ArrayRef<ColumnAttr> inputs) {
  // TODO: Verify input column count and type(s)
  return mlir::success();
}

// Need to define this here to avoid depending on GARelAttr in
// GARelDialect and creating a cycle.
void GARelDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "garel/GARelAttr.cpp.inc"
      >();
}

} // namespace garel
