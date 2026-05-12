#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"

#include "garel/GARelEnumAttr.cpp.inc"
#include "garel/GARelTypes.h"
#define GET_ATTRDEF_CLASSES
#include "garel/GARelAttr.cpp.inc"

namespace garel {

mlir::Type AggregatorAttr::getResultType(mlir::Type inputRel) {
  switch (getFunc()) {
  case AggregateFunc::SUM:
  case AggregateFunc::MIN:
  case AggregateFunc::MAX:
  case AggregateFunc::LOR:
  case AggregateFunc::ARGMIN:
    // NOTE: argmin(arg, val) also uses first input column as output type.
    return llvm::cast<RelationType>(inputRel).getColumns()[getInputs()[0]];
  case AggregateFunc::COUNT:
    return mlir::IntegerType::get(inputRel.getContext(), 64);
  }
}

static std::size_t expectedNumInputs(AggregateFunc f) {
  switch (f) {
  case AggregateFunc::SUM:
  case AggregateFunc::MIN:
  case AggregateFunc::MAX:
  case AggregateFunc::LOR:
    return 1;
  case AggregateFunc::ARGMIN:
    return 2;
  case AggregateFunc::COUNT:
    return 0;
  }
}

mlir::LogicalResult
AggregatorAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       AggregateFunc func, llvm::ArrayRef<ColumnIdx> inputs) {
  if (inputs.size() != expectedNumInputs(func)) {
    return emitError() << stringifyAggregateFunc(func) << " expects exactly "
                       << expectedNumInputs(func) << " inputs, got "
                       << inputs.size();
  }

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
