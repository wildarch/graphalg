#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "garel/GARelDialect.h"
#include "garel/GARelTypes.h"

#define GET_TYPEDEF_CLASSES
#include "garel/GARelOpsTypes.cpp.inc"

namespace garel {

static mlir::LogicalResult
verifyColumnsUnique(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    llvm::ArrayRef<ColumnAttr> columns) {
  // Columns must be unique
  llvm::SmallDenseSet<ColumnAttr, 4> columnSet;
  for (auto c : columns) {
    auto [_, newlyAdded] = columnSet.insert(c);
    if (!newlyAdded) {
      return emitError() << "column " << c
                         << " specified multiple times in the same column set";
    }
  }

  return mlir::success();
}

mlir::LogicalResult
RelationType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                     llvm::ArrayRef<ColumnAttr> columns) {
  return verifyColumnsUnique(emitError, columns);
}

mlir::LogicalResult
TupleType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<ColumnAttr> columns) {
  return verifyColumnsUnique(emitError, columns);
}

bool isColumnType(mlir::Type t) {
  // Allow i1, si64, f64, index
  return t.isSignlessInteger(1) || t.isSignedInteger(64) || t.isF64() ||
         t.isIndex();
}

// Need to define this here to avoid depending on IPRTypes in
// IPRDialect and creating a cycle.
void GARelDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "garel/GARelOpsTypes.cpp.inc"
      >();
}

} // namespace garel
