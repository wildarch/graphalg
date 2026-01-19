#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "garel/GARelDialect.h"
#include "garel/GARelTypes.h"

#define GET_TYPEDEF_CLASSES
#include "garel/GARelOpsTypes.cpp.inc"

namespace garel {

// Need to define this here to avoid depending on IPRTypes in
// IPRDialect and creating a cycle.
void GARelDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "garel/GARelOpsTypes.cpp.inc"
      >();
}

} // namespace garel
