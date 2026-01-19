#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "garel/GARelAttr.h"
#include "garel/GARelDialect.h"

#define GET_ATTRDEF_CLASSES
#include "garel/GARelAttr.cpp.inc"

namespace garel {

// Need to define this here to avoid depending on GraphAlgAttr in
// GARelDialect and creating a cycle.
void GARelDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "garel/GARelAttr.cpp.inc"
      >();
}

} // namespace garel
