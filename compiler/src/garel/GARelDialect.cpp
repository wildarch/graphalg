#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"

#include "garel/GARelOpsDialect.cpp.inc"

namespace garel {

void GARelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "garel/GARelOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}

} // namespace garel
