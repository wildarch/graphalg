#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include "garel/GARelAttr.h"

#define GET_TYPEDEF_CLASSES
#include "garel/GARelOpsTypes.h.inc"

namespace garel {

bool isColumnType(mlir::Type t);

RelationType getI64RelationType(mlir::MLIRContext *ctx);

} // namespace garel
