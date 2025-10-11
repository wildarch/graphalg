#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

#include "graphalg/GraphAlgEnumAttr.h.inc"

#define GET_ATTRDEF_CLASSES
#include "graphalg/GraphAlgAttr.h.inc"

namespace graphalg {

bool binaryOpIsCompare(BinaryOp op);

}
