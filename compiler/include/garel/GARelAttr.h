#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "garel/GARelEnumAttr.h.inc"

namespace garel {

/** Reference to a column inside of \c RelationType or \c TupleType. */
using ColumnIdx = unsigned;

} // namespace garel

#define GET_ATTRDEF_CLASSES
#include "garel/GARelAttr.h.inc"
