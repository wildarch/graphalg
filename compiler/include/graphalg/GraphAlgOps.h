#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <graphalg/GraphAlgAttr.h>
#include <graphalg/GraphAlgInterfaces.h>
#include <graphalg/GraphAlgTypes.h>

#define GET_OP_CLASSES
#include "graphalg/GraphAlgOps.h.inc"

namespace graphalg {} // namespace graphalg
