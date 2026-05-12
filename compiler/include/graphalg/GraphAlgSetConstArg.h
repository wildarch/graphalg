#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/Support/LLVM.h>

namespace graphalg {

/**
 * Inlines constant values for function parameters.
 *
 * The function parameters to set to constants must be scalar.
 *
 * Constant parameters are NOT removed from the function signature,
 * but the corresponding block arguments will have zero uses.
 *
 * @param op the function for which to inline argument values.
 * @param values for each parameter, the constant value to inline, or null if
 *        the argument is not constant.
 */
mlir::LogicalResult
setConstantArguments(mlir::func::FuncOp op,
                     llvm::ArrayRef<mlir::TypedAttr> values);

} // namespace graphalg
