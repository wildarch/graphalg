#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/DataFlowFramework.h>

#include <graphalg/analysis/DenseAnalysis.h>
#include <graphalg/GraphAlgInterfaces.h>
#include <graphalg/GraphAlgTypes.h>

#include "graphalg/GraphAlgTypeInterfaces.cpp.inc"

namespace graphalg {

mlir::LogicalResult verifySameOperandsAndResultSemiring(mlir::Operation* op) {
    for (auto r : op->getResultTypes()) {
        auto resultType = llvm::cast<MatrixType>(r);
        for (auto o : op->getOperandTypes()) {
            auto operandType = llvm::cast<MatrixType>(o);
            if (resultType.getSemiring() != operandType.getSemiring()) {
                return op->emitOpError()
                        << "requires the same semiring for all operands and"
                        << " results";
            }
        }
    }

    return mlir::success();
}

#include "graphalg/GraphAlgOpInterfaces.cpp.inc"

} // namespace graphalg
