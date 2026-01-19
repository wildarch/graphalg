#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include "garel/GARelDialect.h"
#include "garel/GARelOps.h"

#include "garel/GARelOpsDialect.cpp.inc"

namespace garel {

namespace {

class GARelOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute attr,
                       mlir::raw_ostream &os) const override {
    // Assign aliases to columns.
    if (auto colAttr = llvm::dyn_cast<ColumnAttr>(attr)) {
      os << "col";
      return AliasResult::FinalAlias;
    }

    return AliasResult::NoAlias;
  }
};

} // namespace

void GARelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "garel/GARelOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
  addInterface<GARelOpAsmDialectInterface>();
}

} // namespace garel
