#ifndef GAREL_OPS
#define GAREL_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "GARelDialect.td"
include "GARelTypes.td"

def ProjectOp : GARel_Op<"project", [Pure]> {
    let summary = "Remaps, reorders, drops and computes columns";

    let arguments = (ins Relation:$input);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];
}

#endif // GAREL_OPS
