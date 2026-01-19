#ifndef GAREL_OPS
#define GAREL_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "GARelDialect.td"
include "GARelTypes.td"

def ProjectOp : GARel_Op<"project", [Pure]> {
    let summary = "Remaps, reorders, drops and computes columns";

    let arguments = (ins Relation:$input);

    let regions = (region SizedRegion<1>:$projections);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) $projections attr-dict
    }];

    let hasRegionVerifier = 1;
}

def ProjectReturnOp : GARel_Op<"project.return", [
        Pure,
        Terminator,
        HasParent<"ProjectOp">]> {
    let summary = "The output projections";

    let arguments = (ins Variadic<ColumnType>:$projections);

    let assemblyFormat = [{
        $projections `:` type($projections) attr-dict
    }];

    // NOTE: verification performed by ProjectOp
}

def ExtractOp : GARel_Op<"extract", [Pure, InferTypeOpAdaptor]> {
    let summary = "Extract the value of one column from a tuple";

    let arguments = (ins ColumnAttr:$column, Tuple:$tuple);

    let results = (outs ColumnType:$result);

    let assemblyFormat = [{
        $column $tuple `:` type($tuple) attr-dict
    }];

    let hasVerifier = 1;
}

#endif // GAREL_OPS
