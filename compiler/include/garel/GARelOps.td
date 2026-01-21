#ifndef GAREL_OPS
#define GAREL_OPS

/**
 * Relation-level ops in the GARel dialect.
 *
 * See ./GARelTupleOps.td for the tuple-level ops.
 */

include "mlir/Interfaces/InferTypeOpInterface.td"

include "GARelDialect.td"
include "GARelTypes.td"

// Per-tuple ops kept in a separate file.
include "GARelTupleOps.td"

def ProjectOp : GARel_Op<"project", [IsolatedFromAbove]> {
    let summary = "Remaps, reorders, drops and computes columns";

    let arguments = (ins Relation:$input);

    let regions = (region SizedRegion<1>:$projections);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) $projections attr-dict
    }];

    let hasRegionVerifier = 1;

    let extraClassDeclaration = [{
        mlir::Block& createProjectionsBlock();
        ProjectReturnOp getTerminator();
    }];
}

def ProjectReturnOp : GARel_Op<"project.return", [
        Terminator,
        HasParent<"ProjectOp">]> {
    let summary = "The output projections";

    let arguments = (ins Variadic<ColumnType>:$projections);

    let assemblyFormat = [{
        $projections `:` type($projections) attr-dict
    }];

    // NOTE: verification performed by ProjectOp
}

def SelectOp : GARel_Op<"select", [
        SameOperandsAndResultType,
        IsolatedFromAbove]> {
    let summary = "Removes tuples that fail (one of) the predicates";

    let arguments = (ins Relation:$input);
    let regions = (region SizedRegion<1>:$predicates);

    let builders = [OpBuilder<(ins "mlir::Value":$child)>];
    let skipDefaultBuilders = 1;

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $input `:` type($input) $predicates attr-dict
    }];

    let hasRegionVerifier = 1;

    let extraClassDeclaration = [{
        SelectReturnOp getTerminator();
    }];
}

def SelectReturnOp : GARel_Op<"select.return", [
        Terminator,
        HasParent<"SelectOp">]> {
    let summary = "Return the select predicates";

    let arguments = (ins Variadic<I1>:$predicates);

    let assemblyFormat = [{
        $predicates attr-dict
    }];
}

def JoinOp : GARel_Op<"join", [InferTypeOpAdaptor]> {
    let summary = "Natural (equi)join of relations";

    let arguments = (ins
        // NOTE: All inputs must have distinct columns
        Variadic<Relation>:$inputs,
        // NOTE: Equality predicates only
        JoinPredicates:$predicates);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $inputs `:` type($inputs)
        $predicates
        attr-dict
    }];

    let hasVerifier = 1;
}

def UnionOp : GARel_Op<"union", [SameOperandsAndResultType]> {
    let summary = "Union of relations";

    let arguments = (ins Variadic<Relation>:$inputs);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $inputs `:` type($inputs)
        attr-dict
    }];
}

def AggregateOp : GARel_Op<"aggregate", [InferTypeOpAdaptor]> {
    let summary = "Groups tuples by key columns, aggregating values of other columns";

    let arguments = (ins
        Relation:$input,
        ColumnList:$groupBy,
        Aggregators:$aggregators);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $input `:` type($input)
        `group_by` `` `=` `` $groupBy
        `aggregators` `` `=` `` $aggregators
        attr-dict
    }];
}

def ForOp : GARel_Op<"for", [InferTypeOpAdaptor]> {
    let summary = "Bounded iteration";

    let arguments = (ins
        Variadic<Relation>:$init,
        I64Attr:$iters,
        I64Attr:$resultIdx);

    let regions = (region SizedRegion<1>:$body);

    let results = (outs Relation:$result);

    let assemblyFormat = [{
        $init `:` type($init)
        `iters` `` `=` `` $iters
        `result_idx` `` `=` `` $resultIdx
        $body
        attr-dict
    }];

    let hasVerifier = 1;
    let hasRegionVerifier = 1;
}

def ForYieldOp : GARel_Op<"for.yield", [
        Terminator,
        HasParent<"ForOp">]> {
    let summary = "Produces the iter args for the next iteration";

    let arguments = (ins Variadic<Relation>:$inputs);

    let assemblyFormat = [{
        $inputs `:` type($inputs) attr-dict
    }];

    // Note: verification performed by parent ForOp.
}

#endif // GAREL_OPS
