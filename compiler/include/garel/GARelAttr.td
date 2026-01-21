#ifndef GAREL_ATTR
#define GAREL_ATTR

include "mlir/IR/BuiltinAttributeInterfaces.td"

include "GARelDialect.td"

class GARel_Attr<string name, string attrMnemonic, list<Trait> traits = []>
        : AttrDef<GARel_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

// Also often referred to as 'attribute' in the literature, but that conflicts
// with MLIRs use of the term 'attribute', so name this Column instead.
def ColumnAttr : GARel_Attr<"Column", "column"> {
    let summary = "A column (also referred to as 'attribute') of a relation";

    let parameters = (ins
        "::mlir::DistinctAttr":$id,
        TypeParameter<"::mlir::Type", "Type of this column">:$type);

    let assemblyFormat = [{
        `<` $id `` `:` `` $type `>`
    }];

    let extraClassDeclaration = [{
        static ColumnAttr newOfType(mlir::Type type);
    }];
}

def ColumnListParameter : OptionalArrayRefParameter<"ColumnAttr">;

def JoinPredicate : GARel_Attr<"JoinPredicate", "join_pred"> {
    let summary = "A binary equality join predicate";

    let parameters = (ins "ColumnAttr":$lhs, "ColumnAttr":$rhs);

    let assemblyFormat = [{
        `<` $lhs `=` $rhs `>`
    }];
}

def JoinPredicates : ArrayOfAttr<
        GARel_Dialect,
        "JoinPredicates",
        "join_preds",
        "JoinPredicateAttr">;

def ColumnList : GARel_Attr<"ColumnList", "columns"> {
    let summary = "Array of columns";

    let parameters = (ins
        ColumnListParameter:$columns);

    let assemblyFormat = [{
        `<` (`>`) : (`` $columns^ `>`)?
    }];
}

def AggregateFunc : I64EnumAttr<
    "AggregateFunc", "",
    [
        I64EnumAttrCase<"SUM", 0>,
        I64EnumAttrCase<"MIN", 1>,
        I64EnumAttrCase<"MAX", 2>,
        I64EnumAttrCase<"LOR", 3>, /* Logical OR (over i1) */
        I64EnumAttrCase<"ARGMIN", 4>,
    ]
> {
    let cppNamespace = "::garel";
}

// NOTE: assumes an aggregator produces exactly one output column.
def Aggregator : GARel_Attr<"Aggregator", "aggregator"> {
    let summary = "Aggregate function with bound input columns";

    let parameters = (ins
        "AggregateFunc":$func,
        ColumnListParameter:$inputs);

    let assemblyFormat = [{
        `<` $func $inputs `>`
    }];

    let genVerifyDecl = 1;

    let extraClassDeclaration = [{
        /** Type for values in the output column. */
        mlir::Type getResultType();
    }];
}

def Aggregators : ArrayOfAttr<
        GARel_Dialect,
        "Aggregators",
        "aggregators",
        "AggregatorAttr">;

#endif // GAREL_ATTR
