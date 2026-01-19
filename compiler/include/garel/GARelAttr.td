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
}

def ColumnList : OptionalArrayRefParameter<"ColumnAttr">;

/*
def ColumnSetAttr : GARel_Attr<"ColumnSet", "column_set"> {
    let summary = "The set of columns in a relation";

    let description = [{
        An ordered set of columns.

        Order is significant: `<a, b>` is a different set than `<b, a>`.

        A column may only appear once in the set: `<a, a>` is invalid.
    }];

    let parameters = (ins
        OptionalArrayRefParameter<"ColumnAttr">:$columns);

    let assemblyFormat = [{
        `<` (`>`) : (`` $columns^ `>`)?
    }];

    let genVerifyDecl = 1;
}
*/

#endif // GAREL_ATTR
