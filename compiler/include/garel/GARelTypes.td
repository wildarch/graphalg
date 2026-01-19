
#ifndef GAREL_TYPES
#define GAREL_TYPES

include "mlir/IR/AttrTypeBase.td"

include "GARelDialect.td"
include "GARelAttr.td"

class GARel_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<GARel_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Relation : GARel_Type<"Relation", "rel"> {
    let summary = "A set of tuples";

    let parameters = (ins ColumnSetAttr:$columns);

    let assemblyFormat = [{
        `<` $columns `>`
    }];
}

#endif // GAREL_TYPES
