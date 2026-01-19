#ifndef GAREL_DIALECT
#define GAREL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def GARel_Dialect : Dialect {
    let name = "garel";
    let cppNamespace = "::garel";

    let extraClassDeclaration = [{
    private:
        void registerAttributes();
        void registerTypes();
    }];

    let usePropertiesForAttributes = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

class GARel_Op<string mnemonic, list<Trait> traits = []> :
        Op<GARel_Dialect, mnemonic, traits>;

#endif // GAREL_DIALECT
