var factory = require('/workspaces/graphalg/compiler/build-wasm/wasm_parse.js');

factory().then((instance) => {
    parse = instance.cwrap('ga_parse', null, ['string']);
    parse(`
        func Hello(a:int) -> int {
            return a;
        }
    `);
});
