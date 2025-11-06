var factory = require('/workspaces/graphalg/compiler/build-wasm/graphalg-playground.js');

factory().then((instance) => {
    ga_new = instance.cwrap('ga_new', 'number', []);
    ga_try_parse = instance.cwrap('ga_try_parse', 'number', ['number', 'string']);
    ga_get_printed = instance.cwrap('ga_get_printed', 'number', ['number']);
    ga_free = instance.cwrap('ga_free', null, ['number'])

    inst = ga_new();
    ga_try_parse(inst, `
        func Hello(a:int) -> int {
            return a;
        }
    `);
    printed = instance.UTF8ToString(ga_get_printed(inst));
    console.log(printed);
    ga_free(inst);
});
