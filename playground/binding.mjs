import playgroundWasm from "../compiler/build-wasm/graphalg-playground.wasm"
import playgroundWasmFactory from "../compiler/build-wasm/graphalg-playground.js"

class PlaygroundWasmBindings {
    constructor() {
        // Flipped to true once bindings have been registered
        this.loaded = false;
        this.callbacks = [];
    }

    onLoaded(cb) {
        if (this.loaded) {
            // Already loaded, invoke callback immediately.
            cb(this);
        } else {
            // Will be called once loaded.
            this.callbacks.push(cb);
        }
    }
}

export function loadPlaygroundWasm() {
    // Load and register all webassembly bindings
    let bindings = new PlaygroundWasmBindings();
    playgroundWasmFactory({
        locateFile: function (path, prefix) {
            return playgroundWasm;
        },
    }).then((instance) => {
        bindings.ga_new = instance.cwrap('ga_new', 'number', []);
        bindings.ga_free = instance.cwrap('ga_free', null, ['number']);
        bindings.ga_parse = instance.cwrap('ga_parse', 'number', ['number', 'string']);
        bindings.ga_diag_count = instance.cwrap('ga_diag_count', 'number', ['number']);
        bindings.ga_diag_line_start = instance.cwrap('ga_diag_line_start', 'number', ['number', 'number']);
        bindings.ga_diag_line_end = instance.cwrap('ga_diag_line_end', 'number', ['number', 'number']);
        bindings.ga_diag_col_start = instance.cwrap('ga_diag_col_start', 'number', ['number', 'number']);
        bindings.ga_diag_col_end = instance.cwrap('ga_diag_col_end', 'number', ['number', 'number']);
        bindings.ga_diag_msg = instance.cwrap('ga_diag_msg', 'number', ['number', 'number']);
        bindings.ga_desugar = instance.cwrap('ga_desugar', 'number', ['number'])
        bindings.ga_add_arg = instance.cwrap('ga_add_arg', null, ['number', 'number', 'number']);
        bindings.ga_set_dims = instance.cwrap('ga_set_dims', 'number', ['number', 'string']);
        bindings.ga_set_arg_bool = instance.cwrap('ga_set_arg_bool', 'number', ['number', 'number', 'number', 'number', 'number']);
        bindings.ga_set_arg_int = instance.cwrap('ga_set_arg_int', 'number', ['number', 'number', 'number', 'number', 'number']);
        bindings.ga_set_arg_real = instance.cwrap('ga_set_arg_real', 'number', ['number', 'number', 'number', 'number', 'number']);
        bindings.ga_evaluate = instance.cwrap('ga_evaluate', 'number', ['number']);
        bindings.ga_get_res_ring = instance.cwrap('ga_get_res_ring', 'number', ['number']);
        bindings.ga_get_res_rows = instance.cwrap('ga_get_res_rows', 'number', ['number']);
        bindings.ga_get_res_cols = instance.cwrap('ga_get_res_cols', 'number', ['number']);
        bindings.ga_get_res_bool = instance.cwrap('ga_get_res_bool', 'boolean', ['number', 'number', 'number']);
        bindings.ga_get_res_int = instance.cwrap('ga_get_res_int', 'number', ['number', 'number', 'number']);
        bindings.ga_get_res_real = instance.cwrap('ga_get_res_real', 'number', ['number', 'number', 'number']);
        bindings.ga_get_res_inf = instance.cwrap('ga_get_res_inf', 'boolean', ['number', 'number', 'number']);
        bindings.UTF8ToString = instance.UTF8ToString;
        bindings.loaded = true;

        for (let cb of bindings.callbacks) {
            cb(bindings);
        }

        bindings.callbacks = [];
    });

    return bindings;
}
