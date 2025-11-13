import { EditorView, basicSetup } from "codemirror"
import { vim } from "@replit/codemirror-vim"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { GraphAlg } from "codemirror-lang-graphalg"
import playgroundWasm from "/workspaces/graphalg/compiler/build-wasm/graphalg-playground.wasm"
import playgroundWasmFactory from "/workspaces/graphalg/compiler/build-wasm/graphalg-playground.js"

// Find graphalg-editor elements
// Create a graphalg-editor-container and initialize the editor on them.

// Load the playground wasm
// Create run/compile buttons

const editorContainer = document.getElementsByClassName("graphalg-editor-container");

let editor = new EditorView({
  extensions: [
    //vim(),
    keymap.of([indentWithTab]),
    basicSetup,
    GraphAlg()
  ],
  parent: editorContainer[0]
})

editor.state.doc.toString()

playgroundWasmFactory({
  locateFile: function (path, prefix) {
    return playgroundWasm;
  },
}).then((instance) => {
  const ga_new = instance.cwrap('ga_new', 'number', []);
  const ga_free = instance.cwrap('ga_free', null, ['number']);
  const ga_parse = instance.cwrap('ga_parse', 'number', ['number', 'string']);
  const ga_desugar = instance.cwrap('ga_desugar', 'number', ['number'])
  const ga_add_arg = instance.cwrap('ga_add_arg', null, ['number', 'number', 'number']);
  const ga_set_dims = instance.cwrap('ga_set_dims', 'number', ['number', 'string']);
  const ga_set_arg_bool = instance.cwrap('ga_set_arg_bool', 'number', ['number', 'number', 'number', 'number', 'number']);
  const ga_set_arg_int = instance.cwrap('ga_set_arg_int', 'number', ['number', 'number', 'number', 'number', 'number']);
  const ga_set_arg_real = instance.cwrap('ga_set_arg_real', 'number', ['number', 'number', 'number', 'number', 'number']);
  const ga_evaluate = instance.cwrap('ga_evaluate', 'number', ['number']);
  const ga_get_res_inf = instance.cwrap('ga_get_res_inf', 'number', ['number', 'number', 'number']);
  const ga_get_res_int = instance.cwrap('ga_get_res_int', 'number', ['number', 'number', 'number']);
  const ga_get_res_real = instance.cwrap('ga_get_res_real', 'number', ['number', 'number', 'number']);

  window.run = function () {

    const pg = ga_new();
    const program = editor.state.doc.toString();

    if (!ga_parse(pg, program)) {
      console.error("Parse failed");
      return;
    }

    if (!ga_desugar(pg)) {
      console.error("Desugar failed");
      return;
    }

    ga_add_arg(pg, 2, 2); // lhs
    ga_add_arg(pg, 2, 2); // rhs
    ga_set_dims(pg, "MatMul");

    ga_set_arg_int(pg, 0, 0, 0, 3n);
    ga_set_arg_int(pg, 0, 0, 1, 5n);
    ga_set_arg_int(pg, 0, 1, 0, 7n);
    ga_set_arg_int(pg, 0, 1, 1, 1n);

    ga_set_arg_int(pg, 1, 0, 0, 13n);
    ga_set_arg_int(pg, 1, 0, 1, 17n);
    ga_set_arg_int(pg, 1, 1, 0, 19n);
    ga_set_arg_int(pg, 1, 1, 1, 23n);

    if (!ga_evaluate(pg)) {
      console.error("Evaluate failed");
    }

    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const v = ga_get_res_int(pg, r, c);
        console.log(r, c, v);
      }
    }

    ga_free(pg);
  }
});
