import { EditorView, basicSetup } from "codemirror"
import { vim } from "@replit/codemirror-vim"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { GraphAlg } from "codemirror-lang-graphalg"
import playgroundWasm from "/workspaces/graphalg/compiler/build-wasm/graphalg-playground.wasm"
import playgroundWasmFactory from "/workspaces/graphalg/compiler/build-wasm/graphalg-playground.js"

// Set as class=.. on the container for the editor view
const EDITOR_CONTAINER_CLASS = 'pt-1';

class GraphAlgEditor {
  constructor(rootElem) {
    this.root = rootElem;

    // Container for toolbar buttons above the editor
    this.toolbar = document.createElement("div");

    // Container to host the editor view
    this.editorContainer = document.createElement("div");
    this.editorContainer.setAttribute('class', EDITOR_CONTAINER_CLASS);

    // Container for output
    this.outputContainer = document.createElement("div");

    this.root.append(this.toolbar, this.editorContainer, this.outputContainer);
  }

  initializeEditorView() {
    this.editorView = new EditorView({
      extensions: [
        //vim(),
        keymap.of([indentWithTab]),
        basicSetup,
        GraphAlg()
      ],
      parent: this.editorContainer
    });
  }
}

// Find graphalg-editor elements
const editorElems = document.getElementsByClassName("graphalg-editor");
const editors = []
for (let elem of editorElems) {
  editors.push(new GraphAlgEditor(elem));
}

// Initialize editor views
for (let editor of editors) {
  editor.initializeEditorView();
}

// Load the playground wasm
// Create run/compile buttons

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

  function run(editor) {
    const pg = ga_new();
    const program = editor.editorView.state.doc.toString();

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

    // Create an output table.
    const table = document.createElement("table");

    // Header
    const thead = document.createElement("thead");
    const tr = document.createElement("tr");
    const thRow = document.createElement("th");
    thRow.textContent = "Row";
    const thCol = document.createElement("th");
    thCol.textContent = "Column";
    const thVal = document.createElement("th");
    thVal.textContent = "Value";
    tr.append(thRow, thCol, thVal);
    thead.appendChild(tr);

    // Body
    const tbody = document.createElement("tbody");
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const tr = document.createElement("tr");
        const tdRow = document.createElement("td");
        tdRow.textContent = r;
        const tdCol = document.createElement("td");
        tdCol.textContent = c;

        const tdVal = document.createElement("td");
        const v = ga_get_res_int(pg, r, c);
        tdVal.textContent = v;
        tr.append(tdRow, tdCol, tdVal);
        tbody.appendChild(tr);
      }
    }

    table.append(thead, tbody);
    editor.outputContainer.replaceChildren(table);
    editor.outputContainer.append(table);

    ga_free(pg);
  }

  for (let editor of editors) {
    const runButton = document.createElement("button");
    runButton.setAttribute('type', 'button');
    runButton.setAttribute('name', 'run');
    runButton.setAttribute('class', 'btn');
    runButton.textContent = "Run";
    runButton.addEventListener('click', () => {
      run(editor);
    });
    editor.toolbar.appendChild(runButton);
  }
});
