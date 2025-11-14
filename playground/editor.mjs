import { EditorView, basicSetup } from "codemirror"
import { vim } from "@replit/codemirror-vim"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { linter, Diagnostic } from "@codemirror/lint"
import { GraphAlg } from "codemirror-lang-graphalg"
import playgroundWasm from "/workspaces/graphalg/compiler/build-wasm/graphalg-playground.wasm"
import playgroundWasmFactory from "/workspaces/graphalg/compiler/build-wasm/graphalg-playground.js"

// Load and register all webassembly bindings
let playgroundWasmBindings = {
  // Flipped to true once bindings have been registered
  loaded: false,
};
playgroundWasmFactory({
  locateFile: function (path, prefix) {
    return playgroundWasm;
  },
}).then((instance) => {
  playgroundWasmBindings.ga_new = instance.cwrap('ga_new', 'number', []);
  playgroundWasmBindings.ga_free = instance.cwrap('ga_free', null, ['number']);
  playgroundWasmBindings.ga_parse = instance.cwrap('ga_parse', 'number', ['number', 'string']);
  playgroundWasmBindings.ga_diag_count = instance.cwrap('ga_diag_count', 'number', ['number']);
  playgroundWasmBindings.ga_diag_line_start = instance.cwrap('ga_diag_line_start', 'number', ['number', 'number']);
  playgroundWasmBindings.ga_diag_line_end = instance.cwrap('ga_diag_line_end', 'number', ['number', 'number']);
  playgroundWasmBindings.ga_diag_col_start = instance.cwrap('ga_diag_col_start', 'number', ['number', 'number']);
  playgroundWasmBindings.ga_diag_col_end = instance.cwrap('ga_diag_col_end', 'number', ['number', 'number']);
  playgroundWasmBindings.ga_diag_msg = instance.cwrap('ga_diag_msg', 'number', ['number', 'number']);
  playgroundWasmBindings.ga_desugar = instance.cwrap('ga_desugar', 'number', ['number'])
  playgroundWasmBindings.ga_add_arg = instance.cwrap('ga_add_arg', null, ['number', 'number', 'number']);
  playgroundWasmBindings.ga_set_dims = instance.cwrap('ga_set_dims', 'number', ['number', 'string']);
  playgroundWasmBindings.ga_set_arg_bool = instance.cwrap('ga_set_arg_bool', 'number', ['number', 'number', 'number', 'number', 'number']);
  playgroundWasmBindings.ga_set_arg_int = instance.cwrap('ga_set_arg_int', 'number', ['number', 'number', 'number', 'number', 'number']);
  playgroundWasmBindings.ga_set_arg_real = instance.cwrap('ga_set_arg_real', 'number', ['number', 'number', 'number', 'number', 'number']);
  playgroundWasmBindings.ga_evaluate = instance.cwrap('ga_evaluate', 'number', ['number']);
  playgroundWasmBindings.ga_get_res_inf = instance.cwrap('ga_get_res_inf', 'number', ['number', 'number', 'number']);
  playgroundWasmBindings.ga_get_res_int = instance.cwrap('ga_get_res_int', 'number', ['number', 'number', 'number']);
  playgroundWasmBindings.ga_get_res_real = instance.cwrap('ga_get_res_real', 'number', ['number', 'number', 'number']);
  playgroundWasmBindings.UTF8ToString = instance.UTF8ToString;
  playgroundWasmBindings.loaded = true;
});

class GraphAlgDiagnostic {
  constructor(startLine, endLine, startColumn, endColumn, message) {
    this.startLine = startLine;
    this.endLine = endLine;
    this.startColumn = startColumn;
    this.endColumn = endColumn;
    this.message = message;
  }
}

class PlaygroundInstance {
  constructor(bindings) {
    this.bindings = bindings;
  }

  init() {
    this.handle = this.bindings.ga_new();
  }

  lint(program) {
    const ga_new = this.bindings.ga_new;
    const ga_free = this.bindings.ga_free;
    const ga_parse = this.bindings.ga_parse;
    const ga_desugar = this.bindings.ga_desugar;
    const ga_diag_count = this.bindings.ga_diag_count;
    const ga_diag_line_start = this.bindings.ga_diag_line_start;
    const ga_diag_line_end = this.bindings.ga_diag_line_end;
    const ga_diag_col_start = this.bindings.ga_diag_col_start;
    const ga_diag_col_end = this.bindings.ga_diag_col_end;
    const ga_diag_msg = this.bindings.ga_diag_msg;
    const UTF8ToString = this.bindings.UTF8ToString;

    const pg = ga_new();

    let diagnostics = [];
    if (ga_parse(pg, program)) {
      // Also desugar
      // TODO: Catch desugar error
      ga_desugar(pg);
    }

    const ndiag = ga_diag_count(pg);
    for (let i = 0; i < ndiag; i++) {
      diagnostics.push(new GraphAlgDiagnostic(
        ga_diag_line_start(pg, i),
        ga_diag_line_end(pg, i),
        ga_diag_col_start(pg, i),
        ga_diag_col_end(pg, i),
        UTF8ToString(ga_diag_msg(pg, i))));
    }

    ga_free(pg);
    return diagnostics;
  }
}

const GraphAlgLinter = linter(view => {
  let diagnostics = [];
  if (!playgroundWasmBindings.loaded) {
    // Need to have wasm bindings loaded before we can lint.
    return diagnostics;
  }

  const inst = new PlaygroundInstance(playgroundWasmBindings);
  const program = view.state.doc.toString();
  for (let diag of inst.lint(program)) {
    const fromLine = view.state.doc.line(diag.startLine);
    const toLine = view.state.doc.line(diag.endLine);
    if (!fromLine || !toLine) {
      console.error("Diagnostic at invalid line: ", diag);
      continue;
    }

    diagnostics.push({
      from: fromLine.from + diag.startColumn - 1,
      to: toLine.from + diag.endColumn - 1,
      severity: "error",
      message: diag.message,
    });
  }

  return diagnostics;
});

class GraphAlgEditor {
  constructor(rootElem) {
    this.root = rootElem;

    // Container for toolbar buttons above the editor
    this.toolbar = document.createElement("div");

    // Container to host the editor view
    this.editorContainer = document.createElement("div");
    // NOTE: pt-1 is a just-the-docs class to add padding at the top.
    // This helps separate it from the toolbar.
    this.editorContainer.setAttribute('class', 'pt-1');

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
        GraphAlg(),
        GraphAlgLinter,
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

playgroundWasmFactory({
  locateFile: function (path, prefix) {
    return playgroundWasm;
  },
}).then((instance) => {
  const ga_new = instance.cwrap('ga_new', 'number', []);
  const ga_free = instance.cwrap('ga_free', null, ['number']);
  const ga_parse = instance.cwrap('ga_parse', 'number', ['number', 'string']);
  const ga_diag_count = instance.cwrap('ga_diag_count', 'number', ['number']);
  const ga_diag_line_start = instance.cwrap('ga_diag_line_start', 'number', ['number', 'number']);
  const ga_diag_line_end = instance.cwrap('ga_diag_line_end', 'number', ['number', 'number']);
  const ga_diag_col_start = instance.cwrap('ga_diag_col_start', 'number', ['number', 'number']);
  const ga_diag_col_end = instance.cwrap('ga_diag_col_end', 'number', ['number', 'number']);
  const ga_diag_msg = instance.cwrap('ga_diag_msg', 'number', ['number', 'number']);
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
      const ndiag = ga_diag_count(pg);
      for (let i = 0; i < ndiag; i++) {
        const line = ga_diag_line_start(pg, i);
        const col = ga_diag_col_start(pg, i);
        const msg = instance.UTF8ToString(ga_diag_msg(pg, i));
        console.error(line, col, msg);
      }
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
