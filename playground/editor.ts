import { EditorView, basicSetup } from "codemirror"
import { vim } from "@replit/codemirror-vim"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { linter, Diagnostic } from "@codemirror/lint"
import { GraphAlg } from "codemirror-lang-graphalg"
import { loadPlaygroundWasm } from "./binding.mjs"

// Load and register all webassembly bindings
let playgroundWasmBindings = loadPlaygroundWasm();

interface GraphAlgDiagnostic {
    startLine: number;
    endLine: number;
    startColumn: number;
    endColumn: number;
    message: string;
}

interface GraphAlgMatrixEntry {
    row: number;
    col: number;
    val: boolean | bigint | number;
}
interface GraphAlgMatrix {
    rows: number;
    cols: number;
    type: 'bool' | 'int' | 'real';
    values: GraphAlgMatrixEntry[],
}

interface RunResults {
    result?: GraphAlgMatrix;
    diagnostics: GraphAlgDiagnostic[];
}

class PlaygroundInstance {
    bindings: any;

    constructor(bindings: any) {
        this.bindings = bindings;
    }

    getDiagnosticsAndFree(pg: number): GraphAlgDiagnostic[] {
        const ga_diag_count = this.bindings.ga_diag_count;
        const ga_diag_line_start = this.bindings.ga_diag_line_start;
        const ga_diag_line_end = this.bindings.ga_diag_line_end;
        const ga_diag_col_start = this.bindings.ga_diag_col_start;
        const ga_diag_col_end = this.bindings.ga_diag_col_end;
        const ga_diag_msg = this.bindings.ga_diag_msg;
        const ga_free = this.bindings.ga_free;
        const UTF8ToString = this.bindings.UTF8ToString;

        const diagnostics: GraphAlgDiagnostic[] = [];
        const ndiag = ga_diag_count(pg);
        for (let i = 0; i < ndiag; i++) {
            diagnostics.push({
                startLine: ga_diag_line_start(pg, i),
                endLine: ga_diag_line_end(pg, i),
                startColumn: ga_diag_col_start(pg, i),
                endColumn: ga_diag_col_end(pg, i),
                message: UTF8ToString(ga_diag_msg(pg, i))
            });
        }

        ga_free(pg);
        return diagnostics;
    }

    lint(program: string): GraphAlgDiagnostic[] {
        const ga_new = this.bindings.ga_new;
        const ga_free = this.bindings.ga_free;
        const ga_parse = this.bindings.ga_parse;
        const ga_desugar = this.bindings.ga_desugar;

        const pg = ga_new();
        if (ga_parse(pg, program)) {
            // Also desugar
            // TODO: Catch desugar error
            ga_desugar(pg);
        }

        return this.getDiagnosticsAndFree(pg);
    }

    run(program: string, func: string, args: GraphAlgMatrix[]): RunResults {
        const ga_new = this.bindings.ga_new;
        const ga_free = this.bindings.ga_free;
        const ga_parse = this.bindings.ga_parse;
        const ga_desugar = this.bindings.ga_desugar;
        const ga_add_arg = this.bindings.ga_add_arg;
        const ga_set_dims = this.bindings.ga_set_dims;
        const ga_set_arg_bool = this.bindings.ga_set_arg_bool;
        const ga_set_arg_int = this.bindings.ga_set_arg_int;
        const ga_set_arg_real = this.bindings.ga_set_arg_real;
        const ga_evaluate = this.bindings.ga_evaluate;
        const ga_get_res_int = this.bindings.ga_get_res_int;

        const pg = ga_new();

        if (!ga_parse(pg, program)) {
            return {
                diagnostics: this.getDiagnosticsAndFree(pg),
            };
        }

        if (!ga_desugar(pg)) {
            return {
                diagnostics: this.getDiagnosticsAndFree(pg),
            };
        }

        for (let arg of args) {
            ga_add_arg(pg, arg.rows, arg.cols);
        }

        if (!ga_set_dims(pg, func)) {
            return {
                diagnostics: this.getDiagnosticsAndFree(pg),
            };
        }

        args.forEach((arg, idx) => {
            for (let val of arg.values) {
                switch (arg.type) {
                    case 'bool':
                        ga_set_arg_bool(pg, idx, val.row, val.col, val.val);
                        break;
                    case 'int':
                        ga_set_arg_int(pg, idx, val.row, val.col, val.val);
                        break;
                    case 'real':
                        ga_set_arg_real(pg, idx, val.row, val.col, val.val);
                        break;
                }
            }
        });

        if (!ga_evaluate(pg)) {
            return {
                diagnostics: this.getDiagnosticsAndFree(pg),
            };
        }

        // TODO: Infer the result dimensions and type.
        const resultRows = 2;
        const resultCols = 2;
        const resultType = 'int';
        let resultVals: GraphAlgMatrixEntry[] = [];
        for (let r = 0; r < resultRows; r++) {
            for (let c = 0; c < resultCols; c++) {
                resultVals.push({
                    row: r,
                    col: c,
                    val: ga_get_res_int(pg, r, c),
                });
            }
        }

        return {
            result: {
                rows: resultRows,
                cols: resultCols,
                type: resultType,
                values: resultVals,
            },
            diagnostics: this.getDiagnosticsAndFree(pg),
        };
    }
}

const GraphAlgLinter = linter(view => {
    let diagnostics: Diagnostic[] = [];
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

function parseMatrix(input: string): GraphAlgMatrix {
    const lines = input.split(';');
    const header = lines[0].split(',');
    const rows = parseInt(header[0]);
    const cols = parseInt(header[1]);
    const type = header[2].trim();
    if (type != 'bool' && type != 'int' && type != 'real') {
        throw new Error(`invalid type ${type}`);
    }

    let values: GraphAlgMatrixEntry[] = [];
    for (let line of lines.slice(1)) {
        if (!line) {
            // Skip empty lines
            continue
        }

        const parts = line.split(',');

        let val: boolean | bigint | number;
        if (type == 'bool') {
            val = true;
        } else if (type == 'int') {
            val = BigInt(parts[2]);
        } else if (type == 'real') {
            val = parseFloat(parts[2]);
        } else {
            throw new Error(`invalid type ${type}`);
        }

        values.push({
            row: parseInt(parts[0]),
            col: parseInt(parts[1]),
            val,
        })
    }

    return {
        rows: rows,
        cols: cols,
        type: type,
        values: values,
    }
}

class GraphAlgEditor {
    root: Element;
    toolbar: Element;
    editorContainer: Element;
    outputContainer: Element;

    initialProgram: string;
    functionName?: string;
    arguments: GraphAlgMatrix[] = [];

    editorView?: EditorView;

    constructor(rootElem: Element, program: string) {
        this.root = rootElem;
        this.initialProgram = program;

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
            parent: this.editorContainer,
            doc: this.initialProgram,
        });
    }
}

// Find code snippets to turn into editors.
let editors: GraphAlgEditor[] = [];
const codeElems = document.getElementsByClassName("language-graphalg");
for (let elem of Array.from(codeElems)) {
    // An empty div for the editor to own as its root.
    const editorRoot = document.createElement("div");
    const program = elem.textContent.trim();
    const editor = new GraphAlgEditor(editorRoot, program);

    if (elem.parentElement?.tagName == 'PRE') {
        // Have additional annotations in a pre wrapper
        elem = elem.parentElement;

        const func = elem.getAttribute('data-ga-func');
        if (func) {
            editor.functionName = func;
        }

        for (let i = 0; ; i++) {
            const arg = elem.getAttribute('data-ga-arg-' + i);
            if (!arg) {
                break;
            }

            const parsed = parseMatrix(arg);
            editor.arguments.push(parsed);
        }
    }

    // Replace the code snippet with the editor view.
    elem.replaceWith(editorRoot);

    editors.push(editor);
}

// Initialize editor views
for (let editor of editors) {
    editor.initializeEditorView();
}

function buildTable(m: GraphAlgMatrix): HTMLTableElement {
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
    for (let val of m.values) {
        const tr = document.createElement("tr");
        const tdRow = document.createElement("td");
        tdRow.textContent = val.row.toString();
        const tdCol = document.createElement("td");
        tdCol.textContent = val.col.toString();

        const tdVal = document.createElement("td");
        tdVal.textContent = val.val.toString();
        tr.append(tdRow, tdCol, tdVal);
        tbody.appendChild(tr);
    }

    table.append(thead, tbody);
    return table;
}

function buildErrorNote(diagnostics: GraphAlgDiagnostic[]): HTMLQuoteElement {
    const quote = document.createElement("blockquote");
    quote.setAttribute('class', 'error-title');
    const title = document.createElement("p");
    title.textContent = "Run failed";
    quote.appendChild(title);

    for (let diag of diagnostics) {
        const pelem = document.createElement("p");
        pelem.textContent = `line ${diag.startLine}: ${diag.message}`;
        quote.appendChild(pelem);
    }

    return quote;
}

function run(editor: GraphAlgEditor, inst: PlaygroundInstance) {
    const program = editor.editorView?.state.doc.toString();
    if (!program) {
        throw new Error("No program to run");
    }

    const result = inst.run(program, editor.functionName!!, editor.arguments);
    if (result.result) {
        const table = buildTable(result.result);
        editor.outputContainer.replaceChildren(table);
    } else {
        const note = buildErrorNote(result.diagnostics);
        editor.outputContainer.replaceChildren(note);
    }

}

// Add run buttons
playgroundWasmBindings.onLoaded((bindings: any) => {
    const instance = new PlaygroundInstance(bindings);

    for (let editor of editors) {
        if (!editor.functionName) {
            // No function to run
            continue;
        }

        const runButton = document.createElement("button");
        runButton.setAttribute('type', 'button');
        runButton.setAttribute('name', 'run');
        runButton.setAttribute('class', 'btn');
        runButton.textContent = `Run '${editor.functionName}'`;
        runButton.addEventListener('click', () => {
            run(editor, instance);
        });
        editor.toolbar.appendChild(runButton);
    }
});
