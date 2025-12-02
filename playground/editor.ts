import { EditorView, basicSetup } from "codemirror"
import { vim } from "@replit/codemirror-vim"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { linter, Diagnostic } from "@codemirror/lint"
import { GraphAlg } from "codemirror-lang-graphalg"
import { loadPlaygroundWasm } from "./binding.mjs"
import { DataSet, Network } from "vis-network/standalone"
import katex from "katex"
import renderMathInElement from "katex/contrib/auto-render"

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
    ring: string;
    rows: number;
    cols: number;
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

    compile(program: string): GraphAlgDiagnostic[] {
        const ga_new = this.bindings.ga_new;
        const ga_parse = this.bindings.ga_parse;
        const ga_desugar = this.bindings.ga_desugar;

        const pg = ga_new();

        if (!ga_parse(pg, program)) {
            return this.getDiagnosticsAndFree(pg);
        }

        if (!ga_desugar(pg)) {
            return this.getDiagnosticsAndFree(pg);
        }

        return [];
    }

    run(program: string, func: string, args: GraphAlgMatrix[]): RunResults {
        const ga_new = this.bindings.ga_new;
        const ga_parse = this.bindings.ga_parse;
        const ga_desugar = this.bindings.ga_desugar;
        const ga_add_arg = this.bindings.ga_add_arg;
        const ga_set_dims = this.bindings.ga_set_dims;
        const ga_set_arg_bool = this.bindings.ga_set_arg_bool;
        const ga_set_arg_int = this.bindings.ga_set_arg_int;
        const ga_set_arg_real = this.bindings.ga_set_arg_real;
        const ga_evaluate = this.bindings.ga_evaluate;
        const ga_get_res_ring = this.bindings.ga_get_res_ring;
        const ga_get_res_rows = this.bindings.ga_get_res_rows;
        const ga_get_res_cols = this.bindings.ga_get_res_cols;
        const ga_get_res_bool = this.bindings.ga_get_res_bool;
        const ga_get_res_int = this.bindings.ga_get_res_int;
        const ga_get_res_real = this.bindings.ga_get_res_real;
        const ga_get_res_inf = this.bindings.ga_get_res_inf;
        const UTF8ToString = this.bindings.UTF8ToString;

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
                switch (arg.ring) {
                    case 'i1':
                        ga_set_arg_bool(pg, idx, val.row, val.col, val.val);
                        break;
                    case 'i64':
                    case '!graphalg.trop_i64':
                    case '!graphalg.trop_max_i64':
                        ga_set_arg_int(pg, idx, val.row, val.col, val.val);
                        break;
                    case 'f64':
                    case '!graphalg.trop_f64':
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

        const resultRing = UTF8ToString(ga_get_res_ring(pg));
        const resultRows = ga_get_res_rows(pg);
        const resultCols = ga_get_res_cols(pg);
        let resultVals: GraphAlgMatrixEntry[] = [];
        for (let r = 0; r < resultRows; r++) {
            for (let c = 0; c < resultCols; c++) {
                let val;
                switch (resultRing) {
                    case 'i1':
                        val = ga_get_res_bool(pg, r, c);
                        break;
                    case '!graphalg.trop_i64':
                    case '!graphalg.trop_max_i64':
                        if (ga_get_res_inf(pg, r, c)) {
                            // Skip infinity
                            continue;
                        }
                    case 'i64':
                        val = ga_get_res_int(pg, r, c);
                        break;
                    case '!graphalg.trop_f64':
                        if (ga_get_res_inf(pg, r, c)) {
                            // Skip infinity
                            continue;
                        }
                    case 'f64':
                        val = ga_get_res_real(pg, r, c);
                        break;
                    default:
                        throw Error(`Invalid result semiring '${resultRing}'`);
                }

                resultVals.push({
                    row: r,
                    col: c,
                    val: val,
                });
            }
        }

        return {
            result: {
                ring: resultRing,
                rows: resultRows,
                cols: resultCols,
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
    const ring = header[2].trim();

    let values: GraphAlgMatrixEntry[] = [];
    for (let line of lines.slice(1)) {
        if (!line) {
            // Skip empty lines
            continue
        }

        const parts = line.split(',');

        let val: boolean | bigint | number;
        switch (ring) {
            case 'i1':
                val = true;
                break;
            case 'i64':
            case '!graphalg.trop_i64':
            case '!graphalg.trop_max_i64':
                val = BigInt(parts[2]);
                break;
            case 'f64':
            case '!graphalg.trop_f64':
                val = parseFloat(parts[2]);
                break;
            default:
                throw new Error(`invalid ring ${ring}`);
        }

        values.push({
            row: parseInt(parts[0]),
            col: parseInt(parts[1]),
            val,
        })
    }

    return {
        ring,
        rows,
        cols,
        values,
    }
}

function renderValue(entry: boolean | bigint | number, ring: string) {
    switch (ring) {
        case 'f64':
        case '!graphalg.trop_f64':
            return (entry as number).toFixed(3);
        case 'i1':
            return (entry as boolean) ? "1" : "0";
        default:
            return entry.toString();
    }
}

function renderMatrixLatex(m: GraphAlgMatrix): HTMLElement {
    let defaultCellValue;
    switch (m.ring) {
        case "i1":
        case "i64":
        case "f64":
            defaultCellValue = "0";
            break;
        case "!graphalg.trop_i64":
        case "!graphalg.trop_f64":
        case "!graphalg.trop_max_i64":
            defaultCellValue = "\\infty";
            break;
        default:
            defaultCellValue = "";
            break;
    }

    let rows: string[][] = [];
    for (let r = 0; r < m.rows; r++) {
        let cols: string[] = [];
        for (let c = 0; c < m.cols; c++) {
            cols.push(defaultCellValue);
        }

        rows.push(cols);
    }

    for (let val of m.values) {
        rows[val.row][val.col] = renderValue(val.val, m.ring);
    }

    const tex =
        "\\begin{bmatrix}\n" +
        rows.map((row) => row.join(" & ")).join("\\\\")
        + "\n\\end{bmatrix}";

    const katexCont = document.createElement("div");
    // TODO: We could generate MathML directly, skipping katex entirely.
    katex.render(tex, katexCont, { output: "mathml" });
    return katexCont;
}

function renderMatrixTable(m: GraphAlgMatrix): HTMLTableElement {
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
        tdVal.textContent = renderValue(val.val, m.ring);
        tr.append(tdRow, tdCol, tdVal);
        tbody.appendChild(tr);
    }

    table.append(thead, tbody);
    return table;
}

function renderMatrixVisGraph(m: GraphAlgMatrix): HTMLElement {
    if (m.rows != m.cols) {
        throw Error("renderMatrixVisGraph called with a non-square matrix");
    }

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "300px";
    container.style.border = "1px solid black";

    interface NodeItem {
        id: number;
        label: string;
    }
    const nodes = new DataSet<NodeItem>();
    for (let r = 0; r < m.rows; r++) {
        nodes.add({ id: r, label: r.toString() });
    }

    // create an array with edges
    interface EdgeItem {
        id: number;
        from: number;
        to: number;
        label: string;
    }
    const edges = new DataSet<EdgeItem>();
    for (let val of m.values) {
        let label = renderValue(val.val, m.ring);
        if (m.ring == "i1" && val.val == true) {
            label = "";
        }

        edges.add({
            id: edges.length,
            from: val.row,
            to: val.col,
            label: label
        });
    }

    // create a network
    const data = {
        nodes: nodes,
        edges: edges,
    };
    var options = {
        layout: {
            // Deterministic layout of graphs
            randomSeed: 42
        },
        edges: {
            arrows: {
                to: {
                    enabled: true
                }
            }
        }
    };
    var network = new Network(container, data, options);

    return container;
}

function renderVectorAsNodeProperty(
    vector: GraphAlgMatrix,
    graph: GraphAlgMatrix): HTMLElement {
    if (vector.rows != graph.rows
        || graph.rows != graph.cols
        || vector.cols != 1) {
        console.warn("cannot render as node property due to incompatible dimensions, falling back to default output rendering");
        return renderMatrixAuto(vector);
    }

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "300px";
    container.style.border = "1px solid black";

    interface NodeItem {
        id: number;
        label: string;
        color?: string;
    }
    let nodes: NodeItem[] = [];
    for (let r = 0; r < vector.rows; r++) {
        let label = "Node " + r.toString();
        if (vector.ring == '!graphalg.trop_f64'
            || vector.ring == '!graphalg.trop_i64') {
            label = `Node ${r}\nvalue: ∞`;
        }

        nodes.push({ id: r, label: label });
    }

    for (let val of vector.values) {
        const renderVal = renderValue(val.val, vector.ring);
        nodes[val.row].label = `Node ${val.row}\nvalue: ${renderVal}`;
        if (vector.ring == 'i1' && val.val) {
            // A shade of red to complement default blue.
            nodes[val.row].color = '#FB7E81';
        }
    }

    const nodeDataSet = new DataSet(nodes);

    // create an array with edges
    interface EdgeItem {
        id: number;
        from: number;
        to: number;
        label: string;
    }
    const edges = new DataSet<EdgeItem>();
    for (let val of graph.values) {
        edges.add({
            id: edges.length,
            from: val.row,
            to: val.col,
            label: renderValue(val.val, graph.ring)
        });
    }

    // create a network
    const data = {
        nodes: nodes,
        edges: edges,
    };
    var options = {
        layout: {
            // Deterministic layout of graphs
            randomSeed: 42
        },
        edges: {
            arrows: {
                to: {
                    enabled: true
                }
            }
        }
    };
    var network = new Network(container, data, options);

    return container;
}

function renderMatrixAuto(m: GraphAlgMatrix): HTMLElement {
    if (m.rows == 1 && m.cols == 1) {
        // Simple scalar
        return renderMatrixLatex(m);
    } else if (m.rows == m.cols && m.rows < 20) {
        return renderMatrixVisGraph(m);
    } else if (m.rows < 20 && m.cols < 20) {
        return renderMatrixLatex(m);
    } else {
        return renderMatrixTable(m);
    }
}

enum MatrixRenderMode {
    AUTO,
    LATEX,
    VIS_GRAPH,
    TABLE,
    VERTEX_PROPERTY,
}

function renderMatrix(m: GraphAlgMatrix, mode: MatrixRenderMode) {
    switch (mode) {
        case MatrixRenderMode.LATEX:
            return renderMatrixLatex(m);
        case MatrixRenderMode.VIS_GRAPH:
            return renderMatrixVisGraph(m);
        case MatrixRenderMode.TABLE:
            return renderMatrixTable(m);
        default:
            return renderMatrixAuto(m);
    }
}

class GraphAlgEditor {
    root: Element;
    toolbar: Element;
    editorContainer: Element;
    argumentContainer: Element;
    outputContainer: Element;

    initialProgram: string;
    functionName?: string;
    arguments: GraphAlgMatrix[] = [];
    renderMode: MatrixRenderMode = MatrixRenderMode.AUTO;
    resultRenderMode: MatrixRenderMode = MatrixRenderMode.AUTO;

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

        this.argumentContainer = document.createElement("div");

        // Container for output
        this.outputContainer = document.createElement("div");

        this.root.append(
            this.toolbar,
            this.editorContainer,
            this.argumentContainer,
            this.outputContainer);
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

    addArgument(arg: GraphAlgMatrix) {
        this.arguments.push(arg);

        // Display in accordion below the editor.
        const argDetails = document.createElement("details");
        const argSummary = document.createElement("summary");
        argSummary.textContent = `Argument ${this.arguments.length} (${arg.ring} x ${arg.rows} x ${arg.cols})`;
        const table = renderMatrix(arg, this.renderMode);
        argDetails.append(argSummary, table);
        this.argumentContainer.appendChild(argDetails);
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

        // NOTE: Need to configure this before we add arguments.
        const defaultRender = elem.getAttribute('data-ga-render');
        if (defaultRender == 'latex') {
            editor.renderMode = MatrixRenderMode.LATEX;
        }

        for (let i = 0; ; i++) {
            const arg = elem.getAttribute('data-ga-arg-' + i);
            if (!arg) {
                break;
            }

            const parsed = parseMatrix(arg);
            editor.addArgument(parsed);
        }

        const resultRender = elem.getAttribute('data-ga-result-render');
        if (!resultRender) {
            editor.resultRenderMode = editor.renderMode;
        } if (resultRender == 'vertex-property') {
            editor.resultRenderMode = MatrixRenderMode.VERTEX_PROPERTY;
        } if (resultRender == 'latex') {
            editor.resultRenderMode = MatrixRenderMode.LATEX;
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

function buildCompileSuccessNote(): HTMLQuoteElement {
    const quote = document.createElement("blockquote");
    quote.setAttribute('class', 'success-title');
    const messages = [
        "Compiled successfully",
        "Parser: OK, syntax is valid",
        "Type checker: OK, types are valid",
    ];

    for (let msg of messages) {
        const pelem = document.createElement("p");
        pelem.textContent = msg;
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
    let resultElem;
    if (result.result) {
        if (editor.resultRenderMode == MatrixRenderMode.VERTEX_PROPERTY) {
            resultElem = renderVectorAsNodeProperty(result.result, editor.arguments[0]);
        } else {
            resultElem = renderMatrix(result.result, editor.resultRenderMode);
        }
    } else {
        resultElem = buildErrorNote(result.diagnostics);
    }

    // Place output in a default-open accordion
    const details = document.createElement("details");
    details.setAttribute('open', 'true');
    const summary = document.createElement("summary");
    summary.textContent = "Output";
    details.append(summary, resultElem);
    editor.outputContainer.replaceChildren(details);
}

function compile(editor: GraphAlgEditor, inst: PlaygroundInstance) {
    const program = editor.editorView?.state.doc.toString();
    if (!program) {
        throw new Error("No program to compile");
    }

    const diagnostics = inst.compile(program);
    let resultElem;
    if (diagnostics.length > 0) {
        resultElem = buildErrorNote(diagnostics);
    } else {
        resultElem = buildCompileSuccessNote();
    }

    // Place output in a default-open accordion
    const details = document.createElement("details");
    details.setAttribute('open', 'true');
    const summary = document.createElement("summary");
    summary.textContent = "Output";
    details.append(summary, resultElem);
    editor.outputContainer.replaceChildren(details);
}

// Add run buttons
playgroundWasmBindings.onLoaded((bindings: any) => {
    const instance = new PlaygroundInstance(bindings);

    for (let editor of editors) {
        if (editor.functionName) {
            const runButton = document.createElement("button");
            runButton.setAttribute('type', 'button');
            runButton.setAttribute('name', 'run');
            runButton.setAttribute('class', 'btn');
            runButton.textContent = `Run '${editor.functionName}'`;
            runButton.addEventListener('click', () => {
                run(editor, instance);
            });
            editor.toolbar.appendChild(runButton);
        } else {
            // No function to run, compile only
            const compileButton = document.createElement("button");
            compileButton.setAttribute('type', 'button');
            compileButton.setAttribute('name', 'compile');
            compileButton.setAttribute('class', 'btn');
            compileButton.textContent = "Compile";
            compileButton.addEventListener('click', () => {
                compile(editor, instance);
            });
            editor.toolbar.appendChild(compileButton);
        }
    }
});

// Initialize graph views
const graphElems = document.getElementsByClassName("language-graphalg-matrix");
for (let elem of Array.from(graphElems)) {
    const mat = parseMatrix(elem.textContent.trim());

    let mode: string | null = null;
    if (elem.parentElement?.tagName == 'PRE') {
        // Have additional annotations in a pre wrapper
        elem = elem.parentElement;

        mode = elem.getAttribute('data-ga-mode');
    }

    let rendered: HTMLElement;
    if (mode == "vis") {
        rendered = renderMatrixVisGraph(mat);
    } else if (mode == "coo") {
        rendered = renderMatrixTable(mat);
    } else {
        rendered = renderMatrixLatex(mat);
    }

    elem.replaceWith(rendered);
}

// Initialize math views
const mathElems = document.getElementsByClassName("language-math");
for (let elem of Array.from(mathElems)) {
    const container = document.createElement("div");
    katex.render(elem.textContent, container, { output: "mathml", displayMode: true });
    elem.replaceWith(container);
}

// Initialize inline math.
renderMathInElement(document.body, {
    output: 'mathml',
    delimiters: [
        { left: "$", right: "$", display: false },
    ]
})
