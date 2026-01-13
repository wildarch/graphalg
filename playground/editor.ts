import { linter, Diagnostic } from "@codemirror/lint"
import { loadPlaygroundWasm } from "./binding.mjs"
import katex from "katex"
import renderMathInElement from "katex/contrib/auto-render"
import { PlaygroundInstance } from "./src/PlaygroundInstance"
import { GraphAlgMatrix, GraphAlgMatrixEntry } from "./src/GraphAlgMatrix"
import { GraphAlgEditor, GraphAlgEditorMode } from "./src/GraphAlgEditor"
import { MatrixRenderMode, renderMatrix } from "./src/matrixRendering"

// Load and register all webassembly bindings
let playgroundWasmBindings = loadPlaygroundWasm();

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

        const editorMode = elem.getAttribute('data-ga-editor');
        if (editorMode && editorMode == 'playground') {
            editor.editorMode = GraphAlgEditorMode.PLAYGROUND;
        }

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
    editor.initializeEditorView(GraphAlgLinter);
}

// Add run buttons
playgroundWasmBindings.onLoaded((bindings: any) => {
    const instance = new PlaygroundInstance(bindings);
    for (let editor of editors) {
        editor.bindPlayground(instance);
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

    let renderMode = MatrixRenderMode.LATEX;
    if (mode == "vis") {
        renderMode = MatrixRenderMode.VIS_GRAPH;
    } else if (mode == "coo") {
        renderMode = MatrixRenderMode.TABLE;
    }

    elem.replaceWith(renderMatrix(mat, renderMode));
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
