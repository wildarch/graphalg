import { GraphAlgMatrix } from "./GraphAlgMatrix"
import { GraphAlgDiagnostic } from "./GraphAlgDiagnostic"
import { PlaygroundInstance } from "./PlaygroundInstance"
import { renderVectorAsNodeProperty, renderMatrix, MatrixRenderMode } from "./matrixRendering"
import { EditorView, basicSetup } from "codemirror"
import { Extension } from "@codemirror/state"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { GraphAlg } from "codemirror-lang-graphalg"

export enum GraphAlgEditorMode {
    TUTORIAL,
    PLAYGROUND,
}

export class GraphAlgEditor {
    root: Element;
    toolbar: Element;
    editorContainer: Element;
    argumentContainer: Element;
    outputContainer: Element;

    initialProgram: string;
    functionName?: string;
    arguments: GraphAlgMatrix[] = [];
    editorMode: GraphAlgEditorMode = GraphAlgEditorMode.TUTORIAL;
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

    initializeEditorView(linter: Extension) {
        this.editorView = new EditorView({
            extensions: [
                //vim(),
                keymap.of([indentWithTab]),
                basicSetup,
                GraphAlg(),
                linter,
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
        argDetails.appendChild(argSummary);

        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND) {
            // Allow uploading a replacement file.
            const inputFile = document.createElement("input");
            inputFile.setAttribute("type", "file");
            console.log(inputFile);
            argDetails.appendChild(inputFile);

            inputFile.addEventListener("change", async () => {
                if (inputFile.files?.length != 1) {
                    return;
                }

                const file = inputFile.files[0];
                const content = await file.text();
                console.log(content);
            });
        }

        const table = renderMatrix(arg, this.renderMode);
        argDetails.appendChild(table);
        this.argumentContainer.appendChild(argDetails);
    }

    bindPlayground(instance: PlaygroundInstance) {
        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND
            || this.functionName) {
            const runButton = document.createElement("button");
            runButton.setAttribute('type', 'button');
            runButton.setAttribute('name', 'run');
            runButton.setAttribute('class', 'btn');
            runButton.textContent =
                this.editorMode == GraphAlgEditorMode.TUTORIAL
                    ?
                    `Run '${this.functionName}'`
                    : "Run";
            runButton.addEventListener('click', () => {
                this.run(instance);
            });
            this.toolbar.appendChild(runButton);
        } else {
            // No function to run, compile only
            const compileButton = document.createElement("button");
            compileButton.setAttribute('type', 'button');
            compileButton.setAttribute('name', 'compile');
            compileButton.setAttribute('class', 'btn');
            compileButton.textContent = "Compile";
            compileButton.addEventListener('click', () => {
                this.compile(instance);
            });
            this.toolbar.appendChild(compileButton);
        }
    }

    run(inst: PlaygroundInstance) {
        const program = this.editorView?.state.doc.toString();
        if (!program) {
            throw new Error("No program to run");
        }

        const result = inst.run(program, this.functionName!!, this.arguments);
        let resultElem;
        if (result.result) {
            if (this.resultRenderMode == MatrixRenderMode.VERTEX_PROPERTY) {
                resultElem = renderVectorAsNodeProperty(result.result, this.arguments[0]);
            } else {
                resultElem = renderMatrix(result.result, this.resultRenderMode);
            }
        } else {
            resultElem = buildErrorNote(result.diagnostics);
        }

        let outputElems: Node[] = [];
        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND && result.coreIR) {
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = "Core IR";
            const pre = document.createElement("pre");
            const code = document.createElement("code");
            code.textContent = result.coreIR;
            pre.appendChild(code);
            details.append(summary, pre);
            outputElems.push(details);
        }

        // Place output in a default-open accordion
        const details = document.createElement("details");
        details.setAttribute('open', 'true');
        const summary = document.createElement("summary");
        summary.textContent = "Output";
        details.append(summary, resultElem);
        outputElems.push(details);
        this.outputContainer.replaceChildren(...outputElems);
    }

    compile(inst: PlaygroundInstance) {
        const program = this.editorView?.state.doc.toString();
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
        this.outputContainer.replaceChildren(details);
    }
}

function buildErrorNote(diagnostics: GraphAlgDiagnostic[]): HTMLQuoteElement {
    const quote = document.createElement("blockquote");
    quote.setAttribute('class', 'error-title');
    const title = document.createElement("p");
    title.textContent = "Compiler error";
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
    quote.innerHTML = `
        <p>Compiled successfully</p>
        <p>
            Parser: Syntax valid ✓
            <br/>
            Type checker: Types valid ✓
        </p>`;

    return quote;
}
