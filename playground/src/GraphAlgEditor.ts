import { GraphAlgMatrix } from "./GraphAlgMatrix"
import { GraphAlgDiagnostic } from "./GraphAlgDiagnostic"
import { PlaygroundInstance } from "./PlaygroundInstance"
import { renderVectorAsNodeProperty, renderMatrix, MatrixRenderMode } from "./matrixRendering"
import { EditorView, basicSetup } from "codemirror"
import { Extension } from "@codemirror/state"
import { keymap } from "@codemirror/view"
import { indentWithTab } from "@codemirror/commands"
import { GraphAlg } from "codemirror-lang-graphalg"
import { parseMatrix, ParseMatrixError } from "./matrixParsing"
import { highlightMLIR } from './highlightMLIR'

export enum GraphAlgEditorMode {
    TUTORIAL,
    PLAYGROUND,
}

class EditorArgument {
    rootElem: HTMLDetailsElement;
    value?: GraphAlgMatrix;

    constructor(root: HTMLDetailsElement) {
        this.rootElem = root;
    }

    destroy() {
        // TODO: Cleanup network vis etc.
    }
};

export class GraphAlgEditor {
    root: Element;
    toolbar: Element;
    editorContainer: Element;
    argumentContainer: Element;
    argumentToolbar: HTMLElement;
    outputContainer: Element;

    initialProgram: string;
    functionName?: string;
    arguments: EditorArgument[] = [];
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

        this.argumentToolbar = document.createElement("div");

        // Container for output
        this.outputContainer = document.createElement("div");

        this.root.append(
            this.toolbar,
            this.editorContainer,
            this.argumentContainer,
            this.argumentToolbar,
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

        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND) {
            this.argumentToolbar.style.marginTop = '.5em';
            this.argumentToolbar.style.marginBottom = '.5em';

            // Add argument button
            const addButton = document.createElement("button");
            addButton.setAttribute('type', 'button');
            addButton.setAttribute('class', 'btn');
            addButton.textContent = "Add Argument";
            addButton.addEventListener('click', () => {
                this.addArgument();
            });
            this.argumentToolbar.appendChild(addButton);

            // Remove argument button
            const removeButton = document.createElement("button");
            removeButton.setAttribute('type', 'button');
            removeButton.setAttribute('class', 'btn');
            removeButton.textContent = "Remove Argument";
            removeButton.addEventListener('click', () => {
                this.dropArgument();
            });
            this.argumentToolbar.appendChild(removeButton);
        }
    }

    addArgument(value?: GraphAlgMatrix) {
        const argElem = document.createElement("details");
        const argument = new EditorArgument(argElem);
        argument.value = value;

        this.argumentContainer.appendChild(argElem);
        this.arguments.push(argument);
        this.renderArgument(this.arguments.length - 1);
    }

    dropArgument() {
        // TODO: If there are no more arguments left, disable the remove
        // argument button.
        const arg = this.arguments.pop();
        if (arg) {
            arg.destroy();
            arg.rootElem.remove();
        }
    }

    bindPlayground(instance: PlaygroundInstance) {
        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND) {
            // Allow changing the name of called function.
            const funcNameContainer = document.createElement("div");
            funcNameContainer.style.backgroundColor = '#f7f7f7';
            funcNameContainer.style.borderRadius = '4px';
            funcNameContainer.style.padding = '0.3em 1em';
            funcNameContainer.style.borderWidth = '0';
            funcNameContainer.style.boxShadow = 'rgba(0, 0, 0, 0.12) 0px 1px 2px 0px, rgba(0, 0, 0, 0.08) 0px 3px 10px 0px';
            funcNameContainer.style.color = '#7253ed';
            funcNameContainer.style.lineHeight = '1.5';
            funcNameContainer.style.display = 'inline-block';
            funcNameContainer.style.fontWeight = '500';
            funcNameContainer.textContent = "Function:";

            const funcNameInput = document.createElement("input");
            funcNameInput.setAttribute('type', 'text');
            funcNameInput.style.backgroundColor = '#f7f7f7';
            funcNameInput.style.color = '#7253ed';
            funcNameInput.style.borderWidth = '0';
            funcNameContainer.appendChild(funcNameInput);

            if (this.functionName) {
                funcNameInput.value = this.functionName;
            }

            funcNameInput.addEventListener('change', () => {
                console.log(funcNameInput.value);
                this.functionName = funcNameInput.value;
            });

            this.toolbar.appendChild(funcNameContainer);
        }

        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND
            || this.functionName) {
            // Add run button
            const runButton = document.createElement("button");
            runButton.setAttribute('type', 'button');
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
            compileButton.setAttribute('class', 'btn');
            compileButton.textContent = "Compile";
            compileButton.addEventListener('click', () => {
                this.compile(instance);
            });
            this.toolbar.appendChild(compileButton);
        }
    }

    tryRun(inst: PlaygroundInstance): HTMLElement[] {
        let outputElems: HTMLElement[] = [];
        const program = this.editorView?.state.doc.toString();
        if (!program) {
            outputElems.push(buildErrorNote("No program to run"));
        }

        const args: GraphAlgMatrix[] = [];
        this.arguments.forEach((arg, idx) => {
            if (arg.value) {
                args.push(arg.value);
            } else {
                outputElems.push(buildErrorNote(`Argument ${idx} has no value set`));
            }
        });

        if (!this.functionName) {
            outputElems.push(buildErrorNote("No function name set"));
        }

        if (outputElems.length > 0) {
            // Collected some errors.
            return outputElems;
        }

        const result = inst.run(program!!, this.functionName!!, args);
        let resultElem;
        if (result.result) {
            if (this.resultRenderMode == MatrixRenderMode.VERTEX_PROPERTY
                && this.arguments.length >= 1) {
                const height = this.editorMode == GraphAlgEditorMode.PLAYGROUND ?
                    "600px"
                    : "300px";
                resultElem = renderVectorAsNodeProperty(
                    result.result,
                    this.arguments[0].value!!,
                    height);
            } else {
                resultElem = renderMatrix(result.result, this.resultRenderMode);
            }
        } else {
            resultElem = buildDiagnosticsNote(result.diagnostics);
        }

        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND && result.parsedIR) {
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = "GraphAlg IR";
            details.append(summary, renderIR(result.parsedIR));
            outputElems.push(details);
        }

        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND && result.coreIR) {
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = "Core IR";
            details.append(summary, renderIR(result.coreIR));
            outputElems.push(details);
        }

        // Place output in a default-open accordion
        const details = document.createElement("details");
        details.setAttribute('open', 'true');
        const summary = document.createElement("summary");
        summary.textContent = "Output";
        details.append(summary, resultElem);
        outputElems.push(details);

        return outputElems;
    }

    run(inst: PlaygroundInstance) {
        const outputElems = this.tryRun(inst);
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
            resultElem = buildDiagnosticsNote(diagnostics);
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

    renderArgument(argIndex: number) {
        const arg = this.arguments[argIndex];
        // Cleanup visualizations etc.
        arg.destroy();

        const argSummary = document.createElement("summary");
        if (arg.value) {
            argSummary.textContent = `Argument ${argIndex} (${arg.value.ring} x ${arg.value.rows} x ${arg.value.cols})`;
        } else {
            argSummary.textContent = `Argument ${argIndex}`;
        }

        // NOTE: Also cleans up previously rendered nodes.
        arg.rootElem.replaceChildren(argSummary);

        if (this.editorMode == GraphAlgEditorMode.PLAYGROUND) {
            // Allow uploading a replacement file.
            const inputFile = document.createElement("input");
            inputFile.setAttribute("type", "file");
            arg.rootElem.appendChild(inputFile);

            inputFile.addEventListener("change", async () => {
                if (inputFile.files?.length != 1) {
                    return;
                }

                const file = inputFile.files[0];
                const content = await file.text();
                const mat = parseMatrix(content);
                if (mat instanceof ParseMatrixError) {
                    window.alert(`Invalid input matrix: ${mat.message}`);
                } else {
                    arg.value = mat;
                    this.renderArgument(argIndex);
                }
            });
        }

        if (arg.value) {
            console.log(arg.value);
            arg.rootElem.appendChild(renderMatrix(arg.value, this.renderMode));
        }
    }
}

function renderIR(ir: string): HTMLElement {
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = ir;
    code.classList.add('language-mlir');
    highlightMLIR(code);
    pre.appendChild(code);
    return pre;
}

function buildDiagnosticsNote(diagnostics: GraphAlgDiagnostic[]): HTMLQuoteElement {
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

function buildErrorNote(message: string): HTMLQuoteElement {
    const quote = document.createElement("blockquote");
    quote.setAttribute('class', 'error-title');
    const title = document.createElement("p");
    title.textContent = "Compiler error";
    quote.appendChild(title);

    const pelem = document.createElement("p");
    pelem.textContent = message;
    quote.appendChild(pelem);

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
