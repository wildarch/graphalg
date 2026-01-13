import { DataSet, Network } from "vis-network/standalone"
import { GraphAlgMatrix } from "./GraphAlgMatrix"
import katex from "katex"

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

export function renderVectorAsNodeProperty(
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

export enum MatrixRenderMode {
    AUTO,
    LATEX,
    VIS_GRAPH,
    TABLE,
    VERTEX_PROPERTY,
}

export function renderMatrix(m: GraphAlgMatrix, mode: MatrixRenderMode) {
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
