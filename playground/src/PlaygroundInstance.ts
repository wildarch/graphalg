import { GraphAlgDiagnostic } from "./GraphAlgDiagnostic"
import { GraphAlgMatrix, GraphAlgMatrixEntry } from "./GraphAlgMatrix"

export interface RunResults {
    result?: GraphAlgMatrix;
    diagnostics: GraphAlgDiagnostic[];
    coreIR?: string;
}

export class PlaygroundInstance {
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
        const ga_print_module = this.bindings.ga_print_module;
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

        const coreIR = UTF8ToString(ga_print_module(pg));

        for (let arg of args) {
            ga_add_arg(pg, arg.rows, arg.cols);
        }

        if (!ga_set_dims(pg, func)) {
            return {
                diagnostics: this.getDiagnosticsAndFree(pg),
                coreIR: coreIR,
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
                coreIR: coreIR,
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
            coreIR: coreIR,
        };
    }
}
