import { GraphAlgMatrix, GraphAlgMatrixEntry } from "./GraphAlgMatrix";

export class ParseMatrixError extends Error {
    cause?: Error;

    constructor(message: string) {
        super(message);
        this.name = "ParseMatrixError";
    }
}

const VALID_RINGS = new Set([
    "i1",
    "i64",
    "f64",
    "!graphalg.trop_i64",
    "!graphalg.trop_f64",
    "!graphalg.trop_max_i64",
]);

export function parseMatrix(input: string): GraphAlgMatrix | ParseMatrixError {
    const lines = input.split(';');
    if (lines.length == 0) {
        return new ParseMatrixError("Empty input");
    }

    const header = lines[0].split(',');
    if (header.length != 3) {
        return new ParseMatrixError(`Header should have 3 values separated by commas, got ${header.length}`);
    }

    const rows = parseInt(header[0]);
    if (isNaN(rows)) {
        return new ParseMatrixError(`Invalid number of rows '${header[0]}' in header`);
    }

    const cols = parseInt(header[1]);
    if (isNaN(cols)) {
        return new ParseMatrixError(`Invalid number of columns '${header[1]}' in header`);
    }

    const ring = header[2].trim();
    if (!VALID_RINGS.has(ring)) {
        return new ParseMatrixError(`Invalid semiring '${header[2]}' in header`);
    }

    let values: GraphAlgMatrixEntry[] = [];
    for (let line of lines.slice(1)) {
        if (!line.trim()) {
            // Skip empty lines
            continue
        }

        const parts = line.split(',');
        if (ring == 'i1' && parts.length != 2) {
            return new ParseMatrixError(`Expected two values (row, col) per entry, got ${parts.length} in '${line}'`);
        } else if (ring != 'i1' && parts.length != 3) {
            return new ParseMatrixError(`Expected three values (row, col, val) per entry, got ${parts.length} in '${line}'`);
        }

        let val: boolean | bigint | number;
        switch (ring) {
            case 'i1':
                val = true;
                break;
            case 'i64':
            case '!graphalg.trop_i64':
            case '!graphalg.trop_max_i64':
                try {
                    val = BigInt(parts[2]);
                } catch (err) {
                    const parseErr = new ParseMatrixError(`Invalid value for ring ${ring} '${parts[2]}'`);
                    if (err instanceof Error) {
                        parseErr.cause = err;
                    }

                    return parseErr;
                }
                break;
            case 'f64':
            case '!graphalg.trop_f64':
                val = parseFloat(parts[2]);
                if (isNaN(val)) {
                    return new ParseMatrixError(`Invalid floatig-point value '${parts[2]}'`);
                }
                break;
            default:
                return new ParseMatrixError(`invalid ring ${ring}`);
        }

        const row = parseInt(parts[0]);
        if (isNaN(row)) {
            return new ParseMatrixError(`Invalid row index '${parts[0]}'`);
        } else if (row >= rows) {
            return new ParseMatrixError(`Row index ${row} exceeds matrix dimensions ${rows} x ${cols}`);
        }

        const col = parseInt(parts[1]);
        if (isNaN(col)) {
            return new ParseMatrixError(`Invalid column index '${parts[1]}'`);
        } else if (col >= cols) {
            return new ParseMatrixError(`Column index ${col} exceeds matrix dimensions ${rows} x ${cols}`);
        }

        values.push({
            row,
            col,
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
