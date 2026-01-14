export interface GraphAlgMatrixEntry {
    row: number;
    col: number;
    val: boolean | bigint | number;
}

export interface GraphAlgMatrix {
    ring: string;
    rows: number;
    cols: number;
    values: GraphAlgMatrixEntry[],
}
