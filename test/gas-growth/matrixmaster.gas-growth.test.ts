// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromInt(n: bigint): Promise<string>;
    qFromUInt(n: bigint): Promise<string>;
    qFromFrac(num: bigint, den: bigint): Promise<string>;
    toFloat(x: string): Promise<bigint>;

    mulMatrixHarness(
        aRows: bigint,
        aCols: bigint,
        aData: string[],
        bRows: bigint,
        bCols: bigint,
        bData: string[]
    ): Promise<[bigint, bigint, string[]]>;

    mulSparseMatrixVectorHarness(
        aRows: bigint,
        aCols: bigint,
        rowPtr: bigint[],
        colInd: bigint[],
        values: string[],
        xRows: bigint,
        xData: string[]
    ): Promise<[bigint, bigint, string[]]>;

    powerIterationHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[],
        seed: string,
        tol: string
    ): Promise<[string, bigint, bigint, string[]]>;
};

// ------------------------------------------------------------
// Constants & Helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function toGasBigInt(gas: unknown): bigint {
    if (typeof gas === "bigint") return gas;
    return BigInt(gas as string);
}

function idx(cols: number, i: number, j: number): number {
    return i * cols + j;
}

function flatten(mat: bigint[][]): bigint[] {
    return mat.flat();
}

async function qArrayFromBigints(h: MatrixMasterHarness, arr: bigint[]): Promise<string[]> {
    return Promise.all(arr.map(v => h.qFromInt(v)));
}

async function fromQuadArray(h: MatrixMasterHarness, arr: string[]): Promise<bigint[]> {
    return Promise.all(arr.map(v => h.toFloat(v)));
}

function headTail(arr: (string | bigint)[], limit = 3): string {
    if (arr.length <= limit * 2) return `[${arr.join(", ")}]`;
    return `[${arr.slice(0, limit).join(", ")}, ..., ${arr.slice(-limit).join(", ")}]`;
}

function makeDenseMatrix(n: number, pattern: "dense" | "identity" | "diagonal" | "banded" | "mixed"): bigint[] {
    const A = new Array<bigint>(n * n).fill(0n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (pattern === "identity") {
                A[idx(n, i, j)] = i === j ? 1n : 0n;
            } else if (pattern === "diagonal") {
                A[idx(n, i, j)] = i === j ? BigInt(i + 2) : 0n;
            } else if (pattern === "banded") {
                const d = Math.abs(i - j);
                A[idx(n, i, j)] = d <= 1 ? BigInt((i + 1) + (j + 1)) : 0n;
            } else if (pattern === "mixed") {
                const sign = (i + j) % 2 === 0 ? 1n : -1n;
                A[idx(n, i, j)] = sign * BigInt(i + j + 1);
            } else {
                A[idx(n, i, j)] = BigInt((i + 1) * (j + 2));
            }
        }
    }

    return A;
}

function makeDenseVector(n: number, pattern: "ones" | "increasing" | "alternating" | "mixed"): bigint[] {
    const x: bigint[] = [];
    for (let i = 0; i < n; i++) {
        if (pattern === "ones") x.push(1n);
        else if (pattern === "alternating") x.push(i % 2 === 0 ? 1n : -1n);
        else if (pattern === "mixed") x.push(i % 3 === 0 ? -BigInt(i + 1) : BigInt(i + 1));
        else x.push(BigInt(i + 1));
    }
    return x;
}

function denseMatMul(A: bigint[], B: bigint[], aRows: number, aCols: number, bCols: number): bigint[] {
    const C = new Array<bigint>(aRows * bCols).fill(0n);
    for (let i = 0; i < aRows; i++) {
        for (let j = 0; j < bCols; j++) {
            let s = 0n;
            for (let k = 0; k < aCols; k++) {
                s += A[idx(aCols, i, k)] * B[idx(bCols, k, j)];
            }
            C[idx(bCols, i, j)] = s;
        }
    }
    return C;
}

function buildSparseDiagonal(n: number, diagValue: bigint): {
    rowPtr: bigint[];
    colInd: bigint[];
    values: bigint[];
    vector: bigint[];
} {
    const rowPtr: bigint[] = [];
    const colInd: bigint[] = [];
    const values: bigint[] = [];
    const vector: bigint[] = [];

    for (let i = 0; i <= n; i++) rowPtr.push(BigInt(i));
    for (let i = 0; i < n; i++) {
        colInd.push(BigInt(i));
        values.push(diagValue);
        vector.push(BigInt(i + 1));
    }

    return { rowPtr, colInd, values, vector };
}

function buildSparsePattern(
    n: number,
    pattern: "verySparse" | "sparse" | "medium" | "denseLike" | "diagonal"
): {
    rowPtr: bigint[];
    colInd: bigint[];
    values: bigint[];
    vector: bigint[];
} {
    const rowPtr: bigint[] = [0n];
    const colInd: bigint[] = [];
    const values: bigint[] = [];
    const vector = makeDenseVector(n, "increasing");

    for (let i = 0; i < n; i++) {
        let countBefore = colInd.length;

        for (let j = 0; j < n; j++) {
            let include = false;

            if (pattern === "diagonal") include = i === j;
            else if (pattern === "verySparse") include = i === j || (j === 0 && i % 4 === 0);
            else if (pattern === "sparse") include = i === j || Math.abs(i - j) === 1;
            else if (pattern === "medium") include = Math.abs(i - j) <= 2;
            else include = Math.abs(i - j) <= 3 || (i + j) % 5 === 0;

            if (include) {
                colInd.push(BigInt(j));
                values.push(BigInt(i + j + 1));
            }
        }

        rowPtr.push(BigInt(colInd.length));
        if (colInd.length === countBefore) {
            // keep structural sanity; not expected here
            colInd.push(BigInt(i));
            values.push(1n);
            rowPtr[rowPtr.length - 1] = BigInt(colInd.length);
        }
    }

    return { rowPtr, colInd, values, vector };
}

function makeSymmetricMatrix(n: number, pattern: "diagDominant" | "identity" | "clustered" | "weighted" | "mixed"): bigint[] {
    const A = new Array<bigint>(n * n).fill(0n);

    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let v: bigint;
            if (pattern === "identity") v = 0n;
            else if (pattern === "clustered") v = BigInt((i + j) % 2 === 0 ? 1 : 2);
            else if (pattern === "weighted") v = BigInt((i % 3) + 1);
            else if (pattern === "mixed") v = BigInt(((i + j) % 3) + 1);
            else v = 1n;

            A[idx(n, i, j)] = v;
            A[idx(n, j, i)] = v;
        }
    }

    for (let i = 0; i < n; i++) {
        let rowAbsSum = 0n;
        for (let j = 0; j < n; j++) {
            if (i !== j) rowAbsSum += A[idx(n, i, j)] < 0n ? -A[idx(n, i, j)] : A[idx(n, i, j)];
        }

        if (pattern === "identity") A[idx(n, i, i)] = 1n;
        else if (pattern === "clustered") A[idx(n, i, i)] = rowAbsSum + 5n + BigInt(i);
        else if (pattern === "weighted") A[idx(n, i, i)] = rowAbsSum + 10n + BigInt(2 * i);
        else if (pattern === "mixed") A[idx(n, i, i)] = rowAbsSum + 7n + BigInt(i % 4);
        else A[idx(n, i, i)] = rowAbsSum + 3n;
    }

    return A;
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("MatrixMasterHarness - Gas Growth Tests", function () {
    let harness: MatrixMasterHarness;
    let t = 0;
    let tolPower: string;

    before(async function () {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": await mathlib.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();

        tolPower = await harness.qFromFrac(1n, 1_000_000n); // 1e-6
    });

    // ------------------------------------------------------------
    // Section 1: Dense Matrix-Matrix Multiplication
    // ------------------------------------------------------------

    describe("Section 1: Gas Growth for Dense Matrix-Matrix Multiplication", function () {
        const DIM_CASES = [2, 4, 8, 16];

        for (const n of DIM_CASES) {
            it(`Test ${++t}: Dense matrix-matrix multiplication gas growth with dimension n=${n}`, async function () {
                const A = makeDenseMatrix(n, "dense");
                const B = makeDenseMatrix(n, "banded");
                const Aq = await qArrayFromBigints(harness, A);
                const Bq = await qArrayFromBigints(harness, B);

                await touchGas(harness, "mulMatrixHarness", [BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq]);
                const gas = await estimateGas(harness, "mulMatrixHarness", [BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq]);

                const [, , out] = await harness.mulMatrixHarness(BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq);
                const outDec = await fromQuadArray(harness, out);
                const expected = denseMatMul(A, B, n, n, n);

                printBlockRegular({
                    t,
                    method: "matrix-matrix multiplication",
                    explanation: `Gas growth with respect to matrix dimension for dense ${n}x${n} × ${n}x${n} multiplication.`,
                    gas,
                    inHex: `A_shape=${n}x${n}, B_shape=${n}x${n}`,
                    expectedHex: "N/A",
                    outHex: headTail(out),
                    expectedDec: headTail(expected.map(formatScaledInt)),
                    outDec: headTail(outDec.map(formatScaledInt)),
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }

        const PATTERN_CASES: Array<{ patternA: "identity" | "diagonal" | "banded" | "dense" | "mixed"; patternB: "dense" | "banded"; label: string }> = [
            { patternA: "identity", patternB: "dense", label: "identity-left" },
            { patternA: "diagonal", patternB: "dense", label: "diagonal-left" },
            { patternA: "banded", patternB: "dense", label: "banded-left" },
            { patternA: "dense", patternB: "banded", label: "dense-times-banded" },
            { patternA: "mixed", patternB: "dense", label: "mixed-sign-left" },
        ];

        for (const c of PATTERN_CASES) {
            it(`Test ${++t}: Dense matrix-matrix multiplication gas sensitivity for ${c.label}`, async function () {
                const n = 8;
                const A = makeDenseMatrix(n, c.patternA);
                const B = makeDenseMatrix(n, c.patternB);
                const Aq = await qArrayFromBigints(harness, A);
                const Bq = await qArrayFromBigints(harness, B);

                await touchGas(harness, "mulMatrixHarness", [8n, 8n, Aq, 8n, 8n, Bq]);
                const gas = await estimateGas(harness, "mulMatrixHarness", [8n, 8n, Aq, 8n, 8n, Bq]);

                const [, , out] = await harness.mulMatrixHarness(8n, 8n, Aq, 8n, 8n, Bq);
                const outDec = await fromQuadArray(harness, out);
                const expected = denseMatMul(A, B, n, n, n);

                printBlockRegular({
                    t,
                    method: "matrix-matrix multiplication",
                    explanation: `Gas sensitivity to matrix structure using ${c.label} pattern with fixed dimension n=8.`,
                    gas,
                    inHex: `pattern=${c.label}, shape=8x8 × 8x8`,
                    expectedHex: "N/A",
                    outHex: headTail(out),
                    expectedDec: headTail(expected.map(formatScaledInt)),
                    outDec: headTail(outDec.map(formatScaledInt)),
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 2: Sparse Matrix-Vector Multiplication
    // ------------------------------------------------------------

    describe("Section 2: Gas Growth for Sparse Matrix-Vector Multiplication", function () {
        const DIM_CASES = [8, 16, 32, 64];

        for (const n of DIM_CASES) {
            it(`Test ${++t}: Sparse matrix-vector multiplication gas growth with dimension n=${n}`, async function () {
                const s = buildSparseDiagonal(n, 2n);
                const vq = await qArrayFromBigints(harness, s.values);
                const xq = await qArrayFromBigints(harness, s.vector);

                await touchGas(harness, "mulSparseMatrixVectorHarness", [
                    BigInt(n), BigInt(n), s.rowPtr, s.colInd, vq, BigInt(n), xq
                ]);
                const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [
                    BigInt(n), BigInt(n), s.rowPtr, s.colInd, vq, BigInt(n), xq
                ]);

                const [, , out] = await harness.mulSparseMatrixVectorHarness(
                    BigInt(n), BigInt(n), s.rowPtr, s.colInd, vq, BigInt(n), xq
                );
                const outDec = await fromQuadArray(harness, out);

                printBlockRegular({
                    t,
                    method: "sparse matrix-vector multiplication",
                    explanation: `Gas growth with respect to dimension for diagonal sparse ${n}x${n} matrix times dense vector.`,
                    gas,
                    inHex: `shape=${n}x${n}, nnz=${n}`,
                    expectedHex: "N/A",
                    outHex: headTail(out),
                    expectedDec: headTail(s.vector.map(v => formatScaledInt(2n * v * SCALE))),
                    outDec: headTail(outDec.map(formatScaledInt)),
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }

        const SPARSITY_CASES: Array<{ pattern: "verySparse" | "sparse" | "medium" | "denseLike" | "diagonal"; label: string }> = [
            { pattern: "verySparse", label: "verySparse" },
            { pattern: "sparse", label: "sparse" },
            { pattern: "medium", label: "medium" },
            { pattern: "denseLike", label: "denseLike" },
            { pattern: "diagonal", label: "diagonal" },
        ];

        for (const c of SPARSITY_CASES) {
            it(`Test ${++t}: Sparse matrix-vector multiplication gas sensitivity for ${c.label}`, async function () {
                const n = 32;
                const s = buildSparsePattern(n, c.pattern);
                const vq = await qArrayFromBigints(harness, s.values);
                const xq = await qArrayFromBigints(harness, s.vector);

                await touchGas(harness, "mulSparseMatrixVectorHarness", [
                    32n, 32n, s.rowPtr, s.colInd, vq, 32n, xq
                ]);
                const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [
                    32n, 32n, s.rowPtr, s.colInd, vq, 32n, xq
                ]);

                const [, , out] = await harness.mulSparseMatrixVectorHarness(
                    32n, 32n, s.rowPtr, s.colInd, vq, 32n, xq
                );
                const outDec = await fromQuadArray(harness, out);

                printBlockRegular({
                    t,
                    method: "sparse matrix-vector multiplication",
                    explanation: `Gas sensitivity to sparsity pattern using fixed dimension n=32 with ${c.label} sparse matrix structure.`,
                    gas,
                    inHex: `pattern=${c.label}, shape=32x32, nnz=${s.values.length}`,
                    expectedHex: "N/A",
                    outHex: headTail(out),
                    expectedDec: "N/A",
                    outDec: headTail(outDec.map(formatScaledInt)),
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 3: Power Iteration
    // ------------------------------------------------------------

    describe("Section 3: Gas Growth for Power Iteration", function () {
        const DIM_CASES = [2, 4, 8, 16];
        const seed = ethers.id("matrix-power-seed");

        for (const n of DIM_CASES) {
            it(`Test ${++t}: Power iteration gas growth with dimension n=${n}`, async function () {
                const A = makeSymmetricMatrix(n, "diagDominant");
                const Aq = await qArrayFromBigints(harness, A);

                await touchGas(harness, "powerIterationHarness", [BigInt(n), BigInt(n), Aq, seed, tolPower]);
                const gas = await estimateGas(harness, "powerIterationHarness", [BigInt(n), BigInt(n), Aq, seed, tolPower]);

                const [lambda, , , x] = await harness.powerIterationHarness(BigInt(n), BigInt(n), Aq, seed, tolPower);
                const lambdaDec = await harness.toFloat(lambda);
                const xDec = await fromQuadArray(harness, x);

                printBlockRegular({
                    t,
                    method: "power iteration",
                    explanation: `Gas growth with respect to matrix dimension for dominant eigenvalue approximation on symmetric ${n}x${n} matrix.`,
                    gas,
                    inHex: `shape=${n}x${n}, seed=fixed`,
                    expectedHex: "N/A",
                    outHex: `lambda=${lambda}, eigenvector=${headTail(x)}`,
                    expectedDec: "N/A",
                    outDec: `lambda=${formatScaledInt(lambdaDec)}, eigenvector=${headTail(xDec.map(formatScaledInt))}`,
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }

        const STRUCTURE_CASES: Array<{ pattern: "identity" | "diagDominant" | "clustered" | "weighted" | "mixed"; label: string }> = [
            { pattern: "identity", label: "identity" },
            { pattern: "diagDominant", label: "diagDominant" },
            { pattern: "clustered", label: "clustered" },
            { pattern: "weighted", label: "weighted" },
            { pattern: "mixed", label: "mixed" },
        ];

        for (const c of STRUCTURE_CASES) {
            it(`Test ${++t}: Power iteration gas sensitivity for ${c.label} matrix structure`, async function () {
                const n = 8;
                const A = makeSymmetricMatrix(n, c.pattern);
                const Aq = await qArrayFromBigints(harness, A);

                await touchGas(harness, "powerIterationHarness", [8n, 8n, Aq, seed, tolPower]);
                const gas = await estimateGas(harness, "powerIterationHarness", [8n, 8n, Aq, seed, tolPower]);

                const [lambda, , , x] = await harness.powerIterationHarness(8n, 8n, Aq, seed, tolPower);
                const lambdaDec = await harness.toFloat(lambda);
                const xDec = await fromQuadArray(harness, x);

                printBlockRegular({
                    t,
                    method: "power iteration",
                    explanation: `Gas sensitivity to matrix eigen-structure using fixed dimension n=8 and ${c.label} symmetric matrix.`,
                    gas,
                    inHex: `pattern=${c.label}, shape=8x8`,
                    expectedHex: "N/A",
                    outHex: `lambda=${lambda}, eigenvector=${headTail(x)}`,
                    expectedDec: "N/A",
                    outDec: `lambda=${formatScaledInt(lambdaDec)}, eigenvector=${headTail(xDec.map(formatScaledInt))}`,
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }

        const TOL_CASES: Array<{ label: string; num: bigint; den: bigint }> = [
            { label: "loose", num: 1n, den: 1_000n },          // 1e-3
            { label: "medium", num: 1n, den: 1_000_000n },     // 1e-6
            { label: "tight", num: 1n, den: 1_000_000_000n },  // 1e-9
        ];

        for (const c of TOL_CASES) {
            it(`Test ${++t}: Power iteration gas sensitivity for tolerance ${c.label}`, async function () {
                const n = 8;
                const A = makeSymmetricMatrix(n, "weighted");
                const Aq = await qArrayFromBigints(harness, A);
                const tol = await harness.qFromFrac(c.num, c.den);

                await touchGas(harness, "powerIterationHarness", [8n, 8n, Aq, seed, tol]);
                const gas = await estimateGas(harness, "powerIterationHarness", [8n, 8n, Aq, seed, tol]);

                const [lambda, , , x] = await harness.powerIterationHarness(8n, 8n, Aq, seed, tol);
                const lambdaDec = await harness.toFloat(lambda);
                const xDec = await fromQuadArray(harness, x);

                printBlockRegular({
                    t,
                    method: "power iteration",
                    explanation: `Gas sensitivity to convergence tolerance using fixed weighted symmetric 8x8 matrix and ${c.label} tolerance.`,
                    gas,
                    inHex: `shape=8x8, tol=${c.label}`,
                    expectedHex: "N/A",
                    outHex: `lambda=${lambda}, eigenvector=${headTail(x)}`,
                    expectedDec: "N/A",
                    outDec: `lambda=${formatScaledInt(lambdaDec)}, eigenvector=${headTail(xDec.map(formatScaledInt))}`,
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }
    });
});