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

    transposeHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;
    detHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<string>;
    inverseHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;
    mulSparseMatrixVectorHarness(aRows: bigint, aCols: bigint, rowPtr: bigint[], colInd: bigint[], values: string[], xRows: bigint, xData: string[]): Promise<[bigint, bigint, string[]]>;

    powerIterationHarness(rows: bigint, cols: bigint, dataFlat: string[], seed: string, tol: string): Promise<[string, bigint, bigint, string[]]>;
    powerIterationWithIterHarness(rows: bigint, cols: bigint, dataFlat: string[], seed: string, tol: string, maxIter: bigint): Promise<[string, bigint, bigint, string[], bigint]>;
};

// ------------------------------------------------------------
// Constants & Helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;
const Q_SCALE = 1_000_000n;

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

function idx(n: number, i: number, j: number): number {
    return i * n + j;
}

function flatten(mat: number[][]): number[] {
    return mat.flat();
}

function headTail(arr: (string | bigint)[], limit = 3): string {
    if (arr.length <= limit * 2) return `[${arr.join(", ")}]`;
    return `[${arr.slice(0, limit).join(", ")}, ..., ${arr.slice(-limit).join(", ")}]`;
}

async function qFromNumber(h: MatrixMasterHarness, x: number): Promise<string> {
    if (Number.isInteger(x)) {
        return h.qFromInt(BigInt(x));
    }
    const scaled = Math.round(x * Number(Q_SCALE));
    return h.qFromFrac(BigInt(scaled), Q_SCALE);
}

async function qArrayFromNumbers(h: MatrixMasterHarness, arr: number[]): Promise<string[]> {
    return Promise.all(arr.map(v => qFromNumber(h, v)));
}

async function fromQuad(h: MatrixMasterHarness, q: string): Promise<bigint> {
    return h.toFloat(q);
}

async function fromQuadArray(h: MatrixMasterHarness, arr: string[]): Promise<bigint[]> {
    return Promise.all(arr.map(v => h.toFloat(v)));
}

function absNumber(x: number): number {
    return x < 0 ? -x : x;
}

function makeDenseMatrix(
    n: number,
    pattern: "dense" | "identity" | "diagonal" | "banded" | "mixed"
): bigint[] {
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

async function qArrayFromBigints(
    h: MatrixMasterHarness,
    arr: bigint[]
): Promise<string[]> {
    return Promise.all(arr.map(v => h.qFromInt(v)));
}

// ------------------------------------------------------------
// Matrix Builders
// ------------------------------------------------------------

function makeIdentity(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) A[i][i] = 1;
    return A;
}

function makeDiagonal(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) A[i][i] = i + 2;
    return A;
}

function makeTriangular(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            A[i][j] = i + j + 1;
        }
    }
    return A;
}

function makeDiagonallyDominant(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const v = ((i + j) % 2) + 1;
            A[i][j] = v;
            rowSum += absNumber(v);
        }
        A[i][i] = rowSum + 5 + i;
    }
    return A;
}

function makePermutation(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        A[i][j] = 1;
    }
    return A;
}

function makeDenseRandom(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const raw = ((i * 17 + j * 13 + 7) % 7) - 3; // [-3, 3]
            const v = raw === 0 ? 1 : raw;
            A[i][j] = v;
            rowSum += absNumber(v);
        }
        A[i][i] = rowSum + 10;
    }
    return A;
}

function makeSparseRandomDense(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const include = ((i * 11 + j * 7) % 9) === 0;
            const v = include ? ((i + j) % 3) + 1 : 0;
            A[i][j] = v;
            rowSum += absNumber(v);
        }
        A[i][i] = rowSum + 3;
    }
    return A;
}

function makeSingleNonzeroSparseDense(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    A[0][0] = 7;
    return A;
}

function makeWeighted(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const v = ((i % 3) + 1) * ((j % 2) + 1);
            A[i][j] = v;
            rowSum += absNumber(v);
        }
        A[i][i] = rowSum + 8 + 2 * i;
    }
    return A;
}

function makeSingular(n: number): number[][] {
    const A = makeDiagonallyDominant(n);
    for (let j = 0; j < n; j++) {
        A[n - 1][j] = A[0][j];
    }
    return A;
}

function makeNearSingular(n: number): number[][] {
    const A = makeIdentity(n);
    for (let j = 0; j < n; j++) {
        A[n - 1][j] = A[0][j];
    }
    A[n - 1][n - 1] = 1.0001; // almost duplicate row -> near singular
    return A;
}

function makeWellConditioned(n: number): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        A[i][i] = 2 + i;
    }
    return A;
}

function makePivotingRequired(n: number): number[][] {
    const A = makeIdentity(n);
    A[0][0] = 0;
    A[0][1] = 2;
    A[1][0] = 1;
    A[1][1] = 3;
    for (let i = 2; i < n; i++) {
        A[i][i] = i + 2;
    }
    return A;
}

function buildClassMatrix(
    n: number,
    cls: "well-conditioned" | "singular" | "near-singular" | "pivoting-required" | "diagonally-dominant"
): number[][] {
    if (cls === "well-conditioned") return makeWellConditioned(n);
    if (cls === "singular") return makeSingular(n);
    if (cls === "near-singular") return makeNearSingular(n);
    if (cls === "pivoting-required") return makePivotingRequired(n);
    return makeDiagonallyDominant(n);
}

function buildPatternMatrix(
    n: number,
    pattern:
        | "identity"
        | "diagonal"
        | "triangular"
        | "dense_random"
        | "sparse_random"
        | "single_nonzero_sparse"
        | "weighted"
        | "singular"
        | "near_singular"
        | "well-conditioned"
        | "pivoting-required"
        | "diagonally_dominant"
): number[][] {
    if (pattern === "identity") return makeIdentity(n);
    if (pattern === "diagonal") return makeDiagonal(n);
    if (pattern === "triangular") return makeTriangular(n);
    if (pattern === "dense_random") return makeDenseRandom(n);
    if (pattern === "sparse_random") return makeSparseRandomDense(n);
    if (pattern === "single_nonzero_sparse") return makeSingleNonzeroSparseDense(n);
    if (pattern === "weighted") return makeWeighted(n);
    if (pattern === "singular") return makeSingular(n);
    if (pattern === "near_singular") return makeNearSingular(n);
    if (pattern === "well-conditioned") return makeWellConditioned(n);
    if (pattern === "pivoting-required") return makePivotingRequired(n);
    return makeDiagonallyDominant(n);
}

// ------------------------------------------------------------
// Sparse Builders for nnz growth
// ------------------------------------------------------------

function makeDenseVector(n: number): number[] {
    return Array.from({ length: n }, (_, i) => i + 1);
}

function buildSparseBandCSR(n: number, bandwidth: number): {
    rowPtr: bigint[];
    colInd: bigint[];
    values: number[];
    vector: number[];
} {
    const rowPtr: bigint[] = [0n];
    const colInd: bigint[] = [];
    const values: number[] = [];
    const vector = makeDenseVector(n);

    for (let i = 0; i < n; i++) {
        for (let j = Math.max(0, i - bandwidth); j <= Math.min(n - 1, i + bandwidth); j++) {
            colInd.push(BigInt(j));
            values.push(i === j ? i + 2 : 1);
        }
        rowPtr.push(BigInt(colInd.length));
    }

    return { rowPtr, colInd, values, vector };
}

// ------------------------------------------------------------
// Expected dominant eigenvalue for diag-dominant symmetric family
// ------------------------------------------------------------

function makePowerMatrix(n: number): number[][] {
    // Same family used before: off-diagonal 1, diagonal = rowAbsSum + 3
    // Dominant eigenvalue becomes 2n + 1 for this construction.
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            A[i][j] = 1;
            rowSum += 1;
        }
        A[i][i] = rowSum + 3;
    }
    return A;
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("MatrixMasterHarness - Gas Growth Tests (Part 2)", function () {
    let harness: MatrixMasterHarness;
    let t = 0;
    let powerSeed: string;

    before(async function () {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": await mathlib.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();

        powerSeed = ethers.keccak256(ethers.toUtf8Bytes("matrixmaster2-fixed-seed"));
    });

    // ------------------------------------------------------------
    // Section 0: Matrix Creation
    // ------------------------------------------------------------

    describe("Section 0: Matrix Creation", function () {
        const CREATION_CASES = [
            { sub: "0.1", kind: "zeros", label: "zeros", n: 8 },
            { sub: "0.2", kind: "ones", label: "ones", n: 8 },
            { sub: "0.3", kind: "identity", label: "identity", n: 8 },
            { sub: "0.4", kind: "diagonal", label: "diagonal", n: 8 },
            { sub: "0.5", kind: "random", label: "random", n: 8 },
        ] as const;

        for (const c of CREATION_CASES) {
            it(`${c.sub} Matrix creation gas sensitivity for ${c.label}`, async function () {
                t++;

                if (c.kind === "zeros") {
                    await touchGas(harness, "zerosHarness", [BigInt(c.n), BigInt(c.n)]);
                    const gas = await estimateGas(harness, "zerosHarness", [BigInt(c.n), BigInt(c.n)]);
                    const [, , out] = await harness.zerosHarness(BigInt(c.n), BigInt(c.n));
                    const outDec = await fromQuadArray(harness, out);

                    printBlockRegular({
                        t,
                        method: "matrix creation",
                        explanation: `Gas sensitivity of dense matrix creation using zeros pattern at fixed dimension n=${c.n}.`,
                        gas,
                        inHex: `pattern=zeros, shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: headTail(out),
                        expectedDec: "all zeros",
                        outDec: headTail(outDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }

                else if (c.kind === "ones") {
                    await touchGas(harness, "onesHarness", [BigInt(c.n), BigInt(c.n)]);
                    const gas = await estimateGas(harness, "onesHarness", [BigInt(c.n), BigInt(c.n)]);
                    const [, , out] = await harness.onesHarness(BigInt(c.n), BigInt(c.n));
                    const outDec = await fromQuadArray(harness, out);

                    printBlockRegular({
                        t,
                        method: "matrix creation",
                        explanation: `Gas sensitivity of dense matrix creation using ones pattern at fixed dimension n=${c.n}.`,
                        gas,
                        inHex: `pattern=ones, shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: headTail(out),
                        expectedDec: "all ones",
                        outDec: headTail(outDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }

                else if (c.kind === "identity") {
                    await touchGas(harness, "createIdentityMatrixHarness", [BigInt(c.n)]);
                    const gas = await estimateGas(harness, "createIdentityMatrixHarness", [BigInt(c.n)]);
                    const [, , out] = await harness.createIdentityMatrixHarness(BigInt(c.n));
                    const outDec = await fromQuadArray(harness, out);

                    printBlockRegular({
                        t,
                        method: "matrix creation",
                        explanation: `Gas sensitivity of dense matrix creation using identity pattern at fixed dimension n=${c.n}.`,
                        gas,
                        inHex: `pattern=identity, shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: headTail(out),
                        expectedDec: "identity matrix",
                        outDec: headTail(outDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }

                else if (c.kind === "diagonal") {
                    const diag = Array.from({ length: c.n }, (_, i) => i + 2);
                    const diagQ = await qArrayFromNumbers(harness, diag);

                    await touchGas(harness, "fromDiagonalHarness", [diagQ]);
                    const gas = await estimateGas(harness, "fromDiagonalHarness", [diagQ]);
                    const [, , out] = await harness.fromDiagonalHarness(diagQ);
                    const outDec = await fromQuadArray(harness, out);

                    printBlockRegular({
                        t,
                        method: "matrix creation",
                        explanation: `Gas sensitivity of dense matrix creation from diagonal vector at fixed dimension n=${c.n}.`,
                        gas,
                        inHex: `pattern=diagonal, shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: headTail(out),
                        expectedDec: `diag=${headTail(diag.map(v => v.toString()))}`,
                        outDec: headTail(outDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }

                else if (c.kind === "random") {
                    const seed = ethers.keccak256(ethers.toUtf8Bytes(`matrix-creation-${c.n}`));

                    await touchGas(harness, "randomMatrixHarness", [BigInt(c.n), BigInt(c.n), seed]);
                    const gas = await estimateGas(harness, "randomMatrixHarness", [BigInt(c.n), BigInt(c.n), seed]);
                    const [, , out] = await harness.randomMatrixHarness(BigInt(c.n), BigInt(c.n), seed);
                    const outDec = await fromQuadArray(harness, out);

                    printBlockRegular({
                        t,
                        method: "matrix creation",
                        explanation: `Gas sensitivity of dense matrix creation using pseudo-random pattern at fixed dimension n=${c.n}.`,
                        gas,
                        inHex: `pattern=random, shape=${c.n}x${c.n}, seed=fixed`,
                        expectedHex: "N/A",
                        outHex: headTail(out),
                        expectedDec: "pseudo-random values",
                        outDec: headTail(outDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }
            });
        }
    });

    // ------------------------------------------------------------
    // Section 1: Basic Matrix Ops
    // ------------------------------------------------------------

    describe("Section 1: Basic Matrix Ops", function () {
        describe("Section 0.1: Matrix Creation", function () {
            const CREATION_CASES = [
                { sub: "1.1", kind: "zeros", label: "zeros", n: 8 },
                { sub: "1.2", kind: "ones", label: "ones", n: 8 },
                { sub: "1.3", kind: "identity", label: "identity", n: 8 },
                { sub: "1.4", kind: "diagonal", label: "diagonal", n: 8 },
                { sub: "1.5", kind: "random", label: "random", n: 8 },
            ] as const;

            for (const c of CREATION_CASES) {
                it(`${c.sub} Matrix creation gas sensitivity for ${c.label}`, async function () {
                    t++;

                    if (c.kind === "zeros") {
                        await touchGas(harness, "zerosHarness", [BigInt(c.n), BigInt(c.n)]);
                        const gas = await estimateGas(harness, "zerosHarness", [BigInt(c.n), BigInt(c.n)]);
                        const [, , out] = await harness.zerosHarness(BigInt(c.n), BigInt(c.n));
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix creation",
                            explanation: `Gas sensitivity of dense matrix creation using zeros pattern at fixed dimension n=${c.n}.`,
                            gas,
                            inHex: `pattern=zeros, shape=${c.n}x${c.n}`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: "all zeros",
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    } else if (c.kind === "ones") {
                        await touchGas(harness, "onesHarness", [BigInt(c.n), BigInt(c.n)]);
                        const gas = await estimateGas(harness, "onesHarness", [BigInt(c.n), BigInt(c.n)]);
                        const [, , out] = await harness.onesHarness(BigInt(c.n), BigInt(c.n));
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix creation",
                            explanation: `Gas sensitivity of dense matrix creation using ones pattern at fixed dimension n=${c.n}.`,
                            gas,
                            inHex: `pattern=ones, shape=${c.n}x${c.n}`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: "all ones",
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    } else if (c.kind === "identity") {
                        await touchGas(harness, "createIdentityMatrixHarness", [BigInt(c.n)]);
                        const gas = await estimateGas(harness, "createIdentityMatrixHarness", [BigInt(c.n)]);
                        const [, , out] = await harness.createIdentityMatrixHarness(BigInt(c.n));
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix creation",
                            explanation: `Gas sensitivity of dense matrix creation using identity pattern at fixed dimension n=${c.n}.`,
                            gas,
                            inHex: `pattern=identity, shape=${c.n}x${c.n}`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: "identity matrix",
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    } else if (c.kind === "diagonal") {
                        const diag = Array.from({ length: c.n }, (_, i) => i + 2);
                        const diagQ = await qArrayFromNumbers(harness, diag);

                        await touchGas(harness, "fromDiagonalHarness", [diagQ]);
                        const gas = await estimateGas(harness, "fromDiagonalHarness", [diagQ]);
                        const [, , out] = await harness.fromDiagonalHarness(diagQ);
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix creation",
                            explanation: `Gas sensitivity of dense matrix creation from diagonal vector at fixed dimension n=${c.n}.`,
                            gas,
                            inHex: `pattern=diagonal, shape=${c.n}x${c.n}`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: `diag=${headTail(diag.map(v => v.toString()))}`,
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    } else if (c.kind === "random") {
                        const seed = ethers.keccak256(ethers.toUtf8Bytes(`matrix-creation-${c.n}`));

                        await touchGas(harness, "randomMatrixHarness", [BigInt(c.n), BigInt(c.n), seed]);
                        const gas = await estimateGas(harness, "randomMatrixHarness", [BigInt(c.n), BigInt(c.n), seed]);
                        const [, , out] = await harness.randomMatrixHarness(BigInt(c.n), BigInt(c.n), seed);
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix creation",
                            explanation: `Gas sensitivity of dense matrix creation using pseudo-random pattern at fixed dimension n=${c.n}.`,
                            gas,
                            inHex: `pattern=random, shape=${c.n}x${c.n}, seed=fixed`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: "pseudo-random values",
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    }
                });
            }
        });

        describe("Section 2: Matrix Add/Sub", function () {
            const ADD_SUB_CASES = [
                { sub: "2.1", patternA: "dense", patternB: "dense", label: "dense+dense", op: "add" },
                { sub: "2.2", patternA: "identity", patternB: "dense", label: "identity+dense", op: "add" },
                { sub: "2.3", patternA: "diagonal", patternB: "banded", label: "diagonal+banded", op: "add" },
                { sub: "2.4", patternA: "dense", patternB: "banded", label: "dense-banded", op: "sub" },
                { sub: "2.5", patternA: "mixed", patternB: "dense", label: "mixed-dense", op: "sub" },
            ] as const;

            for (const c of ADD_SUB_CASES) {
                it(`${c.sub} Matrix ${c.op} gas sensitivity for ${c.label}`, async function () {
                    t++;
                    const n = 8;

                    const A = makeDenseMatrix(n, c.patternA);
                    const B = makeDenseMatrix(n, c.patternB);
                    const Aq = await qArrayFromBigints(harness, A);
                    const Bq = await qArrayFromBigints(harness, B);

                    if (c.op === "add") {
                        await touchGas(harness, "addHarness", [BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq]);
                        const gas = await estimateGas(harness, "addHarness", [BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq]);

                        const [, , out] = await harness.addHarness(BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq);
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix add",
                            explanation: `Gas sensitivity of matrix addition using ${c.label} operand structure at fixed dimension n=${n}.`,
                            gas,
                            inHex: `op=add, pattern=${c.label}, shape=${n}x${n}`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: "N/A",
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    } else {
                        await touchGas(harness, "subHarness", [BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq]);
                        const gas = await estimateGas(harness, "subHarness", [BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq]);

                        const [, , out] = await harness.subHarness(BigInt(n), BigInt(n), Aq, BigInt(n), BigInt(n), Bq);
                        const outDec = await fromQuadArray(harness, out);

                        printBlockRegular({
                            t,
                            method: "matrix sub",
                            explanation: `Gas sensitivity of matrix subtraction using ${c.label} operand structure at fixed dimension n=${n}.`,
                            gas,
                            inHex: `op=sub, pattern=${c.label}, shape=${n}x${n}`,
                            expectedHex: "N/A",
                            outHex: headTail(out),
                            expectedDec: "N/A",
                            outDec: headTail(outDec.map(formatScaledInt)),
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    }
                });
            }
        });
    });

    // ------------------------------------------------------------
    // Section 3: Gas vs number of nonzero entries (nnz)
    // ------------------------------------------------------------

    describe("Section 3: Gas vs number of nonzero entries (nnz)", function () {
        const NNZ_CASES = [
            { sub: "3.1", bandwidth: 0, label: "diagonal" },
            { sub: "3.2", bandwidth: 1, label: "band-1" },
            { sub: "3.3", bandwidth: 2, label: "band-2" },
            { sub: "3.4", bandwidth: 3, label: "band-3" },
            { sub: "3.5", bandwidth: 4, label: "band-4" },
        ];

        for (const c of NNZ_CASES) {
            it(`${c.sub} Gas vs nnz using ${c.label} sparse pattern`, async function () {
                t++;
                const n = 32;
                const sparse = buildSparseBandCSR(n, c.bandwidth);
                const valuesQ = await qArrayFromNumbers(harness, sparse.values);
                const xQ = await qArrayFromNumbers(harness, sparse.vector);

                await touchGas(harness, "mulSparseMatrixVectorHarness", [
                    BigInt(n), BigInt(n), sparse.rowPtr, sparse.colInd, valuesQ, BigInt(n), xQ,
                ]);
                const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [
                    BigInt(n), BigInt(n), sparse.rowPtr, sparse.colInd, valuesQ, BigInt(n), xQ,
                ]);

                const [, , out] = await harness.mulSparseMatrixVectorHarness(
                    BigInt(n), BigInt(n), sparse.rowPtr, sparse.colInd, valuesQ, BigInt(n), xQ
                );
                const outDec = await fromQuadArray(harness, out);

                printBlockRegular({
                    t,
                    method: "sparse matrix-vector multiplication",
                    explanation: `Gas growth with respect to number of nonzero entries (nnz) using ${c.label} sparse structure at fixed dimension n=32.`,
                    gas,
                    inHex: `shape=32x32, pattern=${c.label}, nnz=${sparse.colInd.length}`,
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
    // Section 4: Matrix size vs gas (transpose / determinant / inversion)
    // ------------------------------------------------------------

    describe("Section 4: Matrix size vs gas", function () {
        const SIZE_CASES = [
            { sub: "4.1", n: 2 },
            { sub: "4.2", n: 4 },
            { sub: "4.3", n: 6 },
            { sub: "4.4", n: 8 },
            { sub: "4.5", n: 10 },
        ];

        describe("Section 4.1: Transpose", function () {
            for (const c of SIZE_CASES) {
                it(`${c.sub} Transpose gas growth for n=${c.n}`, async function () {
                    t++;
                    const A = makeDiagonallyDominant(c.n);
                    const Aq = await qArrayFromNumbers(harness, flatten(A));

                    await touchGas(harness, "transposeHarness", [BigInt(c.n), BigInt(c.n), Aq]);
                    const gas = await estimateGas(harness, "transposeHarness", [BigInt(c.n), BigInt(c.n), Aq]);

                    const [, , out] = await harness.transposeHarness(BigInt(c.n), BigInt(c.n), Aq);
                    const outDec = await fromQuadArray(harness, out);

                    printBlockRegular({
                        t,
                        method: "transpose",
                        explanation: `Gas growth of transpose operation for square matrix size n=${c.n}.`,
                        gas,
                        inHex: `shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: headTail(out),
                        expectedDec: "N/A",
                        outDec: headTail(outDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        });

        describe("Section 4.2: Determinant", function () {
            for (const c of SIZE_CASES) {
                it(`${c.sub} Determinant gas growth for n=${c.n}`, async function () {
                    t++;
                    const A = makeDiagonallyDominant(c.n);
                    const Aq = await qArrayFromNumbers(harness, flatten(A));

                    await touchGas(harness, "detHarness", [BigInt(c.n), BigInt(c.n), Aq]);
                    const gas = await estimateGas(harness, "detHarness", [BigInt(c.n), BigInt(c.n), Aq]);

                    const detHex = await harness.detHarness(BigInt(c.n), BigInt(c.n), Aq);
                    const detDec = await fromQuad(harness, detHex);

                    printBlockRegular({
                        t,
                        method: "determinant",
                        explanation: `Gas growth of determinant computation for square matrix size n=${c.n}.`,
                        gas,
                        inHex: `shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: detHex,
                        expectedDec: "N/A",
                        outDec: formatScaledInt(detDec),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        });

        describe("Section 4.3: Inversion", function () {
            for (const c of SIZE_CASES) {
                it(`${c.sub} Inversion gas growth for n=${c.n}`, async function () {
                    t++;
                    const A = makeDiagonallyDominant(c.n);
                    const Aq = await qArrayFromNumbers(harness, flatten(A));

                    await touchGas(harness, "inverseHarness", [BigInt(c.n), BigInt(c.n), Aq]);
                    const gas = await estimateGas(harness, "inverseHarness", [BigInt(c.n), BigInt(c.n), Aq]);

                    const [, , invHex] = await harness.inverseHarness(BigInt(c.n), BigInt(c.n), Aq);
                    const invDec = await fromQuadArray(harness, invHex);

                    printBlockRegular({
                        t,
                        method: "inverse",
                        explanation: `Gas growth of matrix inversion for square matrix size n=${c.n}.`,
                        gas,
                        inHex: `shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: headTail(invHex),
                        expectedDec: "N/A",
                        outDec: headTail(invDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        });
    });

    // ------------------------------------------------------------
    // Section 5: Power Iteration
    // ------------------------------------------------------------

    function makeSymmetricMatrix(
        n: number,
        pattern: "diagDominant" | "identity" | "clustered" | "weighted" | "mixed"
    ): number[][] {
        const A = Array.from({ length: n }, () => Array(n).fill(0));

        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                let v: number;

                if (pattern === "identity") v = 0;
                else if (pattern === "clustered") v = (i + j) % 2 === 0 ? 1 : 2;
                else if (pattern === "weighted") v = (i % 3) + 1;
                else if (pattern === "mixed") v = ((i + j) % 3) + 1;
                else v = 1; // diagDominant default off-diagonal

                A[i][j] = v;
                A[j][i] = v;
            }
        }

        for (let i = 0; i < n; i++) {
            let rowAbsSum = 0;
            for (let j = 0; j < n; j++) {
                if (i !== j) rowAbsSum += absNumber(A[i][j]);
            }

            if (pattern === "identity") A[i][i] = 1;
            else if (pattern === "clustered") A[i][i] = rowAbsSum + 5 + i;
            else if (pattern === "weighted") A[i][i] = rowAbsSum + 10 + 2 * i;
            else if (pattern === "mixed") A[i][i] = rowAbsSum + 7 + (i % 4);
            else A[i][i] = rowAbsSum + 3; // diagDominant
        }

        return A;
    }

    describe("Section 5: Power Iteration", function () {

        describe("Section 5.1: Gas vs matrix dimension", function () {
            const DIM_CASES = [
                { sub: "5.1", n: 2 },
                { sub: "5.2", n: 4 },
                { sub: "5.3", n: 6 },
                { sub: "5.4", n: 9 },
                { sub: "5.5", n: 12 },
            ];

            for (const c of DIM_CASES) {
                it(`${c.sub} Power iteration gas growth for n=${c.n}`, async function () {
                    t++;

                    const A = makeSymmetricMatrix(c.n, "diagDominant");
                    const Aq = await qArrayFromNumbers(harness, flatten(A));
                    const tol = await harness.qFromFrac(1n, 1_000_000n); // 1e-6

                    await touchGas(harness, "powerIterationWithIterHarness", [
                        BigInt(c.n),
                        BigInt(c.n),
                        Aq,
                        powerSeed,
                        tol,
                        200n
                    ]);

                    const gas = await estimateGas(harness, "powerIterationWithIterHarness", [
                        BigInt(c.n),
                        BigInt(c.n),
                        Aq,
                        powerSeed,
                        tol,
                        200n
                    ]);

                    const [lambdaHex, xRows, xCols, xData, iterCount] =
                        await harness.powerIterationWithIterHarness(
                            BigInt(c.n),
                            BigInt(c.n),
                            Aq,
                            powerSeed,
                            tol,
                            200n
                        );

                    const lambdaDec = await fromQuad(harness, lambdaHex);
                    const xDec = await fromQuadArray(harness, xData);

                    printBlockRegular({
                        t,
                        method: "power iteration",
                        explanation: `Gas growth with respect to matrix dimension for dominant eigenvalue approximation using diagDominant symmetric matrix.`,
                        gas,
                        inHex: `shape=${c.n}x${c.n}, pattern=diagDominant, seed=fixed, tol=1e-6`,
                        expectedHex: "N/A",
                        outHex: `lambda=${lambdaHex}, iter=${iterCount}, eigenvector=${headTail(xData)}`,
                        expectedDec: "N/A",
                        outDec: `lambda=${formatScaledInt(lambdaDec)}, iter=${iterCount}, eigenvector=${headTail(xDec.map(formatScaledInt))}`,
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        });

        describe("Section 5.2: Gas by matrix pattern", function () {
            const PATTERN_CASES: Array<{
                sub: string;
                pattern: "identity" | "diagDominant" | "clustered" | "weighted" | "mixed";
            }> = [
                    { sub: "5.6", pattern: "identity" },
                    { sub: "5.7", pattern: "diagDominant" },
                    { sub: "5.8", pattern: "clustered" },
                    { sub: "5.9", pattern: "weighted" },
                    { sub: "5.10", pattern: "mixed" },
                ];

            for (const c of PATTERN_CASES) {
                it(`${c.sub} Power iteration gas sensitivity for ${c.pattern}`, async function () {
                    t++;

                    const n = 8;
                    const A = makeSymmetricMatrix(n, c.pattern);
                    const Aq = await qArrayFromNumbers(harness, flatten(A));
                    const tol = await harness.qFromFrac(1n, 1_000_000n); // 1e-6

                    await touchGas(harness, "powerIterationWithIterHarness", [
                        8n,
                        8n,
                        Aq,
                        powerSeed,
                        tol,
                        200n
                    ]);

                    const gas = await estimateGas(harness, "powerIterationWithIterHarness", [
                        8n,
                        8n,
                        Aq,
                        powerSeed,
                        tol,
                        200n
                    ]);

                    const [lambdaHex, xRows, xCols, xData, iterCount] =
                        await harness.powerIterationWithIterHarness(
                            8n,
                            8n,
                            Aq,
                            powerSeed,
                            tol,
                            200n
                        );

                    const lambdaDec = await fromQuad(harness, lambdaHex);
                    const xDec = await fromQuadArray(harness, xData);

                    printBlockRegular({
                        t,
                        method: "power iteration",
                        explanation: `Gas sensitivity of power iteration under different symmetric matrix patterns at fixed dimension n=8.`,
                        gas,
                        inHex: `shape=8x8, pattern=${c.pattern}, seed=fixed, tol=1e-6`,
                        expectedHex: "N/A",
                        outHex: `lambda=${lambdaHex}, iter=${iterCount}, eigenvector=${headTail(xData)}`,
                        expectedDec: "N/A",
                        outDec: `lambda=${formatScaledInt(lambdaDec)}, iter=${iterCount}, eigenvector=${headTail(xDec.map(formatScaledInt))}`,
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        });
    });

    // ------------------------------------------------------------
    // Section 6: input class vs determinant
    // ------------------------------------------------------------

    describe("Section 6: input class vs determinant", function () {
        const CLASS_CASES: Array<{
            sub: string;
            cls: "well-conditioned" | "singular" | "near-singular" | "pivoting-required" | "diagonally-dominant";
        }> = [
                { sub: "6.1", cls: "well-conditioned" },
                { sub: "6.2", cls: "singular" },
                { sub: "6.3", cls: "near-singular" },
                { sub: "6.4", cls: "pivoting-required" },
                { sub: "6.5", cls: "diagonally-dominant" },
            ];

        for (const c of CLASS_CASES) {
            it(`${c.sub} Determinant gas sensitivity for ${c.cls}`, async function () {
                t++;
                const n = 4;
                const A = buildClassMatrix(n, c.cls);
                const Aq = await qArrayFromNumbers(harness, flatten(A));

                await touchGas(harness, "detHarness", [BigInt(n), BigInt(n), Aq]);
                const gas = await estimateGas(harness, "detHarness", [BigInt(n), BigInt(n), Aq]);

                const detHex = await harness.detHarness(BigInt(n), BigInt(n), Aq);
                const detDec = await fromQuad(harness, detHex);

                printBlockRegular({
                    t,
                    method: "determinant",
                    explanation: `Gas sensitivity of determinant computation for ${c.cls} matrix class.`,
                    gas,
                    inHex: `class=${c.cls}, shape=4x4`,
                    expectedHex: "N/A",
                    outHex: detHex,
                    expectedDec: "N/A",
                    outDec: formatScaledInt(detDec),
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 7: input class vs inverse
    // ------------------------------------------------------------

    describe("Section 7: input class vs inverse", function () {
        const CLASS_CASES: Array<{
            sub: string;
            cls: "well-conditioned" | "singular" | "near-singular" | "pivoting-required" | "diagonally-dominant";
        }> = [
                { sub: "7.1", cls: "well-conditioned" },
                { sub: "7.2", cls: "singular" },
                { sub: "7.3", cls: "near-singular" },
                { sub: "7.4", cls: "pivoting-required" },
                { sub: "7.5", cls: "diagonally-dominant" },
            ];

        for (const c of CLASS_CASES) {
            it(`${c.sub} Inverse gas sensitivity for ${c.cls}`, async function () {
                t++;
                const n = 4;
                const A = buildClassMatrix(n, c.cls);
                const Aq = await qArrayFromNumbers(harness, flatten(A));

                if (c.cls === "singular") {
                    await expect(
                        harness.inverseHarness(BigInt(n), BigInt(n), Aq)
                    ).to.be.reverted;

                    printBlockRegular({
                        t,
                        method: "inverse",
                        explanation: `Inverse computation for ${c.cls} matrix class should revert because the matrix is not invertible.`,
                        gas: "N/A",
                        inHex: `class=${c.cls}, shape=4x4`,
                        expectedHex: "Reverted",
                        outHex: "Reverted",
                        expectedDec: "Reverted",
                        outDec: "Reverted",
                    });
                } else {
                    await touchGas(harness, "inverseHarness", [BigInt(n), BigInt(n), Aq]);
                    const gas = await estimateGas(harness, "inverseHarness", [BigInt(n), BigInt(n), Aq]);

                    const [, , invHex] = await harness.inverseHarness(BigInt(n), BigInt(n), Aq);
                    const invDec = await fromQuadArray(harness, invHex);

                    printBlockRegular({
                        t,
                        method: "inverse",
                        explanation: `Gas sensitivity of matrix inversion for ${c.cls} matrix class.`,
                        gas,
                        inHex: `class=${c.cls}, shape=4x4`,
                        expectedHex: "N/A",
                        outHex: headTail(invHex),
                        expectedDec: "N/A",
                        outDec: headTail(invDec.map(formatScaledInt)),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }
            });
        }
    });

    // ------------------------------------------------------------
    // Section 8: matrix pattern vs gas
    // ------------------------------------------------------------

    describe("Section 8: matrix pattern vs gas", function () {
        const PATTERN_CASES = [
            { sub: "8.1", pattern: "identity" as const },
            { sub: "8.2", pattern: "triangular" as const },
            { sub: "8.3", pattern: "dense_random" as const },
            { sub: "8.4", pattern: "sparse_random" as const },
            { sub: "8.5", pattern: "pivoting-required" as const },
        ];

        for (const c of PATTERN_CASES) {
            it(`${c.sub} Matrix pattern vs gas for ${c.pattern}`, async function () {
                t++;
                const n = 8;
                const A = buildPatternMatrix(n, c.pattern);
                const Aq = await qArrayFromNumbers(harness, flatten(A));

                // determinant chosen as the common gas probe across different matrix patterns
                await touchGas(harness, "detHarness", [BigInt(n), BigInt(n), Aq]);
                const gas = await estimateGas(harness, "detHarness", [BigInt(n), BigInt(n), Aq]);

                const detHex = await harness.detHarness(BigInt(n), BigInt(n), Aq);
                const detDec = await fromQuad(harness, detHex);

                printBlockRegular({
                    t,
                    method: "matrix pattern vs gas",
                    explanation: `Gas sensitivity to matrix pattern using determinant as the common structural gas probe for ${c.pattern} pattern.`,
                    gas,
                    inHex: `pattern=${c.pattern}, shape=8x8`,
                    expectedHex: "N/A",
                    outHex: detHex,
                    expectedDec: "N/A",
                    outDec: formatScaledInt(detDec),
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 9: Dense Matrix-Matrix Multiplication
    // ------------------------------------------------------------

    describe("Section 9: Dense Matrix-Matrix Multiplication", function () {
        const DIM_CASES = [
            { sub: "9.1", n: 2 },
            { sub: "9.2", n: 4 },
            { sub: "9.3", n: 8 },
            { sub: "9.4", n: 12 },
            { sub: "9.5", n: 16 },
        ];

        for (const c of DIM_CASES) {
            it(`${c.sub} Dense matmul gas growth for n=${c.n}`, async function () {
                t++;

                const A = makeDenseRandom(c.n);
                const B = makeDenseRandom(c.n);

                const Aq = await qArrayFromNumbers(harness, flatten(A));
                const Bq = await qArrayFromNumbers(harness, flatten(B));

                await touchGas(harness, "mulMatrixHarness", [
                    BigInt(c.n), BigInt(c.n), Aq,
                    BigInt(c.n), BigInt(c.n), Bq
                ]);

                const gas = await estimateGas(
                    harness,
                    "mulMatrixHarness",
                    [BigInt(c.n), BigInt(c.n), Aq, BigInt(c.n), BigInt(c.n), Bq]
                );

                if (gas === "revert") {
                    printBlockRegular({
                        t,
                        method: "matrix-matrix multiplication",
                        explanation: `Gas growth with respect to matrix dimension for dense ${c.n}x${c.n} × ${c.n}x${c.n} multiplication.`,
                        gas: "revert",
                        inHex: `A_shape=${c.n}x${c.n}, B_shape=${c.n}x${c.n}`,
                        expectedHex: "N/A",
                        outHex: "Reverted",
                        expectedDec: "N/A",
                        outDec: "Reverted",
                    });
                    return;
                }

                const [, , out] = await harness.mulMatrixHarness(
                    BigInt(c.n), BigInt(c.n), Aq,
                    BigInt(c.n), BigInt(c.n), Bq
                );
                const outDec = await fromQuadArray(harness, out);

                printBlockRegular({
                    t,
                    method: "matrix-matrix multiplication",
                    explanation: `Gas growth with respect to matrix dimension for dense ${c.n}x${c.n} × ${c.n}x${c.n} multiplication.`,
                    gas,
                    inHex: `A_shape=${c.n}x${c.n}, B_shape=${c.n}x${c.n}`,
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
    // Section 10: Sparse Matrix-Vector Multiplication
    // ------------------------------------------------------------

    describe("Section 10: Sparse Matrix-Vector Multiplication", function () {

        const CASES = [
            { sub: "10.1", bw: 0 },
            { sub: "10.2", bw: 1 },
            { sub: "10.3", bw: 2 },
            { sub: "10.4", bw: 3 },
            { sub: "10.5", bw: 4 },
        ];

        for (const c of CASES) {
            it(`${c.sub} Sparse matvec gas vs bandwidth=${c.bw}`, async function () {
                t++;

                const n = 32;
                const s = buildSparseBandCSR(n, c.bw);

                const valuesQ = await qArrayFromNumbers(harness, s.values);
                const xQ = await qArrayFromNumbers(harness, s.vector);

                await touchGas(harness, "mulSparseMatrixVectorHarness", [
                    BigInt(n), BigInt(n), s.rowPtr, s.colInd, valuesQ, BigInt(n), xQ
                ]);

                const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [
                    BigInt(n), BigInt(n), s.rowPtr, s.colInd, valuesQ, BigInt(n), xQ
                ]);

                printBlockRegular({
                    t,
                    method: "sparse matrix-vector multiplication",
                    explanation: `Gas vs sparsity (bandwidth=${c.bw}).`,
                    gas,
                    inHex: `nnz=${s.colInd.length}`,
                    expectedHex: "N/A",
                    outHex: "N/A",
                    expectedDec: "N/A",
                    outDec: "N/A",
                });

                expect(toGasBigInt(gas) > 0n).to.equal(true);
            });
        }
    });
});