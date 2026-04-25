import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Contract Type
// ------------------------------------------------------------

type LinearSolversHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;
    fromFloat(n: bigint): Promise<string>;

    gradientDescentLeastSquares(
        m: bigint,
        n: bigint,
        Adata: string[],
        bdata: string[],
        x0data: string[],
        alpha: string,
        maxIter: bigint,
        tol: string
    ): Promise<[string[], bigint]>;

    jacobi(
        n: bigint,
        Adata: string[],
        bdata: string[],
        x0data: string[],
        maxIter: bigint,
        tolDiff: string
    ): Promise<[string[], bigint]>;

    gaussSeidel(
        n: bigint,
        Adata: string[],
        bdata: string[],
        x0data: string[],
        maxIter: bigint,
        tolDiff: string
    ): Promise<[string[], bigint]>;

    gaussianElimination(
        n: bigint,
        Adata: string[],
        bdata: string[]
    ): Promise<string[]>;

    luDecomposition(
        n: bigint,
        Adata: string[]
    ): Promise<[string[], string[]]>;
};

// ------------------------------------------------------------
// Global Numeric Settings
// ------------------------------------------------------------

let SCALE_DECIMALS = 32n;
let SCALE = 10n ** SCALE_DECIMALS;

// ------------------------------------------------------------
// Basic Helpers
// ------------------------------------------------------------

function asBigInt(v: unknown): bigint {
    if (typeof v === "bigint") return v;
    if (typeof v === "number") return BigInt(v);
    if (typeof v === "string") return BigInt(v);

    if (v && typeof v === "object") {
        const maybeToString = (v as { toString?: () => string }).toString;
        if (typeof maybeToString === "function") {
            return BigInt(maybeToString.call(v));
        }
    }

    throw new Error(`Cannot convert value to bigint: ${String(v)}`);
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

function inferScaleDecimals(scale: bigint): bigint {
    const s = scale.toString();
    if (!/^10*$/.test(s) || s[0] !== "1") {
        return SCALE_DECIMALS;
    }
    return BigInt(s.length - 1);
}

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function sumBigInt(arr: bigint[]): bigint {
    return arr.reduce((a, b) => a + b, 0n);
}

function avgBigInt(arr: bigint[]): bigint {
    return arr.length === 0 ? 0n : sumBigInt(arr) / BigInt(arr.length);
}

function minBigInt(arr: bigint[]): bigint {
    return arr.reduce((a, b) => (b < a ? b : a), arr[0]);
}

function maxBigInt(arr: bigint[]): bigint {
    return arr.reduce((a, b) => (b > a ? b : a), arr[0]);
}

function vecInfNorm(v: bigint[]): bigint {
    let m = 0n;
    for (const x of v) {
        const ax = absBigInt(x);
        if (ax > m) m = ax;
    }
    return m;
}

function subVec(a: bigint[], b: bigint[]): bigint[] {
    return a.map((x, i) => x - b[i]);
}

function flattenMatrix(A: bigint[][]): bigint[] {
    return A.flat();
}

function reshapeMatrix(flat: bigint[], rows: number, cols: number): bigint[][] {
    const out: bigint[][] = [];
    for (let i = 0; i < rows; i++) {
        out.push(flat.slice(i * cols, (i + 1) * cols));
    }
    return out;
}

function fmtVec(v: bigint[]): string {
    return `[${v.map(formatScaledInt).join(", ")}]`;
}

// ------------------------------------------------------------
// Fixed-Point Linear Algebra Helpers
// ------------------------------------------------------------

function matVecMulScaled(A: bigint[][], x: bigint[]): bigint[] {
    return A.map(row => {
        let sum = 0n;
        for (let i = 0; i < row.length; i++) {
            sum += (row[i] * x[i]) / SCALE;
        }
        return sum;
    });
}

function matMulScaled(A: bigint[][], B: bigint[][]): bigint[][] {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;

    const out: bigint[][] = Array.from({ length: rows }, () => Array<bigint>(cols).fill(0n));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let sum = 0n;
            for (let k = 0; k < inner; k++) {
                sum += (A[i][k] * B[k][j]) / SCALE;
            }
            out[i][j] = sum;
        }
    }

    return out;
}

function matInfNormEntries(A: bigint[][]): bigint {
    let m = 0n;
    for (const row of A) {
        for (const x of row) {
            const ax = absBigInt(x);
            if (ax > m) m = ax;
        }
    }
    return m;
}

function matSub(A: bigint[][], B: bigint[][]): bigint[][] {
    return A.map((row, i) => row.map((x, j) => x - B[i][j]));
}

function forwardSubstitutionScaled(L: bigint[][], b: bigint[]): bigint[] {
    const n = L.length;
    const y = Array<bigint>(n).fill(0n);

    for (let i = 0; i < n; i++) {
        let sum = 0n;
        for (let j = 0; j < i; j++) {
            sum += (L[i][j] * y[j]) / SCALE;
        }

        const rhs = b[i] - sum;
        y[i] = (rhs * SCALE) / L[i][i];
    }

    return y;
}

function backwardSubstitutionScaled(U: bigint[][], y: bigint[]): bigint[] {
    const n = U.length;
    const x = Array<bigint>(n).fill(0n);

    for (let i = n - 1; i >= 0; i--) {
        let sum = 0n;
        for (let j = i + 1; j < n; j++) {
            sum += (U[i][j] * x[j]) / SCALE;
        }

        const rhs = y[i] - sum;
        x[i] = (rhs * SCALE) / U[i][i];
    }

    return x;
}

// ------------------------------------------------------------
// Quad Conversion Helpers
// ------------------------------------------------------------

async function outScaled(harness: LinearSolversHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

async function scaledVec(harness: LinearSolversHarness, values: string[]): Promise<bigint[]> {
    return Promise.all(values.map(v => outScaled(harness, v)));
}

async function qScaled(harness: LinearSolversHarness, scaledValue: bigint): Promise<string> {
    return await harness.fromFloat(asBigInt(scaledValue));
}

async function qVecFromScaledInts(harness: LinearSolversHarness, vals: bigint[]): Promise<string[]> {
    return Promise.all(vals.map(v => qScaled(harness, v)));
}

async function qMatFromScaledInts(harness: LinearSolversHarness, A: bigint[][]): Promise<string[]> {
    return qVecFromScaledInts(harness, flattenMatrix(A));
}

// ------------------------------------------------------------
// Gas Estimation Helpers
// ------------------------------------------------------------

async function estimateJacobiGas(
    harness: LinearSolversHarness,
    n: bigint,
    Adata: string[],
    bdata: string[],
    x0data: string[],
    maxIter: bigint,
    tol: string
): Promise<bigint> {
    const gas = await (harness.jacobi as any).estimateGas(n, Adata, bdata, x0data, maxIter, tol);
    return asBigInt(gas);
}

async function estimateGaussSeidelGas(
    harness: LinearSolversHarness,
    n: bigint,
    Adata: string[],
    bdata: string[],
    x0data: string[],
    maxIter: bigint,
    tol: string
): Promise<bigint> {
    const gas = await (harness.gaussSeidel as any).estimateGas(n, Adata, bdata, x0data, maxIter, tol);
    return asBigInt(gas);
}

async function estimateGaussianEliminationGas(
    harness: LinearSolversHarness,
    n: bigint,
    Adata: string[],
    bdata: string[]
): Promise<bigint> {
    const gas = await (harness.gaussianElimination as any).estimateGas(n, Adata, bdata);
    return asBigInt(gas);
}

async function estimateLUGas(
    harness: LinearSolversHarness,
    n: bigint,
    Adata: string[]
): Promise<bigint> {
    const gas = await (harness.luDecomposition as any).estimateGas(n, Adata);
    return asBigInt(gas);
}

async function estimateGDGas(
    harness: LinearSolversHarness,
    m: bigint,
    n: bigint,
    Adata: string[],
    bdata: string[],
    x0data: string[],
    alpha: string,
    maxIter: bigint,
    tol: string
): Promise<bigint> {
    const gas = await (harness.gradientDescentLeastSquares as any).estimateGas(
        m, n, Adata, bdata, x0data, alpha, maxIter, tol
    );
    return asBigInt(gas);
}

// ------------------------------------------------------------
// Deterministic Pseudo-Random Generation
// ------------------------------------------------------------

function hashToBigInt(tag: string): bigint {
    return BigInt(ethers.keccak256(ethers.toUtf8Bytes(tag)));
}

function boundedRandInt(tag: string, min: number, max: number): bigint {
    const h = hashToBigInt(tag);
    const range = BigInt(max - min + 1);
    return BigInt(min) + (h % range);
}

function boundedScaled(tag: string, min: number, max: number): bigint {
    return boundedRandInt(tag, min, max) * SCALE;
}

function makeVectorScaled(len: number, baseTag: string, min: number, max: number): bigint[] {
    return Array.from({ length: len }, (_, i) =>
        boundedScaled(`${baseTag}:v:${i}`, min, max)
    );
}

function makeDiagonallyDominantMatrixScaled(n: number, baseTag: string): bigint[][] {
    const A: bigint[][] = [];

    for (let i = 0; i < n; i++) {
        const row = Array<bigint>(n).fill(0n);
        let rowAbsSum = 0n;

        for (let j = 0; j < n; j++) {
            if (i === j) continue;

            const v = boundedScaled(`${baseTag}:a:${i}:${j}`, -5, 5);
            row[j] = v;
            rowAbsSum += absBigInt(v);
        }

        row[i] = rowAbsSum + boundedScaled(`${baseTag}:diag:${i}`, 5, 10);
        A.push(row);
    }

    return A;
}

function makeFullRankRectMatrixScaled(m: number, n: number, baseTag: string): bigint[][] {
    const A: bigint[][] = [];

    for (let i = 0; i < m; i++) {
        const row: bigint[] = [];
        for (let j = 0; j < n; j++) {
            let v = boundedScaled(`${baseTag}:a:${i}:${j}`, -1, 1);
            if (i === j) {
                v += 1n * SCALE;
            }
            row.push(v);
        }
        A.push(row);
    }

    return A;
}

// ------------------------------------------------------------
// Helpers for case base tests
// ------------------------------------------------------------

type NamedMatrixCase = {
    caseName: string;
    n: number;
    makeMatrix: (tag: string) => bigint[][];
};

function makeIdentityMatrixScaled(n: number): bigint[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? SCALE : 0n))
    );
}

function makeDiagonalMatrixScaled(n: number, tag: string): bigint[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            if (i !== j) return 0n;
            return boundedScaled(`${tag}:diag:${i}`, 2, 9);
        })
    );
}

function makeUpperTriangularMatrixScaled(n: number, tag: string): bigint[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            if (j < i) return 0n;
            if (i === j) return boundedScaled(`${tag}:diag:${i}`, 3, 9);
            return boundedScaled(`${tag}:u:${i}:${j}`, -3, 3);
        })
    );
}

function makeStrongDiagDominantMatrixScaled(n: number, tag: string): bigint[][] {
    const A: bigint[][] = [];

    for (let i = 0; i < n; i++) {
        const row = Array<bigint>(n).fill(0n);
        let rowAbsSum = 0n;

        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const v = boundedScaled(`${tag}:a:${i}:${j}`, -3, 3);
            row[j] = v;
            rowAbsSum += absBigInt(v);
        }

        row[i] = rowAbsSum + boundedScaled(`${tag}:diag:${i}`, 10, 15);
        A.push(row);
    }

    return A;
}

function makeWeakDiagDominantMatrixScaled(n: number, tag: string): bigint[][] {
    const A: bigint[][] = [];

    for (let i = 0; i < n; i++) {
        const row = Array<bigint>(n).fill(0n);
        let rowAbsSum = 0n;

        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const v = boundedScaled(`${tag}:a:${i}:${j}`, -4, 4);
            row[j] = v;
            rowAbsSum += absBigInt(v);
        }

        row[i] = rowAbsSum + boundedScaled(`${tag}:diag:${i}`, 1, 2);
        A.push(row);
    }

    return A;
}

function makePivotStressDenseMatrixScaled(n: number, tag: string): bigint[][] {
    const A: bigint[][] = [];

    for (let i = 0; i < n; i++) {
        const row = Array<bigint>(n).fill(0n);
        let rowAbsSum = 0n;

        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const v = boundedScaled(`${tag}:a:${i}:${j}`, -5, 5);
            row[j] = v;
            rowAbsSum += absBigInt(v);
        }

        // First diagonal intentionally smaller, others regular
        if (i === 0) {
            row[i] = 1n * SCALE;
        } else {
            row[i] = rowAbsSum + boundedScaled(`${tag}:diag:${i}`, 4, 7);
        }

        A.push(row);
    }

    // Make first column heavier to stress pivoting behavior
    for (let i = 1; i < n; i++) {
        A[i][0] += boundedScaled(`${tag}:col0:${i}`, 3, 7);
    }

    return A;
}

function makeBandedMatrixScaled(n: number, tag: string): bigint[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            const d = Math.abs(i - j);
            if (d > 1) return 0n;
            if (i === j) return boundedScaled(`${tag}:diag:${i}`, 4, 8);
            return boundedScaled(`${tag}:band:${i}:${j}`, -2, 2);
        })
    );
}

function makeSPDLikeMatrixScaled(n: number, tag: string): bigint[][] {
    const M = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => boundedScaled(`${tag}:m:${i}:${j}`, -2, 2))
    );

    const MT = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => M[j][i])
    );

    let A = matMulScaled(MT, M);

    for (let i = 0; i < n; i++) {
        A[i][i] += BigInt(n + 2) * SCALE;
    }

    return A;
}

type NamedLeastSquaresCase = {
    caseName: string;
    m: number;
    n: number;
    makeMatrix: (tag: string) => bigint[][];
};

function makeTallIdentityLikeMatrixScaled(m: number, n: number): bigint[][] {
    return Array.from({ length: m }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            if (i < n && i === j) return SCALE;
            return 0n;
        })
    );
}

function makeWellConditionedRectMatrixScaled(m: number, n: number, tag: string): bigint[][] {
    return Array.from({ length: m }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            let v = boundedScaled(`${tag}:a:${i}:${j}`, -2, 2);
            if (i === j) v += 2n * SCALE;
            return v;
        })
    );
}

function makeColumnCorrelatedRectMatrixScaled(m: number, n: number, tag: string): bigint[][] {
    const baseCol = Array.from({ length: m }, (_, i) =>
        boundedScaled(`${tag}:base:${i}`, -2, 2)
    );

    return Array.from({ length: m }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            if (j === 0) return baseCol[i];
            return baseCol[i] + boundedScaled(`${tag}:noise:${i}:${j}`, -1, 1);
        })
    );
}

function makeScaledColumnsRectMatrixScaled(m: number, n: number, tag: string): bigint[][] {
    const scales = [1n, 2n, 5n, 10n, 20n, 50n, 100n, 200n];

    return Array.from({ length: m }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            const base = boundedScaled(`${tag}:a:${i}:${j}`, -1, 1);
            return (base * scales[j % scales.length]);
        })
    );
}

function makeNearDependentRectMatrixScaled(m: number, n: number, tag: string): bigint[][] {
    const firstCol = Array.from({ length: m }, (_, i) =>
        boundedScaled(`${tag}:c0:${i}`, -2, 2)
    );

    return Array.from({ length: m }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            if (j === 0) return firstCol[i];
            if (j === 1) return firstCol[i] + boundedScaled(`${tag}:eps:${i}`, -1, 1);
            return boundedScaled(`${tag}:a:${i}:${j}`, -2, 2);
        })
    );
}

// ------------------------------------------------------------
// Reporting Helpers
// ------------------------------------------------------------

function printSectionExplanation(args: {
    section: string;
    objective: string;
    setup: string;
    metrics: string;
}) {
    console.log("############################################################");
    console.log(`SECTION EXPLANATION   : ${args.section}`);
    console.log("############################################################");
    console.log(`Objective            : ${args.objective}`);
    console.log(`Experimental Setup   : ${args.setup}`);
    console.log(`Reported Metrics     : ${args.metrics}`);
    console.log("############################################################");
}

function printTestExplanation(args: {
    method: string;
    testExplanation: string;
}) {
    console.log("************************************************************");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log("************************************************************");
}

function printCaseBlock(args: {
    method: string;
    testExplanation: string;
    gas: bigint;
    err: bigint;
    res: bigint;
    iters?: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log(`Gas Usage            : ${args.gas.toString()}`);
    console.log(`Solution Error       : ${formatScaledInt(args.err)}`);
    console.log(`Residual             : ${formatScaledInt(args.res)}`);
    if (args.iters !== undefined) {
        console.log(`Iterations           : ${args.iters.toString()}`);
    }
    console.log("------------------------------------------------------------");
}

function printLUCaseBlock(args: {
    method: string;
    testExplanation: string;
    gas: bigint;
    recon: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log(`Gas Usage            : ${args.gas.toString()}`);
    console.log(`Reconstruction Error : ${formatScaledInt(args.recon)}`);
    console.log("------------------------------------------------------------");
}

function printSummaryBlock(args: {
    method: string;
    testExplanation: string;
    title: string;
    count: number;
    avgErr: bigint;
    avgRes: bigint;
    avgGas: bigint;
    minGas: bigint;
    maxGas: bigint;
    avgIter?: bigint;
}) {
    console.log("============================================================");
    console.log(args.title);
    console.log("============================================================");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log(`Test Count           : ${args.count}`);
    console.log(`Average Abs. Error   : ${formatScaledInt(args.avgErr)}`);
    console.log(`Average Residual     : ${formatScaledInt(args.avgRes)}`);
    if (args.avgIter !== undefined) {
        console.log(`Average Iterations   : ${args.avgIter.toString()}`);
    }
    console.log(`Average Gas          : ${args.avgGas.toString()}`);
    console.log(`Min Gas              : ${args.minGas.toString()}`);
    console.log(`Max Gas              : ${args.maxGas.toString()}`);
    console.log("============================================================");
}

function printLUSummaryBlock(args: {
    method: string;
    testExplanation: string;
    title: string;
    count: number;
    avgRecon: bigint;
    avgGas: bigint;
    minGas: bigint;
    maxGas: bigint;
}) {
    console.log("============================================================");
    console.log(args.title);
    console.log("============================================================");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log(`Test Count           : ${args.count}`);
    console.log(`Average Recon. Error : ${formatScaledInt(args.avgRecon)}`);
    console.log(`Average Gas          : ${args.avgGas.toString()}`);
    console.log(`Min Gas              : ${args.minGas.toString()}`);
    console.log(`Max Gas              : ${args.maxGas.toString()}`);
    console.log("============================================================");
}

function printFeasibilityLine(args: {
    method: string;
    testExplanation: string;
    label: string;
    status: string;
    gas?: bigint;
    err?: bigint;
    res?: bigint;
    iters?: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log(`Label                : ${args.label}`);
    console.log(`Status               : ${args.status}`);
    if (args.gas !== undefined) console.log(`Estimated Gas        : ${args.gas.toString()}`);
    if (args.err !== undefined) console.log(`Solution Error       : ${formatScaledInt(args.err)}`);
    if (args.res !== undefined) console.log(`Residual             : ${formatScaledInt(args.res)}`);
    if (args.iters !== undefined) console.log(`Iterations           : ${args.iters.toString()}`);
    console.log("------------------------------------------------------------");
}

function printSanitySummaryBlock(args: {
    method: string;
    testExplanation: string;
    title: string;
    count: number;
    avgErr: bigint;
    avgRes: bigint;
    maxErr: bigint;
    maxRes: bigint;
}) {
    console.log("============================================================");
    console.log(args.title);
    console.log("============================================================");
    console.log(`Method               : ${args.method}`);
    console.log(`Test Explanation     : ${args.testExplanation}`);
    console.log(`Test Count           : ${args.count}`);
    console.log(`Average Abs. Error   : ${formatScaledInt(args.avgErr)}`);
    console.log(`Average Residual     : ${formatScaledInt(args.avgRes)}`);
    console.log(`Max Abs. Error       : ${formatScaledInt(args.maxErr)}`);
    console.log(`Max Residual         : ${formatScaledInt(args.maxRes)}`);
    console.log("============================================================");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("LinearSolvers Library - Randomized Accuracy, Iteration, and Gas Tests", function () {
    this.timeout(300000); // 5 minutes
    let harness: LinearSolversHarness;

    let TOL_1E_15: string;

    const SIZE_NS = [20];
    const T = 15;
    const MAX_ITER = 10000n;

    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    // Case based scenario constants
    const CASE_REPEAT = 10;

    const NAMED_CASES: NamedMatrixCase[] = [
        {
            caseName: "Identity",
            n: 8,
            makeMatrix: () => makeIdentityMatrixScaled(8),
        },
        {
            caseName: "Diagonal",
            n: 8,
            makeMatrix: (tag: string) => makeDiagonalMatrixScaled(8, tag),
        },
        {
            caseName: "Upper Triangular",
            n: 8,
            makeMatrix: (tag: string) => makeUpperTriangularMatrixScaled(8, tag),
        },
        {
            caseName: "Strongly Diagonally Dominant",
            n: 8,
            makeMatrix: (tag: string) => makeStrongDiagDominantMatrixScaled(8, tag),
        },
        {
            caseName: "Weakly Diagonally Dominant",
            n: 8,
            makeMatrix: (tag: string) => makeWeakDiagDominantMatrixScaled(8, tag),
        },
        {
            caseName: "Pivot-Stress Dense",
            n: 8,
            makeMatrix: (tag: string) => makePivotStressDenseMatrixScaled(8, tag),
        },
        {
            caseName: "Banded",
            n: 8,
            makeMatrix: (tag: string) => makeBandedMatrixScaled(8, tag),
        },
        {
            caseName: "SPD-Like",
            n: 8,
            makeMatrix: (tag: string) => makeSPDLikeMatrixScaled(8, tag),
        },
    ];

    const NAMED_GDLS_CASES: NamedLeastSquaresCase[] = [
        {
            caseName: "Identity-Like Tall",
            m: 12,
            n: 8,
            makeMatrix: () => makeTallIdentityLikeMatrixScaled(12, 8),
        },
        {
            caseName: "Well-Conditioned Full Rank",
            m: 16,
            n: 8,
            makeMatrix: (tag: string) => makeWellConditionedRectMatrixScaled(16, 8, tag),
        },
        {
            caseName: "Column-Correlated",
            m: 16,
            n: 8,
            makeMatrix: (tag: string) => makeColumnCorrelatedRectMatrixScaled(16, 8, tag),
        },
        {
            caseName: "Scaled Columns",
            m: 16,
            n: 8,
            makeMatrix: (tag: string) => makeScaledColumnsRectMatrixScaled(16, 8, tag),
        },
        {
            caseName: "Near-Dependent Columns",
            m: 16,
            n: 8,
            makeMatrix: (tag: string) => makeNearDependentRectMatrixScaled(16, 8, tag),
        },
    ];

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const MatrixMasterFactory = await ethers.getContractFactory("MatrixMaster");
        await MatrixMasterFactory.deploy();

        const HF = await ethers.getContractFactory("LinearSolversHarness", {
            libraries: {
                MathLib: await mathlib.getAddress(),
            },
        });

        harness = (await HF.deploy()) as unknown as LinearSolversHarness;
        await (harness as any).waitForDeployment?.();

        TOL_1E_15 = await qFrac(1, 1_000_000_000_000_000);

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    describe("Section 1: Size-based accuracy and gas tests", function () {
        before(function () {
            printSectionExplanation({
                section: "Section 1: Size-based accuracy and gas tests",
                objective: "To evaluate how matrix dimension affects numerical accuracy, residual behavior, iteration count, and gas usage of the implemented linear solver methods.",
                setup: "For each matrix size n in {2, 4, 8, 12, 16}, 15 deterministic diagonally dominant systems are generated. Direct and iterative methods are executed on these systems, and gas is estimated for each case.",
                metrics: "Solution error, residual norm, iteration count for iterative methods, reconstruction error for LU decomposition, and estimated gas usage."
            });
        });

        it("Gaussian Elimination: average solution error, residual, and gas across matrix sizes", async function () {
            const method = "Gaussian Elimination";
            const testExplanation =
                "Size-based gas and accuracy evaluation on deterministic diagonally dominant systems. For each matrix dimension, 15 cases are solved and per-case gas, solution error, and residual are recorded.";

            printTestExplanation({ method, testExplanation });

            for (const n of SIZE_NS) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `GE:n=${n}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);

                    const out = await harness.gaussianElimination(BigInt(n), Adata, bdata);
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateGaussianEliminationGas(harness, BigInt(n), Adata, bdata);

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | n=${n} | case=${t + 1}/${T}`,
                        gas,
                        err,
                        res,
                    });

                    errs.push(err);
                    residuals.push(res);
                    gases.push(gas);
                }

                printSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for n=${n}`,
                    title: `Gaussian Elimination Summary for n=${n}`,
                    count: T,
                    avgErr: avgBigInt(errs),
                    avgRes: avgBigInt(residuals),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                });

                expect(errs.length).to.equal(T);
            }
        });

        it("LU Decomposition: average reconstruction error and gas across matrix sizes", async function () {
            const method = "LU Decomposition";
            const testExplanation =
                "Size-based reconstruction and gas evaluation on deterministic diagonally dominant systems. For each matrix dimension, 15 cases are factorized and LU reconstruction error together with gas usage is recorded.";

            printTestExplanation({ method, testExplanation });

            for (const n of SIZE_NS) {
                const reconErrs: bigint[] = [];
                const gases: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `LU:n=${n}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const Adata = await qMatFromScaledInts(harness, A);

                    const [Lraw, Uraw] = await harness.luDecomposition(BigInt(n), Adata);
                    const Lflat = await scaledVec(harness, Lraw);
                    const Uflat = await scaledVec(harness, Uraw);

                    const L = reshapeMatrix(Lflat, n, n);
                    const U = reshapeMatrix(Uflat, n, n);

                    const LU = matMulScaled(L, U);
                    const diff = matSub(LU, A);
                    const reconErr = matInfNormEntries(diff);
                    const gas = await estimateLUGas(harness, BigInt(n), Adata);

                    printLUCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | n=${n} | case=${t + 1}/${T}`,
                        gas,
                        recon: reconErr,
                    });

                    reconErrs.push(reconErr);
                    gases.push(gas);
                }

                printLUSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for n=${n}`,
                    title: `LU Decomposition Summary for n=${n}`,
                    count: T,
                    avgRecon: avgBigInt(reconErrs),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                });

                expect(reconErrs.length).to.equal(T);
            }
        });

        it("Jacobi: average solution error, residual, iterations, and gas across matrix sizes", async function () {
            const method = "Jacobi";
            const testExplanation =
                "Size-based iteration, gas, and accuracy evaluation on deterministic diagonally dominant systems. For each matrix dimension, 15 cases are solved using a zero initial guess, fixed tolerance, and fixed maximum iteration budget.";

            printTestExplanation({ method, testExplanation });

            for (const n of SIZE_NS) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `JAC:n=${n}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    const [out, iters] = await harness.jacobi(BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateJacobiGas(harness, BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | n=${n} | case=${t + 1}/${T}`,
                        gas,
                        err,
                        res,
                        iters: asBigInt(iters),
                    });

                    errs.push(err);
                    residuals.push(res);
                    gases.push(gas);
                    itersArr.push(asBigInt(iters));
                }

                printSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for n=${n}`,
                    title: `Jacobi Summary for n=${n}`,
                    count: T,
                    avgErr: avgBigInt(errs),
                    avgRes: avgBigInt(residuals),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                    avgIter: avgBigInt(itersArr),
                });

                expect(errs.length).to.equal(T);
            }
        });

        it("Gauss-Seidel: average solution error, residual, iterations, and gas across matrix sizes", async function () {
            const method = "Gauss-Seidel";
            const testExplanation =
                "Size-based iteration, gas, and accuracy evaluation on deterministic diagonally dominant systems. For each matrix dimension, 15 cases are solved using a zero initial guess, fixed tolerance, and fixed maximum iteration budget.";

            printTestExplanation({ method, testExplanation });

            for (const n of SIZE_NS) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];
                let failedCount = 0;

                for (let t = 0; t < T; t++) {
                    const tag = `GS:n=${n}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    try {
                        const [out, iters] = await harness.gaussSeidel(
                            BigInt(n),
                            Adata,
                            bdata,
                            x0data,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        const xComp = await scaledVec(harness, out);
                        const gas = await estimateGaussSeidelGas(
                            harness,
                            BigInt(n),
                            Adata,
                            bdata,
                            x0data,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        const errVal = vecInfNorm(subVec(xComp, xTrue));
                        const resVal = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

                        printCaseBlock({
                            method,
                            testExplanation: `${testExplanation} | n=${n} | case=${t + 1}/${T}`,
                            gas,
                            err: errVal,
                            res: resVal,
                            iters: asBigInt(iters),
                        });

                        errs.push(errVal);
                        residuals.push(resVal);
                        gases.push(gas);
                        itersArr.push(asBigInt(iters));
                    } catch (err) {
                        failedCount++;

                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : ${testExplanation} | n=${n} | case=${t + 1}/${T}`);
                        console.log(`Status               : failed`);
                        console.log(`Failure Reason       : ${(err as Error).message}`);
                        console.log("------------------------------------------------------------");
                    }
                }

                if (gases.length > 0) {
                    printSummaryBlock({
                        method,
                        testExplanation: `${testExplanation} | summary for n=${n}`,
                        title: `Gauss-Seidel Summary for n=${n}`,
                        count: gases.length,
                        avgErr: avgBigInt(errs),
                        avgRes: avgBigInt(residuals),
                        avgGas: avgBigInt(gases),
                        minGas: minBigInt(gases),
                        maxGas: maxBigInt(gases),
                        avgIter: avgBigInt(itersArr),
                    });
                }
            }
        });

        it("Gradient Descent for Least Squares: average solution error, residual, iterations, and gas across matrix sizes", async function () {
            const method = "Gradient Descent for Least Squares";
            const testExplanation =
                "Size-based iteration, gas, and accuracy evaluation on deterministic full-rank least-squares systems. For each problem size, 15 cases are solved using a zero initial guess, fixed tolerance, fixed maximum iteration budget, and fixed step size.";

            printTestExplanation({ method, testExplanation });

            // GDLS settings.
            const GD_MAX_ITER = 50n;
            const alpha = await qFrac(1, 50);
            const gdTol = await qFrac(1, 1_000_000_000_000); // 1e-12

            for (const n of [2, 3, 4, 5, 6]) {
                const m = n + 4;

                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];
                let failedCount = 0;

                for (let t = 0; t < T; t++) {
                    const tag = `GDLS:m=${m}:n=${n}:t=${t}`;

                    const A = makeFullRankRectMatrixScaled(m, n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    try {
                        const [out, iters] = await harness.gradientDescentLeastSquares(
                            BigInt(m),
                            BigInt(n),
                            Adata,
                            bdata,
                            x0data,
                            alpha,
                            GD_MAX_ITER,
                            gdTol
                        );

                        const xComp = await scaledVec(harness, out);

                        const gas = await estimateGDGas(
                            harness,
                            BigInt(m),
                            BigInt(n),
                            Adata,
                            bdata,
                            x0data,
                            alpha,
                            GD_MAX_ITER,
                            gdTol
                        );

                        const errVal = vecInfNorm(subVec(xComp, xTrue));
                        const resVal = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

                        printCaseBlock({
                            method,
                            testExplanation: `${testExplanation} | m=${m} | n=${n} | case=${t + 1}/${T}`,
                            gas,
                            err: errVal,
                            res: resVal,
                            iters: asBigInt(iters),
                        });

                        errs.push(errVal);
                        residuals.push(resVal);
                        gases.push(gas);
                        itersArr.push(asBigInt(iters));
                    } catch (err: any) {
                        failedCount++;

                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : ${testExplanation} | m=${m} | n=${n} | case=${t + 1}/${T}`);
                        console.log(`Status               : failed`);
                        console.log(`Failure Reason       : ${err?.shortMessage ?? err?.message ?? "unknown error"}`);
                        console.log("------------------------------------------------------------");

                        // If too many runs fail, stop wasting time on this size.
                        if (failedCount >= 3) {
                            console.log("------------------------------------------------------------");
                            console.log(`Method               : ${method}`);
                            console.log(`Status               : early stop for size m=${m}, n=${n}`);
                            console.log(`Reason               : too many failed runs`);
                            console.log("------------------------------------------------------------");
                            break;
                        }
                    }
                }

                if (gases.length > 0) {
                    printSummaryBlock({
                        method,
                        testExplanation: `${testExplanation} | summary for m=${m}, n=${n}`,
                        title: `Gradient Descent Least Squares Summary for m=${m}, n=${n}`,
                        count: gases.length,
                        avgErr: avgBigInt(errs),
                        avgRes: avgBigInt(residuals),
                        avgGas: avgBigInt(gases),
                        minGas: minBigInt(gases),
                        maxGas: maxBigInt(gases),
                        avgIter: avgBigInt(itersArr),
                    });
                } else {
                    console.log("============================================================");
                    console.log(`Gradient Descent Least Squares Summary for m=${m}, n=${n}`);
                    console.log("============================================================");
                    console.log(`Method               : ${method}`);
                    console.log(`Test Explanation     : ${testExplanation} | summary for m=${m}, n=${n}`);
                    console.log(`Test Count           : 0`);
                    console.log(`Failed Runs          : ${failedCount}`);
                    console.log(`Status               : no successful run`);
                    console.log("============================================================");
                }
            }
        });
    });

    describe("Section 2: Initial guess sensitivity tests", function () {
        before(function () {
            printSectionExplanation({
                section: "Section 2: Initial guess sensitivity tests",
                objective: "To examine how the initial guess vector affects convergence behavior, solution accuracy, residuals, iteration count, and gas usage of iterative methods.",
                setup: "At fixed matrix size n = 8, deterministic diagonally dominant systems are generated. Three initial guess categories are tested: zero, moderate, and large.",
                metrics: "Solution error, residual norm, iteration count, and estimated gas usage."
            });
        });

        it("Jacobi: x0 sensitivity at fixed size with average accuracy, iterations, and gas", async function () {
            const method = "Jacobi";
            const testExplanation =
                "Initial guess sensitivity analysis at fixed matrix size n = 8. For each initial guess category, 15 deterministic systems are solved and the effect on iteration count, gas usage, solution error, and residual is recorded.";

            printTestExplanation({ method, testExplanation });

            const n = 8;
            const x0Configs = [
                { name: "zero", min: 0, max: 0 },
                { name: "moderate", min: -1, max: 1 },
                { name: "large", min: -10, max: 10 },
            ];

            for (const cfg of x0Configs) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `JAC_X0:${cfg.name}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = makeVectorScaled(n, `${tag}:x0`, cfg.min, cfg.max);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    const [out, iters] = await harness.jacobi(BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateJacobiGas(harness, BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | x0=${cfg.name} | case=${t + 1}/${T}`,
                        gas,
                        err,
                        res,
                        iters: asBigInt(iters),
                    });

                    errs.push(err);
                    residuals.push(res);
                    gases.push(gas);
                    itersArr.push(asBigInt(iters));
                }

                printSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for x0=${cfg.name}`,
                    title: `Jacobi x0 Sensitivity (${cfg.name})`,
                    count: T,
                    avgErr: avgBigInt(errs),
                    avgRes: avgBigInt(residuals),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                    avgIter: avgBigInt(itersArr),
                });

                expect(errs.length).to.equal(T);
            }
        });

        it("Gauss-Seidel: x0 sensitivity at fixed size with average accuracy, iterations, and gas", async function () {
            const method = "Gauss-Seidel";
            const testExplanation =
                "Initial guess sensitivity analysis at fixed matrix size n = 8. For each initial guess category, 15 deterministic systems are solved and the effect on iteration count, gas usage, solution error, and residual is recorded.";

            printTestExplanation({ method, testExplanation });

            const n = 8;
            const x0Configs = [
                { name: "zero", min: 0, max: 0 },
                { name: "moderate", min: -1, max: 1 },
                { name: "large", min: -10, max: 10 },
            ];

            for (const cfg of x0Configs) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `GS_X0:${cfg.name}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = makeVectorScaled(n, `${tag}:x0`, cfg.min, cfg.max);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    const [out, iters] = await harness.gaussSeidel(BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateGaussSeidelGas(harness, BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | x0=${cfg.name} | case=${t + 1}/${T}`,
                        gas,
                        err,
                        res,
                        iters: asBigInt(iters),
                    });

                    errs.push(err);
                    residuals.push(res);
                    gases.push(gas);
                    itersArr.push(asBigInt(iters));
                }

                printSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for x0=${cfg.name}`,
                    title: `Gauss-Seidel x0 Sensitivity (${cfg.name})`,
                    count: T,
                    avgErr: avgBigInt(errs),
                    avgRes: avgBigInt(residuals),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                    avgIter: avgBigInt(itersArr),
                });

                expect(errs.length).to.equal(T);
            }
        });
    });

    describe("Section 3: LU Sanity checks", function () {
        before(function () {
            printSectionExplanation({
                section: "Section 3: Sanity checks",
                objective: "To verify that the LU factors can recover correct solutions on multiple named deterministic structured systems.",
                setup: "A set of named structured matrices is generated at different sizes. The produced LU factors are used with forward and backward substitution to recover the solution vectors, and gas usage is also recorded.",
                metrics: "Estimated gas usage, recovered solution vector, infinity-norm solution error, and infinity-norm residual."
            });
        });

        it("LU factors can recover deterministic solutions on named structured systems", async function () {
            const method = "LU Decomposition";
            const testExplanation =
                "Sanity check on named deterministic structured systems. The LU factors are used with forward and backward substitution to recover the solution and verify numerical consistency, while gas usage is also recorded.";

            printTestExplanation({ method, testExplanation });

            const sanityCases = [
                {
                    caseName: "Identity 3x3",
                    n: 3,
                    makeMatrix: (_tag: string, n: number) => makeIdentityMatrixScaled(n),
                },
                {
                    caseName: "Identity 5x5",
                    n: 5,
                    makeMatrix: (_tag: string, n: number) => makeIdentityMatrixScaled(n),
                },
                {
                    caseName: "Diagonal 4x4",
                    n: 4,
                    makeMatrix: (tag: string, n: number) => makeDiagonalMatrixScaled(n, tag),
                },
                {
                    caseName: "Diagonal 6x6",
                    n: 6,
                    makeMatrix: (tag: string, n: number) => makeDiagonalMatrixScaled(n, tag),
                },
                {
                    caseName: "Upper Triangular 4x4",
                    n: 4,
                    makeMatrix: (tag: string, n: number) => makeUpperTriangularMatrixScaled(n, tag),
                },
                {
                    caseName: "Upper Triangular 6x6",
                    n: 6,
                    makeMatrix: (tag: string, n: number) => makeUpperTriangularMatrixScaled(n, tag),
                },
                {
                    caseName: "Strongly Diagonally Dominant 5x5",
                    n: 5,
                    makeMatrix: (tag: string, n: number) => makeStrongDiagDominantMatrixScaled(n, tag),
                },
                {
                    caseName: "Banded 6x6",
                    n: 6,
                    makeMatrix: (tag: string, n: number) => makeBandedMatrixScaled(n, tag),
                },
                {
                    caseName: "Pivot-Stress Dense 5x5",
                    n: 5,
                    makeMatrix: (tag: string, n: number) => makePivotStressDenseMatrixScaled(n, tag),
                },
                {
                    caseName: "Strongly Diagonally Dominant 8x8",
                    n: 8,
                    makeMatrix: (tag: string, n: number) => makeStrongDiagDominantMatrixScaled(n, tag),
                },
            ];

            const errArr: bigint[] = [];
            const residualArr: bigint[] = [];
            const gasArr: bigint[] = [];

            for (let i = 0; i < sanityCases.length; i++) {
                const sanityCase = sanityCases[i];
                const { caseName, n } = sanityCase;
                const tag = `LU_SANITY:${caseName.replace(/\s+/g, "_")}`;

                const A = sanityCase.makeMatrix(tag, n);
                const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -2, 2);
                const b = matVecMulScaled(A, xTrue);

                const Adata = await qMatFromScaledInts(harness, A);
                const [Lraw, Uraw] = await harness.luDecomposition(BigInt(n), Adata);

                const gas = await estimateLUGas(harness, BigInt(n), Adata);

                const L = reshapeMatrix(await scaledVec(harness, Lraw), n, n);
                const U = reshapeMatrix(await scaledVec(harness, Uraw), n, n);

                const y = forwardSubstitutionScaled(L, b);
                const xComp = backwardSubstitutionScaled(U, y);

                const solErr = vecInfNorm(subVec(xComp, xTrue));
                const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

                printFeasibilityLine({
                    method,
                    testExplanation: `${testExplanation} | case=${caseName}`,
                    label: "LU sanity check",
                    status: "success",
                    gas,
                    err: solErr,
                    res: residual,
                });

                console.log("------------------------------------------------------------");
                console.log(`Method               : ${method}`);
                console.log(`Test Explanation     : ${testExplanation} | case=${caseName}`);
                console.log(`Estimated Gas        : ${gas.toString()}`);
                console.log(`x_true               : ${fmtVec(xTrue)}`);
                console.log(`x_comp               : ${fmtVec(xComp)}`);
                console.log(`Solution Error       : ${formatScaledInt(solErr)}`);
                console.log(`Residual Norm        : ${formatScaledInt(residual)}`);
                console.log("------------------------------------------------------------");

                errArr.push(solErr);
                residualArr.push(residual);
                gasArr.push(gas);

                expect(solErr >= 0n).to.equal(true);
                expect(residual >= 0n).to.equal(true);
            }

            printSummaryBlock({
                method,
                testExplanation,
                title: "LU Sanity Check Summary",
                count: sanityCases.length,
                avgErr: avgBigInt(errArr),
                avgRes: avgBigInt(residualArr),
                avgGas: avgBigInt(gasArr),
                minGas: minBigInt(gasArr),
                maxGas: maxBigInt(gasArr),
            });
        });
    });

    describe("Section 4: Gas sensitivity to iteration count", function () {
        before(function () {
            printSectionExplanation({
                section: "Section 4: Gas sensitivity to iteration count",
                objective: "To isolate the relationship between actual iteration count and gas usage for iterative methods on a fixed linear system.",
                setup: "A single deterministic diagonally dominant system of size n = 8 is held fixed. The initial guess is fixed as the zero vector. The iteration budget is swept from 1 to 100 with a very tight tolerance.",
                metrics: "Actual iteration count, estimated gas usage, solution error, and residual norm."
            });
        });

        it("Jacobi and Gauss-Seidel: gas growth versus actual iteration count on a fixed system", async function () {
            const n = 12;
            const ITER_MIN = 1;
            const ITER_MAX = 300;

            const tag = "ITER_GROWTH_FIXED_SYSTEM";
            const A = makeDiagonallyDominantMatrixScaled(n, tag);
            const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
            const b = matVecMulScaled(A, xTrue);
            const x0 = Array<bigint>(n).fill(0n);

            const TOL_1E_30 = await qFrac(1, 1_000_000_000_000_000_000_000_000_000_000n);

            const Adata = await qMatFromScaledInts(harness, A);
            const bdata = await qVecFromScaledInts(harness, b);
            const x0data = await qVecFromScaledInts(harness, x0);

            // -------------------- Jacobi --------------------
            {
                const method = "Jacobi";
                const testExplanation =
                    "Iteration-growth experiment on a fixed deterministic system. The matrix, right-hand side, and initial guess are held constant while maxIter is swept upward, and actual iterations, gas usage, solution error, and residual are recorded until convergence saturates.";

                printTestExplanation({ method, testExplanation });

                const actualItersArr: bigint[] = [];
                const gasArr: bigint[] = [];
                const errArr: bigint[] = [];
                const residualArr: bigint[] = [];

                for (let k = ITER_MIN; k <= ITER_MAX; k++) {
                    const [out, iters] = await harness.jacobi(
                        BigInt(n),
                        Adata,
                        bdata,
                        x0data,
                        BigInt(k),
                        TOL_1E_30
                    );

                    const xComp = await scaledVec(harness, out);
                    const actualIters = asBigInt(iters);

                    const gas = await estimateJacobiGas(
                        harness,
                        BigInt(n),
                        Adata,
                        bdata,
                        x0data,
                        BigInt(k),
                        TOL_1E_30
                    );

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | maxIter=${k}`,
                        gas,
                        err,
                        res,
                        iters: actualIters,
                    });

                    actualItersArr.push(actualIters);
                    gasArr.push(gas);
                    errArr.push(err);
                    residualArr.push(res);

                    if (actualIters < BigInt(k)) {
                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : Early stop triggered because convergence was reached before maxIter.`);
                        console.log(`Stopping Point       : maxIter=${k}`);
                        console.log(`Actual Iterations    : ${actualIters.toString()}`);
                        console.log("------------------------------------------------------------");
                        break;
                    }
                }

                printSummaryBlock({
                    method,
                    testExplanation,
                    title: "Jacobi Iteration Growth Summary",
                    count: gasArr.length,
                    avgErr: avgBigInt(errArr),
                    avgRes: avgBigInt(residualArr),
                    avgGas: avgBigInt(gasArr),
                    minGas: minBigInt(gasArr),
                    maxGas: maxBigInt(gasArr),
                    avgIter: avgBigInt(actualItersArr),
                });

                expect(gasArr.length).to.be.greaterThan(0);
            }

            // -------------------- Gauss-Seidel --------------------
            {
                const method = "Gauss-Seidel";
                const testExplanation =
                    "Iteration-growth experiment on a fixed deterministic system. The matrix, right-hand side, and initial guess are held constant while maxIter is swept upward, and actual iterations, gas usage, solution error, and residual are recorded until convergence saturates.";

                printTestExplanation({ method, testExplanation });

                const actualItersArr: bigint[] = [];
                const gasArr: bigint[] = [];
                const errArr: bigint[] = [];
                const residualArr: bigint[] = [];

                for (let k = ITER_MIN; k <= ITER_MAX; k++) {
                    const [out, iters] = await harness.gaussSeidel(
                        BigInt(n),
                        Adata,
                        bdata,
                        x0data,
                        BigInt(k),
                        TOL_1E_30
                    );

                    const xComp = await scaledVec(harness, out);
                    const actualIters = asBigInt(iters);

                    const gas = await estimateGaussSeidelGas(
                        harness,
                        BigInt(n),
                        Adata,
                        bdata,
                        x0data,
                        BigInt(k),
                        TOL_1E_30
                    );

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | maxIter=${k}`,
                        gas,
                        err,
                        res,
                        iters: actualIters,
                    });

                    actualItersArr.push(actualIters);
                    gasArr.push(gas);
                    errArr.push(err);
                    residualArr.push(res);

                    if (actualIters < BigInt(k)) {
                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : Early stop triggered because convergence was reached before maxIter.`);
                        console.log(`Stopping Point       : maxIter=${k}`);
                        console.log(`Actual Iterations    : ${actualIters.toString()}`);
                        console.log("------------------------------------------------------------");
                        break;
                    }
                }

                printSummaryBlock({
                    method,
                    testExplanation,
                    title: "Gauss-Seidel Iteration Growth Summary",
                    count: gasArr.length,
                    avgErr: avgBigInt(errArr),
                    avgRes: avgBigInt(residualArr),
                    avgGas: avgBigInt(gasArr),
                    minGas: minBigInt(gasArr),
                    maxGas: maxBigInt(gasArr),
                    avgIter: avgBigInt(actualItersArr),
                });

                expect(gasArr.length).to.be.greaterThan(0);
            }

            // -------------------- Gradient Descent for Least Squares --------------------
            {
                const method = "Gradient Descent for Least Squares";
                const testExplanation =
                    "Iteration-growth experiment on a fixed deterministic least-squares system. The matrix, right-hand side, initial guess, and step size are held constant while maxIter is swept upward, and actual iterations, gas usage, solution error, and residual are recorded until convergence saturates.";

                printTestExplanation({ method, testExplanation });

                const m = 16;
                const n_ls = 8;

                const gdTag = "GDLS_ITER_GROWTH_FIXED_SYSTEM";
                const A_ls = makeFullRankRectMatrixScaled(m, n_ls, gdTag);
                const xTrue_ls = makeVectorScaled(n_ls, `${gdTag}:xtrue`, -5, 5);
                const b_ls = matVecMulScaled(A_ls, xTrue_ls);
                const x0_ls = Array<bigint>(n_ls).fill(0n);

                const Adata_ls = await qMatFromScaledInts(harness, A_ls);
                const bdata_ls = await qVecFromScaledInts(harness, b_ls);
                const x0data_ls = await qVecFromScaledInts(harness, x0_ls);

                // Step size choice
                const alpha = await qFrac(1, 20);
                const tol = TOL_1E_30;

                const actualItersArr: bigint[] = [];
                const gasArr: bigint[] = [];
                const errArr: bigint[] = [];
                const residualArr: bigint[] = [];

                for (let k = ITER_MIN; k <= ITER_MAX; k++) {
                    const [out, iters] = await harness.gradientDescentLeastSquares(
                        BigInt(m),
                        BigInt(n_ls),
                        Adata_ls,
                        bdata_ls,
                        x0data_ls,
                        alpha,
                        BigInt(k),
                        tol
                    );

                    const xComp = await scaledVec(harness, out);
                    const actualIters = asBigInt(iters);

                    const gas = await estimateGDGas(
                        harness,
                        BigInt(m),
                        BigInt(n_ls),
                        Adata_ls,
                        bdata_ls,
                        x0data_ls,
                        alpha,
                        BigInt(k),
                        tol
                    );

                    const err = vecInfNorm(subVec(xComp, xTrue_ls));
                    const res = vecInfNorm(subVec(matVecMulScaled(A_ls, xComp), b_ls));

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | maxIter=${k}`,
                        gas,
                        err,
                        res,
                        iters: actualIters,
                    });

                    actualItersArr.push(actualIters);
                    gasArr.push(gas);
                    errArr.push(err);
                    residualArr.push(res);

                    if (actualIters < BigInt(k)) {
                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : Early stop triggered because convergence was reached before maxIter.`);
                        console.log(`Stopping Point       : maxIter=${k}`);
                        console.log(`Actual Iterations    : ${actualIters.toString()}`);
                        console.log("------------------------------------------------------------");
                        break;
                    }
                }

                printSummaryBlock({
                    method,
                    testExplanation,
                    title: "Gradient Descent Least Squares Iteration Growth Summary",
                    count: gasArr.length,
                    avgErr: avgBigInt(errArr),
                    avgRes: avgBigInt(residualArr),
                    avgGas: avgBigInt(gasArr),
                    minGas: minBigInt(gasArr),
                    maxGas: maxBigInt(gasArr),
                    avgIter: avgBigInt(actualItersArr),
                });

                expect(gasArr.length).to.be.greaterThan(0);
            }
        });
    });

    describe("Section 5: Case-based structured system tests", function () {
        before(function () {
            printSectionExplanation({
                section: "Section 5: Case-based structured system tests",
                objective: "To evaluate how different structured matrix cases affect gas usage and numerical behavior of direct and iterative linear solver methods.",
                setup: "Eight structured matrix cases are defined at fixed size n = 8. Each case is tested 10 times using deterministic data generation. Direct methods are evaluated with solution or reconstruction metrics, while iterative methods are evaluated with solution error, residual, iteration count, and gas usage.",
                metrics: "Gas usage, solution error, residual norm, iteration count for iterative methods, and reconstruction error for LU decomposition."
            });
        });

        it("Gaussian Elimination: case-based gas and solution behavior", async function () {
            const method = "Gaussian Elimination";
            const testExplanation =
                "Case-based structured matrix evaluation at fixed size n = 8. Each named matrix case is tested 10 times and gas usage, solution error, and residual are recorded.";

            printTestExplanation({ method, testExplanation });

            for (const matrixCase of NAMED_CASES) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];

                for (let t = 0; t < CASE_REPEAT; t++) {
                    const tag = `GE_CASE:${matrixCase.caseName}:t=${t}`;
                    const A = matrixCase.makeMatrix(tag);
                    const xTrue = makeVectorScaled(matrixCase.n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);

                    const out = await harness.gaussianElimination(BigInt(matrixCase.n), Adata, bdata);
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateGaussianEliminationGas(harness, BigInt(matrixCase.n), Adata, bdata);

                    printCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | case=${matrixCase.caseName} | run=${t + 1}/${CASE_REPEAT}`,
                        gas,
                        err,
                        res,
                    });

                    errs.push(err);
                    residuals.push(res);
                    gases.push(gas);
                }

                printSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for case=${matrixCase.caseName}`,
                    title: `Gaussian Elimination Case Summary (${matrixCase.caseName})`,
                    count: CASE_REPEAT,
                    avgErr: avgBigInt(errs),
                    avgRes: avgBigInt(residuals),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                });
            }
        });

        it("LU Decomposition: case-based reconstruction and gas behavior", async function () {
            const method = "LU Decomposition";
            const testExplanation =
                "Case-based structured matrix evaluation at fixed size n = 8. Each named matrix case is tested 10 times and gas usage together with LU reconstruction error is recorded.";

            printTestExplanation({ method, testExplanation });

            for (const matrixCase of NAMED_CASES) {
                const reconErrs: bigint[] = [];
                const gases: bigint[] = [];

                for (let t = 0; t < CASE_REPEAT; t++) {
                    const tag = `LU_CASE:${matrixCase.caseName}:t=${t}`;
                    const A = matrixCase.makeMatrix(tag);
                    const Adata = await qMatFromScaledInts(harness, A);

                    const [Lraw, Uraw] = await harness.luDecomposition(BigInt(matrixCase.n), Adata);
                    const Lflat = await scaledVec(harness, Lraw);
                    const Uflat = await scaledVec(harness, Uraw);

                    const L = reshapeMatrix(Lflat, matrixCase.n, matrixCase.n);
                    const U = reshapeMatrix(Uflat, matrixCase.n, matrixCase.n);

                    const LU = matMulScaled(L, U);
                    const diff = matSub(LU, A);
                    const reconErr = matInfNormEntries(diff);
                    const gas = await estimateLUGas(harness, BigInt(matrixCase.n), Adata);

                    printLUCaseBlock({
                        method,
                        testExplanation: `${testExplanation} | case=${matrixCase.caseName} | run=${t + 1}/${CASE_REPEAT}`,
                        gas,
                        recon: reconErr,
                    });

                    reconErrs.push(reconErr);
                    gases.push(gas);
                }

                printLUSummaryBlock({
                    method,
                    testExplanation: `${testExplanation} | summary for case=${matrixCase.caseName}`,
                    title: `LU Decomposition Case Summary (${matrixCase.caseName})`,
                    count: CASE_REPEAT,
                    avgRecon: avgBigInt(reconErrs),
                    avgGas: avgBigInt(gases),
                    minGas: minBigInt(gases),
                    maxGas: maxBigInt(gases),
                });
            }
        });

        it("Jacobi: case-based gas, iteration, and accuracy behavior", async function () {
            const method = "Jacobi";
            const testExplanation =
                "Case-based structured matrix evaluation at fixed size n = 8. Each named matrix case is tested 10 times using zero initialization, and gas usage, solution error, residual, and iteration count are recorded.";

            printTestExplanation({ method, testExplanation });

            for (const matrixCase of NAMED_CASES) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < CASE_REPEAT; t++) {
                    const tag = `JAC_CASE:${matrixCase.caseName}:t=${t}`;
                    const A = matrixCase.makeMatrix(tag);
                    const xTrue = makeVectorScaled(matrixCase.n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(matrixCase.n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    try {
                        const [out, iters] = await harness.jacobi(
                            BigInt(matrixCase.n),
                            Adata,
                            bdata,
                            x0data,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        const xComp = await scaledVec(harness, out);

                        const err = vecInfNorm(subVec(xComp, xTrue));
                        const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                        const gas = await estimateJacobiGas(
                            harness,
                            BigInt(matrixCase.n),
                            Adata,
                            bdata,
                            x0data,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        printCaseBlock({
                            method,
                            testExplanation: `${testExplanation} | case=${matrixCase.caseName} | run=${t + 1}/${CASE_REPEAT}`,
                            gas,
                            err,
                            res,
                            iters: asBigInt(iters),
                        });

                        errs.push(err);
                        residuals.push(res);
                        gases.push(gas);
                        itersArr.push(asBigInt(iters));
                    } catch (e: any) {
                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : ${testExplanation} | case=${matrixCase.caseName} | run=${t + 1}/${CASE_REPEAT}`);
                        console.log(`Status               : failed`);
                        console.log(`Reason               : ${e?.shortMessage ?? e?.message ?? "unknown error"}`);
                        console.log("------------------------------------------------------------");
                        continue;
                    }
                }

                if (gases.length > 0) {
                    printSummaryBlock({
                        method,
                        testExplanation: `${testExplanation} | summary for case=${matrixCase.caseName}`,
                        title: `Jacobi Case Summary (${matrixCase.caseName})`,
                        count: gases.length,
                        avgErr: avgBigInt(errs),
                        avgRes: avgBigInt(residuals),
                        avgGas: avgBigInt(gases),
                        minGas: minBigInt(gases),
                        maxGas: maxBigInt(gases),
                        avgIter: avgBigInt(itersArr),
                    });
                } else {
                    console.log("============================================================");
                    console.log(`Jacobi Case Summary (${matrixCase.caseName})`);
                    console.log("============================================================");
                    console.log(`Method               : ${method}`);
                    console.log(`Test Explanation     : ${testExplanation} | summary for case=${matrixCase.caseName}`);
                    console.log("Test Count           : 0");
                    console.log("Status               : all runs failed");
                    console.log("============================================================");
                }
            }
        });

        it("Gauss-Seidel: case-based gas, iteration, and accuracy behavior", async function () {
            const method = "Gauss-Seidel";
            const testExplanation =
                "Case-based structured matrix evaluation at fixed size n = 8. Each named matrix case is tested 10 times using zero initialization, and gas usage, solution error, residual, and iteration count are recorded.";

            printTestExplanation({ method, testExplanation });

            for (const matrixCase of NAMED_CASES) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                let failedRuns = 0;

                for (let t = 0; t < CASE_REPEAT; t++) {
                    const tag = `GS_CASE:${matrixCase.caseName}:t=${t}`;
                    const caseRunExplanation = `${testExplanation} | case=${matrixCase.caseName} | run=${t + 1}/${CASE_REPEAT}`;

                    try {
                        const A = matrixCase.makeMatrix(tag);
                        const xTrue = makeVectorScaled(matrixCase.n, `${tag}:xtrue`, -5, 5);
                        const b = matVecMulScaled(A, xTrue);
                        const x0 = Array<bigint>(matrixCase.n).fill(0n);

                        const Adata = await qMatFromScaledInts(harness, A);
                        const bdata = await qVecFromScaledInts(harness, b);
                        const x0data = await qVecFromScaledInts(harness, x0);

                        const [out, iters] = await harness.gaussSeidel(
                            BigInt(matrixCase.n),
                            Adata,
                            bdata,
                            x0data,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        const xComp = await scaledVec(harness, out);

                        const err = vecInfNorm(subVec(xComp, xTrue));
                        const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                        const gas = await estimateGaussSeidelGas(
                            harness,
                            BigInt(matrixCase.n),
                            Adata,
                            bdata,
                            x0data,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        printCaseBlock({
                            method,
                            testExplanation: caseRunExplanation,
                            gas,
                            err,
                            res,
                            iters: asBigInt(iters),
                        });

                        errs.push(err);
                        residuals.push(res);
                        gases.push(gas);
                        itersArr.push(asBigInt(iters));
                    } catch (e: any) {
                        failedRuns++;

                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : ${caseRunExplanation}`);
                        console.log(`Status               : failed`);
                        console.log(`Reason               : ${e?.message ?? String(e)}`);
                        console.log("------------------------------------------------------------");
                    }
                }

                if (gases.length > 0) {
                    printSummaryBlock({
                        method,
                        testExplanation: `${testExplanation} | summary for case=${matrixCase.caseName}`,
                        title: `Gauss-Seidel Case Summary (${matrixCase.caseName})`,
                        count: gases.length,
                        avgErr: avgBigInt(errs),
                        avgRes: avgBigInt(residuals),
                        avgGas: avgBigInt(gases),
                        minGas: minBigInt(gases),
                        maxGas: maxBigInt(gases),
                        avgIter: avgBigInt(itersArr),
                    });
                } else {
                    console.log("============================================================");
                    console.log(`Gauss-Seidel Case Summary (${matrixCase.caseName})`);
                    console.log("============================================================");
                    console.log(`Method               : ${method}`);
                    console.log(`Test Explanation     : ${testExplanation} | summary for case=${matrixCase.caseName}`);
                    console.log(`Test Count           : 0`);
                    console.log(`Failed Runs          : ${failedRuns}`);
                    console.log(`Status               : no successful run`);
                    console.log("============================================================");
                }
            }
        });

        it("Gradient Descent for Least Squares: case-based gas, iteration, and accuracy behavior", async function () {
            const method = "Gradient Descent for Least Squares";
            const testExplanation =
                "Case-based structured least-squares evaluation. Each named rectangular matrix case is tested multiple times using zero initialization, and gas usage, solution error, residual, and iteration count are recorded.";

            printTestExplanation({ method, testExplanation });

            const alpha = await qFrac(1, 30);

            for (const matrixCase of NAMED_GDLS_CASES) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];
                let failedRuns = 0;

                for (let t = 0; t < CASE_REPEAT; t++) {
                    const tag = `GDLS_CASE:${matrixCase.caseName}:t=${t}`;
                    const caseRunExplanation = `${testExplanation} | case=${matrixCase.caseName} | run=${t + 1}/${CASE_REPEAT}`;

                    try {
                        const A = matrixCase.makeMatrix(tag);
                        const xTrue = makeVectorScaled(matrixCase.n, `${tag}:xtrue`, -5, 5);
                        const b = matVecMulScaled(A, xTrue);
                        const x0 = Array<bigint>(matrixCase.n).fill(0n);

                        const Adata = await qMatFromScaledInts(harness, A);
                        const bdata = await qVecFromScaledInts(harness, b);
                        const x0data = await qVecFromScaledInts(harness, x0);

                        const [out, iters] = await harness.gradientDescentLeastSquares(
                            BigInt(matrixCase.m),
                            BigInt(matrixCase.n),
                            Adata,
                            bdata,
                            x0data,
                            alpha,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        const xComp = await scaledVec(harness, out);

                        const err = vecInfNorm(subVec(xComp, xTrue));
                        const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                        const gas = await estimateGDGas(
                            harness,
                            BigInt(matrixCase.m),
                            BigInt(matrixCase.n),
                            Adata,
                            bdata,
                            x0data,
                            alpha,
                            MAX_ITER,
                            TOL_1E_15
                        );

                        printCaseBlock({
                            method,
                            testExplanation: caseRunExplanation,
                            gas,
                            err,
                            res,
                            iters: asBigInt(iters),
                        });

                        errs.push(err);
                        residuals.push(res);
                        gases.push(gas);
                        itersArr.push(asBigInt(iters));
                    } catch (e: any) {
                        failedRuns++;

                        console.log("------------------------------------------------------------");
                        console.log(`Method               : ${method}`);
                        console.log(`Test Explanation     : ${caseRunExplanation}`);
                        console.log(`Status               : failed`);
                        console.log(`Reason               : ${e?.shortMessage ?? e?.message ?? "unknown error"}`);
                        console.log("------------------------------------------------------------");
                    }
                }

                if (gases.length > 0) {
                    printSummaryBlock({
                        method,
                        testExplanation: `${testExplanation} | summary for case=${matrixCase.caseName}`,
                        title: `Gradient Descent Least Squares Case Summary (${matrixCase.caseName})`,
                        count: gases.length,
                        avgErr: avgBigInt(errs),
                        avgRes: avgBigInt(residuals),
                        avgGas: avgBigInt(gases),
                        minGas: minBigInt(gases),
                        maxGas: maxBigInt(gases),
                        avgIter: avgBigInt(itersArr),
                    });
                } else {
                    console.log("============================================================");
                    console.log(`Gradient Descent Least Squares Case Summary (${matrixCase.caseName})`);
                    console.log("============================================================");
                    console.log(`Method               : ${method}`);
                    console.log(`Test Explanation     : ${testExplanation} | summary for case=${matrixCase.caseName}`);
                    console.log(`Test Count           : 0`);
                    console.log(`Failed Runs          : ${failedRuns}`);
                    console.log(`Status               : no successful run`);
                    console.log("============================================================");
                }
            }
        });
    });
});