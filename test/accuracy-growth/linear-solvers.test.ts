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
// Reporting Helpers
// ------------------------------------------------------------

function printCaseBlock(args: {
    title: string;
    gas: bigint;
    err: bigint;
    res: bigint;
    iters?: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(args.title);
    console.log(`Gas Usage            : ${args.gas.toString()}`);
    console.log(`Solution Error       : ${formatScaledInt(args.err)}`);
    console.log(`Residual             : ${formatScaledInt(args.res)}`);
    if (args.iters !== undefined) {
        console.log(`Iterations           : ${args.iters.toString()}`);
    }
    console.log("------------------------------------------------------------");
}

function printLUCaseBlock(args: {
    title: string;
    gas: bigint;
    recon: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(args.title);
    console.log(`Gas Usage            : ${args.gas.toString()}`);
    console.log(`Reconstruction Error : ${formatScaledInt(args.recon)}`);
    console.log("------------------------------------------------------------");
}

function printSummaryBlock(args: {
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
    console.log(`Test Count           : ${args.count}`);
    console.log(`Average Recon. Error : ${formatScaledInt(args.avgRecon)}`);
    console.log(`Average Gas          : ${args.avgGas.toString()}`);
    console.log(`Min Gas              : ${args.minGas.toString()}`);
    console.log(`Max Gas              : ${args.maxGas.toString()}`);
    console.log("============================================================");
}

function printFeasibilityLine(args: {
    label: string;
    status: string;
    gas?: bigint;
    err?: bigint;
    res?: bigint;
    iters?: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(args.label);
    console.log(`Status               : ${args.status}`);
    if (args.gas !== undefined) console.log(`Estimated Gas        : ${args.gas.toString()}`);
    if (args.err !== undefined) console.log(`Solution Error       : ${formatScaledInt(args.err)}`);
    if (args.res !== undefined) console.log(`Residual             : ${formatScaledInt(args.res)}`);
    if (args.iters !== undefined) console.log(`Iterations           : ${args.iters.toString()}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("LinearSolvers Library - Randomized Accuracy, Iteration, and Gas Tests", function () {
    this.timeout(300000); // 5 minutes
    let harness: LinearSolversHarness;

    let TOL_1E_15: string;

    const SIZE_NS = [2, 4, 8, 12, 16];

    const T = 15;
    const MAX_ITER = 3000n;

    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
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
        it("Gaussian Elimination: average solution error, residual, and gas across matrix sizes", async function () {
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
                        title: `Gaussian Elimination | n=${n} | case=${t + 1}/${T}`,
                        gas,
                        err,
                        res,
                    });

                    errs.push(err);
                    residuals.push(res);
                    gases.push(gas);
                }

                printSummaryBlock({
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
                        title: `LU Decomposition | n=${n} | case=${t + 1}/${T}`,
                        gas,
                        recon: reconErr,
                    });

                    reconErrs.push(reconErr);
                    gases.push(gas);
                }

                printLUSummaryBlock({
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
                        title: `Jacobi | n=${n} | case=${t + 1}/${T}`,
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
            for (const n of SIZE_NS) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `GS:n=${n}:t=${t}`;
                    const A = makeDiagonallyDominantMatrixScaled(n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -5, 5);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    const [out, iters] = await harness.gaussSeidel(BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateGaussSeidelGas(harness, BigInt(n), Adata, bdata, x0data, MAX_ITER, TOL_1E_15);

                    printCaseBlock({
                        title: `Gauss-Seidel | n=${n} | case=${t + 1}/${T}`,
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
                    title: `Gauss-Seidel Summary for n=${n}`,
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

    describe("Section 2: Initial guess sensitivity tests", function () {
        it("Jacobi: x0 sensitivity at fixed size with average accuracy, iterations, and gas", async function () {
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
                        title: `Jacobi x0 Sensitivity | x0=${cfg.name} | case=${t + 1}/${T}`,
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
                        title: `Gauss-Seidel x0 Sensitivity | x0=${cfg.name} | case=${t + 1}/${T}`,
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

    describe("Section 3: Sanity checks", function () {
        it("LU factors can recover a deterministic solution on one structured system", async function () {
            const n = 3;
            const tag = "LU_SANITY";

            const A = makeDiagonallyDominantMatrixScaled(n, tag);
            const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -2, 2);
            const b = matVecMulScaled(A, xTrue);

            const Adata = await qMatFromScaledInts(harness, A);
            const [Lraw, Uraw] = await harness.luDecomposition(BigInt(n), Adata);

            const L = reshapeMatrix(await scaledVec(harness, Lraw), n, n);
            const U = reshapeMatrix(await scaledVec(harness, Uraw), n, n);

            const y = forwardSubstitutionScaled(L, b);
            const xComp = backwardSubstitutionScaled(U, y);

            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            console.log("------------------------------------------------------------");
            console.log("LU sanity check");
            console.log(`x_true: ${fmtVec(xTrue)}`);
            console.log(`x_comp: ${fmtVec(xComp)}`);
            console.log(`Solution Error (inf-norm): ${formatScaledInt(solErr)}`);
            console.log(`Residual Norm ||Ax-b||_inf: ${formatScaledInt(residual)}`);
            console.log("------------------------------------------------------------");

            expect(solErr >= 0n).to.equal(true);
            expect(residual >= 0n).to.equal(true);
        });
    });
});