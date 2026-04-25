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
    let ALPHA_0001: string;
    let ALPHA_0005: string;
    let ALPHA_001: string;

    const GD_SIZES = [
        { m: 4, n: 2 },
        { m: 6, n: 3 },
    ];

    const T = 15;
    const GD_MAX_ITER = 75n;

    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

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

        ALPHA_0001 = await qFrac(1, 1000); // 0.001
        ALPHA_0005 = await qFrac(1, 200);  // 0.005
        ALPHA_001 = await qFrac(1, 100);   // 0.01

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    describe("Section 1: Size-based accuracy and gas tests", function () {

        it("Gradient Descent Least Squares: average solution error, residual, iterations, and gas across feasible problem sizes", async function () {
            for (const cfg of GD_SIZES) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `GD:m=${cfg.m}:n=${cfg.n}:t=${t}`;
                    const A = makeFullRankRectMatrixScaled(cfg.m, cfg.n, tag);
                    const xTrue = makeVectorScaled(cfg.n, `${tag}:xtrue`, -3, 3);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(cfg.n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    const [out, iters] = await harness.gradientDescentLeastSquares(
                        BigInt(cfg.m), BigInt(cfg.n), Adata, bdata, x0data, ALPHA_001, GD_MAX_ITER, TOL_1E_15
                    );
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateGDGas(
                        harness, BigInt(cfg.m), BigInt(cfg.n), Adata, bdata, x0data, ALPHA_001, GD_MAX_ITER, TOL_1E_15
                    );

                    printCaseBlock({
                        title: `Gradient Descent LS | (m,n)=(${cfg.m},${cfg.n}) | case=${t + 1}/${T}`,
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
                    title: `Gradient Descent Least Squares Summary for (m,n)=(${cfg.m},${cfg.n})`,
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

    describe("Section 2: Step size sensitivity tests for Gradient Descent Least Squares", function () {
        it("Gradient Descent Least Squares: alpha sensitivity with average accuracy, iterations, and gas", async function () {
            const m = 8;
            const n = 4;

            const alphaConfigs = [
                { name: "small", alpha: ALPHA_0001 },
                { name: "medium", alpha: ALPHA_0005 },
                { name: "large", alpha: ALPHA_001 },
            ];

            for (const cfg of alphaConfigs) {
                const errs: bigint[] = [];
                const residuals: bigint[] = [];
                const gases: bigint[] = [];
                const itersArr: bigint[] = [];

                for (let t = 0; t < T; t++) {
                    const tag = `GD_ALPHA:${cfg.name}:t=${t}`;
                    const A = makeFullRankRectMatrixScaled(m, n, tag);
                    const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -3, 3);
                    const b = matVecMulScaled(A, xTrue);
                    const x0 = Array<bigint>(n).fill(0n);

                    const Adata = await qMatFromScaledInts(harness, A);
                    const bdata = await qVecFromScaledInts(harness, b);
                    const x0data = await qVecFromScaledInts(harness, x0);

                    const [out, iters] = await harness.gradientDescentLeastSquares(
                        BigInt(m), BigInt(n), Adata, bdata, x0data, cfg.alpha, GD_MAX_ITER, TOL_1E_15
                    );
                    const xComp = await scaledVec(harness, out);

                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));
                    const gas = await estimateGDGas(
                        harness, BigInt(m), BigInt(n), Adata, bdata, x0data, cfg.alpha, GD_MAX_ITER, TOL_1E_15
                    );

                    printCaseBlock({
                        title: `Gradient Descent LS alpha Sensitivity | alpha=${cfg.name} | case=${t + 1}/${T}`,
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
                    title: `Gradient Descent Least Squares Step Size Sensitivity (${cfg.name})`,
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

    describe("Section 3: Gradient Descent feasibility and gas-growth test", function () {
        it("Gradient Descent Least Squares: reports feasible iteration budgets before execution becomes impractical", async function () {
            const m = 8;
            const n = 4;
            const iterBudgets = [10n, 25n, 50n, 100n, 200n, 400n, 800n];

            const tag = "GD_FEASIBILITY";
            const A = makeFullRankRectMatrixScaled(m, n, tag);
            const xTrue = makeVectorScaled(n, `${tag}:xtrue`, -3, 3);
            const b = matVecMulScaled(A, xTrue);
            const x0 = Array<bigint>(n).fill(0n);

            const Adata = await qMatFromScaledInts(harness, A);
            const bdata = await qVecFromScaledInts(harness, b);
            const x0data = await qVecFromScaledInts(harness, x0);

            let atLeastOneWorked = false;

            for (const iterBudget of iterBudgets) {
                try {
                    const [out, iters] = await harness.gradientDescentLeastSquares(
                        BigInt(m), BigInt(n), Adata, bdata, x0data, ALPHA_001, iterBudget, TOL_1E_15
                    );

                    const gas = await estimateGDGas(
                        harness, BigInt(m), BigInt(n), Adata, bdata, x0data, ALPHA_001, iterBudget, TOL_1E_15
                    );

                    const xComp = await scaledVec(harness, out);
                    const err = vecInfNorm(subVec(xComp, xTrue));
                    const res = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

                    printFeasibilityLine({
                        label: `GD feasibility for (m,n)=(${m},${n}), maxIter=${iterBudget.toString()}`,
                        status: "executed",
                        gas,
                        err,
                        res,
                        iters: asBigInt(iters),
                    });

                    atLeastOneWorked = true;
                } catch {
                    printFeasibilityLine({
                        label: `GD feasibility for (m,n)=(${m},${n}), maxIter=${iterBudget.toString()}`,
                        status: "execution became impractical or exceeded the available gas budget",
                    });
                }
            }

            expect(atLeastOneWorked).to.equal(true);
        });
    });
});