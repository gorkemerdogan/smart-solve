// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
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
// Helpers
// ------------------------------------------------------------

let SCALE_DECIMALS = 32n;
let SCALE = 10n ** SCALE_DECIMALS;

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

async function outScaled(harness: LinearSolversHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

async function qScaled(harness: LinearSolversHarness, scaledValue: bigint): Promise<string> {
    return await harness.fromFloat(asBigInt(scaledValue));
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

function matVecMulScaled(A: bigint[][], x: bigint[]): bigint[] {
    return A.map(row => {
        let sum = 0n;
        for (let i = 0; i < row.length; i++) {
            sum += (row[i] * x[i]) / SCALE;
        }
        return sum;
    });
}

function fmtVec(v: bigint[]): string {
    return `[${v.map(formatScaledInt).join(", ")}]`;
}

function fmtHexVec(v: string[]): string {
    return `[${v.join(", ")}]`;
}

async function qVecFromScaled(harness: LinearSolversHarness, vals: bigint[]): Promise<string[]> {
    return Promise.all(vals.map(v => qScaled(harness, v)));
}

/**
 * Forward substitution in local fixed-point arithmetic.
 * This is used only to validate the numerical quality of returned LU factors
 * on carefully chosen benchmarks.
 */
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

/**
 * Backward substitution in local fixed-point arithmetic.
 * This remains an off-chain validator, not a replacement for on-chain quad logic.
 */
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

function printSolverBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expectedHex: string;
    outputHex: string;
    expectedX: string;
    outputX: string;
    solutionError: string;
    residualNorm: string;
    iterations?: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected Output (hex): ${args.expectedHex}`);
    console.log(`Output (hex): ${args.outputHex}`);
    console.log(`Expected Output (dec): ${args.expectedX}`);
    console.log(`Output (dec): ${args.outputX}`);
    console.log(`Solution Error (inf-norm): ${args.solutionError}`);
    console.log(`Residual Norm ||Ax-b||_inf: ${args.residualNorm}`);
    if (args.iterations !== undefined) {
        console.log(`Iterations: ${args.iterations}`);
    }
    console.log("------------------------------------------------------------");
}

function printLUBlock(args: {
    t: string;
    explanation: string;
    input: string;
    expectedHex: string;
    lHex: string;
    uHex: string;
    L: string;
    U: string;
    reconstructionError: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log("Method: LU");
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected Output (hex): ${args.expectedHex}`);
    console.log(`L (hex): ${args.lHex}`);
    console.log(`U (hex): ${args.uHex}`);
    console.log(`L: ${args.L}`);
    console.log(`U: ${args.U}`);
    console.log(`Reconstruction Error ||LU-A||_inf: ${args.reconstructionError}`);
    console.log("------------------------------------------------------------");
}

function printComparisonBlock(args: {
    title: string;
    looseErr: bigint;
    tightErr: bigint;
    looseRes: bigint;
    tightRes: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(args.title);
    console.log(`Loose Solution Error (dec): ${formatScaledInt(args.looseErr)}`);
    console.log(`Tight Solution Error (dec): ${formatScaledInt(args.tightErr)}`);
    console.log(`Loose Residual (dec): ${formatScaledInt(args.looseRes)}`);
    console.log(`Tight Residual (dec): ${formatScaledInt(args.tightRes)}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("LinearSolvers Library - Decimal Accuracy Behavior Tests", function () {
    let harness: LinearSolversHarness;

    let TOL_1E_9: string;
    let TOL_1E_15: string;

    let ALPHA_01: string;
    let ALPHA_025: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    async function qVecFromInts(values: Array<number | bigint>): Promise<string[]> {
        return Promise.all(values.map(v => qInt(v)));
    }

    async function qVecFromFracs(values: Array<[number | bigint, number | bigint]>): Promise<string[]> {
        return Promise.all(values.map(([n, d]) => qFrac(n, d)));
    }

    async function scaledVec(values: string[]): Promise<bigint[]> {
        return Promise.all(values.map(v => outScaled(harness, v)));
    }

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const MatrixMasterFactory = await ethers.getContractFactory("MatrixMaster");
        const matrixMaster = await MatrixMasterFactory.deploy();
        await matrixMaster.waitForDeployment();

        const HF = await ethers.getContractFactory("LinearSolversHarness", {
            libraries: {
                MathLib: await mathlib.getAddress(),
            },
        });

        harness = (await HF.deploy()) as unknown as LinearSolversHarness;

        TOL_1E_9 = await qFrac(1, 1_000_000_000);
        TOL_1E_15 = await qFrac(1, 1_000_000_000_000_000);

        ALPHA_01 = await qFrac(1, 10);
        ALPHA_025 = await qFrac(1, 4);

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    // ------------------------------------------------------------
    // Section 1: Direct exactness benchmarks
    // ------------------------------------------------------------

    describe("Section 1: Direct exactness benchmarks", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: Gaussian elimination solves a 2x2 exact integer system`, async function () {
            const A = [
                [2n * SCALE, 1n * SCALE],
                [1n * SCALE, 3n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE];
            const b = [4n * SCALE, 7n * SCALE];

            const Adata = await qVecFromInts([2, 1, 1, 3]);
            const bdata = await qVecFromInts([4, 7]);

            const out = await harness.gaussianElimination(2n, Adata, bdata);
            const xComp = await scaledVec(out);

            const expectedHex = await qVecFromInts([1, 2]);
            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `1.${testNo}`,
                method: "Gaussian Elimination",
                explanation: "Exact integer benchmark.",
                input: "A=[[2,1],[1,3]], b=[4,7]",
                expectedHex: fmtHexVec(expectedHex),
                outputHex: fmtHexVec(out),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
            });

            expect(solErr).to.equal(0n);
            expect(residual).to.equal(0n);
        });

        it(`Test 1.${++testNo}: Gaussian elimination solves an exact upper-triangular 3x3 system`, async function () {
            const A = [
                [2n * SCALE, 1n * SCALE, 0n],
                [0n, 3n * SCALE, 1n * SCALE],
                [0n, 0n, 4n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE, 3n * SCALE];
            const b = [4n * SCALE, 9n * SCALE, 12n * SCALE];

            const Adata = await qVecFromInts([2, 1, 0, 0, 3, 1, 0, 0, 4]);
            const bdata = await qVecFromInts([4, 9, 12]);

            const out = await harness.gaussianElimination(3n, Adata, bdata);
            const xComp = await scaledVec(out);

            const expectedHex = await qVecFromInts([1, 2, 3]);
            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `1.${testNo}`,
                method: "Gaussian Elimination",
                explanation: "Exact triangular benchmark.",
                input: "A=[[2,1,0],[0,3,1],[0,0,4]], b=[4,9,12]",
                expectedHex: fmtHexVec(expectedHex),
                outputHex: fmtHexVec(out),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
            });

            expect(solErr).to.equal(0n);
            expect(residual).to.equal(0n);
        });

        it(`Test 1.${++testNo}: Gaussian elimination handles a decimal 2x2 system accurately`, async function () {
            // A = [[0.5, 0.25],
            //      [0.125, 0.75]]
            // x = [2, 4]
            // b = [2, 3.25]
            const A = [
                [SCALE / 2n, SCALE / 4n],
                [SCALE / 8n, (3n * SCALE) / 4n],
            ];
            const xTrue = [2n * SCALE, 4n * SCALE];
            const b = [2n * SCALE, (13n * SCALE) / 4n];

            const Adata = await qVecFromFracs([
                [1, 2], [1, 4],
                [1, 8], [3, 4],
            ]);
            const bdata = await qVecFromFracs([
                [2, 1],
                [13, 4],
            ]);

            const out = await harness.gaussianElimination(2n, Adata, bdata);
            const xComp = await scaledVec(out);

            const expectedHex = await qVecFromInts([2, 4]);
            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `1.${testNo}`,
                method: "Gaussian Elimination",
                explanation: "Decimal benchmark with binary-exact fractional coefficients.",
                input: "A=[[0.5,0.25],[0.125,0.75]], b=[2,3.25]",
                expectedHex: fmtHexVec(expectedHex),
                outputHex: fmtHexVec(out),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
            });

            expect(residual <= 1n).to.equal(true);
            expect(solErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: LU decomposition reconstructs a simple exact-factor 2x2 matrix`, async function () {
            const A = [
                [2n * SCALE, 1n * SCALE],
                [4n * SCALE, 3n * SCALE],
            ];

            const Adata = await qVecFromInts([2, 1, 4, 3]);
            const [Lraw, Uraw] = await harness.luDecomposition(2n, Adata);

            const L = await scaledVec(Lraw);
            const U = await scaledVec(Uraw);

            const Lm = [
                [L[0], L[1]],
                [L[2], L[3]],
            ];
            const Um = [
                [U[0], U[1]],
                [U[2], U[3]],
            ];

            const LU = [
                [
                    (Lm[0][0] * Um[0][0]) / SCALE + (Lm[0][1] * Um[1][0]) / SCALE,
                    (Lm[0][0] * Um[0][1]) / SCALE + (Lm[0][1] * Um[1][1]) / SCALE,
                ],
                [
                    (Lm[1][0] * Um[0][0]) / SCALE + (Lm[1][1] * Um[1][0]) / SCALE,
                    (Lm[1][0] * Um[0][1]) / SCALE + (Lm[1][1] * Um[1][1]) / SCALE,
                ],
            ];

            const diff = [
                [LU[0][0] - A[0][0], LU[0][1] - A[0][1]],
                [LU[1][0] - A[1][0], LU[1][1] - A[1][1]],
            ];
            const reconErr = vecInfNorm([diff[0][0], diff[0][1], diff[1][0], diff[1][1]]);

            printLUBlock({
                t: `1.${testNo}`,
                explanation: "Exact-factor reconstruction benchmark.",
                input: "A=[[2,1],[4,3]]",
                expectedHex: "N/A",
                lHex: fmtHexVec(Lraw),
                uHex: fmtHexVec(Uraw),
                L: fmtVec(L),
                U: fmtVec(U),
                reconstructionError: formatScaledInt(reconErr),
            });

            expect(reconErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: LU factors recover the exact solution on an exact-factor 3x3 system`, async function () {
            const A = [
                [2n * SCALE, 1n * SCALE, 0n],
                [4n * SCALE, 5n * SCALE, 1n * SCALE],
                [2n * SCALE, 10n * SCALE, 7n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE, 3n * SCALE];
            const b = [4n * SCALE, 17n * SCALE, 43n * SCALE];

            const Adata = await qVecFromInts([2, 1, 0, 4, 5, 1, 2, 10, 7]);
            const [Lraw, Uraw] = await harness.luDecomposition(3n, Adata);

            const Lflat = await scaledVec(Lraw);
            const Uflat = await scaledVec(Uraw);

            const L = [
                [Lflat[0], Lflat[1], Lflat[2]],
                [Lflat[3], Lflat[4], Lflat[5]],
                [Lflat[6], Lflat[7], Lflat[8]],
            ];
            const U = [
                [Uflat[0], Uflat[1], Uflat[2]],
                [Uflat[3], Uflat[4], Uflat[5]],
                [Uflat[6], Uflat[7], Uflat[8]],
            ];

            const y = forwardSubstitutionScaled(L, b);
            const xComp = backwardSubstitutionScaled(U, y);

            const expectedHex = await qVecFromInts([1, 2, 3]);
            const outputHex = await qVecFromScaled(harness, xComp);

            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `1.${testNo}`,
                method: "LU Solve",
                explanation: "Exact-factor solve benchmark.",
                input: "A=[[2,1,0],[4,5,1],[2,10,7]], b=[4,17,43]",
                expectedHex: fmtHexVec(expectedHex),
                outputHex: fmtHexVec(outputHex),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
            });

            console.log(`Intermediate y: ${fmtVec(y)}`);

            expect(solErr).to.equal(0n);
            expect(residual).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Iterative decimal behavior
    // ------------------------------------------------------------

    describe("Section 2: Iterative decimal behavior on a strongly diagonally dominant system", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: Jacobi and Gauss-Seidel improve decimal solution/residual relative to the initial guess`, async function () {
            const A = [
                [10n * SCALE, 1n * SCALE],
                [1n * SCALE, 10n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE];
            const b = [12n * SCALE, 21n * SCALE];
            const x0 = [0n, 0n];

            const Adata = await qVecFromInts([10, 1, 1, 10]);
            const bdata = await qVecFromInts([12, 21]);
            const x0data = await qVecFromInts([0, 0]);

            const initErr = vecInfNorm(subVec(x0, xTrue));
            const initRes = vecInfNorm(subVec(matVecMulScaled(A, x0), b));

            const [xJacLooseRaw, itJacLoose] = await harness.jacobi(2n, Adata, bdata, x0data, 2000n, TOL_1E_9);
            const [xJacTightRaw, itJacTight] = await harness.jacobi(2n, Adata, bdata, x0data, 2000n, TOL_1E_15);

            const [xGSLooseRaw, itGSLoose] = await harness.gaussSeidel(2n, Adata, bdata, x0data, 2000n, TOL_1E_9);
            const [xGSTightRaw, itGSTight] = await harness.gaussSeidel(2n, Adata, bdata, x0data, 2000n, TOL_1E_15);

            const xJacLoose = await scaledVec(xJacLooseRaw);
            const xJacTight = await scaledVec(xJacTightRaw);
            const xGSLoose = await scaledVec(xGSLooseRaw);
            const xGSTight = await scaledVec(xGSTightRaw);

            const jacLooseErr = vecInfNorm(subVec(xJacLoose, xTrue));
            const jacTightErr = vecInfNorm(subVec(xJacTight, xTrue));
            const gsLooseErr = vecInfNorm(subVec(xGSLoose, xTrue));
            const gsTightErr = vecInfNorm(subVec(xGSTight, xTrue));

            const jacLooseRes = vecInfNorm(subVec(matVecMulScaled(A, xJacLoose), b));
            const jacTightRes = vecInfNorm(subVec(matVecMulScaled(A, xJacTight), b));
            const gsLooseRes = vecInfNorm(subVec(matVecMulScaled(A, xGSLoose), b));
            const gsTightRes = vecInfNorm(subVec(matVecMulScaled(A, xGSTight), b));

            printSolverBlock({
                t: `2.${testNo}.1`,
                method: "Jacobi (loose)",
                explanation: "Decimal behavior with tol=1e-9.",
                input: "A=[[10,1],[1,10]], b=[12,21], x0=[0,0], tol=1e-9",
                expectedHex: fmtHexVec(await qVecFromInts([1, 2])),
                outputHex: fmtHexVec(xJacLooseRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xJacLoose),
                solutionError: formatScaledInt(jacLooseErr),
                residualNorm: formatScaledInt(jacLooseRes),
                iterations: itJacLoose.toString(),
            });

            printSolverBlock({
                t: `2.${testNo}.2`,
                method: "Jacobi (tight)",
                explanation: "Decimal behavior with tol=1e-15.",
                input: "A=[[10,1],[1,10]], b=[12,21], x0=[0,0], tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromInts([1, 2])),
                outputHex: fmtHexVec(xJacTightRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xJacTight),
                solutionError: formatScaledInt(jacTightErr),
                residualNorm: formatScaledInt(jacTightRes),
                iterations: itJacTight.toString(),
            });

            printSolverBlock({
                t: `2.${testNo}.3`,
                method: "Gauss-Seidel (loose)",
                explanation: "Decimal behavior with tol=1e-9.",
                input: "A=[[10,1],[1,10]], b=[12,21], x0=[0,0], tol=1e-9",
                expectedHex: fmtHexVec(await qVecFromInts([1, 2])),
                outputHex: fmtHexVec(xGSLooseRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xGSLoose),
                solutionError: formatScaledInt(gsLooseErr),
                residualNorm: formatScaledInt(gsLooseRes),
                iterations: itGSLoose.toString(),
            });

            printSolverBlock({
                t: `2.${testNo}.4`,
                method: "Gauss-Seidel (tight)",
                explanation: "Decimal behavior with tol=1e-15.",
                input: "A=[[10,1],[1,10]], b=[12,21], x0=[0,0], tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromInts([1, 2])),
                outputHex: fmtHexVec(xGSTightRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xGSTight),
                solutionError: formatScaledInt(gsTightErr),
                residualNorm: formatScaledInt(gsTightRes),
                iterations: itGSTight.toString(),
            });

            printComparisonBlock({
                title: "Decimal comparison: Jacobi",
                looseErr: jacLooseErr,
                tightErr: jacTightErr,
                looseRes: jacLooseRes,
                tightRes: jacTightRes,
            });

            printComparisonBlock({
                title: "Decimal comparison: Gauss-Seidel",
                looseErr: gsLooseErr,
                tightErr: gsTightErr,
                looseRes: gsLooseRes,
                tightRes: gsTightRes,
            });

            expect(jacLooseErr < initErr).to.equal(true);
            expect(jacTightErr < initErr).to.equal(true);
            expect(gsLooseErr < initErr).to.equal(true);
            expect(gsTightErr < initErr).to.equal(true);

            expect(jacLooseRes < initRes).to.equal(true);
            expect(jacTightRes < initRes).to.equal(true);
            expect(gsLooseRes < initRes).to.equal(true);
            expect(gsTightRes < initRes).to.equal(true);
        });

        it(`Test 2.${++testNo}: Jacobi and Gauss-Seidel improve decimal solution/residual on a decimal matrix`, async function () {
            // A = [[8, 0.5],
            //      [0.25, 9]]
            // x = [1.5, 2.5]
            // b = [13.25, 22.875]
            const A = [
                [8n * SCALE, SCALE / 2n],
                [SCALE / 4n, 9n * SCALE],
            ];
            const xTrue = [(3n * SCALE) / 2n, (5n * SCALE) / 2n];
            const b = [(53n * SCALE) / 4n, (183n * SCALE) / 8n];
            const x0 = [0n, 0n];

            const Adata = await qVecFromFracs([
                [8, 1], [1, 2],
                [1, 4], [9, 1],
            ]);
            const bdata = await qVecFromFracs([
                [53, 4],
                [183, 8],
            ]);
            const x0data = await qVecFromInts([0, 0]);

            const initErr = vecInfNorm(subVec(x0, xTrue));
            const initRes = vecInfNorm(subVec(matVecMulScaled(A, x0), b));

            const [xJacRaw, itJac] = await harness.jacobi(2n, Adata, bdata, x0data, 3000n, TOL_1E_15);
            const [xGSRaw, itGS] = await harness.gaussSeidel(2n, Adata, bdata, x0data, 3000n, TOL_1E_15);

            const xJac = await scaledVec(xJacRaw);
            const xGS = await scaledVec(xGSRaw);

            const errJac = vecInfNorm(subVec(xJac, xTrue));
            const errGS = vecInfNorm(subVec(xGS, xTrue));

            const resJac = vecInfNorm(subVec(matVecMulScaled(A, xJac), b));
            const resGS = vecInfNorm(subVec(matVecMulScaled(A, xGS), b));

            printSolverBlock({
                t: `2.${testNo}.5`,
                method: "Jacobi",
                explanation: "Decimal matrix benchmark with binary-exact fractional coefficients.",
                input: "A=[[8,0.5],[0.25,9]], b=[13.25,22.875], x0=[0,0], tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromFracs([[3, 2], [5, 2]])),
                outputHex: fmtHexVec(xJacRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xJac),
                solutionError: formatScaledInt(errJac),
                residualNorm: formatScaledInt(resJac),
                iterations: itJac.toString(),
            });

            printSolverBlock({
                t: `2.${testNo}.6`,
                method: "Gauss-Seidel",
                explanation: "Decimal matrix benchmark with binary-exact fractional coefficients.",
                input: "A=[[8,0.5],[0.25,9]], b=[13.25,22.875], x0=[0,0], tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromFracs([[3, 2], [5, 2]])),
                outputHex: fmtHexVec(xGSRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xGS),
                solutionError: formatScaledInt(errGS),
                residualNorm: formatScaledInt(resGS),
                iterations: itGS.toString(),
            });

            expect(errJac < initErr).to.equal(true);
            expect(errGS < initErr).to.equal(true);
            expect(resJac < initRes).to.equal(true);
            expect(resGS < initRes).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Gradient descent decimal behavior
    // ------------------------------------------------------------

    describe("Section 3: Gradient descent least squares decimal behavior", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: gradient descent improves decimal solution/residual relative to the initial guess`, async function () {
            const A = [
                [1n * SCALE, 0n],
                [0n, 1n * SCALE],
                [1n * SCALE, 0n],
                [0n, 1n * SCALE],
            ];
            const xTrue = [2n * SCALE, 3n * SCALE];
            const b = [2n * SCALE, 3n * SCALE, 2n * SCALE, 3n * SCALE];
            const x0 = [0n, 0n];

            const Adata = await qVecFromInts([1, 0, 0, 1, 1, 0, 0, 1]);
            const bdata = await qVecFromInts([2, 3, 2, 3]);
            const x0data = await qVecFromInts([0, 0]);

            const initErr = vecInfNorm(subVec(x0, xTrue));
            const initRes = vecInfNorm(subVec(matVecMulScaled(A, x0), b));

            const [xLooseRaw, itLoose] = await harness.gradientDescentLeastSquares(
                4n,
                2n,
                Adata,
                bdata,
                x0data,
                ALPHA_025,
                2000n,
                TOL_1E_9
            );

            const [xTightRaw, itTight] = await harness.gradientDescentLeastSquares(
                4n,
                2n,
                Adata,
                bdata,
                x0data,
                ALPHA_025,
                2000n,
                TOL_1E_15
            );

            const xLoose = await scaledVec(xLooseRaw);
            const xTight = await scaledVec(xTightRaw);

            const looseErr = vecInfNorm(subVec(xLoose, xTrue));
            const tightErr = vecInfNorm(subVec(xTight, xTrue));

            const looseRes = vecInfNorm(subVec(matVecMulScaled(A, xLoose), b));
            const tightRes = vecInfNorm(subVec(matVecMulScaled(A, xTight), b));

            printSolverBlock({
                t: `3.${testNo}.1`,
                method: "Gradient Descent Least Squares (loose)",
                explanation: "Decimal behavior with tol=1e-9.",
                input: "A=[[1,0],[0,1],[1,0],[0,1]], b=[2,3,2,3], x0=[0,0], alpha=0.25, tol=1e-9",
                expectedHex: fmtHexVec(await qVecFromInts([2, 3])),
                outputHex: fmtHexVec(xLooseRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xLoose),
                solutionError: formatScaledInt(looseErr),
                residualNorm: formatScaledInt(looseRes),
                iterations: itLoose.toString(),
            });

            printSolverBlock({
                t: `3.${testNo}.2`,
                method: "Gradient Descent Least Squares (tight)",
                explanation: "Decimal behavior with tol=1e-15.",
                input: "A=[[1,0],[0,1],[1,0],[0,1]], b=[2,3,2,3], x0=[0,0], alpha=0.25, tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromInts([2, 3])),
                outputHex: fmtHexVec(xTightRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xTight),
                solutionError: formatScaledInt(tightErr),
                residualNorm: formatScaledInt(tightRes),
                iterations: itTight.toString(),
            });

            printComparisonBlock({
                title: "Decimal comparison: Gradient Descent Least Squares",
                looseErr,
                tightErr,
                looseRes,
                tightRes,
            });

            expect(looseErr < initErr).to.equal(true);
            expect(tightErr < initErr).to.equal(true);
            expect(looseRes < initRes).to.equal(true);
            expect(tightRes < initRes).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Initial guess decimal behavior
    // ------------------------------------------------------------

    describe("Section 4: Initial guess decimal behavior", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: different initial guesses both improve decimal error/residual on Jacobi`, async function () {
            const A = [
                [10n * SCALE, 1n * SCALE],
                [1n * SCALE, 10n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE];
            const b = [12n * SCALE, 21n * SCALE];

            const Adata = await qVecFromInts([10, 1, 1, 10]);
            const bdata = await qVecFromInts([12, 21]);

            const x0Zero = [0n, 0n];
            const x0Far = [100n * SCALE, -100n * SCALE];

            const x0ZeroData = await qVecFromInts([0, 0]);
            const x0FarData = await qVecFromInts([100, -100]);

            const initErrZero = vecInfNorm(subVec(x0Zero, xTrue));
            const initErrFar = vecInfNorm(subVec(x0Far, xTrue));

            const initResZero = vecInfNorm(subVec(matVecMulScaled(A, x0Zero), b));
            const initResFar = vecInfNorm(subVec(matVecMulScaled(A, x0Far), b));

            const [xZeroRaw] = await harness.jacobi(2n, Adata, bdata, x0ZeroData, 2000n, TOL_1E_15);
            const [xFarRaw] = await harness.jacobi(2n, Adata, bdata, x0FarData, 2000n, TOL_1E_15);

            const xZero = await scaledVec(xZeroRaw);
            const xFar = await scaledVec(xFarRaw);

            const errZero = vecInfNorm(subVec(xZero, xTrue));
            const errFar = vecInfNorm(subVec(xFar, xTrue));
            const between = vecInfNorm(subVec(xZero, xFar));

            const resZero = vecInfNorm(subVec(matVecMulScaled(A, xZero), b));
            const resFar = vecInfNorm(subVec(matVecMulScaled(A, xFar), b));

            printSolverBlock({
                t: `4.${testNo}.1`,
                method: "Jacobi",
                explanation: "Decimal behavior from x0=[0,0].",
                input: "A=[[10,1],[1,10]], b=[12,21], x0=[0,0], tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromInts([1, 2])),
                outputHex: fmtHexVec(xZeroRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xZero),
                solutionError: formatScaledInt(errZero),
                residualNorm: formatScaledInt(resZero),
            });

            printSolverBlock({
                t: `4.${testNo}.2`,
                method: "Jacobi",
                explanation: "Decimal behavior from x0=[100,-100].",
                input: "A=[[10,1],[1,10]], b=[12,21], x0=[100,-100], tol=1e-15",
                expectedHex: fmtHexVec(await qVecFromInts([1, 2])),
                outputHex: fmtHexVec(xFarRaw),
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xFar),
                solutionError: formatScaledInt(errFar),
                residualNorm: formatScaledInt(resFar),
            });

            console.log("------------------------------------------------------------");
            console.log("Decimal comparison: Initial guess sensitivity");
            console.log(`Distance between final solutions (dec): ${formatScaledInt(between)}`);
            console.log("------------------------------------------------------------");

            expect(errZero < initErrZero).to.equal(true);
            expect(errFar < initErrFar).to.equal(true);

            expect(resZero < initResZero).to.equal(true);
            expect(resFar < initResFar).to.equal(true);
        });
    });
});