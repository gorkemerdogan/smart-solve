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
    toFloat(q: string): Promise<bigint>;
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

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
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
    return await harness.toFloat(q);
}

async function qScaled(harness: LinearSolversHarness, scaledValue: bigint): Promise<string> {
    return await harness.fromFloat(scaledValue);
}

function scaledAbsError(actual: bigint, expected: bigint): bigint {
    return absBigInt(actual - expected);
}

function scaledRelError(actual: bigint, expected: bigint): string {
    const num = absBigInt(actual - expected);
    const den = absBigInt(expected);

    if (den === 0n) {
        return num === 0n ? "0" : "undefined (reference is zero)";
    }

    const relScaled = (num * SCALE) / den;
    return formatScaledInt(relScaled);
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
    // A and x are already scaled by 1e12.
    // Real multiplication -> divide one SCALE after each multiply-accumulate term.
    return A.map(row => {
        let sum = 0n;
        for (let i = 0; i < row.length; i++) {
            sum += (row[i] * x[i]) / SCALE;
        }
        return sum;
    });
}

function flatten<T>(m: T[][]): T[] {
    return m.flat();
}

function printSolverBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
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
    console.log(`Expected x: ${args.expectedX}`);
    console.log(`Output x: ${args.outputX}`);
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
    L: string;
    U: string;
    reconstructionError: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log("Method: LU");
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`L: ${args.L}`);
    console.log(`U: ${args.U}`);
    console.log(`Reconstruction Error ||LU-A||_inf: ${args.reconstructionError}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("LinearSolvers Library - Numerical Accuracy Tests", function () {
    let harness: LinearSolversHarness;

    let TOL_1E_9: string;
    let ALPHA_01: string;
    let ALPHA_005: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    async function qVecFromInts(values: Array<number | bigint>): Promise<string[]> {
        return Promise.all(values.map(v => qInt(v)));
    }

    async function qVecFromFracs(values: Array<[number | bigint, number | bigint]>): Promise<string[]> {
        return Promise.all(values.map(([n, d]) => qFrac(n, d)));
    }

    async function scaledVec(harness: LinearSolversHarness, values: string[]): Promise<bigint[]> {
        return Promise.all(values.map(v => outScaled(harness, v)));
    }

    function fmtVec(v: bigint[]): string {
        return `[${v.map(formatScaledInt).join(", ")}]`;
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
        ALPHA_01 = await qFrac(1, 10);
        ALPHA_005 = await qFrac(1, 20);
    });

    // ------------------------------------------------------------
    // Section 1: Direct solvers on exact small systems
    // ------------------------------------------------------------

    describe("Section 1: Direct solvers on exact small systems", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: Gaussian elimination solves a 2x2 system exactly`, async function () {
            // A = [[2,1],[1,3]], x=[1,2], b=[4,7]
            const A = [
                [2n * SCALE, 1n * SCALE],
                [1n * SCALE, 3n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE];
            const b = [4n * SCALE, 7n * SCALE];

            const Adata = await qVecFromInts([2, 1, 1, 3]);
            const bdata = await qVecFromInts([4, 7]);

            const out = await harness.gaussianElimination(2n, Adata, bdata);
            const xComp = await scaledVec(harness, out);

            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `1.${testNo}`,
                method: "Gaussian Elimination",
                explanation: "Direct solve on a well-conditioned 2x2 benchmark with known exact solution.",
                input: "A=[[2,1],[1,3]], b=[4,7]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
            });

            expect(solErr).to.equal(0n);
            expect(residual).to.equal(0n);
        });

        it(`Test 1.${++testNo}: LU decomposition reconstructs the original matrix accurately`, async function () {
            // A = [[4,3],[6,3]]
            const A = [
                [4n * SCALE, 3n * SCALE],
                [6n * SCALE, 3n * SCALE],
            ];

            const Adata = await qVecFromInts([4, 3, 6, 3]);
            const [Lraw, Uraw] = await harness.luDecomposition(2n, Adata);

            const L = await scaledVec(harness, Lraw);
            const U = await scaledVec(harness, Uraw);

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
                explanation: "LU factors should reconstruct the original matrix up to numerical roundoff.",
                input: "A=[[4,3],[6,3]]",
                L: fmtVec(L),
                U: fmtVec(U),
                reconstructionError: formatScaledInt(reconErr),
            });

            expect(reconErr < 10n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Iterative solvers on diagonally dominant system
    // ------------------------------------------------------------

    describe("Section 2: Iterative solvers on diagonally dominant system", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: Jacobi and Gauss-Seidel converge to the correct solution`, async function () {
            // A = [[4,1],[2,3]], x=[1,2], b=[6,8]
            const A = [
                [4n * SCALE, 1n * SCALE],
                [2n * SCALE, 3n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE];
            const b = [6n * SCALE, 8n * SCALE];

            const Adata = await qVecFromInts([4, 1, 2, 3]);
            const bdata = await qVecFromInts([6, 8]);
            const x0data = await qVecFromInts([0, 0]);

            const [xJacobiRaw, itJacobi] = await harness.jacobi(2n, Adata, bdata, x0data, 200n, TOL_1E_9);
            const [xGSRaw, itGS] = await harness.gaussSeidel(2n, Adata, bdata, x0data, 200n, TOL_1E_9);

            const xJacobi = await scaledVec(harness, xJacobiRaw);
            const xGS = await scaledVec(harness, xGSRaw);

            const errJacobi = vecInfNorm(subVec(xJacobi, xTrue));
            const errGS = vecInfNorm(subVec(xGS, xTrue));

            const resJacobi = vecInfNorm(subVec(matVecMulScaled(A, xJacobi), b));
            const resGS = vecInfNorm(subVec(matVecMulScaled(A, xGS), b));

            printSolverBlock({
                t: `2.${testNo}.1`,
                method: "Jacobi",
                explanation: "Jacobi should converge on a strictly diagonally dominant 2x2 system.",
                input: "A=[[4,1],[2,3]], b=[6,8], x0=[0,0]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xJacobi),
                solutionError: formatScaledInt(errJacobi),
                residualNorm: formatScaledInt(resJacobi),
                iterations: itJacobi.toString(),
            });

            printSolverBlock({
                t: `2.${testNo}.2`,
                method: "Gauss-Seidel",
                explanation: "Gauss-Seidel should converge on the same diagonally dominant system, often faster than Jacobi.",
                input: "A=[[4,1],[2,3]], b=[6,8], x0=[0,0]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xGS),
                solutionError: formatScaledInt(errGS),
                residualNorm: formatScaledInt(resGS),
                iterations: itGS.toString(),
            });

            expect(errJacobi < 100_000_000n).to.equal(true); // 1e-4
            expect(errGS < 100_000_000n).to.equal(true);
            expect(resJacobi < 100_000_000n).to.equal(true);
            expect(resGS < 100_000_000n).to.equal(true);

            expect(itGS <= itJacobi).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Gradient descent least squares
    // ------------------------------------------------------------

    describe("Section 3: Gradient descent least squares", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: Gradient descent recovers a consistent least-squares solution`, async function () {
            // Choose a consistent overdetermined system:
            // A = [[1,0],[0,1],[1,1]], x=[2,3], b=[2,3,5]
            const A = [
                [1n * SCALE, 0n],
                [0n, 1n * SCALE],
                [1n * SCALE, 1n * SCALE],
            ];
            const xTrue = [2n * SCALE, 3n * SCALE];
            const b = [2n * SCALE, 3n * SCALE, 5n * SCALE];

            const Adata = await qVecFromInts([1, 0, 0, 1, 1, 1]);
            const bdata = await qVecFromInts([2, 3, 5]);
            const x0data = await qVecFromInts([0, 0]);

            const [xRaw, iters] = await harness.gradientDescentLeastSquares(
                3n,
                2n,
                Adata,
                bdata,
                x0data,
                ALPHA_01,
                500n,
                TOL_1E_9
            );

            const xComp = await scaledVec(harness, xRaw);
            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `3.${testNo}`,
                method: "Gradient Descent Least Squares",
                explanation: "Gradient descent should recover the least-squares solution on a consistent overdetermined system.",
                input: "A=[[1,0],[0,1],[1,1]], b=[2,3,5], x0=[0,0], alpha=0.1",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
                iterations: iters.toString(),
            });

            expect(solErr < 100_000_000n).to.equal(true); // 1e-4
            expect(residual < 100_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Initial guess sensitivity for iterative solvers
    // ------------------------------------------------------------

    describe("Section 4: Initial guess sensitivity", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: Different initial guesses still converge to the same Jacobi solution`, async function () {
            // A = [[4,1],[2,3]], x=[1,2], b=[6,8]
            const A = [
                [4n * SCALE, 1n * SCALE],
                [2n * SCALE, 3n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE];
            const b = [6n * SCALE, 8n * SCALE];

            const Adata = await qVecFromInts([4, 1, 2, 3]);
            const bdata = await qVecFromInts([6, 8]);

            const x0Zero = await qVecFromInts([0, 0]);
            const x0Far = await qVecFromInts([10, -10]);

            const [xZeroRaw] = await harness.jacobi(2n, Adata, bdata, x0Zero, 300n, TOL_1E_9);
            const [xFarRaw] = await harness.jacobi(2n, Adata, bdata, x0Far, 300n, TOL_1E_9);

            const xZero = await scaledVec(harness, xZeroRaw);
            const xFar = await scaledVec(harness, xFarRaw);

            const errZero = vecInfNorm(subVec(xZero, xTrue));
            const errFar = vecInfNorm(subVec(xFar, xTrue));
            const between = vecInfNorm(subVec(xZero, xFar));

            const resZero = vecInfNorm(subVec(matVecMulScaled(A, xZero), b));
            const resFar = vecInfNorm(subVec(matVecMulScaled(A, xFar), b));

            printSolverBlock({
                t: `4.${testNo}.1`,
                method: "Jacobi",
                explanation: "Jacobi result starting from x0=[0,0].",
                input: "A=[[4,1],[2,3]], b=[6,8], x0=[0,0]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xZero),
                solutionError: formatScaledInt(errZero),
                residualNorm: formatScaledInt(resZero),
            });

            printSolverBlock({
                t: `4.${testNo}.2`,
                method: "Jacobi",
                explanation: "Jacobi result starting from x0=[10,-10].",
                input: "A=[[4,1],[2,3]], b=[6,8], x0=[10,-10]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xFar),
                solutionError: formatScaledInt(errFar),
                residualNorm: formatScaledInt(resFar),
            });

            expect(errZero < 100_000_000n).to.equal(true); // 1e-4
            expect(errFar < 100_000_000n).to.equal(true);
            expect(between < 100_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 5: 3x3 direct solve benchmark
    // ------------------------------------------------------------

    describe("Section 5: 3x3 benchmark", function () {
        let testNo = 0;

        it(`Test 5.${++testNo}: Gaussian elimination solves a 3x3 system accurately`, async function () {
            // Let x=[1,2,3]
            // A = [[4,1,1],[1,5,1],[1,1,6]]
            // b = [9,14,21]
            const A = [
                [4n * SCALE, 1n * SCALE, 1n * SCALE],
                [1n * SCALE, 5n * SCALE, 1n * SCALE],
                [1n * SCALE, 1n * SCALE, 6n * SCALE],
            ];
            const xTrue = [1n * SCALE, 2n * SCALE, 3n * SCALE];
            const b = [9n * SCALE, 14n * SCALE, 21n * SCALE];

            const Adata = await qVecFromInts([4, 1, 1, 1, 5, 1, 1, 1, 6]);
            const bdata = await qVecFromInts([9, 14, 21]);

            const out = await harness.gaussianElimination(3n, Adata, bdata);
            const xComp = await scaledVec(harness, out);

            const solErr = vecInfNorm(subVec(xComp, xTrue));
            const residual = vecInfNorm(subVec(matVecMulScaled(A, xComp), b));

            printSolverBlock({
                t: `5.${testNo}`,
                method: "Gaussian Elimination",
                explanation: "Direct solve on a 3x3 symmetric positive system with known exact solution.",
                input: "A=[[4,1,1],[1,5,1],[1,1,6]], b=[9,14,21]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                solutionError: formatScaledInt(solErr),
                residualNorm: formatScaledInt(residual),
            });

            expect(solErr < 10n).to.equal(true);
            expect(residual < 10n).to.equal(true);
        });
    });
});