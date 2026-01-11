// test/LinearSolvers.test.ts
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type LinearSolversHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

    gradientDescentLeastSquares(m: bigint, n: bigint, Adata: string[], bdata: string[], x0data: string[], alpha: string, maxIter: bigint, tol: string): Promise<[string[], bigint]>;
    jacobi(n: bigint, Adata: string[], bdata: string[], x0data: string[], maxIter: bigint, tolDiff: string): Promise<[string[], bigint]>;
    gaussSeidel(n: bigint, Adata: string[], bdata: string[], x0data: string[], maxIter: bigint, tolDiff: string): Promise<[string[], bigint]>;
    gaussianElimination(n: bigint, Adata: string[], bdata: string[]): Promise<string[]>;
    luDecomposition(n: bigint, Adata: string[]): Promise<[string[], string[]]>;
};

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

function toBigIntArray(arr: number[]): bigint[] {
    return arr.map((x) => BigInt(x));
}

// Convert Array<number> to Array<BigInt> scaled
function toScaledArray(arr: number[]): bigint[] {
    return arr.map(x => BigInt(Math.round(x * Number(SCALE))));
}

// Check if scalar values are close
function expectClose(actual: bigint, expected: bigint, tol: bigint, msg: string = "") {
    let diff = actual - expected;
    if (diff < 0n) diff = -diff;
    expect(diff).to.be.below(tol, `${msg} | Expected ${expected}, got ${actual}, diff ${diff}`);
}

/// Helper to multiply A (n×n) by x (n×1) in JS for verification.
function multiplyMatrixVector(n: number, A: bigint[], x: bigint[]): bigint[] {
    const out: bigint[] = new Array(n).fill(0n);
    for (let i = 0; i < n; ++i) {
        let sum = 0n;
        for (let j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        out[i] = sum;
    }
    return out;
}

// Matrix Multiplication (A * B) / SCALE
function multiplyMatrices(n: number, A: bigint[], B: bigint[]): bigint[] {
    const C: bigint[] = new Array(n * n).fill(0n);
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < n; ++j) {
            let sum = 0n;
            for (let k = 0; k < n; ++k) {
                sum += (A[i * n + k] * B[k * n + j]) / SCALE;
            }
            C[i * n + j] = sum;
        }
    }
    return C;
}

function expectMatrixSimilar(n: number, A: bigint[], B: bigint[], tol: bigint) {
    for (let i = 0; i < n * n; i++) {
        expectClose(A[i], B[i], tol, `Matrix mismatch at index ${i}`);
    }
}

/// Helper to assert vector equality.
function expectVecEq(actual: bigint[], expected: number[]) {
    const expBig = toBigIntArray(expected);
    expect(actual.length).to.equal(expBig.length);
    for (let i = 0; i < expBig.length; ++i) {
        expect(actual[i]).to.equal(expBig[i], `Vector element at index ${i} mismatch. Expected ${expBig[i]}, got ${actual[i]}`);
    }
}

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;
const TOL_DIRECT = 500n; // For direct solvers (Gaussian/LU) (1e-10)
const TOL_ITERATIVE = 1_000_000n; // For iterative solvers (GD/Jacobi) (1e-6)
const MAT_TOL = 1_000n; // Matrix reconstruction tolerance (1e-9)

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("LinearSolversHarness", function () {
    let harness: LinearSolversHarness;
    let t = 0;

    beforeEach(async function () {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const Factory = await ethers.getContractFactory("LinearSolversHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as LinearSolversHarness;
        await harness.waitForDeployment();
    });

    // ------------------------------------------------------------
    //  Gradient Descent
    // ------------------------------------------------------------

    describe("Section 1: Gradient Descent Least Squares", function () {

        it("Test 1: 1D identity A=1 converges in one step", async function () {
            t++;
            const valA = 1n;
            const valB = 5n;

            const expectedVal = (valB * SCALE) / valA;

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)];
            const alpha = await harness.qFromInt(1);
            const tol = await harness.qFromInt(0);

            const args = [1n, 1n, A, b, x0, alpha, 10n, tol];
            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x, iters] = await harness.gradientDescentLeastSquares(1n, 1n, A, b, x0, alpha, 10n, tol);
            const actualVal = await harness.toFloat(x[0]);

            expectClose(actualVal, expectedVal, TOL_ITERATIVE, "Result x0");
            expect(iters).to.be.lessThanOrEqual(1n);

            printBlockRegular({
                t,
                method: "Gradient Descent",
                explanation: `1D identity (A=${valA}, b=${valB}) | x = b/A.`,
                inHex: `A=[${valA}], b=[${valB}]`,
                expectedHex: await harness.qFromInt(5),
                outHex: x[0],
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 2: 1D scaled identity A=2", async function () {
            t++;
            const valA = 2n;
            const valB = 8n;
            const expectedVal = (valB * SCALE) / valA;

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)];
            const alpha = await harness.qFromFrac(1, 4);

            const [x] = await harness.gradientDescentLeastSquares(1n, 1n, A, b, x0, alpha, 50n, await harness.qFromInt(0));
            const actualVal = await harness.toFloat(x[0]);

            expectClose(actualVal, expectedVal, TOL_ITERATIVE);

            await touchGas(harness, "gradientDescentLeastSquares", [1n, 1n, A, b, x0, alpha, 50n, await harness.qFromInt(0)]);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", [1n, 1n, A, b, x0, alpha, 50n, await harness.qFromInt(0)]);

            printBlockRegular({
                t,
                method: "Gradient Descent",
                explanation: "Should converge to 4 for 1D scaled.",
                inHex: `A=[${valA}], b=[${valB}]`,
                expectedHex: await harness.qFromInt(4),
                outHex: x[0],
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 3: 2D identity system", async function () {
            t++;

            const bVals = [3n, -7n];
            const expectedVec = bVals.map(v => v * SCALE);

            const A = [await harness.qFromInt(1), await harness.qFromInt(0), await harness.qFromInt(0), await harness.qFromInt(1)];
            const b = [await harness.qFromInt(bVals[0]), await harness.qFromInt(bVals[1])];
            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];

            const [x] = await harness.gradientDescentLeastSquares(2n, 2n, A, b, x0, await harness.qFromInt(1), 10n, await harness.qFromInt(0));

            const actual0 = await harness.toFloat(x[0]);
            const actual1 = await harness.toFloat(x[1]);

            expectClose(actual0, expectedVec[0], TOL_ITERATIVE);
            expectClose(actual1, expectedVec[1], TOL_ITERATIVE);

            const gas = await estimateGas(harness, "gradientDescentLeastSquares", [2n, 2n, A, b, x0, await harness.qFromInt(1), 10n, await harness.qFromInt(0)]);

            printBlockRegular({
                t,
                method: "Gradient Descent",
                explanation: "Result matches input b for 2D Identity.",
                inHex: "A=I, b=[3,-7]",
                expectedHex: `[${b[0]}, ${b[1]}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${bVals.join(", ")}]`,
                outDec: `[${formatScaledInt(actual0)}, ${formatScaledInt(actual1)}]`,
                gas,
            });
        });

        it("Test 4: Overdetermined least-squares system", async function () {
            t++;

            const m = 2n;
            const n = 1n;

            const valA = [1n, 1n];
            const valB = [2n, 4n];

            const numerator = valA.reduce((sum, ai, i) => sum + ai * valB[i], 0n);
            const denominator = valA.reduce((sum, ai) => sum + ai * ai, 0n);
            const expectedVal = (numerator * SCALE) / denominator;

            const A = await Promise.all(valA.map(v => harness.qFromInt(v)));
            const b = await Promise.all(valB.map(v => harness.qFromInt(v)));
            const x0 = [await harness.qFromInt(0)];
            const alpha = await harness.qFromFrac(1, 2);
            const tol = await harness.qFromInt(0);
            const maxIter = 50n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x] = await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const actualVal = await harness.toFloat(x[0]);

            // Use TOL_ITERATIVE (1e-6)
            expectClose(actualVal, expectedVal, TOL_ITERATIVE, "Least Squares Solution Mismatch");

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "Minimal residual found at the mean for unit A. (m=2, n=1)",
                inHex: `A=[${valA.join(",")}], b=[${valB.join(",")}]`,
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 5: Zero gradient at start", async function () {
            t++;

            const m = 1n;
            const n = 1n;

            const valA = [1n];
            const valB = [0n];
            const valX0 = [0n];

            // For 1D: grad = A * (A * x0 - b)
            const initialGrad = valA[0] * (valA[0] * valX0[0] - valB[0]);
            const expectedVal = valX0[0] * SCALE;

            const A = [await harness.qFromInt(valA[0])];
            const b = [await harness.qFromInt(valB[0])];
            const x0 = [await harness.qFromInt(valX0[0])];

            const alpha = await harness.qFromInt(1);
            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x, iters] = await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const actualVal = await harness.toFloat(x[0]);

            expect(actualVal).to.equal(expectedVal);

            // If JS shows gradient is 0, iterations must be 0
            if (initialGrad === 0n) {
                expect(iters).to.equal(0n, "Should terminate immediately with zero gradient");
            }

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "If initial gradient is null, iterations must be zero.",
                inHex: `A=[${valA}], b=[${valB}], x0=[${valX0}]`,
                expectedHex: await harness.qFromInt(valX0[0]),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 6: Very small alpha", async function () {
            t++;

            const m = 1n;
            const n = 1n;

            const valA = 1n;
            const valB = 1n;
            const valX0 = 0n;
            const maxIter = 5n;

            const alphaNum = 1n;
            const alphaDen = 1_000_000n;

            // x_next = x - alpha * (A * (A * x - b))
            let xJS = Number(valX0);
            const aJS = Number(valA);
            const bJS = Number(valB);
            const lrJS = Number(alphaNum) / Number(alphaDen);

            for (let i = 0; i < Number(maxIter); i++) {
                const grad = aJS * (aJS * xJS - bJS);
                xJS = xJS - lrJS * grad;
            }

            const expectedVal = BigInt(Math.round(xJS * Number(SCALE)));

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(valX0)];
            const alpha = await harness.qFromFrac(alphaNum, alphaDen);
            const tol = await harness.qFromInt(0);

            const args = [m, n, A, b, x0, alpha, maxIter, tol];
            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x] = await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);
            const actualVal = await harness.toFloat(x[0]);

            expectClose(actualVal, expectedVal, TOL_DIRECT, "Slow convergence mismatch");

            printBlockRegular({
                t,
                method: "Gradient Descent",
                explanation: "Small alpha should result in minimal but precise progress.",
                inHex: `A=[1], b=[1], α=1e-6, iters=${maxIter}`,
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });
    });

    // ------------------------------------------------------------
    //  Jacobi
    // ------------------------------------------------------------

    describe("Section 2: Jacobi Iteration", function () {

        it("Test 7: 1D identity A=1 solves immediately", async function () {
            t++;

            const n = 1n;
            const valA = 1n;
            const valB = 7n;

            // x = b / A
            const expectedVal = (valB * SCALE) / valA;

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)];
            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x, iters] = await harness.jacobi(n, A, b, x0, maxIter, tol);
            const actualVal = await harness.toFloat(x[0]);

            expectClose(actualVal, expectedVal, TOL_DIRECT, "Jacobi 1D Solution Mismatch");
            expect(iters).to.be.lessThanOrEqual(2n); // Iter 1 jumps to solution. Iter 2 confirms stability.

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "1D identity convergence. Iteration 1 solves, Iteration 2 verifies.",
                inHex: `A=[${valA}], b=[${valB}]`,
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 8: 2D diagonal dominant system", async function () {
            t++;

            const n = 2n;
            const m = [[4n, 1n], [2n, 3n]];
            const bVals = [1n, 2n];

            const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const x0Exp = (bVals[0] * m[1][1] - bVals[1] * m[0][1]) * SCALE / det;
            const x1Exp = (m[0][0] * bVals[1] - m[1][0] * bVals[0]) * SCALE / det;

            const A = [
                await harness.qFromInt(m[0][0]), await harness.qFromInt(m[0][1]),
                await harness.qFromInt(m[1][0]), await harness.qFromInt(m[1][1]),
            ];
            const b = [await harness.qFromInt(bVals[0]), await harness.qFromInt(bVals[1])];
            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];

            const tol = await harness.qFromFrac(1, 1_000_000_000_000n); // 1e-12 precision
            const maxIter = 200n;

            const args = [n, A, b, x0, maxIter, tol];
            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x] = await harness.jacobi(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            expectClose(x0Int, x0Exp, TOL_ITERATIVE, "x0 result mismatch");
            expectClose(x1Int, x1Exp, TOL_ITERATIVE, "x1 result mismatch");

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "Verification of a 2x2 diagonally dominant system.",
                inHex: "A=[[4,1],[2,3]], b=[1,2]",
                expectedHex: `[${await harness.fromFloat(x0Exp)}, ${await harness.fromFloat(x1Exp)}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${formatScaledInt(x0Exp)}, ${formatScaledInt(x1Exp)}]`,
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas,
            });
        });

        it("Test 9: Already-solved system remains stable", async function () {
            t++;

            const n = 2n;

            const bVals = [3n, 4n];
            const x0Vals = [3n, 4n];

            const A = [
                await harness.qFromInt(1), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(1),
            ];

            const b = await Promise.all(bVals.map(v => harness.qFromInt(v)));
            const x0 = await Promise.all(x0Vals.map(v => harness.qFromInt(v)));

            const tol = await harness.qFromInt(0);
            const maxIter = 5n; // Should converge immediately

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x, iters] = await harness.jacobi(n, A, b, x0, maxIter, tol);

            const actual0 = await harness.toFloat(x[0]);
            const actual1 = await harness.toFloat(x[1]);

            // Since A=I, x should exactly equal to b.
            const expected0 = bVals[0] * SCALE;
            const expected1 = bVals[1] * SCALE;

            expectClose(actual0, expected0, TOL_DIRECT, "x[0] drifted from solution");
            expectClose(actual1, expected1, TOL_DIRECT, "x[1] drifted from solution");

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "Initial guess x0 matches exact solution -> output should remain unchanged.",
                inHex: `A=I, b=[${bVals}], x0=[${x0Vals}]`,
                expectedHex: `[${await harness.fromFloat(expected0)}, ${await harness.fromFloat(expected1)}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${formatScaledInt(expected0)}, ${formatScaledInt(expected1)}]`,
                outDec: `[${formatScaledInt(actual0)}, ${formatScaledInt(actual1)}]`,
                gas,
            });
        });

        it("Test 10: Zero diagonal element reverts", async function () {
            t++;

            const n = 2n;

            const A_vals = [0n, 1n, 1n, 1n]; // Matrix with explicit Zero on diagonal at (0,0)
            const A = await Promise.all(A_vals.map(v => harness.qFromInt(v)));

            const b = [await harness.qFromInt(1), await harness.qFromInt(1)];
            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];
            const tol = await harness.qFromInt(0);

            await expect(harness.jacobi(n, A, b, x0, 10n, tol)).to.be.reverted;

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "Zero element on diagonal (index 0,0) must trigger revert.",
                inHex: `A=[[${A_vals[0]},${A_vals[1]}],[${A_vals[2]},${A_vals[3]}]] (Zero Diag)`,
                expectedHex: "Revert",
                outHex: "Revert",
                expectedDec: "Revert",
                outDec: "Revert",
                gas: "N/A",
            });
        });

        it("Test 11: maxIter=1 performs single iteration only", async function () {
            t++;

            const n = 1n;
            const valA = 2n;
            const valB = 8n;

            const expectedVal = (valB * SCALE) / valA; // 1 iteration of 1D Jacobi is exactly b/A

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)]; // Start at 0
            const tol = await harness.qFromInt(0);
            const maxIter = 1n; // Force single step

            const args = [n, A, b, x0, maxIter, tol];
            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x, iters] = await harness.jacobi(n, A, b, x0, maxIter, tol);
            const actualVal = await harness.toFloat(x[0]);

            expect(iters).to.equal(1n);
            expectClose(actualVal, expectedVal, TOL_DIRECT, "Single iteration value mismatch");

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "maxIter=1 must return result of first update step.",
                inHex: `A=[${valA}], b=[${valB}], maxIter=1`,
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });
    });

    // ------------------------------------------------------------
    //  Gauss–Seidel Method
    // ------------------------------------------------------------

    describe("Section 3: Gauss–Seidel Iteration", function () {

        it("Test 12: 1D identity system (A=1)", async function () {
            t++;
            const valA = 1n;
            const valB = 7n;
            const expectedVal = (valB * SCALE) / valA;

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)];
            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [1n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x, iters] = await harness.gaussSeidel(1n, A, b, x0, maxIter, tol);
            const actualVal = await harness.toFloat(x[0]);

            expectClose(actualVal, expectedVal, TOL_DIRECT);
            // Confirm stability (2 iterations, tol = 0)
            expect(iters).to.be.lessThanOrEqual(2n);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "1D identity system converges immediately.",
                inHex: `A=[${valA}], b=[${valB}]`,
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 13: 2D diagonally dominant system (scaled residual check)", async function () {
            t++;
            const n = 2n;
            const m = [[4n, 1n], [2n, 3n]];
            const bVals = [1n, 2n];

            const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const x0Exp = (bVals[0] * m[1][1] - bVals[1] * m[0][1]) * SCALE / det;
            const x1Exp = (m[0][0] * bVals[1] - m[1][0] * bVals[0]) * SCALE / det;

            const A = await Promise.all([4n, 1n, 2n, 3n].map(v => harness.qFromInt(v)));
            const b = await Promise.all(bVals.map(v => harness.qFromInt(v)));
            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];

            const tol = await harness.qFromFrac(1, 1_000_000_000_000n); // 1e-12
            const maxIter = 200n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x] = await harness.gaussSeidel(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            expectClose(x0Int, x0Exp, TOL_ITERATIVE);
            expectClose(x1Int, x1Exp, TOL_ITERATIVE);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "2×2 diagonally dominant system verifying convergence to analytical point.",
                inHex: "A=[[4,1],[2,3]], b=[1,2]",
                expectedHex: `[${await harness.fromFloat(x0Exp)}, ${await harness.fromFloat(x1Exp)}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${formatScaledInt(x0Exp)}, ${formatScaledInt(x1Exp)}]`,
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas,
            });
        });

        it("Test 14: Already converged initial guess", async function () {
            t++;
            const bVals = [4n, 6n];
            const x0Vals = [2n, 3n];

            const A = [await harness.qFromInt(2), await harness.qFromInt(0), await harness.qFromInt(0), await harness.qFromInt(2)];
            const b = await Promise.all(bVals.map(v => harness.qFromInt(v)));
            const x0 = await Promise.all(x0Vals.map(v => harness.qFromInt(v)));

            const tol = await harness.qFromInt(0);
            const maxIter = 5n;

            const args = [2n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x, iters] = await harness.gaussSeidel(2n, A, b, x0, maxIter, tol);

            expect(iters).to.be.lessThanOrEqual(2n); // Residual is 0

            const out0 = formatScaledInt(await harness.toFloat(x[0]));
            const out1 = formatScaledInt(await harness.toFloat(x[1]));

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "Output must remain at x0 if x0 is already the solution.",
                inHex: "A=2I, b=[4,6], x0=[2,3]",
                expectedHex: `[${x0[0]}, ${x0[1]}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[2, 3]",
                outDec: `[${out0}, ${out1}]`,
                gas,
            });
        });

        it("Test 15: Negative values system", async function () {
            t++;
            const valA = -2n;
            const valB = -8n;
            const expectedVal = (valB * SCALE) / valA;

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [1n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x] = await harness.gaussSeidel(1n, A, b, x0, maxIter, tol);
            const x0Int = await harness.toFloat(x[0]);

            expectClose(x0Int, expectedVal, TOL_DIRECT);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "Signed math check with negative coefficients.",
                inHex: "A=[-2], b=[-8]",
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: "4",
                outDec: formatScaledInt(x0Int),
                gas,
            });
        });

        it("Test 16: Zero diagonal should revert", async function () {
            t++;

            const n = 2n;

            const A = [
                await harness.qFromInt(0), await harness.qFromInt(1),
                await harness.qFromInt(1), await harness.qFromInt(1),
            ];

            const b = [await harness.qFromInt(1), await harness.qFromInt(2)];
            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];
            const tol = await harness.qFromInt(0);
            const maxIter = 5n;

            await expect(harness.gaussSeidel(n, A, b, x0, maxIter, tol)).to.be.reverted;

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "Zero diagonal element should revert.",
                inHex: "A has zero diagonal",
                expectedHex: "Revert",
                outHex: "Revert",
                expectedDec: "Revert",
                outDec: "Revert",
                gas: "N/A",
            });
        });

        it("Test 17: maxIter = 1 executes exactly one iteration", async function () {
            t++;
            const valA = 5n;
            const valB = 10n;
            const expectedVal = (valB * SCALE) / valA; // 2.0

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 1n;

            const [x, iters] = await harness.gaussSeidel(1n, A, b, x0, maxIter, tol);
            const actualVal = await harness.toFloat(x[0]);

            expect(iters).to.equal(1n);
            expectClose(actualVal, expectedVal, TOL_DIRECT);
        });
    });

    describe("Section 4: Gaussian Elimination", function () {

        it("Test 18: 1D trivial system A=[1], b=[7]", async function () {
            t++;
            const n = 1n;
            const valA = 1n;
            const valB = 7n;
            const expectedVal = (valB * SCALE) / valA;

            const A = [await harness.qFromInt(valA)];
            const b = [await harness.qFromInt(valB)];
            const args = [n, A, b];

            await touchGas(harness, "gaussianElimination", args);
            const gas = await estimateGas(harness, "gaussianElimination", args);

            const x = await harness.gaussianElimination(n, A, b);
            const actualVal = await harness.toFloat(x[0]);

            expectClose(actualVal, expectedVal, TOL_DIRECT);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "1D exact system check.",
                inHex: `A=[${valA}], b=[${valB}]`,
                expectedHex: await harness.fromFloat(expectedVal),
                outHex: `[${x.join(", ")}]`,
                expectedDec: formatScaledInt(expectedVal),
                outDec: formatScaledInt(actualVal),
                gas,
            });
        });

        it("Test 19: 2D identity matrix", async function () {
            t++;
            const n = 2n;
            // A = I, so x = b
            const bVals = [3n, -4n];

            const A = [await harness.qFromInt(1), await harness.qFromInt(0), await harness.qFromInt(0), await harness.qFromInt(1)];
            const b = await Promise.all(bVals.map(v => harness.qFromInt(v)));

            const x = await harness.gaussianElimination(n, A, b);

            const actual0 = await harness.toFloat(x[0]);
            const actual1 = await harness.toFloat(x[1]);

            expectClose(actual0, bVals[0] * SCALE, TOL_DIRECT);
            expectClose(actual1, bVals[1] * SCALE, TOL_DIRECT);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Identity matrix should return b directly.",
                inHex: "A=I, b=[3,-4]",
                expectedHex: `[${b[0]}, ${b[1]}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${bVals.join(", ")}]`,
                outDec: `[${formatScaledInt(actual0)}, ${formatScaledInt(actual1)}]`,
                gas: await estimateGas(harness, "gaussianElimination", [n, A, b]),
            });
        });

        it("Test 20: 2D general solvable system (Calculated Expectation)", async function () {
            t++;
            const n = 2n;
            const m = [[2n, 1n], [5n, 7n]];
            const bVals = [11n, 13n];

            const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const x0Exp = (bVals[0] * m[1][1] - bVals[1] * m[0][1]) * SCALE / det;
            const x1Exp = (m[0][0] * bVals[1] - m[1][0] * bVals[0]) * SCALE / det;

            const A = await Promise.all([2n, 1n, 5n, 7n].map(v => harness.qFromInt(v)));
            const b = await Promise.all(bVals.map(v => harness.qFromInt(v)));

            const x = await harness.gaussianElimination(n, A, b);
            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            expectClose(x0Int, x0Exp, TOL_DIRECT, "x0 mismatch");
            expectClose(x1Int, x1Exp, TOL_DIRECT, "x1 mismatch");

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Standard 2×2 solvable system.",
                inHex: "A=[[2,1],[5,7]], b=[11,13]",
                expectedHex: `[${await harness.fromFloat(x0Exp)}, ${await harness.fromFloat(x1Exp)}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${formatScaledInt(x0Exp)}, ${formatScaledInt(x1Exp)}]`,
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas: await estimateGas(harness, "gaussianElimination", [n, A, b]),
            });
        });

        it("Test 21: 3D upper-triangular system (Residual Verification)", async function () {
            t++;
            const n = 3n;
            const A_raw = [
                2n, 1n, 1n,
                0n, 3n, 1n, 
                0n, 0n, 4n
            ];
            const b_raw = [5n, 4n, 8n];

            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));
            const b = await Promise.all(b_raw.map(v => harness.qFromInt(v)));

            const x = await harness.gaussianElimination(n, A, b);

            const x_vals = await Promise.all(x.map(v => harness.toFloat(v)));
            
            const r0 = (2n * x_vals[0] + 1n * x_vals[1] + 1n * x_vals[2]) - (5n * SCALE); // Row 0: 2x + y + z - 5
            const r1 = (3n * x_vals[1] + 1n * x_vals[2]) - (4n * SCALE);                  // Row 1: 3y + z - 4
            const r2 = (4n * x_vals[2]) - (8n * SCALE);                                   // Row 2: 4z - 8

            const abs = (v: bigint) => v < 0n ? -v : v;
            
            expect(abs(r0)).to.be.below(TOL_DIRECT, "Row 0 residual too high");
            expect(abs(r1)).to.be.below(TOL_DIRECT, "Row 1 residual too high");
            expect(abs(r2)).to.be.below(TOL_DIRECT, "Row 2 residual too high");

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "3x3 Upper-triangular system verified via residual |Ax - b| < epsilon.",
                inHex: "Upper Triangular Matrix",
                expectedHex: "Residual ≈ 0",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "Residual < 1e-10",
                outDec: "Pass",
                gas: await estimateGas(harness, "gaussianElimination", [n, A, b]),
            });
        });

        it("Test 22: Singular matrix should revert", async function () {
            t++;

            const n = 2n;

            const A = await Promise.all([1n, 2n, 2n, 4n].map(v => harness.qFromInt(v)));
            const b = await Promise.all([3n, 6n].map(v => harness.qFromInt(v)));

            await expect(harness.gaussianElimination(n, A, b)).to.be.reverted;

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Singular matrix should revert (zero pivot).",
                inHex: "A=[[1,2],[2,4]] (Rank Deficient)",
                expectedHex: "Revert",
                outHex: "Revert",
                expectedDec: "Revert",
                outDec: "Revert",
                gas: "N/A",
            });
        });

        it("Test 23: Zero pivot requiring row swap (Partial Pivoting)", async function () {
            t++;

            const n = 2n;
            const m = [[0n, 1n], [2n, 3n]];
            const bVals = [1n, 5n];

            const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const x0Exp = (bVals[0] * m[1][1] - bVals[1] * m[0][1]) * SCALE / det;
            const x1Exp = (m[0][0] * bVals[1] - m[1][0] * bVals[0]) * SCALE / det;

            const A = await Promise.all([0n, 1n, 2n, 3n].map(v => harness.qFromInt(v)));
            const b = await Promise.all(bVals.map(v => harness.qFromInt(v)));

            const x = await harness.gaussianElimination(n, A, b);
            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            expectClose(x0Int, x0Exp, TOL_DIRECT, "x0 mismatch after swap");
            expectClose(x1Int, x1Exp, TOL_DIRECT, "x1 mismatch after swap");

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Pivot check. A[0,0]=0 requires swapping with R1.",
                inHex: "A=[[0,1],[2,3]], b=[1,5]",
                expectedHex: `[${await harness.fromFloat(x0Exp)}, ${await harness.fromFloat(x1Exp)}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: `[${formatScaledInt(x0Exp)}, ${formatScaledInt(x1Exp)}]`,
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas: await estimateGas(harness, "gaussianElimination", [n, A, b]),
            });
        });
    });

    describe("Section 5: LU Decomposition", function () {

        /**
         * @notice Verifies correctness of an LU decomposition by reconstructing the
         *         original matrix and comparing it against the input.
         *
         * @param n Dimension of the square matrix (n × n).
         * @param A_contract Flattened matrix A as returned
         * @param msg Contextual message used for debugging or test identification.
         * @return L Lower-triangular matrix (bytes16[])
         * @return U Upper-triangular matrix (bytes16[])
         */
        async function verifyLU(n: bigint, A_contract: string[], msg: string) {
            const [L, U] = await harness.luDecomposition(n, A_contract);

            const Ls = await Promise.all(L.map(v => harness.toFloat(v)));
            const Us = await Promise.all(U.map(v => harness.toFloat(v)));
            const As = await Promise.all(A_contract.map(v => harness.toFloat(v)));

            const LU = multiplyMatrices(Number(n), Ls, Us); // Reconstruct A' = L * U

            expectMatrixSimilar(Number(n), LU, As, TOL_DIRECT);

            return { L, U };
        }

        /**
         * @notice Formats a contract-encoded matrix into a readable string
         * @param n Matrix dimension (n × n).
         * @param M_hex Flattened matrix encoded as bytes16 hex strings.
         * @return String representation of the matrix values in scaled-integer form.
         */
        async function formatMatrixForReport(n: number, M_hex: string[]): Promise<string> {
            const vals = await Promise.all(M_hex.map(v => harness.toFloat(v)));
            const str = vals.map(v => formatScaledInt(v)).join(", ");
            return `[${str}]`;
        }

        /**
         * @notice Formats a flattened matrix into a readable grid layout.
         * @param arr Flattened matrix data in row-major order.
         * @return Multi-line formatted string for 2×2 or 3×3 matrices
         */
        const formatMat = (arr: any[]) => {
            const n = Math.sqrt(arr.length);
            if (n === 2) {
                return `[${arr[0]}, ${arr[1]}]\n       [${arr[2]}, ${arr[3]}]`;
            } else if (n === 3) {
                return `[${arr[0]}, ${arr[1]}, ${arr[2]}]\n      
                        [${arr[3]}, ${arr[4]}, ${arr[5]}]\n      
                        [${arr[6]}, ${arr[7]}, ${arr[8]}]`;
            }
            return arr.join(", "); // Fallback
        };

        it("Test 24: LU Decomposition Reconstruction", async function () {
            t++;
            const n = 2n;
            const A_raw = [4n, 3n, 6n, 3n];
            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));

            const [L, U] = await harness.luDecomposition(n, A);

            // Verification
            const Ls = await Promise.all(L.map(v => harness.toFloat(v)));
            const Us = await Promise.all(U.map(v => harness.toFloat(v)));
            const As = await Promise.all(A.map(v => harness.toFloat(v)));

            const LU_product = multiplyMatrices(Number(n), Ls, Us);

            // Show the input A vs the raw output L/U
            const expectedHex = `Original A Hex:\n${formatMat(A)}`;
            const outHex = `Decomposed L/U:\nL:${formatMat(L)}\nU:${formatMat(U)}`;

            const expectedDec = `Original A: [${As.map(v => formatScaledInt(v)).join(", ")}]`;
            const outDec = `Reconstructed A': [${LU_product.map(v => formatScaledInt(v)).join(", ")}]`;

            printBlockRegular({
                t,
                method: "LU Decomposition",
                explanation: "Verifying A = L * U.",
                inHex: "A=[[4,3],[6,3]]",
                expectedHex,
                outHex,
                expectedDec,
                outDec,
                gas: await estimateGas(harness, "luDecomposition", [n, A]),
            });

            expectMatrixSimilar(Number(n), LU_product, As, TOL_DIRECT);
        });

        it("Test 25: 3×3 diagonally dominant matrix (Full Reconstruction)", async function () {
            t++;
            const n = 3n;
            const A_raw = [
                10n, 2n, 1n,
                2n, 8n, 1n,
                1n, 1n, 5n
            ];
            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));

            await touchGas(harness, "luDecomposition", [n, A]);
            const gas = await estimateGas(harness, "luDecomposition", [n, A]);

            const [L, U] = await harness.luDecomposition(n, A);

            // Verification
            const Ls = await Promise.all(L.map(v => harness.toFloat(v)));
            const Us = await Promise.all(U.map(v => harness.toFloat(v)));
            const As = await Promise.all(A.map(v => harness.toFloat(v)));

            const LU_product = multiplyMatrices(3, Ls, Us); // L * U = A'

            const expectedHex = `L Matrix:\n${formatMat(L)}`;
            const outHex = `U Matrix:\n${formatMat(U)}`;

            const expectedDec = `Original A: [${As.map(v => formatScaledInt(v)).join(", ")}]`;
            const outDec = `Reconstructed A': [${LU_product.map(v => formatScaledInt(v)).join(", ")}]`;

            printBlockRegular({
                t,
                method: "LU Decomposition",
                explanation: "L*U reconstruction must match original matrix A. (3x3)",
                inHex: "A=[[10,2,1],[2,8,1],[1,1,5]]",
                expectedHex,
                outHex,
                expectedDec,
                outDec,
                gas,
            });

            expectMatrixSimilar(3, LU_product, As, TOL_DIRECT);
        });

        it("Test 26: Upper triangular matrix (Identity Check)", async function () {
            t++;
            const n = 3n;
            const A_raw = [
                2n, 3n, 4n,
                0n, 5n, 6n,
                0n, 0n, 7n
            ];
            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));

            await touchGas(harness, "luDecomposition", [n, A]);
            const gas = await estimateGas(harness, "luDecomposition", [n, A]);

            const [L, U] = await harness.luDecomposition(n, A);

            // Verification
            const Ls = await Promise.all(L.map(v => harness.toFloat(v)));
            const Us = await Promise.all(U.map(v => harness.toFloat(v)));
            const As = await Promise.all(A.map(v => harness.toFloat(v)));

            const LU_product = multiplyMatrices(3, Ls, Us);

            const expectedHex = `L (Should be Identity):\n${formatMat(L)}`;
            const outHex = `U (Should match A):\n${formatMat(U)}`;

            const expectedDec = `Original A: [${As.map(v => formatScaledInt(v)).join(", ")}]`;
            const outDec = `Rec A':    [${LU_product.map(v => formatScaledInt(v)).join(", ")}]`;

            printBlockRegular({
                t,
                method: "LU Decomposition",
                explanation: "L must be Identity, U must equal A.",
                inHex: `A=[[2,3,4],[0,5,6],[0,0,7]]`,
                expectedHex,
                outHex,
                expectedDec,
                outDec,
                gas,
            });

            expectMatrixSimilar(3, LU_product, As, TOL_DIRECT);
        });

        it("Test 27: Identity matrix (Identity Check)", async function () {
            t++;
            const n = 3n;
            const A_raw = [
                1n, 0n, 0n,
                0n, 1n, 0n,
                0n, 0n, 1n
            ];
            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));

            await touchGas(harness, "luDecomposition", [n, A]);
            const gas = await estimateGas(harness, "luDecomposition", [n, A]);

            const [L, U] = await harness.luDecomposition(n, A);

            // Verification
            const Ls = await Promise.all(L.map(v => harness.toFloat(v)));
            const Us = await Promise.all(U.map(v => harness.toFloat(v)));
            const As = await Promise.all(A.map(v => harness.toFloat(v)));
            const LU_product = multiplyMatrices(3, Ls, Us);

            const expectedHex = `L (Identity):\n${formatMat(L)}`;
            const outHex =      `U (Identity):\n${formatMat(U)}`;

            const expectedDec = `Original A: [${As.map(v => formatScaledInt(v)).join(", ")}]`;
            const outDec =      `Rec A':    [${LU_product.map(v => formatScaledInt(v)).join(", ")}]`;

            printBlockRegular({
                t, 
                method: "LU Decomposition",
                explanation: "Both L and U should be Identity matrices.",
                inHex: "A=I (3x3)",
                expectedHex,
                outHex,
                expectedDec,
                outDec,
                gas,
            });

            expectMatrixSimilar(3, LU_product, As, TOL_DIRECT);
        });

        it("Test 28: Singular matrix (zero pivot)", async function () {
            t++;
            const n = 2n;
            const A_raw = [2n, 4n, 1n, 2n]; // R2 = 0.5 * R1 (Determinant is 0)
            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));

            await expect(harness.luDecomposition(n, A)).to.be.reverted;

            printBlockRegular({
                t, method: "LU Decomposition",
                explanation: "Singular matrix (Det=0) reverts",
                inHex: "A=[[2,4],[1,2]]",
                expectedHex: "Revert",
                outHex: "Revert",
                expectedDec: "Revert",
                outDec: "Revert",
                gas: "N/A",
            });
        });

        it("Test 29: Zero matrix", async function () {
            t++;
            const n = 2n;
            const A_raw = [0n, 0n, 0n, 0n];
            const A = await Promise.all(A_raw.map(v => harness.qFromInt(v)));

            await expect(harness.luDecomposition(n, A)).to.be.reverted;

            printBlockRegular({
                t, method: "LU Decomposition",
                explanation: "Zero matrix: Immediate pivot failure.",
                inHex: "A=[[0,0],[0,0]]",
                expectedHex: "Revert",
                outHex: "Revert",
                expectedDec: "Revert",
                outDec: "Revert",
                gas: "N/A",
            });
        });
    });
});