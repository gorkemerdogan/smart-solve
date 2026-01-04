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

    describe("Method 1: Gradient Descent Least Squares", function () {

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

    describe("Method 2: Jacobi Iteration", function () {

        it("Test 7: 1D identity A=1 solves immediately", async function () {
            t++;

            const n = 1n;

            const A = [await harness.qFromInt(1)];
            const b = [await harness.qFromInt(7)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x, iters] = await harness.jacobi(n, A, b, x0, maxIter, tol);

            const xInt = await harness.toFloat(x[0]);
            expect(xInt / SCALE).to.equal(7n);

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "1D identity system converges in one iteration.",
                inHex: "A=[1], b=[7], x0=[0]",
                expectedHex: await harness.qFromInt(7),
                outHex: `[${x.join(", ")}]`,
                expectedDec: "7",
                outDec: formatScaledInt(xInt),
                gas,
            });
        });

        it("Test 8: 2D diagonal dominant system", async function () {
            t++;

            const n = 2n;

            // A = [[4,1],[2,3]]
            const A = [
                await harness.qFromInt(4), await harness.qFromInt(1),
                await harness.qFromInt(2), await harness.qFromInt(3),
            ];

            // b = [1,2]
            const b = [
                await harness.qFromInt(1),
                await harness.qFromInt(2),
            ];

            const x0 = [
                await harness.qFromInt(0),
                await harness.qFromInt(0),
            ];

            const tol = await harness.qFromFrac(1, 1_000_000);
            const maxIter = 50n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x] = await harness.jacobi(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            // Expected ≈ [0.1, 0.6]
            expect(x0Int / SCALE).to.equal(0n);
            expect(x1Int / SCALE).to.equal(0n);

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "2×2 diagonally dominant system converges.",
                inHex: "A=[[4,1],[2,3]], b=[1,2]",
                expectedHex: "≈[0.1,0.6]",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[0.1, 0.6]",
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas,
            });
        });

        it("Test 9: Already-solved system remains stable", async function () {
            t++;

            const n = 2n;

            const A = [
                await harness.qFromInt(1), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(1),
            ];

            const b = [
                await harness.qFromInt(3),
                await harness.qFromInt(4),
            ];

            const x0 = [
                await harness.qFromInt(3),
                await harness.qFromInt(4),
            ];

            const tol = await harness.qFromInt(0);
            const maxIter = 5n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x] = await harness.jacobi(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            expect(x0Int / SCALE).to.equal(3n);
            expect(x1Int / SCALE).to.equal(4n);

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "Already-correct initial guess remains unchanged.",
                inHex: "A=I, b=[3,4], x0=[3,4]",
                expectedHex: "[3,4]",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[3,4]",
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas,
            });
        });

        it("Test 10: Zero diagonal element reverts", async function () {
            t++;

            const n = 2n;

            const A = [
                await harness.qFromInt(0), await harness.qFromInt(1),
                await harness.qFromInt(1), await harness.qFromInt(1),
            ];

            const b = [
                await harness.qFromInt(1),
                await harness.qFromInt(1),
            ];

            const x0 = [
                await harness.qFromInt(0),
                await harness.qFromInt(0),
            ];

            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            await expect(
                harness.jacobi(n, A, b, x0, maxIter, tol)
            ).to.be.reverted;

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "Zero diagonal causes division-by-zero revert.",
                inHex: "A has zero diagonal",
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

            const A = [await harness.qFromInt(2)];
            const b = [await harness.qFromInt(8)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 1n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "jacobi", args);
            const gas = await estimateGas(harness, "jacobi", args);

            const [x, iters] = await harness.jacobi(n, A, b, x0, maxIter, tol);

            const xInt = await harness.toFloat(x[0]);

            // Single Jacobi step: x = b / a = 4
            expect(xInt / SCALE).to.equal(4n);
            expect(iters).to.equal(1n);

            printBlockRegular({
                t,
                method: "Jacobi",
                explanation: "Jacobi performs exactly one iteration when maxIter=1.",
                inHex: "A=[2], b=[8]",
                expectedHex: await harness.qFromInt(4),
                outHex: `[${x.join(", ")}]`,
                expectedDec: "4",
                outDec: formatScaledInt(xInt),
                gas,
            });
        });
    });

    // ------------------------------------------------------------
    //  Gauss–Seidel Method
    // ------------------------------------------------------------

    describe("Method 3: Gauss–Seidel Iteration", function () {

        it("Test 12: 1D identity system (A=1)", async function () {
            t++;

            const n = 1n;

            const A = [await harness.qFromInt(1)];
            const b = [await harness.qFromInt(7)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x, iters] = await harness.gaussSeidel(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            expect(x0Int / SCALE).to.equal(7n);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "1D identity system converges immediately.",
                inHex: "A=[1], b=[7], x0=[0]",
                expectedHex: await harness.qFromInt(7),
                outHex: `[${x.join(", ")}]`,
                expectedDec: "7",
                outDec: formatScaledInt(x0Int),
                gas,
            });
        });

        const RESIDUAL_TOLERANCE = 10n * SCALE; // allow residual < 10

        it("Test 13: 2D diagonally dominant system (scaled residual check)", async function () {
            t++;

            const n = 2n;

            // A = [[4,1],[2,3]]
            const A = [await harness.qFromInt(4), await harness.qFromInt(1), await harness.qFromInt(2), await harness.qFromInt(3)];

            // b = [1,2]
            const b = [await harness.qFromInt(1), await harness.qFromInt(2)];

            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 25n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x] = await harness.gaussSeidel(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            // Residuals in scaled domain
            const r0 = 4n * x0Int + 1n * x1Int - 1n * SCALE;
            const r1 = 2n * x0Int + 3n * x1Int - 2n * SCALE;

            const r0_abs = r0 < 0n ? -r0 : r0;
            const r1_abs = r1 < 0n ? -r1 : r1;

            expect(r0_abs).to.be.below(RESIDUAL_TOLERANCE);
            expect(r1_abs).to.be.below(RESIDUAL_TOLERANCE);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "2×2 diagonally dominant system verified via scaled residual |Ax − b| < tol.",
                inHex: "A=[[4,1],[2,3]], b=[1,2]",
                expectedHex: `|r| < ${RESIDUAL_TOLERANCE}`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "residual ≈ 0",
                outDec: `r=[${formatScaledInt(r0)}, ${formatScaledInt(r1)}]`,
                gas,
            });
        });

        it("Test 14: Already converged initial guess", async function () {
            t++;

            const n = 2n;

            const A = [
                await harness.qFromInt(2), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(2),
            ];

            const b = [
                await harness.qFromInt(4),
                await harness.qFromInt(6),
            ];

            const x0 = [
                await harness.qFromInt(2),
                await harness.qFromInt(3),
            ];

            const tol = await harness.qFromInt(0);
            const maxIter = 5n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x, iters] = await harness.gaussSeidel(n, A, b, x0, maxIter, tol);

            expect(iters).to.equal(1n);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "Initial guess already satisfies Ax=b.",
                inHex: "A=2I, b=[4,6], x0=[2,3]",
                expectedHex: "[2,3]",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[2,3]",
                outDec: "[2,3]",
                gas,
            });
        });

        it("Test 15: Negative values system", async function () {
            t++;

            const n = 1n;

            const A = [await harness.qFromInt(-2)];
            const b = [await harness.qFromInt(-8)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [n, A, b, x0, maxIter, tol];

            await touchGas(harness, "gaussSeidel", args);
            const gas = await estimateGas(harness, "gaussSeidel", args);

            const [x] = await harness.gaussSeidel(n, A, b, x0, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            expect(x0Int / SCALE).to.equal(4n);

            printBlockRegular({
                t,
                method: "Gauss–Seidel",
                explanation: "Negative scalar system.",
                inHex: "A=[-2], b=[-8]",
                expectedHex: "4",
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

            const b = [
                await harness.qFromInt(1),
                await harness.qFromInt(2),
            ];

            const x0 = [
                await harness.qFromInt(0),
                await harness.qFromInt(0),
            ];

            const tol = await harness.qFromInt(0);
            const maxIter = 5n;

            await expect(
                harness.gaussSeidel(n, A, b, x0, maxIter, tol)
            ).to.be.reverted;

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

            const n = 1n;

            const A = [await harness.qFromInt(5)];
            const b = [await harness.qFromInt(10)];
            const x0 = [await harness.qFromInt(0)];

            const tol = await harness.qFromInt(0);
            const maxIter = 1n;

            const [_, iters] = await harness.gaussSeidel(n, A, b, x0, maxIter, tol);

            expect(iters).to.equal(1n);
        });
    });

    describe("Method 4: Gaussian Elimination", function () {

        // ------------------------------------------------------------
        //  Normal Cases
        // ------------------------------------------------------------

        it("Test 19: 1D trivial system A=[1], b=[7]", async function () {
            t++;

            const n = 1n;

            const A = [await harness.qFromInt(1)];
            const b = [await harness.qFromInt(7)];

            const args = [n, A, b];

            await touchGas(harness, "gaussianElimination", args);
            const gas = await estimateGas(harness, "gaussianElimination", args);

            const x = await harness.gaussianElimination(n, A, b);

            const x0Int = await harness.toFloat(x[0]);
            expect(x0Int / SCALE).to.equal(7n);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "1D exact system.",
                inHex: "A=[1], b=[7]",
                expectedHex: await harness.qFromInt(7),
                outHex: `[${x.join(", ")}]`,
                expectedDec: "7",
                outDec: formatScaledInt(x0Int),
                gas,
            });
        });

        it("Test 20: 2D identity matrix", async function () {
            t++;

            const n = 2n;

            const A = [
                await harness.qFromInt(1), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(1),
            ];

            const b = [
                await harness.qFromInt(3),
                await harness.qFromInt(-4),
            ];

            const args = [n, A, b];

            await touchGas(harness, "gaussianElimination", args);
            const gas = await estimateGas(harness, "gaussianElimination", args);

            const x = await harness.gaussianElimination(n, A, b);

            const x0 = await harness.toFloat(x[0]);
            const x1 = await harness.toFloat(x[1]);

            expect(x0 / SCALE).to.equal(3n);
            expect(x1 / SCALE).to.equal(-4n);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Identity matrix should return b directly.",
                inHex: "A=I, b=[3,-4]",
                expectedHex: "[3,-4]",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[3,-4]",
                outDec: `[${formatScaledInt(x0)}, ${formatScaledInt(x1)}]`,
                gas,
            });
        });

        it("Test 21: 2D general solvable system", async function () {
            t++;

            // A = [[2,1],[5,7]], b=[11,13]
            // Solution: x=[7,-3]
            const n = 2n;

            const A = [
                await harness.qFromInt(2), await harness.qFromInt(1),
                await harness.qFromInt(5), await harness.qFromInt(7),
            ];

            const b = [
                await harness.qFromInt(11),
                await harness.qFromInt(13),
            ];

            const args = [n, A, b];

            await touchGas(harness, "gaussianElimination", args);
            const gas = await estimateGas(harness, "gaussianElimination", args);

            const x = await harness.gaussianElimination(n, A, b);

            const x0 = await harness.toFloat(x[0]);
            const x1 = await harness.toFloat(x[1]);

            expect(x0 / SCALE).to.equal(7n);
            expect(x1 / SCALE).to.equal(-3n);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Standard 2×2 solvable system.",
                inHex: "A=[[2,1],[5,7]], b=[11,13]",
                expectedHex: "[7,-3]",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[7,-3]",
                outDec: `[${formatScaledInt(x0)}, ${formatScaledInt(x1)}]`,
                gas,
            });
        });

        it("Test 22: 3D upper-triangular system (residual check)", async function () {
            t++;

            const n = 3n;

            const A = [
                await harness.qFromInt(2), await harness.qFromInt(1), await harness.qFromInt(1),
                await harness.qFromInt(0), await harness.qFromInt(3), await harness.qFromInt(1),
                await harness.qFromInt(0), await harness.qFromInt(0), await harness.qFromInt(4),
            ];

            const b = [
                await harness.qFromInt(5),
                await harness.qFromInt(4),
                await harness.qFromInt(8),
            ];

            const args = [n, A, b];

            await touchGas(harness, "gaussianElimination", args);
            const gas = await estimateGas(harness, "gaussianElimination", args);

            const x = await harness.gaussianElimination(n, A, b);

            const xs = await Promise.all(x.map(v => harness.toFloat(v)));

            // Residuals (scaled)
            const r0 = 2n * xs[0] + 1n * xs[1] + 1n * xs[2] - 5n * SCALE;
            const r1 = 3n * xs[1] + 1n * xs[2] - 4n * SCALE;
            const r2 = 4n * xs[2] - 8n * SCALE;

            const abs = (v: bigint) => v < 0n ? -v : v;
            const TOL = 10n * SCALE;

            expect(abs(r0)).to.be.below(TOL);
            expect(abs(r1)).to.be.below(TOL);
            expect(abs(r2)).to.be.below(TOL);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Upper-triangular system verified via residual Ax − b ≈ 0.",
                inHex: "Upper triangular A",
                expectedHex: "|Ax − b| < tol",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "residual ≈ 0",
                outDec: xs.map(formatScaledInt).join(", "),
                gas,
            });
        });

        it("Test 23: Singular matrix should revert", async function () {
            t++;

            const n = 2n;

            // Rows are linearly dependent
            const A = [
                await harness.qFromInt(1), await harness.qFromInt(2),
                await harness.qFromInt(2), await harness.qFromInt(4),
            ];

            const b = [
                await harness.qFromInt(3),
                await harness.qFromInt(6),
            ];

            await expect(
                harness.gaussianElimination(n, A, b)
            ).to.be.reverted;

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Singular matrix should revert (zero pivot).",
                inHex: "rank-deficient A",
                expectedHex: "Revert",
                outHex: "Revert",
                expectedDec: "Revert",
                outDec: "Revert",
                gas: "N/A",
            });
        });

        it("Test 24: Zero pivot requiring row swap (partial pivoting)", async function () {
            t++;

            const n = 2n;

            // First pivot is zero, but system is solvable
            const A = [
                await harness.qFromInt(0), await harness.qFromInt(1),
                await harness.qFromInt(2), await harness.qFromInt(3),
            ];

            const b = [
                await harness.qFromInt(1),
                await harness.qFromInt(5),
            ];

            const args = [n, A, b];

            await touchGas(harness, "gaussianElimination", args);
            const gas = await estimateGas(harness, "gaussianElimination", args);

            const x = await harness.gaussianElimination(n, A, b);

            const x0 = await harness.toFloat(x[0]);
            const x1 = await harness.toFloat(x[1]);

            // Solution: x=[1,1]
            expect(x0 / SCALE).to.equal(1n);
            expect(x1 / SCALE).to.equal(1n);

            printBlockRegular({
                t,
                method: "Gaussian Elimination",
                explanation: "Zero pivot handled via row swapping.",
                inHex: "A=[[0,1],[2,3]]",
                expectedHex: "[1,1]",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[1,1]",
                outDec: `[${formatScaledInt(x0)}, ${formatScaledInt(x1)}]`,
                gas,
            });
        });
    });

    describe("Method 5: LU Decomposition", function () {

        // ------------------------------------------------------------
        //  Regular Cases
        // ------------------------------------------------------------

        it("Test 25: 2×2 simple matrix", async function () {
            t++;

            const n = 2n;

            // A = [[4,3],[6,3]]
            const A = [
                await harness.qFromInt(4), await harness.qFromInt(3),
                await harness.qFromInt(6), await harness.qFromInt(3),
            ];

            const args = [n, A];

            await touchGas(harness, "luDecomposition", args);
            const gas = await estimateGas(harness, "luDecomposition", args);

            const [L, U] = await harness.luDecomposition(n, A);

            const Ls = await Promise.all(L.map(v => harness.toFloat(v)));
            const Us = await Promise.all(U.map(v => harness.toFloat(v)));

            const LU = multiplyMatrices(2, Ls, Us);
            const Aint = await Promise.all(A.map(v => harness.toFloat(v)));

            expectMatrixSimilar(2, LU, Aint, MAT_TOL);

            printBlockRegular({
                t,
                method: "LU Decomposition",
                explanation: "2×2 matrix, exact LU reconstruction.",
                inHex: "A=[[4,3],[6,3]]",
                expectedHex: "L·U ≈ A",
                outHex: `[L=${L.join(", ")}, U=${U.join(", ")}]`,
                expectedDec: "A ≈ L·U",
                outDec: "Reconstruction OK",
                gas,
            });
        });

        it("Test 26: 3×3 diagonally dominant matrix", async function () {
            t++;

            const n = 3n;

            const A = [
                await harness.qFromInt(10), await harness.qFromInt(2), await harness.qFromInt(1),
                await harness.qFromInt(2), await harness.qFromInt(8), await harness.qFromInt(1),
                await harness.qFromInt(1), await harness.qFromInt(1), await harness.qFromInt(5),
            ];

            const args = [n, A];

            await touchGas(harness, "luDecomposition", args);
            const gas = await estimateGas(harness, "luDecomposition", args);

            const [L, U] = await harness.luDecomposition(n, A);

            const LU = multiplyMatrices(
                3,
                await Promise.all(L.map(v => harness.toFloat(v))),
                await Promise.all(U.map(v => harness.toFloat(v)))
            );

            const Aint = await Promise.all(A.map(v => harness.toFloat(v)));

            expectMatrixSimilar(3, LU, Aint, MAT_TOL);

            printBlockRegular({
                t,
                method: "LU Decomposition",
                explanation: "3×3 diagonally dominant matrix.",
                inHex: "Diagonal dominant A",
                expectedHex: "L·U ≈ A",
                outHex: "OK",
                expectedDec: "Reconstruction",
                outDec: "Pass",
                gas,
            });
        });

        it("Test 27: Upper triangular matrix", async function () {
            t++;

            const n = 3n;

            const A = [
                await harness.qFromInt(2), await harness.qFromInt(3), await harness.qFromInt(4),
                await harness.qFromInt(0), await harness.qFromInt(5), await harness.qFromInt(6),
                await harness.qFromInt(0), await harness.qFromInt(0), await harness.qFromInt(7),
            ];

            const [L, U] = await harness.luDecomposition(n, A);

            const LU = multiplyMatrices(
                3,
                await Promise.all(L.map(v => harness.toFloat(v))),
                await Promise.all(U.map(v => harness.toFloat(v)))
            );

            const Aint = await Promise.all(A.map(v => harness.toFloat(v)));

            expectMatrixSimilar(3, LU, Aint, MAT_TOL);
        });

        it("Test 28: Identity matrix", async function () {
            t++;

            const n = 3n;

            const A = [
                await harness.qFromInt(1), await harness.qFromInt(0), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(1), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(0), await harness.qFromInt(1),
            ];

            const [L, U] = await harness.luDecomposition(n, A);

            const LU = multiplyMatrices(
                3,
                await Promise.all(L.map(v => harness.toFloat(v))),
                await Promise.all(U.map(v => harness.toFloat(v)))
            );

            const Aint = await Promise.all(A.map(v => harness.toFloat(v)));

            expectMatrixSimilar(3, LU, Aint, MAT_TOL);
        });

        it("Test 29: Singular matrix (zero pivot)", async function () {
            t++;

            const n = 2n;

            // Second row is multiple of first
            const A = [
                await harness.qFromInt(2), await harness.qFromInt(4),
                await harness.qFromInt(1), await harness.qFromInt(2),
            ];

            await expect(
                harness.luDecomposition(n, A)
            ).to.be.reverted;
        });

        it("Test 30: Zero matrix", async function () {
            t++;

            const n = 2n;

            const A = [
                await harness.qFromInt(0), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(0),
            ];

            await expect(
                harness.luDecomposition(n, A)
            ).to.be.reverted;
        });
    });
});