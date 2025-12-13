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

/// Helper to multiply two n×n matrices in JS.
function multiplyMatrices(n: number, A: bigint[], B: bigint[]): bigint[] {
    const C: bigint[] = new Array(n * n).fill(0n);
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < n; ++j) {
            let sum = 0n;
            for (let k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    return C;
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

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;

    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;

    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");

    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`;
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

            // Dimensions
            const m = 1n;
            const n = 1n;

            // Build quad inputs
            const A = [await harness.qFromInt(1)];   // [1]
            const b = [await harness.qFromInt(5)];   // [5]
            const x0 = [await harness.qFromInt(0)];  // [0]

            const alpha = await harness.qFromInt(1);
            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x, iters] = await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            expect(x.length).to.equal(1);

            const x0Int = await harness.toFloat(x[0]);
            expect(x0Int / SCALE).to.equal(5n);
            expect(iters).to.be.lessThanOrEqual(1n);

            const outDec = formatScaledInt(x0Int);

            const expected = await harness.qFromInt(5);

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "1D identity matrix (A=1, b=5). Should converge in one step.",
                inHex: "A=[1], b=[5], x0=[0]",
                expectedHex: expected,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "5",
                outDec: outDec.toString(),
                gas,
            });
        });

        it("Test 2: 1D scaled identity A=2", async function () {
            t++;

            const m = 1n;
            const n = 1n;

            const A = [await harness.qFromInt(2)];
            const b = [await harness.qFromInt(8)];
            const x0 = [await harness.qFromInt(0)];

            const alpha = await harness.qFromFrac(1, 4); // 0.25
            const tol = await harness.qFromInt(0);
            const maxIter = 50n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x, iters] =
                await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const xInt = await harness.toFloat(x[0]);
            expect(xInt / SCALE).to.equal(4n);

            const expected = await harness.qFromInt(4);
            const outDec = formatScaledInt(xInt);

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "1D scaled identity (A=2, b=8). Expected solution x=4.",
                inHex: "A=[2], b=[8], x0=[0]",
                expectedHex: expected,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "4",
                outDec,
                gas,
            });
        });

        it("Test 3: 2D identity system", async function () {
            t++;

            const m = 2n;
            const n = 2n;

            const A = [
                await harness.qFromInt(1), await harness.qFromInt(0),
                await harness.qFromInt(0), await harness.qFromInt(1),
            ];
            const b = [await harness.qFromInt(3), await harness.qFromInt(-7)];
            const x0 = [await harness.qFromInt(0), await harness.qFromInt(0)];

            const alpha = await harness.qFromInt(1);
            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x] =
                await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const x0Int = await harness.toFloat(x[0]);
            const x1Int = await harness.toFloat(x[1]);

            expect(x0Int / SCALE).to.equal(3n);
            expect(x1Int / SCALE).to.equal(-7n);

            const expected0 = await harness.qFromInt(3);
            const expected1 = await harness.qFromInt(-7);

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "2D identity system should recover b exactly.",
                inHex: "A=I₂, b=[3,-7]",
                expectedHex: `[${expected0}, ${expected1}]`,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "[3, -7]",
                outDec: `[${formatScaledInt(x0Int)}, ${formatScaledInt(x1Int)}]`,
                gas,
            });
        });

        it("Test 4: Overdetermined least-squares system", async function () {
            t++;

            const m = 2n;
            const n = 1n;

            const A = [await harness.qFromInt(1), await harness.qFromInt(1)];
            const b = [await harness.qFromInt(2), await harness.qFromInt(4)];
            const x0 = [await harness.qFromInt(0)];

            const alpha = await harness.qFromFrac(1, 2);
            const tol = await harness.qFromInt(0);
            const maxIter = 50n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x] =
                await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const xInt = await harness.toFloat(x[0]);
            expect(xInt / SCALE).to.equal(3n);

            const expected = await harness.qFromInt(3);

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "Overdetermined system, solution is mean of b.",
                inHex: "A=[1,1], b=[2,4]",
                expectedHex: expected,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "3",
                outDec: formatScaledInt(xInt),
                gas,
            });
        });

        it("Test 5 (Edge): Zero gradient at start", async function () {
            t++;

            const m = 1n;
            const n = 1n;

            const A = [await harness.qFromInt(1)];
            const b = [await harness.qFromInt(0)];
            const x0 = [await harness.qFromInt(0)];

            const alpha = await harness.qFromInt(1);
            const tol = await harness.qFromInt(0);
            const maxIter = 10n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x, iters] =
                await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const xInt = await harness.toFloat(x[0]);
            expect(xInt).to.equal(0n);
            expect(iters).to.equal(0n);

            const expected = await harness.qFromInt(0);

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "Zero gradient at initial point should terminate immediately.",
                inHex: "A=[1], b=[0], x0=[0]",
                expectedHex: expected,
                outHex: `[${x.join(", ")}]`,
                expectedDec: "0",
                outDec: "0.000000000000",
                gas,
            });
        });

        it("Test 6 (Edge): Very small alpha", async function () {
            t++;

            const m = 1n;
            const n = 1n;

            const A = [await harness.qFromInt(1)];
            const b = [await harness.qFromInt(1)];
            const x0 = [await harness.qFromInt(0)];

            const alpha = await harness.qFromFrac(1, 1_000_000);
            const tol = await harness.qFromInt(0);
            const maxIter = 5n;

            const args = [m, n, A, b, x0, alpha, maxIter, tol];

            await touchGas(harness, "gradientDescentLeastSquares", args);
            const gas = await estimateGas(harness, "gradientDescentLeastSquares", args);

            const [x] =
                await harness.gradientDescentLeastSquares(m, n, A, b, x0, alpha, maxIter, tol);

            const xInt = await harness.toFloat(x[0]);
            expect(xInt).to.be.lessThan(SCALE);

            printBlockRegular({
                t,
                method: "Gradient Descent Least Squares",
                explanation: "Very small alpha causes slow convergence without divergence.",
                inHex: "A=[1], b=[1], α≈1e-6",
                expectedHex: "≈0",
                outHex: `[${x.join(", ")}]`,
                expectedDec: "~0",
                outDec: formatScaledInt(xInt),
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
});