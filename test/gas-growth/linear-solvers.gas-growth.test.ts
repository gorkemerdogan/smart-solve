// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
//  Types
// ------------------------------------------------------------

type LinearSolversHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
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

    gaussianElimination(n: bigint, Adata: string[], bdata: string[]): Promise<string[]>;
    luDecomposition(n: bigint, Adata: string[]): Promise<[string[], string[]]>;
};

// ------------------------------------------------------------
//  Constants & Helpers
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

type IterMethod = "jacobi" | "gaussSeidel" | "gradientDescentLeastSquares";
type DirectMethod = "gaussianElimination" | "luDecomposition";
type AnyMethod = IterMethod | DirectMethod;

type MatrixCase = {
    n: number;
    A: bigint[];
    b: bigint[];
    xTrue: bigint[];
};

function idx(n: number, i: number, j: number): number {
    return i * n + j;
}

async function qScaledArrayFromBigints(
    harness: LinearSolversHarness,
    arr: bigint[],
    factorNum: bigint,
    factorDen: bigint
): Promise<string[]> {
    return Promise.all(
        arr.map(v => {
            if (factorDen === 1n) {
                return harness.qFromInt(v * factorNum);
            }
            return harness.qFromFrac(v * factorNum, factorDen);
        })
    );
}

/**
 * Build a dense diagonally dominant matrix family with known solution xTrue=[1..n].
 * A(i,j) = scale * (diagBase if i==j else offDiagBase + ((i+j)%2))
 */
function buildSystem(n: number, scale: bigint = 1n): MatrixCase {
    const xTrue = Array.from({ length: n }, (_, i) => BigInt(i + 1));

    const A: bigint[] = new Array(n * n).fill(0n);
    const diagBase = 2n * BigInt(n) + 3n;
    const offDiagBase = 1n;

    for (let i = 0; i < n; i++) {
        let rowSum = 0n;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const off = offDiagBase + BigInt((i + j) % 2);
            A[idx(n, i, j)] = off * scale;
            rowSum += off;
        }
        A[idx(n, i, i)] = (diagBase + rowSum) * scale;
    }

    const b: bigint[] = new Array(n).fill(0n);
    for (let i = 0; i < n; i++) {
        let sum = 0n;
        for (let j = 0; j < n; j++) {
            sum += A[idx(n, i, j)] * xTrue[j];
        }
        b[i] = sum;
    }

    return { n, A, b, xTrue };
}

function makeGuess(kind: "zero" | "near" | "farPositive" | "farNegative" | "mixed", xTrue: bigint[]): bigint[] {
    const n = xTrue.length;

    if (kind === "zero") return new Array(n).fill(0n);
    if (kind === "near") return xTrue.map((v, i) => v + BigInt(i % 2 === 0 ? 1 : -1));
    if (kind === "farPositive") return xTrue.map((v, i) => v + 50n + BigInt(i));
    if (kind === "farNegative") return xTrue.map((v, i) => -(v + 50n + BigInt(i)));
    return xTrue.map((v, i) => (i % 2 === 0 ? v + 25n : -(v + 25n)));
}

async function qArrayFromBigints(harness: LinearSolversHarness, arr: bigint[]): Promise<string[]> {
    return Promise.all(arr.map(v => harness.qFromInt(v)));
}

async function toScaledArray(harness: LinearSolversHarness, arr: string[]): Promise<bigint[]> {
    return Promise.all(arr.map(v => harness.toFloat(v)));
}

async function runDirectCase(
    harness: LinearSolversHarness,
    method: DirectMethod,
    n: number,
    A: string[],
    b?: string[]
) {
    if (method === "gaussianElimination") {
        const args = [BigInt(n), A, b!];
        await touchGas(harness, method, args);
        const gas = await estimateGas(harness, method, args);
        const out = await harness.gaussianElimination(BigInt(n), A, b!);
        return { gas, out };
    }

    const args = [BigInt(n), A];
    await touchGas(harness, method, args);
    const gas = await estimateGas(harness, method, args);
    const out = await harness.luDecomposition(BigInt(n), A);
    return { gas, out };
}

async function runIterCase(
    harness: LinearSolversHarness,
    method: IterMethod,
    n: number,
    A: string[],
    b: string[],
    x0: string[],
    maxIter: bigint,
    tol: string,
    alpha?: string
) {
    if (method === "gradientDescentLeastSquares") {
        const args = [BigInt(n), BigInt(n), A, b, x0, alpha!, maxIter, tol];
        await touchGas(harness, method, args);
        const gas = await estimateGas(harness, method, args);
        const out = await harness.gradientDescentLeastSquares(BigInt(n), BigInt(n), A, b, x0, alpha!, maxIter, tol);
        return { gas, out };
    }

    const args = [BigInt(n), A, b, x0, maxIter, tol];
    await touchGas(harness, method, args);
    const gas = await estimateGas(harness, method, args);
    const out =
        method === "jacobi"
            ? await harness.jacobi(BigInt(n), A, b, x0, maxIter, tol)
            : await harness.gaussSeidel(BigInt(n), A, b, x0, maxIter, tol);

    return { gas, out };
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("LinearSolversHarness - Gas Growth Tests", function () {
    let harness: LinearSolversHarness;

    let GD_ALPHA: string;
    let GD_TOL: string;
    let ITER_TOL: string;

    before(async function () {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const Factory = await ethers.getContractFactory("LinearSolversHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as LinearSolversHarness;
        await harness.waitForDeployment();

        GD_ALPHA = await harness.qFromFrac(1, 500); // 0.002
        GD_TOL = await harness.qFromInt(0); // avoid early stop
        ITER_TOL = await harness.qFromInt(0); // avoid early stop
    });

    // ------------------------------------------------------------
    //  Section 1: Gas Growth with Respect to Matrix Dimension
    // ------------------------------------------------------------

    describe("Section 1: Gas Growth with Respect to Matrix Dimension", function () {
        const DIM_CASES = [2, 3, 4, 5];

        describe("Section 1.1: Direct Methods", function () {
            let testNo = 0;

            const METHODS: Array<{ method: DirectMethod; label: string }> = [
                { method: "gaussianElimination", label: "Gaussian Elimination" },
                { method: "luDecomposition", label: "LU Decomposition" },
            ];

            for (const m of METHODS) {
                for (const n of DIM_CASES) {
                    const t = `1.1.${++testNo}`;

                    it(`Test ${t}: ${m.label} gas growth with matrix dimension n=${n}`, async function () {
                        const sys = buildSystem(n);
                        const Aq = await qArrayFromBigints(harness, sys.A);
                        const bq = await qArrayFromBigints(harness, sys.b);

                        const result =
                            m.method === "gaussianElimination"
                                ? await runDirectCase(harness, m.method, n, Aq, bq)
                                : await runDirectCase(harness, m.method, n, Aq);

                        printBlockRegular({
                            t,
                            method: m.label,
                            explanation: `Gas growth with respect to matrix dimension for dense diagonally dominant system of size n=${n}.`,
                            gas: result.gas,
                            inHex: `n=${n}, dense diagonally dominant`,
                            expectedHex: "N/A",
                            outHex: m.method === "gaussianElimination" ? `[solution vector]` : `[L,U matrices]`,
                            expectedDec: "N/A",
                            outDec: m.method === "gaussianElimination" ? "solution returned" : "decomposition returned",
                        });

                        expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                    });
                }
            }
        });

        describe("Section 1.2: Iterative Methods", function () {
            let testNo = 0;

            const METHODS: Array<{ method: IterMethod; label: string }> = [
                { method: "jacobi", label: "Jacobi" },
                { method: "gaussSeidel", label: "Gauss-Seidel" },
                { method: "gradientDescentLeastSquares", label: "Gradient Descent Least Squares" },
            ];

            const FIXED_MAX_ITER = 20n;

            for (const m of METHODS) {
                for (const n of DIM_CASES) {
                    const t = `1.2.${++testNo}`;

                    it(`Test ${t}: ${m.label} gas growth with matrix dimension n=${n}`, async function () {
                        const sys = buildSystem(n);
                        const Aq = await qArrayFromBigints(harness, sys.A);
                        const bq = await qArrayFromBigints(harness, sys.b);
                        const x0q = await qArrayFromBigints(harness, makeGuess("zero", sys.xTrue));

                        const result = await runIterCase(
                            harness,
                            m.method,
                            n,
                            Aq,
                            bq,
                            x0q,
                            FIXED_MAX_ITER,
                            m.method === "gradientDescentLeastSquares" ? GD_TOL : ITER_TOL,
                            GD_ALPHA
                        );

                        const [, iters] = result.out as [string[], bigint];

                        printBlockRegular({
                            t,
                            method: m.label,
                            explanation: `Gas growth with respect to matrix dimension for dense diagonally dominant system of size n=${n} with fixed iteration budget ${FIXED_MAX_ITER}.`,
                            gas: result.gas,
                            inHex: `n=${n}, maxIter=${FIXED_MAX_ITER}`,
                            expectedHex: "N/A",
                            outHex: `iter=${iters}`,
                            expectedDec: "N/A",
                            outDec: "solution returned",
                        });

                        expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                    });
                }
            }
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Gas Growth with Respect to Iteration Budget
    // ------------------------------------------------------------

    describe("Section 2: Gas Growth with Respect to Iteration Budget", function () {
        let testNo = 0;

        const ITER_CASES = [1n, 5n, 10n, 20n, 50n];
        const n = 3;

        const METHODS: Array<{ method: IterMethod; label: string }> = [
            { method: "jacobi", label: "Jacobi" },
            { method: "gaussSeidel", label: "Gauss-Seidel" },
            { method: "gradientDescentLeastSquares", label: "Gradient Descent Least Squares" },
        ];

        for (const m of METHODS) {
            for (const maxIter of ITER_CASES) {
                const t = `2.${++testNo}`;

                it(`Test ${t}: ${m.label} gas growth with maxIter=${maxIter}`, async function () {
                    const sys = buildSystem(n);
                    const Aq = await qArrayFromBigints(harness, sys.A);
                    const bq = await qArrayFromBigints(harness, sys.b);
                    const x0q = await qArrayFromBigints(harness, makeGuess("zero", sys.xTrue));

                    const result = await runIterCase(
                        harness,
                        m.method,
                        n,
                        Aq,
                        bq,
                        x0q,
                        maxIter,
                        m.method === "gradientDescentLeastSquares" ? GD_TOL : ITER_TOL,
                        GD_ALPHA
                    );

                    const [, iters] = result.out as [string[], bigint];

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas growth with respect to iteration budget for fixed 3x3 dense diagonally dominant system with maxIter=${maxIter}.`,
                        gas: result.gas,
                        inHex: `n=3, maxIter=${maxIter}`,
                        expectedHex: "N/A",
                        outHex: `iter=${iters}`,
                        expectedDec: "N/A",
                        outDec: "solution returned",
                    });

                    expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    //  Section 3: Gas Sensitivity to Initial Guess
    // ------------------------------------------------------------

    describe("Section 3: Gas Sensitivity to Initial Guess", function () {
        let testNo = 0;

        const n = 3;
        const FIXED_MAX_ITER = 20n;

        const GUESS_CASES: Array<{ key: "zero" | "near" | "farPositive" | "farNegative" | "mixed"; label: string }> = [
            { key: "zero", label: "zero vector" },
            { key: "near", label: "near-solution vector" },
            { key: "farPositive", label: "far positive vector" },
            { key: "farNegative", label: "far negative vector" },
            { key: "mixed", label: "mixed-sign vector" },
        ];

        const METHODS: Array<{ method: IterMethod; label: string }> = [
            { method: "jacobi", label: "Jacobi" },
            { method: "gaussSeidel", label: "Gauss-Seidel" },
            { method: "gradientDescentLeastSquares", label: "Gradient Descent Least Squares" },
        ];

        for (const m of METHODS) {
            for (const g of GUESS_CASES) {
                const t = `3.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for ${g.label}`, async function () {
                    const sys = buildSystem(n);
                    const Aq = await qArrayFromBigints(harness, sys.A);
                    const bq = await qArrayFromBigints(harness, sys.b);
                    const x0q = await qArrayFromBigints(harness, makeGuess(g.key, sys.xTrue));

                    const result = await runIterCase(
                        harness,
                        m.method,
                        n,
                        Aq,
                        bq,
                        x0q,
                        FIXED_MAX_ITER,
                        m.method === "gradientDescentLeastSquares" ? GD_TOL : ITER_TOL,
                        GD_ALPHA
                    );

                    const [, iters] = result.out as [string[], bigint];

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to initial guess using fixed 3x3 dense diagonally dominant system, fixed maxIter=${FIXED_MAX_ITER}, and ${g.label}.`,
                        gas: result.gas,
                        inHex: `n=3, x0=${g.label}`,
                        expectedHex: "N/A",
                        outHex: `iter=${iters}`,
                        expectedDec: "N/A",
                        outDec: "solution returned",
                    });

                    expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    //  Section 4: Gas Sensitivity to Coefficient Scale
    // ------------------------------------------------------------

    describe("Section 4: Gas Sensitivity to Coefficient Scale", function () {
        let testNo = 0;

        const n = 3;

        const SCALE_CASES: Array<{ label: string; factorNum: bigint; factorDen: bigint }> = [
            { label: "verySmall", factorNum: 1n, factorDen: 1_000_000n }, // 1e-6
            { label: "small", factorNum: 1n, factorDen: 1_000n }, // 1e-3
            { label: "normal", factorNum: 1n, factorDen: 1n }, // 1
            { label: "large", factorNum: 1_000n, factorDen: 1n }, // 1e3
            { label: "huge", factorNum: 1_000_000n, factorDen: 1n }, // 1e6
        ];

        const DIRECT_METHODS: Array<{ method: DirectMethod; label: string }> = [
            { method: "gaussianElimination", label: "Gaussian Elimination" },
            { method: "luDecomposition", label: "LU Decomposition" },
        ];

        const ITER_METHODS: Array<{ method: IterMethod; label: string }> = [
            { method: "jacobi", label: "Jacobi" },
            { method: "gaussSeidel", label: "Gauss-Seidel" },
            { method: "gradientDescentLeastSquares", label: "Gradient Descent Least Squares" },
        ];

        for (const m of DIRECT_METHODS) {
            for (const s of SCALE_CASES) {
                const t = `4.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for coefficient scale ${s.label}`, async function () {
                    const base = buildSystem(n);

                    const Aq = await qScaledArrayFromBigints(harness, base.A, s.factorNum, s.factorDen);
                    const bq = await qScaledArrayFromBigints(harness, base.b, s.factorNum, s.factorDen);

                    const result =
                        m.method === "gaussianElimination"
                            ? await runDirectCase(harness, m.method, n, Aq, bq)
                            : await runDirectCase(harness, m.method, n, Aq);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to coefficient scale using fixed 3x3 system and ${s.label} scaling.`,
                        gas: result.gas,
                        inHex: `n=3, scale=${s.label}`,
                        expectedHex: "N/A",
                        outHex: m.method === "gaussianElimination" ? `[solution vector]` : `[L,U matrices]`,
                        expectedDec: "N/A",
                        outDec: "result returned",
                    });

                    expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                });
            }
        }

        for (const m of ITER_METHODS) {
            for (const s of SCALE_CASES) {
                const t = `4.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for coefficient scale ${s.label}`, async function () {
                    const base = buildSystem(n);

                    const Aq = await qScaledArrayFromBigints(harness, base.A, s.factorNum, s.factorDen);
                    const bq = await qScaledArrayFromBigints(harness, base.b, s.factorNum, s.factorDen);
                    const x0q = await qArrayFromBigints(harness, makeGuess("zero", base.xTrue));

                    const result = await runIterCase(
                        harness,
                        m.method,
                        n,
                        Aq,
                        bq,
                        x0q,
                        20n,
                        m.method === "gradientDescentLeastSquares" ? GD_TOL : ITER_TOL,
                        GD_ALPHA
                    );

                    const [, iters] = result.out as [string[], bigint];

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to coefficient scale using fixed 3x3 system, fixed maxIter=20, and ${s.label} scaling.`,
                        gas: result.gas,
                        inHex: `n=3, scale=${s.label}`,
                        expectedHex: "N/A",
                        outHex: `iter=${iters}`,
                        expectedDec: "N/A",
                        outDec: "solution returned",
                    });

                    expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                });
            }
        }
    });
});