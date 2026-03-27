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

type IterMethod = "jacobi" | "gaussSeidel" | "gradientDescentLeastSquares";
type DirectMethod = "gaussianElimination" | "luDecomposition";

type MatrixCase = {
    n: number;
    A: bigint[];
    b: bigint[];
    xTrue: bigint[];
};

function idx(n: number, i: number, j: number): number {
    return i * n + j;
}

function toGasBigInt(gas: unknown): bigint {
    if (typeof gas === "bigint") return gas;
    return BigInt(gas as string);
}

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function padRight(s: string, len: number): string {
    return s.length >= len ? s : s + " ".repeat(len - s.length);
}

function splitEvery<T>(arr: T[], width: number): T[][] {
    const out: T[][] = [];
    for (let i = 0; i < arr.length; i += width) {
        out.push(arr.slice(i, i + width));
    }
    return out;
}

function maxAbs(arr: bigint[]): bigint {
    let best = 0n;
    for (const v of arr) {
        const a = v < 0n ? -v : v;
        if (a > best) best = a;
    }
    return best;
}

function subVec(a: bigint[], b: bigint[]): bigint[] {
    return a.map((v, i) => v - b[i]);
}

function matVecMulScaled(A: bigint[], x: bigint[], n: number): bigint[] {
    const out = new Array<bigint>(n).fill(0n);
    for (let i = 0; i < n; i++) {
        let sum = 0n;
        for (let j = 0; j < n; j++) {
            sum += (A[idx(n, i, j)] * x[j]) / SCALE;
        }
        out[i] = sum;
    }
    return out;
}

function matMulScaled(A: bigint[], B: bigint[], n: number): bigint[] {
    const out = new Array<bigint>(n * n).fill(0n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0n;
            for (let k = 0; k < n; k++) {
                sum += (A[idx(n, i, k)] * B[idx(n, k, j)]) / SCALE;
            }
            out[idx(n, i, j)] = sum;
        }
    }
    return out;
}

function maxAbsMatDiff(A: bigint[], B: bigint[]): bigint {
    let best = 0n;
    for (let i = 0; i < A.length; i++) {
        const d = A[i] - B[i];
        const a = d < 0n ? -d : d;
        if (a > best) best = a;
    }
    return best;
}

function formatVectorScaled(arr: bigint[], label = "vec", limit = 8): string {
    const shown = arr.slice(0, limit).map(formatScaledInt);
    const suffix = arr.length > limit ? ", ..." : "";
    return `${label}=[${shown.join(", ")}${suffix}]`;
}

function formatMatrixScaled(flat: bigint[], n: number, label = "M", maxRows = 4, maxCols = 4): string {
    const rows = splitEvery(flat, n);
    const shownRows = rows.slice(0, maxRows).map(r => {
        const shownCols = r.slice(0, maxCols).map(v => padRight(formatScaledInt(v), 12));
        const suffix = r.length > maxCols ? " ..." : "";
        return `[ ${shownCols.join("  ")}${suffix} ]`;
    });

    return `${label}=\n${shownRows.join("\n")}${rows.length > maxRows ? "\n..." : ""}`;
}

function describeScale(factorNum: bigint, factorDen: bigint): string {
    if (factorDen === 1n) return factorNum.toString();
    return `${factorNum.toString()}/${factorDen.toString()}`;
}

async function qArrayFromBigints(harness: LinearSolversHarness, arr: bigint[]): Promise<string[]> {
    return Promise.all(arr.map(v => harness.qFromInt(v)));
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

async function toScaledArray(harness: LinearSolversHarness, arr: string[]): Promise<bigint[]> {
    return Promise.all(arr.map(v => harness.toFloat(v)));
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

function makeGuess(
    kind: "zero" | "near" | "farPositive" | "farNegative" | "mixed",
    xTrue: bigint[]
): bigint[] {
    const n = xTrue.length;

    if (kind === "zero") return new Array(n).fill(0n);
    if (kind === "near") return xTrue.map((v, i) => v + BigInt(i % 2 === 0 ? 1 : -1));
    if (kind === "farPositive") return xTrue.map((v, i) => v + 50n + BigInt(i));
    if (kind === "farNegative") return xTrue.map((v, i) => -(v + 50n + BigInt(i)));
    return xTrue.map((v, i) => (i % 2 === 0 ? v + 25n : -(v + 25n)));
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

async function buildGaussianOutput(
    harness: LinearSolversHarness,
    n: number,
    Aq: string[],
    bq: string[],
    xOutQ: string[],
    xTrueInts: bigint[]
): Promise<{ inHex: string; outHex: string; expectedDec: string; outDec: string }> {
    const AScaled = await toScaledArray(harness, Aq);
    const bScaled = await toScaledArray(harness, bq);
    const xScaled = await toScaledArray(harness, xOutQ);
    const xTrueScaled = xTrueInts.map(v => v * SCALE);

    const xErr = subVec(xScaled, xTrueScaled);
    const residual = subVec(matVecMulScaled(AScaled, xScaled, n), bScaled);

    return {
        inHex: `n=${n}, A=scaled dense system, b=scaled RHS`,
        outHex: `x=[${xOutQ.join(", ")}]`,
        expectedDec: [
            formatVectorScaled(xTrueScaled, "xTrue"),
            formatMatrixScaled(AScaled, n, "A"),
            formatVectorScaled(bScaled, "b"),
        ].join("\n"),
        outDec: [
            formatVectorScaled(xScaled, "xApprox"),
            formatVectorScaled(xErr, "xError"),
            formatVectorScaled(residual, "residual"),
            `max|x-xTrue|=${formatScaledInt(maxAbs(xErr))}`,
            `max|Ax-b|=${formatScaledInt(maxAbs(residual))}`,
        ].join("\n"),
    };
}

async function buildIterativeOutput(
    harness: LinearSolversHarness,
    n: number,
    Aq: string[],
    bq: string[],
    x0q: string[],
    out: [string[], bigint],
    xTrueInts: bigint[],
    extra?: { alpha?: string; maxIter?: bigint }
): Promise<{ inHex: string; outHex: string; expectedDec: string; outDec: string }> {
    const [xOutQ, iters] = out;

    const AScaledSafe = await safeToScaledArray(harness, Aq);
    const bScaledSafe = await safeToScaledArray(harness, bq);
    const x0ScaledSafe = await safeToScaledArray(harness, x0q);
    const xScaledSafe = await safeToScaledArray(harness, xOutQ);
    const xTrueScaled = xTrueInts.map(v => v * SCALE);

    const alphaSafe = extra?.alpha ? await safeToScaled(harness, extra.alpha) : undefined;

    const expectedLines = [
        formatVectorScaled(xTrueScaled, "xTrue"),
        formatSafeVector(x0ScaledSafe, "x0"),
        alphaSafe
            ? alphaSafe.ok
                ? `alpha=${formatScaledInt(alphaSafe.value)}`
                : `alpha=${alphaSafe.raw} (${alphaSafe.reason})`
            : "",
        allSafeOk(AScaledSafe) ? formatMatrixScaled(AScaledSafe.map(v => v.value), n, "A") : "A=[unprintable via toFloat; raw quad values used]",
        formatSafeVector(bScaledSafe, "b"),
    ].filter(Boolean);

    const outLines = [
        formatSafeVector(xScaledSafe, "xApprox"),
        `iterations=${iters.toString()}`,
    ];

    if (allSafeOk(AScaledSafe) && allSafeOk(bScaledSafe) && allSafeOk(xScaledSafe)) {
        const AScaled = AScaledSafe.map(v => v.value);
        const bScaled = bScaledSafe.map(v => v.value);
        const xScaled = xScaledSafe.map(v => v.value);

        const xErr = subVec(xScaled, xTrueScaled);
        const residual = subVec(matVecMulScaled(AScaled, xScaled, n), bScaled);

        outLines.push(
            formatVectorScaled(xErr, "xError"),
            formatVectorScaled(residual, "residual"),
            `max|x-xTrue|=${formatScaledInt(maxAbs(xErr))}`,
            `max|Ax-b|=${formatScaledInt(maxAbs(residual))}`
        );
    } else {
        outLines.push("xError/residual skipped because one or more values could not be converted with toFloat()");
    }

    return {
        inHex: `n=${n}, maxIter=${extra?.maxIter ?? "N/A"}, x0=[${x0q.join(", ")}]`,
        outHex: `x=[${xOutQ.join(", ")}], iter=${iters.toString()}`,
        expectedDec: expectedLines.join("\n"),
        outDec: outLines.join("\n"),
    };
}

async function buildLUOutput(
    harness: LinearSolversHarness,
    n: number,
    Aq: string[],
    out: [string[], string[]]
): Promise<{ inHex: string; outHex: string; expectedDec: string; outDec: string }> {
    const [Lq, Uq] = out;
    const AScaled = await toScaledArray(harness, Aq);
    const LScaled = await toScaledArray(harness, Lq);
    const UScaled = await toScaledArray(harness, Uq);

    const LU = matMulScaled(LScaled, UScaled, n);
    const reconErr = maxAbsMatDiff(LU, AScaled);

    return {
        inHex: `n=${n}, LU factorization`,
        outHex: `L=[${Lq.join(", ")}], U=[${Uq.join(", ")}]`,
        expectedDec: formatMatrixScaled(AScaled, n, "A"),
        outDec: [
            formatMatrixScaled(LScaled, n, "L"),
            formatMatrixScaled(UScaled, n, "U"),
            formatMatrixScaled(LU, n, "L*U"),
            `max|LU-A|=${formatScaledInt(reconErr)}`,
        ].join("\n"),
    };
}

async function safeToScaled(
    harness: LinearSolversHarness,
    q: string
): Promise<{ ok: true; value: bigint } | { ok: false; raw: string; reason: string }> {
    try {
        const value = await harness.toFloat(q);
        return { ok: true, value };
    } catch {
        return { ok: false, raw: q, reason: "toFloat reverted" };
    }
}

async function safeToScaledArray(
    harness: LinearSolversHarness,
    arr: string[]
): Promise<Array<{ ok: true; value: bigint } | { ok: false; raw: string; reason: string }>> {
    return Promise.all(arr.map(v => safeToScaled(harness, v)));
}

function formatSafeValue(
    item: { ok: true; value: bigint } | { ok: false; raw: string; reason: string }
): string {
    if (item.ok) return formatScaledInt(item.value);
    return `${item.raw} (${item.reason})`;
}

function formatSafeVector(
    arr: Array<{ ok: true; value: bigint } | { ok: false; raw: string; reason: string }>,
    label = "vec",
    limit = 8
): string {
    const shown = arr.slice(0, limit).map(formatSafeValue);
    const suffix = arr.length > limit ? ", ..." : "";
    return `${label}=[${shown.join(", ")}${suffix}]`;
}

function allSafeOk(
    arr: Array<{ ok: true; value: bigint } | { ok: false; raw: string; reason: string }>
): arr is Array<{ ok: true; value: bigint }> {
    return arr.every(x => x.ok);
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

                        const printable =
                            m.method === "gaussianElimination"
                                ? await buildGaussianOutput(harness, n, Aq, bq, result.out as string[], sys.xTrue)
                                : await buildLUOutput(harness, n, Aq, result.out as [string[], string[]]);

                        printBlockRegular({
                            t,
                            method: m.label,
                            explanation: `Gas growth with respect to matrix dimension for dense diagonally dominant system of size n=${n}.`,
                            gas: result.gas,
                            inHex: printable.inHex,
                            expectedHex: "Reference solution / factorization structure shown in decimal output",
                            outHex: printable.outHex,
                            expectedDec: printable.expectedDec,
                            outDec: printable.outDec,
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

                        const printable = await buildIterativeOutput(
                            harness,
                            n,
                            Aq,
                            bq,
                            x0q,
                            result.out as [string[], bigint],
                            sys.xTrue,
                            {
                                alpha: m.method === "gradientDescentLeastSquares" ? GD_ALPHA : undefined,
                                maxIter: FIXED_MAX_ITER,
                            }
                        );

                        printBlockRegular({
                            t,
                            method: m.label,
                            explanation: `Gas growth with respect to matrix dimension for dense diagonally dominant system of size n=${n} with fixed iteration budget ${FIXED_MAX_ITER}.`,
                            gas: result.gas,
                            inHex: printable.inHex,
                            expectedHex: "Reference solution shown in decimal output",
                            outHex: printable.outHex,
                            expectedDec: printable.expectedDec,
                            outDec: printable.outDec,
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

                    const printable = await buildIterativeOutput(
                        harness,
                        n,
                        Aq,
                        bq,
                        x0q,
                        result.out as [string[], bigint],
                        sys.xTrue,
                        {
                            alpha: m.method === "gradientDescentLeastSquares" ? GD_ALPHA : undefined,
                            maxIter,
                        }
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas growth with respect to iteration budget for fixed 3x3 dense diagonally dominant system with maxIter=${maxIter}.`,
                        gas: result.gas,
                        inHex: printable.inHex,
                        expectedHex: "Reference solution shown in decimal output",
                        outHex: printable.outHex,
                        expectedDec: printable.expectedDec,
                        outDec: printable.outDec,
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

                    const printable = await buildIterativeOutput(
                        harness,
                        n,
                        Aq,
                        bq,
                        x0q,
                        result.out as [string[], bigint],
                        sys.xTrue,
                        {
                            alpha: m.method === "gradientDescentLeastSquares" ? GD_ALPHA : undefined,
                            maxIter: FIXED_MAX_ITER,
                        }
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to initial guess using fixed 3x3 dense diagonally dominant system, fixed maxIter=${FIXED_MAX_ITER}, and ${g.label}.`,
                        gas: result.gas,
                        inHex: printable.inHex,
                        expectedHex: "Reference solution shown in decimal output",
                        outHex: printable.outHex,
                        expectedDec: printable.expectedDec,
                        outDec: printable.outDec,
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

                    const printable =
                        m.method === "gaussianElimination"
                            ? await buildGaussianOutput(harness, n, Aq, bq, result.out as string[], base.xTrue)
                            : await buildLUOutput(harness, n, Aq, result.out as [string[], string[]]);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to coefficient scale using fixed 3x3 system and ${s.label} scaling (factor=${describeScale(
                            s.factorNum,
                            s.factorDen
                        )}).`,
                        gas: result.gas,
                        inHex: printable.inHex,
                        expectedHex: "Reference solution / factorization structure shown in decimal output",
                        outHex: printable.outHex,
                        expectedDec: printable.expectedDec,
                        outDec: printable.outDec,
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

                    const printable = await buildIterativeOutput(
                        harness,
                        n,
                        Aq,
                        bq,
                        x0q,
                        result.out as [string[], bigint],
                        base.xTrue,
                        {
                            alpha: m.method === "gradientDescentLeastSquares" ? GD_ALPHA : undefined,
                            maxIter: 20n,
                        }
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to coefficient scale using fixed 3x3 system, fixed maxIter=20, and ${s.label} scaling (factor=${describeScale(
                            s.factorNum,
                            s.factorDen
                        )}).`,
                        gas: result.gas,
                        inHex: printable.inHex,
                        expectedHex: "Reference solution shown in decimal output",
                        outHex: printable.outHex,
                        expectedDec: printable.expectedDec,
                        outDec: printable.outDec,
                    });

                    expect(toGasBigInt(result.gas) > 0n).to.equal(true);
                });
            }
        }
    });
});