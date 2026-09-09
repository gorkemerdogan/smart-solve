// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";
import { HARNESS_ESTIMATE_CALL } from "./test-utils";
import type { Contract } from "ethers";
import { binary128ToRational, binary128ToScaledInt, printPrecisionMetadata } from "./precision-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type PolynomialHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<[boolean, bigint]>;
    fromFloat(n: bigint): Promise<string>;

    evaluateHorners(coeffs: string[], x: string): Promise<string>;
    evaluateWithDerivative(coeffs: string[], x: string): Promise<[string, string]>;
    evalHornerMonic(lowerCoeffs: string[], x: string): Promise<string>;

    add(a: string[], b: string[]): Promise<string[]>;
    sub(a: string[], b: string[]): Promise<string[]>;
    mulScalar(coeffs: string[], k: string): Promise<string[]>;
    mul(a: string[], b: string[]): Promise<string[]>;
    syntheticDivide(coeffs: string[], root: string): Promise<[string[], string]>;

    derivative(coeffs: string[]): Promise<string[]>;
    integral(coeffs: string[], C: string): Promise<string[]>;
    degree(coeffs: string[]): Promise<bigint>;
    trimTrailingZeros(coeffs: string[]): Promise<string[]>;
};

// ------------------------------------------------------------
// Global config
// ------------------------------------------------------------

const T = 30;
const FIXED_SEED = 37n;
const DEGREES = [2, 4, 8, 16];

// Direct binary128 decoding at 1e30 -> absolute tolerance 1e-24.
const SCALAR_TOL = 1_000_000n;
const VECTOR_TOL = 1_000_000n;
const SCALAR_REL_TOL_DEN = 10n ** 28n; // relative tolerance 1e-28

// ------------------------------------------------------------
// Scale helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 30n;
const SCALE = 10n ** SCALE_DECIMALS;

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

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

async function outScaled(harness: PolynomialHarness, q: string): Promise<bigint> {
    void harness;
    return binary128ToScaledInt(q, Number(SCALE_DECIMALS));
}

async function scaledVec(harness: PolynomialHarness, values: string[]): Promise<bigint[]> {
    return Promise.all(values.map(v => outScaled(harness, v)));
}

function scaledAbsError(actual: bigint, expected: bigint): bigint {
    return absBigInt(actual - expected);
}

function scalarTolerance(expected: bigint): bigint {
    return SCALAR_TOL + absBigInt(expected) / SCALAR_REL_TOL_DEN;
}

function vecInfNorm(v: bigint[]): bigint {
    let m = 0n;
    for (const x of v) {
        const ax = absBigInt(x);
        if (ax > m) m = ax;
    }
    return m;
}

function vectorTolerance(expected: bigint[]): bigint {
    return VECTOR_TOL + vecInfNorm(expected) / SCALAR_REL_TOL_DEN;
}

function syntheticOperationTolerance(encodedPoly: Frac[], root: Frac): bigint {
    const rootMagnitude = absBigInt(fracToScaledTrunc(root));
    let evaluationMagnitude = 0n;
    for (let i = encodedPoly.length - 1; i >= 0; i--) {
        evaluationMagnitude = (evaluationMagnitude * rootMagnitude) / SCALE
            + absBigInt(fracToScaledTrunc(encodedPoly[i]));
    }
    // Synthetic division forms the zero remainder by repeated multiply-adds;
    // use the absolute Horner sum as a condition scale, plus operation count
    // and a conservative cancellation safety factor.
    return SCALAR_TOL
        + (64n * evaluationMagnitude * BigInt(encodedPoly.length)) / SCALAR_REL_TOL_DEN;
}

function subVec(a: bigint[], b: bigint[]): bigint[] {
    const n = Math.max(a.length, b.length);
    const out: bigint[] = [];
    for (let i = 0; i < n; i++) {
        out.push((a[i] ?? 0n) - (b[i] ?? 0n));
    }
    return out;
}

function trimTrailingZerosBigInt(v: bigint[]): bigint[] {
    if (v.length === 0) return [0n];
    let last = v.length - 1;
    while (last > 0 && v[last] === 0n) last--;
    return v.slice(0, last + 1);
}

// ------------------------------------------------------------
// Gas helpers
// ------------------------------------------------------------

type GasStats = {
    total: bigint;
    min: bigint;
    max: bigint;
    count: bigint;
};

function initGasStats(): GasStats {
    return {
        total: 0n,
        min: 0n,
        max: 0n,
        count: 0n,
    };
}

function recordGas(stats: GasStats, gas: bigint): void {
    stats.total += gas;
    stats.count += 1n;

    if (stats.count === 1n) {
        stats.min = gas;
        stats.max = gas;
        return;
    }

    if (gas < stats.min) stats.min = gas;
    if (gas > stats.max) stats.max = gas;
}

function avgGas(stats: GasStats): bigint {
    return stats.count === 0n ? 0n : stats.total / stats.count;
}

async function estimateMethodGas(
    harness: PolynomialHarness,
    methodName: string,
    args: unknown[]
): Promise<bigint> {
    const method = (harness as any)[methodName];
    if (!method || typeof method.estimateGas !== "function") {
        throw new Error(`estimateGas is not available for method: ${methodName}`);
    }
    return asBigInt(await method.estimateGas(...args));
}

// ------------------------------------------------------------
// Exact rational arithmetic for reference results
// ------------------------------------------------------------

type Frac = {
    num: bigint;
    den: bigint;
};

function gcd(a: bigint, b: bigint): bigint {
    let x = absBigInt(a);
    let y = absBigInt(b);
    while (y !== 0n) {
        const t = x % y;
        x = y;
        y = t;
    }
    return x === 0n ? 1n : x;
}

function normalizeFrac(num: bigint, den: bigint): Frac {
    if (den === 0n) throw new Error("Zero denominator");

    let n = num;
    let d = den;

    if (d < 0n) {
        n = -n;
        d = -d;
    }

    const g = gcd(n, d);
    return { num: n / g, den: d / g };
}

function frac(num: bigint, den: bigint = 1n): Frac {
    return normalizeFrac(num, den);
}

function addF(a: Frac, b: Frac): Frac {
    return normalizeFrac(a.num * b.den + b.num * a.den, a.den * b.den);
}

function subF(a: Frac, b: Frac): Frac {
    return normalizeFrac(a.num * b.den - b.num * a.den, a.den * b.den);
}

function mulF(a: Frac, b: Frac): Frac {
    return normalizeFrac(a.num * b.num, a.den * b.den);
}

function divF(a: Frac, b: Frac): Frac {
    if (b.num === 0n) throw new Error("Division by zero");
    return normalizeFrac(a.num * b.den, a.den * b.num);
}

function negF(a: Frac): Frac {
    return { num: -a.num, den: a.den };
}

function intToFrac(x: bigint): Frac {
    return frac(x, 1n);
}

function fracToScaledTrunc(x: Frac): bigint {
    const scaledNum = x.num * SCALE;
    return scaledNum / x.den;
}

function fracVecToScaled(v: Frac[]): bigint[] {
    return v.map(fracToScaledTrunc);
}

// ------------------------------------------------------------
// Deterministic pseudo-random generation with keccak256
// ------------------------------------------------------------

function keccakBigInt(parts: Array<string | bigint | number>): bigint {
    const packed = ethers.solidityPacked(
        new Array(parts.length).fill("string"),
        parts.map(x => x.toString())
    );
    return BigInt(ethers.keccak256(packed));
}

function randInRange(
    tag: string,
    testIndex: number,
    itemIndex: number,
    min: bigint,
    max: bigint
): bigint {
    if (max < min) throw new Error("Invalid range");
    const span = max - min + 1n;
    const h = keccakBigInt([FIXED_SEED, tag, testIndex, itemIndex]);
    return min + (h % span);
}

function randNonZeroInRange(
    tag: string,
    testIndex: number,
    itemIndex: number,
    min: bigint,
    max: bigint
): bigint {
    let v = 0n;
    let k = 0;
    while (v === 0n) {
        v = randInRange(`${tag}_nz_${k}`, testIndex, itemIndex, min, max);
        k++;
    }
    return v;
}

function randFrac(
    tag: string,
    testIndex: number,
    itemIndex: number,
    numMin: bigint,
    numMax: bigint,
    denMin: bigint,
    denMax: bigint
): Frac {
    const n = randInRange(`${tag}_num`, testIndex, itemIndex, numMin, numMax);
    const d = randNonZeroInRange(`${tag}_den`, testIndex, itemIndex, denMin, denMax);
    return normalizeFrac(n, d);
}

function randCoeff(testIndex: number, itemIndex: number): Frac {
    return randFrac("coeff", testIndex, itemIndex, -25n, 25n, 1n, 5n);
}

function randX(testIndex: number): Frac {
    return randFrac("x", testIndex, 0, -10n, 10n, 1n, 5n);
}

function randScalar(testIndex: number): Frac {
    return randFrac("scalar", testIndex, 0, -15n, 15n, 1n, 5n);
}

function randConst(testIndex: number): Frac {
    return randFrac("const", testIndex, 0, -20n, 20n, 1n, 5n);
}

// ------------------------------------------------------------
// Polynomial exact reference helpers
// Coefficients are in ascending powers.
// ------------------------------------------------------------

function trimTrailingZerosFrac(coeffs: Frac[]): Frac[] {
    if (coeffs.length === 0) return [frac(0n)];
    let last = coeffs.length - 1;
    while (last > 0 && coeffs[last].num === 0n) last--;
    return coeffs.slice(0, last + 1);
}

function polyAdd(a: Frac[], b: Frac[]): Frac[] {
    const n = Math.max(a.length, b.length);
    const out: Frac[] = [];
    for (let i = 0; i < n; i++) {
        out.push(addF(a[i] ?? frac(0n), b[i] ?? frac(0n)));
    }
    return trimTrailingZerosFrac(out);
}

function polySub(a: Frac[], b: Frac[]): Frac[] {
    const n = Math.max(a.length, b.length);
    const out: Frac[] = [];
    for (let i = 0; i < n; i++) {
        out.push(subF(a[i] ?? frac(0n), b[i] ?? frac(0n)));
    }
    return trimTrailingZerosFrac(out);
}

function polyMulScalar(a: Frac[], k: Frac): Frac[] {
    return trimTrailingZerosFrac(a.map(c => mulF(c, k)));
}

function polyMul(a: Frac[], b: Frac[]): Frac[] {
    const out = Array.from({ length: a.length + b.length - 1 }, () => frac(0n));
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < b.length; j++) {
            out[i + j] = addF(out[i + j], mulF(a[i], b[j]));
        }
    }
    return trimTrailingZerosFrac(out);
}

function polyDerivative(a: Frac[]): Frac[] {
    if (a.length <= 1) return [frac(0n)];
    const out: Frac[] = [];
    for (let i = 1; i < a.length; i++) {
        out.push(mulF(a[i], intToFrac(BigInt(i))));
    }
    return trimTrailingZerosFrac(out);
}

function polyIntegral(a: Frac[], C: Frac): Frac[] {
    const out: Frac[] = [C];
    for (let i = 0; i < a.length; i++) {
        out.push(divF(a[i], intToFrac(BigInt(i + 1))));
    }
    return trimTrailingZerosFrac(out);
}

function polyEval(coeffs: Frac[], x: Frac): Frac {
    let acc = frac(0n);
    for (let i = coeffs.length - 1; i >= 0; i--) {
        acc = addF(mulF(acc, x), coeffs[i]);
    }
    return acc;
}

function polyEvalWithDerivative(coeffs: Frac[], x: Frac): { px: Frac; dpx: Frac } {
    if (coeffs.length === 0) {
        return { px: frac(0n), dpx: frac(0n) };
    }

    let b = coeffs[coeffs.length - 1];
    let c = frac(0n);

    for (let i = coeffs.length - 2; i >= 0; i--) {
        c = addF(mulF(c, x), b);
        b = addF(mulF(b, x), coeffs[i]);
    }

    return { px: b, dpx: c };
}

function polyEvalMonic(lowerCoeffs: Frac[], x: Frac): Frac {
    let acc = frac(1n);
    for (let i = lowerCoeffs.length - 1; i >= 0; i--) {
        acc = addF(mulF(acc, x), lowerCoeffs[i]);
    }
    return acc;
}

function polySyntheticDivideAscending(coeffs: Frac[], root: Frac): { q: Frac[]; r: Frac } {
    const n = coeffs.length - 1;
    if (n < 0) return { q: [frac(0n)], r: frac(0n) };
    if (n === 0) return { q: [frac(0n)], r: coeffs[0] };

    const desc = [...coeffs].reverse();
    const b: Frac[] = new Array(desc.length);
    b[0] = desc[0];

    for (let i = 1; i < desc.length; i++) {
        b[i] = addF(desc[i], mulF(root, b[i - 1]));
    }

    const remainder = b[b.length - 1];
    const quotientDesc = b.slice(0, -1);
    const quotientAsc = quotientDesc.reverse();

    return {
        q: trimTrailingZerosFrac(quotientAsc),
        r: remainder,
    };
}

// ------------------------------------------------------------
// Conversion helpers to harness inputs
// ------------------------------------------------------------

async function qFromFrac(harness: PolynomialHarness, x: Frac): Promise<string> {
    return harness.qFromFrac(x.num, x.den);
}

async function qVecFromFrac(harness: PolynomialHarness, values: Frac[]): Promise<string[]> {
    return Promise.all(values.map(v => qFromFrac(harness, v)));
}

function exactEncodedFrac(value: string): Frac {
    return binary128ToRational(value);
}

function exactEncodedVec(values: string[]): Frac[] {
    return values.map(exactEncodedFrac);
}

// ------------------------------------------------------------
// Logging helpers
// ------------------------------------------------------------

function printScalarSummary(args: {
    method: string;
    degree: number;
    tests: number;
    avgError: bigint;
    maxError: bigint;
    minGas: bigint;
    avgGas: bigint;
    maxGas: bigint;
    threshold?: string;
    claim?: string;
}) {
    console.log("============================================================");
    console.log(`Method             : ${args.method}`);
    console.log(`Degree             : ${args.degree}`);
    console.log(`Number of Tests    : ${args.tests}`);
    console.log(`Average Abs. Error : ${formatScaledInt(args.avgError)}`);
    console.log(`Max Abs. Error     : ${formatScaledInt(args.maxError)}`);
    console.log(`Pass Threshold     : ${args.threshold ?? "abs error <= 1e-24 + 1e-28 * |reference|"}`);
    printPrecisionMetadata({
        classification: "binary128-aware comparison",
        comparisonScale: "1e30 (direct bytes16 decoding; truncation toward zero)",
        oraclePrecision: "exact BigInt rational arithmetic on the values actually encoded as binary128",
        conversion: "binary128 bytes16 -> exact BigInt rational -> 1e30 integer",
        claim: args.claim ?? "polynomial output agreement against exact rational references using 1e-24 absolute + 1e-28 relative tolerance",
    });
    console.log(`Min Gas            : ${args.minGas.toString()}`);
    console.log(`Average Gas        : ${args.avgGas.toString()}`);
    console.log(`Max Gas            : ${args.maxGas.toString()}`);
    console.log("============================================================");
}

function printVectorSummary(args: {
    method: string;
    degree: number;
    tests: number;
    avgError: bigint;
    maxError: bigint;
    minGas: bigint;
    avgGas: bigint;
    maxGas: bigint;
    threshold?: string;
    claim?: string;
}) {
    console.log("============================================================");
    console.log(`Method                  : ${args.method}`);
    console.log(`Degree                  : ${args.degree}`);
    console.log(`Number of Tests         : ${args.tests}`);
    console.log(`Average Error Norm (∞)  : ${formatScaledInt(args.avgError)}`);
    console.log(`Max Error Norm (∞)      : ${formatScaledInt(args.maxError)}`);
    console.log(`Pass Threshold          : ${args.threshold ?? "infinity error <= 1e-24 + 1e-28 * ||reference||∞"}`);
    printPrecisionMetadata({
        classification: "binary128-aware comparison",
        comparisonScale: "1e30 (direct bytes16 decoding; truncation toward zero)",
        oraclePrecision: "exact BigInt rational arithmetic on the values actually encoded as binary128",
        conversion: "binary128 bytes16 -> exact BigInt rational -> 1e30 integer",
        claim: args.claim ?? "coefficient agreement against exact rational references using 1e-24 absolute + 1e-28 relative infinity-norm tolerance",
    });
    console.log(`Min Gas                 : ${args.minGas.toString()}`);
    console.log(`Average Gas             : ${args.avgGas.toString()}`);
    console.log(`Max Gas                 : ${args.maxGas.toString()}`);
    console.log("============================================================");
}

// ------------------------------------------------------------
// Test suite
// ------------------------------------------------------------

describe("Polynomial Library - Binary128-Aware Exact-Oracle Benchmarks", function () {
    let harness: PolynomialHarness;
    let resultWriter: BenchmarkResultWriter;

    before(async () => {
        resultWriter = await BenchmarkResultWriter.create({ suite: "polynomial.harness" });
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("PolynomialHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as PolynomialHarness;

    });

    after(async () => {
        await resultWriter.flush();
    });

    for (const degree of DEGREES) {
        describe(`Degree n=${degree}`, function () {
            it(`evaluateHorners - average absolute error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const coeffs = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t, i));
                    const x = randX(t);

                    const qCoeffs = await qVecFromFrac(harness, coeffs);
                    const qx = await qFromFrac(harness, x);

                    const gas = await estimateMethodGas(harness, "evaluateHorners", [qCoeffs, qx]);
                    recordGas(gasStats, gas);

                    const out = await harness.evaluateHorners(qCoeffs, qx);
                    const actual = await outScaled(harness, out);

                    const expectedFrac = polyEval(exactEncodedVec(qCoeffs), exactEncodedFrac(qx));
                    const expected = fracToScaledTrunc(expectedFrac);

                    const err = scaledAbsError(actual, expected);
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(scalarTolerance(expected));
                }

                printScalarSummary({
                    method: "evaluateHorners",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
                resultWriter.record({
                    benchmark: "exact-oracle polynomial accuracy",
                    category: "polynomial",
                    operation: "evaluateHorners",
                    execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
                    input: { degree, tests: T },
                    gas: avgGas(gasStats).toString(),
                    status: "success",
                    errorMetrics: {
                        averageAbsoluteErrorScaled: (totalErr / BigInt(T)).toString(),
                        maxAbsoluteErrorScaled: maxErr.toString(),
                        gasMin: gasStats.min.toString(),
                        gasMax: gasStats.max.toString(),
                    },
                });
            });

            it(`evaluateWithDerivative - average absolute error over ${T} deterministic tests`, async function () {
                let totalErrPx = 0n;
                let totalErrDpx = 0n;
                let maxErrPx = 0n;
                let maxErrDpx = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const coeffs = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 1000, i));
                    const x = randX(t + 1000);

                    const qCoeffs = await qVecFromFrac(harness, coeffs);
                    const qx = await qFromFrac(harness, x);

                    const gas = await estimateMethodGas(harness, "evaluateWithDerivative", [qCoeffs, qx]);
                    recordGas(gasStats, gas);

                    const [px, dpx] = await harness.evaluateWithDerivative(qCoeffs, qx);
                    const actualPx = await outScaled(harness, px);
                    const actualDpx = await outScaled(harness, dpx);

                    const ref = polyEvalWithDerivative(exactEncodedVec(qCoeffs), exactEncodedFrac(qx));
                    const expectedPx = fracToScaledTrunc(ref.px);
                    const expectedDpx = fracToScaledTrunc(ref.dpx);

                    const errPx = scaledAbsError(actualPx, expectedPx);
                    const errDpx = scaledAbsError(actualDpx, expectedDpx);

                    totalErrPx += errPx;
                    totalErrDpx += errDpx;

                    if (errPx > maxErrPx) maxErrPx = errPx;
                    if (errDpx > maxErrDpx) maxErrDpx = errDpx;

                    expect(errPx).to.be.lte(scalarTolerance(expectedPx));
                    expect(errDpx).to.be.lte(scalarTolerance(expectedDpx));
                }

                const avgGasValue = avgGas(gasStats);

                printScalarSummary({
                    method: "evaluateWithDerivative (p(x))",
                    degree,
                    tests: T,
                    avgError: totalErrPx / BigInt(T),
                    maxError: maxErrPx,
                    minGas: gasStats.min,
                    avgGas: avgGasValue,
                    maxGas: gasStats.max,
                });

                printScalarSummary({
                    method: "evaluateWithDerivative (p'(x))",
                    degree,
                    tests: T,
                    avgError: totalErrDpx / BigInt(T),
                    maxError: maxErrDpx,
                    minGas: gasStats.min,
                    avgGas: avgGasValue,
                    maxGas: gasStats.max,
                });
            });

            it(`evalHornerMonic - average absolute error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const lowerCoeffs = Array.from({ length: degree }, (_, i) => randCoeff(t + 2000, i));
                    const x = randX(t + 2000);

                    const qLowerCoeffs = await qVecFromFrac(harness, lowerCoeffs);
                    const qx = await qFromFrac(harness, x);

                    const gas = await estimateMethodGas(harness, "evalHornerMonic", [qLowerCoeffs, qx]);
                    recordGas(gasStats, gas);

                    const out = await harness.evalHornerMonic(qLowerCoeffs, qx);
                    const actual = await outScaled(harness, out);

                    const expectedFrac = polyEvalMonic(exactEncodedVec(qLowerCoeffs), exactEncodedFrac(qx));
                    const expected = fracToScaledTrunc(expectedFrac);

                    const err = scaledAbsError(actual, expected);
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(scalarTolerance(expected));
                }

                printScalarSummary({
                    method: "evalHornerMonic",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });

            it(`add - average infinity-norm error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const a = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 3000, i));
                    const b = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 4000, i));

                    const qa = await qVecFromFrac(harness, a);
                    const qb = await qVecFromFrac(harness, b);

                    const gas = await estimateMethodGas(harness, "add", [qa, qb]);
                    recordGas(gasStats, gas);

                    const out = await harness.add(qa, qb);
                    const actual = trimTrailingZerosBigInt(await scaledVec(harness, out));

                    const expectedFrac = polyAdd(exactEncodedVec(qa), exactEncodedVec(qb));
                    const expected = trimTrailingZerosBigInt(fracVecToScaled(expectedFrac));

                    const err = vecInfNorm(subVec(actual, expected));
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(vectorTolerance(expected));
                }

                printVectorSummary({
                    method: "add",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });

            it(`sub - average infinity-norm error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const a = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 5000, i));
                    const b = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 6000, i));

                    const qa = await qVecFromFrac(harness, a);
                    const qb = await qVecFromFrac(harness, b);

                    const gas = await estimateMethodGas(harness, "sub", [qa, qb]);
                    recordGas(gasStats, gas);

                    const out = await harness.sub(qa, qb);
                    const actual = trimTrailingZerosBigInt(await scaledVec(harness, out));

                    const expectedFrac = polySub(exactEncodedVec(qa), exactEncodedVec(qb));
                    const expected = trimTrailingZerosBigInt(fracVecToScaled(expectedFrac));

                    const err = vecInfNorm(subVec(actual, expected));
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(vectorTolerance(expected));
                }

                printVectorSummary({
                    method: "sub",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });

            it(`mulScalar - average infinity-norm error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const coeffs = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 7000, i));
                    const k = randScalar(t + 7000);

                    const qCoeffs = await qVecFromFrac(harness, coeffs);
                    const qk = await qFromFrac(harness, k);

                    const gas = await estimateMethodGas(harness, "mulScalar", [qCoeffs, qk]);
                    recordGas(gasStats, gas);

                    const out = await harness.mulScalar(qCoeffs, qk);
                    const actual = trimTrailingZerosBigInt(await scaledVec(harness, out));

                    const expectedFrac = polyMulScalar(exactEncodedVec(qCoeffs), exactEncodedFrac(qk));
                    const expected = trimTrailingZerosBigInt(fracVecToScaled(expectedFrac));

                    const err = vecInfNorm(subVec(actual, expected));
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(vectorTolerance(expected));
                }

                printVectorSummary({
                    method: "mulScalar",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });

            it(`mul - average infinity-norm error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const a = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 8000, i));
                    const b = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 9000, i));

                    const qa = await qVecFromFrac(harness, a);
                    const qb = await qVecFromFrac(harness, b);

                    const gas = await estimateMethodGas(harness, "mul", [qa, qb]);
                    recordGas(gasStats, gas);

                    const out = await harness.mul(qa, qb);
                    const actual = trimTrailingZerosBigInt(await scaledVec(harness, out));

                    const expectedFrac = polyMul(exactEncodedVec(qa), exactEncodedVec(qb));
                    const expected = trimTrailingZerosBigInt(fracVecToScaled(expectedFrac));

                    const err = vecInfNorm(subVec(actual, expected));
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(vectorTolerance(expected));
                }

                printVectorSummary({
                    method: "mul",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });

            it(`syntheticDivide - average quotient/remainder error over ${T} deterministic tests`, async function () {
                let totalQErr = 0n;
                let totalRErr = 0n;
                let maxQErr = 0n;
                let maxRErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const root = randX(t + 10000);
                    const quotient = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 11000, i));

                    const factor = [negF(root), frac(1n)];
                    const poly = polyMul(quotient, factor);

                    const qPoly = await qVecFromFrac(harness, poly);
                    const qRoot = await qFromFrac(harness, root);

                    const gas = await estimateMethodGas(harness, "syntheticDivide", [qPoly, qRoot]);
                    recordGas(gasStats, gas);

                    const [qOut, rOut] = await harness.syntheticDivide(qPoly, qRoot);
                    const actualQ = trimTrailingZerosBigInt(await scaledVec(harness, qOut));
                    const actualR = await outScaled(harness, rOut);

                    const encodedPoly = exactEncodedVec(qPoly);
                    const encodedRoot = exactEncodedFrac(qRoot);
                    const ref = polySyntheticDivideAscending(encodedPoly, encodedRoot);
                    const expectedQ = trimTrailingZerosBigInt(fracVecToScaled(ref.q));
                    const expectedR = fracToScaledTrunc(ref.r);

                    const qErr = vecInfNorm(subVec(actualQ, expectedQ));
                    const rErr = scaledAbsError(actualR, expectedR);

                    totalQErr += qErr;
                    totalRErr += rErr;

                    if (qErr > maxQErr) maxQErr = qErr;
                    if (rErr > maxRErr) maxRErr = rErr;

                    const syntheticTolerance = syntheticOperationTolerance(encodedPoly, encodedRoot);
                    expect(qErr).to.be.lte(syntheticTolerance);
                    expect(rErr).to.be.lte(syntheticTolerance);
                }

                const avgGasValue = avgGas(gasStats);

                printVectorSummary({
                    method: "syntheticDivide (quotient)",
                    degree,
                    tests: T,
                    avgError: totalQErr / BigInt(T),
                    maxError: maxQErr,
                    minGas: gasStats.min,
                    avgGas: avgGasValue,
                    maxGas: gasStats.max,
                    threshold: "condition-scaled binary128 multiply-add bound (absolute Horner sum)",
                    claim: "synthetic-division forward error against an exact rational oracle on encoded inputs",
                });

                printScalarSummary({
                    method: "syntheticDivide (remainder)",
                    degree,
                    tests: T,
                    avgError: totalRErr / BigInt(T),
                    maxError: maxRErr,
                    minGas: gasStats.min,
                    avgGas: avgGasValue,
                    maxGas: gasStats.max,
                    threshold: "condition-scaled binary128 cancellation bound (absolute Horner sum)",
                    claim: "synthetic-division remainder error against an exact rational oracle on encoded inputs",
                });
            });

            it(`derivative - average infinity-norm error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const coeffs = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 12000, i));

                    const qCoeffs = await qVecFromFrac(harness, coeffs);

                    const gas = await estimateMethodGas(harness, "derivative", [qCoeffs]);
                    recordGas(gasStats, gas);

                    const out = await harness.derivative(qCoeffs);
                    const actual = trimTrailingZerosBigInt(await scaledVec(harness, out));

                    const expectedFrac = polyDerivative(exactEncodedVec(qCoeffs));
                    const expected = trimTrailingZerosBigInt(fracVecToScaled(expectedFrac));

                    const err = vecInfNorm(subVec(actual, expected));
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(vectorTolerance(expected));
                }

                printVectorSummary({
                    method: "derivative",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });

            it(`integral - average infinity-norm error over ${T} deterministic tests`, async function () {
                let totalErr = 0n;
                let maxErr = 0n;
                const gasStats = initGasStats();

                for (let t = 0; t < T; t++) {
                    const coeffs = Array.from({ length: degree + 1 }, (_, i) => randCoeff(t + 13000, i));
                    const C = randConst(t + 13000);

                    const qCoeffs = await qVecFromFrac(harness, coeffs);
                    const qC = await qFromFrac(harness, C);

                    const gas = await estimateMethodGas(harness, "integral", [qCoeffs, qC]);
                    recordGas(gasStats, gas);

                    const out = await harness.integral(qCoeffs, qC);
                    const actual = trimTrailingZerosBigInt(await scaledVec(harness, out));

                    const expectedFrac = polyIntegral(exactEncodedVec(qCoeffs), exactEncodedFrac(qC));
                    const expected = trimTrailingZerosBigInt(fracVecToScaled(expectedFrac));

                    const err = vecInfNorm(subVec(actual, expected));
                    totalErr += err;
                    if (err > maxErr) maxErr = err;

                    expect(err).to.be.lte(vectorTolerance(expected));
                }

                printVectorSummary({
                    method: "integral",
                    degree,
                    tests: T,
                    avgError: totalErr / BigInt(T),
                    maxError: maxErr,
                    minGas: gasStats.min,
                    avgGas: avgGas(gasStats),
                    maxGas: gasStats.max,
                });
            });
        });
    }
});
