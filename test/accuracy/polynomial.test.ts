// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type PolynomialHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;
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
// Helpers
// ------------------------------------------------------------

let SCALE_DECIMALS = 12n;
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

function inferScaleDecimals(scale: bigint): bigint {
    const s = scale.toString();
    if (!/^10*$/.test(s) || s[0] !== "1") {
        return SCALE_DECIMALS;
    }
    return BigInt(s.length - 1);
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
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

function scaledAbsError(actual: bigint, expected: bigint): bigint {
    return absBigInt(actual - expected);
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

function fmtVec(v: bigint[]): string {
    return `[${v.map(formatScaledInt).join(", ")}]`;
}

async function qVecFromInts(harness: PolynomialHarness, values: Array<number | bigint>): Promise<string[]> {
    return Promise.all(values.map(v => harness.qFromInt(v)));
}

async function qVecFromFracs(
    harness: PolynomialHarness,
    values: Array<[number | bigint, number | bigint]>
): Promise<string[]> {
    return Promise.all(values.map(([n, d]) => harness.qFromFrac(n, d)));
}

async function scaledVec(harness: PolynomialHarness, values: string[]): Promise<bigint[]> {
    return Promise.all(values.map(async v => asBigInt(await harness.toFloat(v))));
}

function printScalarBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expected: string;
    output: string;
    absError: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected: ${args.expected}`);
    console.log(`Output: ${args.output}`);
    console.log(`Absolute Error: ${args.absError}`);
    console.log("------------------------------------------------------------");
}

function printVectorBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expected: string;
    output: string;
    errorNorm: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected: ${args.expected}`);
    console.log(`Output: ${args.output}`);
    console.log(`Error Norm (inf): ${args.errorNorm}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("Polynomial Library - Numerical Accuracy Tests", function () {
    let harness: PolynomialHarness;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("PolynomialHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as PolynomialHarness;

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    // ------------------------------------------------------------
    // Section 1: Polynomial evaluation correctness
    // ------------------------------------------------------------

    describe("Section 1: Evaluation correctness", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: Horner evaluation matches exact scalar result`, async function () {
            // p(x) = 2 + 3x + x^2, evaluated at x=2 => 12
            const coeffs = await qVecFromInts(harness, [2n, 3n, 1n]);
            const x = await qInt(2n);

            const out = await harness.evaluateHorners(coeffs, x);
            const outScaledVal = await outScaled(harness, out);
            const expected = 12n * SCALE;
            const err = scaledAbsError(outScaledVal, expected);

            printScalarBlock({
                t: `1.${testNo}`,
                method: "evaluateHorners",
                explanation: "Horner evaluation should recover the exact value of a simple quadratic polynomial.",
                input: "p(x)=2+3x+x^2, x=2",
                expected: formatScaledInt(expected),
                output: formatScaledInt(outScaledVal),
                absError: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 1.${++testNo}: Extended Horner returns exact value and derivative`, async function () {
            // p(x)=1+2x+3x^2, p'(x)=2+6x
            // at x=2 => p(2)=17, p'(2)=14
            const coeffs = await qVecFromInts(harness, [1n, 2n, 3n]);
            const x = await qInt(2n);

            const [px, dpx] = await harness.evaluateWithDerivative(coeffs, x);
            const pxScaled = await outScaled(harness, px);
            const dpxScaled = await outScaled(harness, dpx);

            const expectedPx = 17n * SCALE;
            const expectedDpx = 14n * SCALE;

            const errPx = scaledAbsError(pxScaled, expectedPx);
            const errDpx = scaledAbsError(dpxScaled, expectedDpx);

            printScalarBlock({
                t: `1.${testNo}.1`,
                method: "evaluateWithDerivative",
                explanation: "Extended Horner should recover the exact polynomial value.",
                input: "p(x)=1+2x+3x^2, x=2",
                expected: formatScaledInt(expectedPx),
                output: formatScaledInt(pxScaled),
                absError: formatScaledInt(errPx),
            });

            printScalarBlock({
                t: `1.${testNo}.2`,
                method: "evaluateWithDerivative",
                explanation: "Extended Horner should recover the exact derivative value.",
                input: "p'(x)=2+6x, x=2",
                expected: formatScaledInt(expectedDpx),
                output: formatScaledInt(dpxScaled),
                absError: formatScaledInt(errDpx),
            });

            expect(errPx).to.equal(0n);
            expect(errDpx).to.equal(0n);
        });

        it(`Test 1.${++testNo}: Monic Horner evaluation matches exact result`, async function () {
            // monic polynomial: x^3 + 2x^2 + 3x + 4
            // lowerCoeffs = [4,3,2], x=2 => 26
            const lowerCoeffs = await qVecFromInts(harness, [4n, 3n, 2n]);
            const x = await qInt(2n);

            const out = await harness.evalHornerMonic(lowerCoeffs, x);
            const outScaledVal = await outScaled(harness, out);
            const expected = 26n * SCALE;
            const err = scaledAbsError(outScaledVal, expected);

            printScalarBlock({
                t: `1.${testNo}`,
                method: "evalHornerMonic",
                explanation: "Monic Horner evaluation should recover the exact value of the encoded monic polynomial.",
                input: "p(x)=x^3+2x^2+3x+4, x=2",
                expected: formatScaledInt(expected),
                output: formatScaledInt(outScaledVal),
                absError: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Polynomial arithmetic
    // ------------------------------------------------------------

    describe("Section 2: Polynomial arithmetic correctness", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: Polynomial addition is exact`, async function () {
            // (1+2x+3x^2) + (4+5x) = 5+7x+3x^2
            const a = await qVecFromInts(harness, [1n, 2n, 3n]);
            const b = await qVecFromInts(harness, [4n, 5n]);
            const expected = [5n * SCALE, 7n * SCALE, 3n * SCALE];

            const out = await harness.add(a, b);
            const outScaledVec = await scaledVec(harness, out);
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `2.${testNo}`,
                method: "add",
                explanation: "Coefficient-wise polynomial addition should be exact.",
                input: "[1,2,3] + [4,5]",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 2.${++testNo}: Polynomial subtraction is exact`, async function () {
            // (5+7x+3x^2) - (1+2x+3x^2) = 4+5x
            const a = await qVecFromInts(harness, [5n, 7n, 3n]);
            const b = await qVecFromInts(harness, [1n, 2n, 3n]);
            const expected = [4n * SCALE, 5n * SCALE, 0n];

            const out = await harness.sub(a, b);
            const outScaledVec = await scaledVec(harness, out);
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `2.${testNo}`,
                method: "sub",
                explanation: "Coefficient-wise polynomial subtraction should be exact.",
                input: "[5,7,3] - [1,2,3]",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 2.${++testNo}: Scalar multiplication is exact`, async function () {
            // 2*(1+2x+3x^2) = 2+4x+6x^2
            const coeffs = await qVecFromInts(harness, [1n, 2n, 3n]);
            const k = await qInt(2n);
            const expected = [2n * SCALE, 4n * SCALE, 6n * SCALE];

            const out = await harness.mulScalar(coeffs, k);
            const outScaledVec = await scaledVec(harness, out);
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `2.${testNo}`,
                method: "mulScalar",
                explanation: "Scalar multiplication should scale all coefficients exactly.",
                input: "2 * [1,2,3]",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 2.${++testNo}: Polynomial multiplication is exact`, async function () {
            // (1+x)(1+2x) = 1+3x+2x^2
            const a = await qVecFromInts(harness, [1n, 1n]);
            const b = await qVecFromInts(harness, [1n, 2n]);
            const expected = [1n * SCALE, 3n * SCALE, 2n * SCALE];

            const out = await harness.mul(a, b);
            const outScaledVec = await scaledVec(harness, out);
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `2.${testNo}`,
                method: "mul",
                explanation: "Polynomial convolution should produce the exact product coefficients.",
                input: "[1,1] * [1,2]",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Synthetic division
    // ------------------------------------------------------------

    describe("Section 3: Synthetic division correctness", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: Synthetic division recovers quotient and zero remainder for exact root`, async function () {
            // p(x)=x^2-3x+2 = (x-1)(x-2)
            // divide by (x-1) => quotient x-2 => [-2,1], remainder 0
            const coeffs = await qVecFromInts(harness, [2n, -3n, 1n]);
            const root = await qInt(1n);

            const [q, r] = await harness.syntheticDivide(coeffs, root);
            const qScaledVec = await scaledVec(harness, q);
            const rScaled = await outScaled(harness, r);

            const expectedQ = [-2n * SCALE, 1n * SCALE];
            const qErr = vecInfNorm(subVec(qScaledVec, expectedQ));
            const rErr = absBigInt(rScaled);

            printVectorBlock({
                t: `3.${testNo}.1`,
                method: "syntheticDivide",
                explanation: "Synthetic division should recover the exact quotient coefficients when dividing by an exact root.",
                input: "p(x)=x^2-3x+2, root=1",
                expected: fmtVec(expectedQ),
                output: fmtVec(qScaledVec),
                errorNorm: formatScaledInt(qErr),
            });

            printScalarBlock({
                t: `3.${testNo}.2`,
                method: "syntheticDivide",
                explanation: "The remainder should vanish at an exact root.",
                input: "p(1)",
                expected: "0",
                output: formatScaledInt(rScaled),
                absError: formatScaledInt(rErr),
            });

            expect(qErr).to.equal(0n);
            expect(rErr).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Calculus operations
    // ------------------------------------------------------------

    describe("Section 4: Calculus operations", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: Derivative coefficients are exact`, async function () {
            // p(x)=1+2x+3x^2 => p'(x)=2+6x
            const coeffs = await qVecFromInts(harness, [1n, 2n, 3n]);
            const expected = [2n * SCALE, 6n * SCALE];

            const out = await harness.derivative(coeffs);
            const outScaledVec = await scaledVec(harness, out);
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `4.${testNo}`,
                method: "derivative",
                explanation: "The derivative operator should return the exact transformed coefficients.",
                input: "d/dx [1,2,3]",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 4.${++testNo}: Integral coefficients are exact`, async function () {
            // p(x)=2+6x ; integral + C=5 => 5+2x+3x^2
            const coeffs = await qVecFromInts(harness, [2n, 6n]);
            const C = await qInt(5n);
            const expected = [5n * SCALE, 2n * SCALE, 3n * SCALE];

            const out = await harness.integral(coeffs, C);
            const outScaledVec = await scaledVec(harness, out);
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `4.${testNo}`,
                method: "integral",
                explanation: "The integral operator should return the exact antiderivative coefficients including the constant of integration.",
                input: "integral([2,6], C=5)",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 4.${++testNo}: Derivative and integral are mutually consistent on polynomial benchmark`, async function () {
            // p(x)=1+2x+3x^2 -> derivative -> [2,6] -> integral(...,1) -> [1,2,3]
            const p = await qVecFromInts(harness, [1n, 2n, 3n]);
            const dp = await harness.derivative(p);
            const dpArray = Array.from(dp);
            const one = await qInt(1n);
            const recovered = await harness.integral(dpArray, one);

            const recoveredScaled = await scaledVec(harness, recovered);
            const expected = [1n * SCALE, 2n * SCALE, 3n * SCALE];
            const err = vecInfNorm(subVec(recoveredScaled, expected));

            printVectorBlock({
                t: `4.${testNo}`,
                method: "derivative/integral consistency",
                explanation: "Integrating the derivative and restoring the original constant term should recover the original polynomial.",
                input: "p=[1,2,3]",
                expected: fmtVec(expected),
                output: fmtVec(recoveredScaled),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 5: Utility correctness
    // ------------------------------------------------------------

    describe("Section 5: Utility correctness", function () {
        let testNo = 0;

        it(`Test 5.${++testNo}: degree ignores trailing zeros`, async function () {
            const coeffs = await qVecFromInts(harness, [1n, 2n, 0n, 0n]);
            const deg = await harness.degree(coeffs);

            console.log("------------------------------------------------------------");
            console.log(`Test: 5.${testNo}`);
            console.log("Method: degree");
            console.log("Explanation: Degree should ignore trailing zero coefficients.");
            console.log("Input: [1,2,0,0]");
            console.log("Expected Degree: 1");
            console.log(`Output Degree: ${deg.toString()}`);
            console.log("------------------------------------------------------------");

            expect(deg).to.equal(1n);
        });

        it(`Test 5.${++testNo}: trimTrailingZeros returns canonical representation`, async function () {
            const coeffs = await qVecFromInts(harness, [1n, 2n, 0n, 0n]);
            const out = await harness.trimTrailingZeros(coeffs);
            const outScaledVec = await scaledVec(harness, out);
            const expected = [1n * SCALE, 2n * SCALE];
            const err = vecInfNorm(subVec(outScaledVec, expected));

            printVectorBlock({
                t: `5.${testNo}`,
                method: "trimTrailingZeros",
                explanation: "Trailing zeros should be removed to obtain the canonical coefficient vector.",
                input: "[1,2,0,0]",
                expected: fmtVec(expected),
                output: fmtVec(outScaledVec),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });
});