// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type TrigonometryHarness = Contract & {
    fromDouble(x: number | bigint): Promise<string>;
    fromUInt(x: number | bigint): Promise<string>;
    toDouble(x: string): Promise<unknown>;

    fromFloat(x: bigint): Promise<string>;
    toFloat(x: string): Promise<unknown>;

    abs(x: string): Promise<string>;
    neg(x: string): Promise<string>;
    cmp(a: string, b: string): Promise<bigint>;
    isNaN(x: string): Promise<boolean>;
    add(a: string, b: string): Promise<string>;
    mul(a: string, b: string): Promise<string>;

    sin(x: string): Promise<string>;
    cos(x: string): Promise<string>;
    tan(x: string): Promise<string>;
    cot(x: string): Promise<string>;
    asin(x: string): Promise<string>;
    acos(x: string): Promise<string>;
    atan(x: string): Promise<string>;

    QPI(): Promise<string>;
    QHALF_PI(): Promise<string>;
    QQUARTER_PI(): Promise<string>;
    QTWO_PI(): Promise<string>;
    QNAN(): Promise<string>;
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

async function outScaled(harness: TrigonometryHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

async function qScaled(harness: TrigonometryHarness, scaledValue: bigint): Promise<string> {
    return await harness.fromFloat(asBigInt(scaledValue));
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

function printTrigAccuracyBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expectedHex: string;
    outputHex: string;
    expectedDec: string;
    outputDec: string;
    absError: string;
    relError: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected Output (hex): ${args.expectedHex}`);
    console.log(`Output (hex): ${args.outputHex}`);
    console.log(`Expected Output (dec): ${args.expectedDec}`);
    console.log(`Output (dec): ${args.outputDec}`);
    console.log(`Absolute Error: ${args.absError}`);
    console.log(`Relative Error: ${args.relError}`);
    console.log("------------------------------------------------------------");
}

async function executeAndPrint(args: {
    harness: TrigonometryHarness;
    t: string;
    method: string;
    explanation: string;
    inputLabel: string;
    outputHex: string;
    expectedScaled: bigint;
}) {
    const expectedHex = await qScaled(args.harness, args.expectedScaled);
    const actualScaled = await outScaled(args.harness, args.outputHex);
    const absErr = scaledAbsError(actualScaled, args.expectedScaled);

    printTrigAccuracyBlock({
        t: args.t,
        method: args.method,
        explanation: args.explanation,
        input: args.inputLabel,
        expectedHex,
        outputHex: args.outputHex,
        expectedDec: formatScaledInt(args.expectedScaled),
        outputDec: formatScaledInt(actualScaled),
        absError: formatScaledInt(absErr),
        relError: scaledRelError(actualScaled, args.expectedScaled),
    });

    return {
        expectedHex,
        actualScaled,
        absErr,
    };
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("Trigonometry Library - Numerical Accuracy Tests", function () {
    let harness: TrigonometryHarness;

    let QZERO: string;
    let QONE: string;
    let QNEG_ONE: string;
    let QHALF: string;

    let QPI: string;
    let QHALF_PI: string;
    let QQUARTER_PI: string;
    let QTWO_PI: string;

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("TrigonometryHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as TrigonometryHarness;

        QZERO = await harness.fromDouble(0n);
        QONE = await harness.fromDouble(1n);
        QNEG_ONE = await harness.fromDouble(-1n);
        QHALF = await harness.fromFloat(500000000000n);

        QPI = await harness.QPI();
        QHALF_PI = await harness.QHALF_PI();
        QQUARTER_PI = await harness.QQUARTER_PI();
        QTWO_PI = await harness.QTWO_PI();

        const oneScaled = asBigInt(await harness.toFloat(await harness.fromDouble(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    // ------------------------------------------------------------
    // Section 1: Exact / near-exact special-angle values
    // ------------------------------------------------------------

    describe("Section 1: Special-angle benchmark cases", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: sin(0)=0`, async function () {
            const out = await harness.sin(QZERO);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "sin",
                explanation: "The sine function should return zero exactly at the origin.",
                inputLabel: "sin(0)",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: cos(0)=1`, async function () {
            const out = await harness.cos(QZERO);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "cos",
                explanation: "The cosine function should return one exactly at the origin.",
                inputLabel: "cos(0)",
                outputHex: out,
                expectedScaled: 1n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: sin(pi/2)=1`, async function () {
            const out = await harness.sin(QHALF_PI);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "sin",
                explanation: "The sine function attains its maximum value at pi/2.",
                inputLabel: "sin(pi/2)",
                outputHex: out,
                expectedScaled: 1n * SCALE,
            });

            expect(result.absErr < 1000n).to.equal(true);
        });

        it(`Test 1.${++testNo}: cos(pi)=-1`, async function () {
            const out = await harness.cos(QPI);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "cos",
                explanation: "The cosine function equals -1 at pi.",
                inputLabel: "cos(pi)",
                outputHex: out,
                expectedScaled: -1n * SCALE,
            });

            expect(result.absErr < 1000n).to.equal(true);
        });

        it(`Test 1.${++testNo}: tan(0)=0`, async function () {
            const out = await harness.tan(QZERO);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "tan",
                explanation: "The tangent function should return zero at the origin.",
                inputLabel: "tan(0)",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: sin(pi/4) and cos(pi/4) are nearly equal`, async function () {
            const s = await harness.sin(QQUARTER_PI);
            const c = await harness.cos(QQUARTER_PI);

            const sScaled = await outScaled(harness, s);
            const cScaled = await outScaled(harness, c);
            const diff = absBigInt(sScaled - cScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 1.${testNo}`);
            console.log("Method: sin/cos");
            console.log("Explanation: At pi/4, sine and cosine should be equal by symmetry.");
            console.log("Input: pi/4");
            console.log(`sin(pi/4): ${formatScaledInt(sScaled)}`);
            console.log(`cos(pi/4): ${formatScaledInt(cScaled)}`);
            console.log(`|sin-cos|: ${formatScaledInt(diff)}`);
            console.log("------------------------------------------------------------");

            expect(diff < 1_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Symmetry properties
    // ------------------------------------------------------------

    describe("Section 2: Symmetry properties", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: sin(-x) = -sin(x)`, async function () {
            const x = QQUARTER_PI;
            const negX = await harness.neg(x);

            const sPos = await harness.sin(x);
            const sNeg = await harness.sin(negX);

            const sPosScaled = await outScaled(harness, sPos);
            const sNegScaled = await outScaled(harness, sNeg);

            const diff = absBigInt(sNegScaled + sPosScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 2.${testNo}`);
            console.log("Method: sin symmetry");
            console.log("Explanation: Sine is an odd function.");
            console.log("Input: x=pi/4");
            console.log(`sin(x): ${formatScaledInt(sPosScaled)}`);
            console.log(`sin(-x): ${formatScaledInt(sNegScaled)}`);
            console.log(`|sin(-x)+sin(x)|: ${formatScaledInt(diff)}`);
            console.log("------------------------------------------------------------");

            expect(diff < 1_000_000n).to.equal(true);
        });

        it(`Test 2.${++testNo}: cos(-x) = cos(x)`, async function () {
            const x = QQUARTER_PI;
            const negX = await harness.neg(x);

            const cPos = await harness.cos(x);
            const cNeg = await harness.cos(negX);

            const cPosScaled = await outScaled(harness, cPos);
            const cNegScaled = await outScaled(harness, cNeg);

            const diff = absBigInt(cNegScaled - cPosScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 2.${testNo}`);
            console.log("Method: cos symmetry");
            console.log("Explanation: Cosine is an even function.");
            console.log("Input: x=pi/4");
            console.log(`cos(x): ${formatScaledInt(cPosScaled)}`);
            console.log(`cos(-x): ${formatScaledInt(cNegScaled)}`);
            console.log(`|cos(-x)-cos(x)|: ${formatScaledInt(diff)}`);
            console.log("------------------------------------------------------------");

            expect(diff < 1_000_000n).to.equal(true);
        });

        it(`Test 2.${++testNo}: atan(-x) = -atan(x)`, async function () {
            const x = QHALF;
            const negX = await harness.neg(x);

            const aPos = await harness.atan(x);
            const aNeg = await harness.atan(negX);

            const aPosScaled = await outScaled(harness, aPos);
            const aNegScaled = await outScaled(harness, aNeg);

            const diff = absBigInt(aNegScaled + aPosScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 2.${testNo}`);
            console.log("Method: atan symmetry");
            console.log("Explanation: Arctangent is an odd function.");
            console.log("Input: x=0.5");
            console.log(`atan(x): ${formatScaledInt(aPosScaled)}`);
            console.log(`atan(-x): ${formatScaledInt(aNegScaled)}`);
            console.log(`|atan(-x)+atan(x)|: ${formatScaledInt(diff)}`);
            console.log("------------------------------------------------------------");

            expect(diff < 1_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Periodicity / range reduction
    // ------------------------------------------------------------

    describe("Section 3: Periodicity and range reduction", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: sin(x + 2pi) = sin(x)`, async function () {
            const x = QQUARTER_PI;
            const xPlusTwoPi = await harness.add(x, QTWO_PI);

            const s1 = await harness.sin(x);
            const s2 = await harness.sin(xPlusTwoPi);

            const s1Scaled = await outScaled(harness, s1);
            const s2Scaled = await outScaled(harness, s2);

            const diff = absBigInt(s2Scaled - s1Scaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 3.${testNo}`);
            console.log("Method: sin periodicity");
            console.log("Explanation: Sine should be periodic with period 2pi.");
            console.log("Input: x=pi/4 and x+2pi");
            console.log(`sin(x): ${formatScaledInt(s1Scaled)}`);
            console.log(`sin(x+2pi): ${formatScaledInt(s2Scaled)}`);
            console.log(`Difference: ${formatScaledInt(diff)}`);
            console.log("------------------------------------------------------------");

            expect(diff < 1_000_000n).to.equal(true);
        });

        it(`Test 3.${++testNo}: cos(x + 2pi) = cos(x)`, async function () {
            const x = QQUARTER_PI;
            const xPlusTwoPi = await harness.add(x, QTWO_PI);

            const c1 = await harness.cos(x);
            const c2 = await harness.cos(xPlusTwoPi);

            const c1Scaled = await outScaled(harness, c1);
            const c2Scaled = await outScaled(harness, c2);

            const diff = absBigInt(c2Scaled - c1Scaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 3.${testNo}`);
            console.log("Method: cos periodicity");
            console.log("Explanation: Cosine should be periodic with period 2pi.");
            console.log("Input: x=pi/4 and x+2pi");
            console.log(`cos(x): ${formatScaledInt(c1Scaled)}`);
            console.log(`cos(x+2pi): ${formatScaledInt(c2Scaled)}`);
            console.log(`Difference: ${formatScaledInt(diff)}`);
            console.log("------------------------------------------------------------");

            expect(diff < 1_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Inverse trigonometric special values
    // ------------------------------------------------------------

    describe("Section 4: Inverse trigonometric benchmark cases", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: asin(0)=0`, async function () {
            const out = await harness.asin(QZERO);

            const result = await executeAndPrint({
                harness,
                t: `4.${testNo}`,
                method: "asin",
                explanation: "The inverse sine function should return zero at the origin.",
                inputLabel: "asin(0)",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 4.${++testNo}: asin(1)=pi/2`, async function () {
            const out = await harness.asin(QONE);

            const expectedScaled = await outScaled(harness, QHALF_PI);
            const result = await executeAndPrint({
                harness,
                t: `4.${testNo}`,
                method: "asin",
                explanation: "The inverse sine function returns pi/2 at x=1.",
                inputLabel: "asin(1)",
                outputHex: out,
                expectedScaled,
            });

            expect(result.absErr < 1_000_000n).to.equal(true);
        });

        it(`Test 4.${++testNo}: acos(1)=0`, async function () {
            const out = await harness.acos(QONE);

            const result = await executeAndPrint({
                harness,
                t: `4.${testNo}`,
                method: "acos",
                explanation: "The inverse cosine function returns zero at x=1.",
                inputLabel: "acos(1)",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr < 1_000_000n).to.equal(true);
        });

        it(`Test 4.${++testNo}: atan(0)=0`, async function () {
            const out = await harness.atan(QZERO);

            const result = await executeAndPrint({
                harness,
                t: `4.${testNo}`,
                method: "atan",
                explanation: "The inverse tangent function returns zero at the origin.",
                inputLabel: "atan(0)",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 5: Forward-inverse consistency
    // ------------------------------------------------------------

    describe("Section 5: Forward-inverse consistency", function () {
        let testNo = 0;

        it(`Test 5.${++testNo}: asin(sin(x)) ≈ x for x in principal range`, async function () {
            const x = QQUARTER_PI;

            const s = await harness.sin(x);
            const recovered = await harness.asin(s);

            const xScaled = await outScaled(harness, x);
            const recoveredScaled = await outScaled(harness, recovered);
            const err = absBigInt(recoveredScaled - xScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 5.${testNo}`);
            console.log("Method: asin(sin(x))");
            console.log("Explanation: In the principal domain, asin(sin(x)) should recover x.");
            console.log("Input: x=pi/4");
            console.log(`x: ${formatScaledInt(xScaled)}`);
            console.log(`asin(sin(x)): ${formatScaledInt(recoveredScaled)}`);
            console.log(`Absolute Error: ${formatScaledInt(err)}`);
            console.log("------------------------------------------------------------");

            expect(err < 1_000_000n).to.equal(true);
        });

        it(`Test 5.${++testNo}: acos(cos(x)) ≈ x for x in principal range`, async function () {
            const x = QQUARTER_PI;

            const c = await harness.cos(x);
            const recovered = await harness.acos(c);

            const xScaled = await outScaled(harness, x);
            const recoveredScaled = await outScaled(harness, recovered);
            const err = absBigInt(recoveredScaled - xScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 5.${testNo}`);
            console.log("Method: acos(cos(x))");
            console.log("Explanation: In the principal domain, acos(cos(x)) should recover x.");
            console.log("Input: x=pi/4");
            console.log(`x: ${formatScaledInt(xScaled)}`);
            console.log(`acos(cos(x)): ${formatScaledInt(recoveredScaled)}`);
            console.log(`Absolute Error: ${formatScaledInt(err)}`);
            console.log("------------------------------------------------------------");

            expect(err < 1_000_000n).to.equal(true);
        });

        it(`Test 5.${++testNo}: tan(atan(x)) ≈ x`, async function () {
            const x = QHALF;

            const a = await harness.atan(x);
            const recovered = await harness.tan(a);

            const xScaled = await outScaled(harness, x);
            const recoveredScaled = await outScaled(harness, recovered);
            const err = absBigInt(recoveredScaled - xScaled);

            console.log("------------------------------------------------------------");
            console.log(`Test: 5.${testNo}`);
            console.log("Method: tan(atan(x))");
            console.log("Explanation: The composition tan(atan(x)) should recover x for finite x away from singularity issues.");
            console.log("Input: x=0.5");
            console.log(`x: ${formatScaledInt(xScaled)}`);
            console.log(`tan(atan(x)): ${formatScaledInt(recoveredScaled)}`);
            console.log(`Absolute Error: ${formatScaledInt(err)}`);
            console.log("------------------------------------------------------------");

            expect(err < 1_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 6: Undefined-value behavior
    // ------------------------------------------------------------

    describe("Section 6: Undefined-value behavior", function () {
        let testNo = 0;

        it(`Test 6.${++testNo}: tan(pi/2) returns NaN`, async function () {
            const out = await harness.tan(QHALF_PI);
            const isNaN = await harness.isNaN(out);

            console.log("------------------------------------------------------------");
            console.log(`Test: 6.${testNo}`);
            console.log("Method: tan");
            console.log("Explanation: Tangent is undefined at pi/2 and should return NaN.");
            console.log("Input: tan(pi/2)");
            console.log(`isNaN: ${isNaN}`);
            console.log("------------------------------------------------------------");

            expect(isNaN).to.equal(true);
        });

        it(`Test 6.${++testNo}: cot(0) returns NaN`, async function () {
            const out = await harness.cot(QZERO);
            const isNaN = await harness.isNaN(out);

            console.log("------------------------------------------------------------");
            console.log(`Test: 6.${testNo}`);
            console.log("Method: cot");
            console.log("Explanation: Cotangent is undefined at zero and should return NaN.");
            console.log("Input: cot(0)");
            console.log(`isNaN: ${isNaN}`);
            console.log("------------------------------------------------------------");

            expect(isNaN).to.equal(true);
        });

        it(`Test 6.${++testNo}: asin outside [-1,1] returns NaN`, async function () {
            const x = await harness.fromDouble(2n);
            const out = await harness.asin(x);
            const isNaN = await harness.isNaN(out);

            console.log("------------------------------------------------------------");
            console.log(`Test: 6.${testNo}`);
            console.log("Method: asin");
            console.log("Explanation: arcsin is undefined outside the closed interval [-1,1].");
            console.log("Input: asin(2)");
            console.log(`isNaN: ${isNaN}`);
            console.log("------------------------------------------------------------");

            expect(isNaN).to.equal(true);
        });

        it(`Test 6.${++testNo}: acos outside [-1,1] returns NaN`, async function () {
            const x = await harness.fromDouble(-2n);
            const out = await harness.acos(x);
            const isNaN = await harness.isNaN(out);

            console.log("------------------------------------------------------------");
            console.log(`Test: 6.${testNo}`);
            console.log("Method: acos");
            console.log("Explanation: arccos is undefined outside the closed interval [-1,1].");
            console.log("Input: acos(-2)");
            console.log(`isNaN: ${isNaN}`);
            console.log("------------------------------------------------------------");

            expect(isNaN).to.equal(true);
        });
    });
});