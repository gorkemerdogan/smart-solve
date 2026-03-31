// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type IntegrationHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;
    fromFloat(n: bigint): Promise<string>;
    PI(): Promise<string>;

    trapezoidal(target: string, selector: string, a: string, b: string, n: bigint): Promise<string>;
    simpson13(target: string, selector: string, a: string, b: string, n: bigint): Promise<string>;
    simpson38(target: string, selector: string, a: string, b: string, n: bigint): Promise<string>;
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

async function outScaled(harness: IntegrationHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

async function qScaled(harness: IntegrationHarness, scaledValue: bigint): Promise<string> {
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

function printAccuracyBlock(args: {
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

type IntegrationMethod = "trapezoidal" | "simpson13" | "simpson38";
type MethodLabel = "TRAP" | "S13" | "S38";

async function runIntegration(
    harness: IntegrationHarness,
    method: IntegrationMethod,
    target: string,
    selector: string,
    a: string,
    b: string,
    n: bigint
): Promise<string> {
    return await harness[method](target, selector, a, b, n);
}

async function executeAndPrint(args: {
    harness: IntegrationHarness;
    t: string;
    method: MethodLabel;
    explanation: string;
    inputLabel: string;
    outputHex: string;
    expectedScaled: bigint;
}) {
    const expectedHex = await qScaled(args.harness, args.expectedScaled);
    const actualScaled = await outScaled(args.harness, args.outputHex);
    const absErr = scaledAbsError(actualScaled, args.expectedScaled);

    printAccuracyBlock({
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

describe("Integration Library - Numerical Accuracy Tests", function () {
    let harness: IntegrationHarness;
    let target: string;

    let selZero: string;
    let selOne: string;
    let selLinear: string;
    let selSquare: string;
    let selCube: string;
    let selConst5: string;
    let selSin: string;
    let selInv: string;
    let selPiecewise: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("IntegrationHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as IntegrationHarness;
        target = await harness.getAddress();

        selZero = harness.interface.getFunction("f_zero")!.selector;
        selOne = harness.interface.getFunction("f_one")!.selector;
        selLinear = harness.interface.getFunction("f_linear")!.selector;
        selSquare = harness.interface.getFunction("f_square")!.selector;
        selCube = harness.interface.getFunction("f_cube")!.selector;
        selConst5 = harness.interface.getFunction("f_const5")!.selector;
        selSin = harness.interface.getFunction("f_sin")!.selector;
        selInv = harness.interface.getFunction("f_inv")!.selector;
        selPiecewise = harness.interface.getFunction("f_piecewise")!.selector;

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    // ------------------------------------------------------------
    // Section 1: Exact / theoretically exact benchmark cases
    // ------------------------------------------------------------

    describe("Section 1: Exact and theoretically exact benchmark cases", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: Trapezoidal is exact for constant function f(x)=1`, async function () {
            const a = await qInt(0n);
            const b = await qInt(5n);

            const out = await runIntegration(harness, "trapezoidal", target, selOne, a, b, 12n);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "TRAP",
                explanation: "Composite trapezoidal rule is exact for constant functions.",
                inputLabel: "f(x)=1, interval=[0,5], n=12",
                outputHex: out,
                expectedScaled: 5n * SCALE,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Simpson 1/3 is exact for constant function f(x)=1`, async function () {
            const a = await qInt(0n);
            const b = await qInt(5n);

            const out = await runIntegration(harness, "simpson13", target, selOne, a, b, 12n);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "S13",
                explanation: "Composite Simpson's 1/3 rule is exact for constant functions.",
                inputLabel: "f(x)=1, interval=[0,5], n=12",
                outputHex: out,
                expectedScaled: 5n * SCALE,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Simpson 3/8 is exact for constant function f(x)=1`, async function () {
            const a = await qInt(0n);
            const b = await qInt(5n);

            const out = await runIntegration(harness, "simpson38", target, selOne, a, b, 12n);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "S38",
                explanation: "Composite Simpson's 3/8 rule is exact for constant functions.",
                inputLabel: "f(x)=1, interval=[0,5], n=12",
                outputHex: out,
                expectedScaled: 5n * SCALE,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Simpson 1/3 is exact for linear function f(x)=x`, async function () {
            const a = await qInt(0n);
            const b = await qInt(2n);

            const out = await runIntegration(harness, "simpson13", target, selLinear, a, b, 12n);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "S13",
                explanation: "Composite Simpson's 1/3 rule is exact for polynomials up to degree 3, including linear functions.",
                inputLabel: "f(x)=x, interval=[0,2], n=12",
                outputHex: out,
                expectedScaled: 2n * SCALE,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Simpson 1/3 is exact for quadratic function f(x)=x^2`, async function () {
            const a = await qInt(0n);
            const b = await qInt(1n);

            const out = await runIntegration(harness, "simpson13", target, selSquare, a, b, 12n);

            const expectedScaled = SCALE / 3n;

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "S13",
                explanation: "Composite Simpson's 1/3 rule is exact for quadratic polynomials.",
                inputLabel: "f(x)=x^2, interval=[0,1], n=12",
                outputHex: out,
                expectedScaled,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Simpson 3/8 is exact for cubic function f(x)=x^3`, async function () {
            const a = await qInt(0n);
            const b = await qInt(1n);

            const out = await runIntegration(harness, "simpson38", target, selCube, a, b, 12n);

            const expectedScaled = SCALE / 4n;

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "S38",
                explanation: "Composite Simpson's 3/8 rule is exact for cubic polynomials.",
                inputLabel: "f(x)=x^3, interval=[0,1], n=12",
                outputHex: out,
                expectedScaled,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Trapezoidal is exact for zero function`, async function () {
            const a = await qInt(-3n);
            const b = await qInt(7n);

            const out = await runIntegration(harness, "trapezoidal", target, selZero, a, b, 20n);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "TRAP",
                explanation: "The integral of the zero function is exactly zero for all intervals and all valid n.",
                inputLabel: "f(x)=0, interval=[-3,7], n=20",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Trapezoidal is exact for constant function f(x)=5`, async function () {
            const a = await qInt(1n);
            const b = await qInt(4n);

            const out = await runIntegration(harness, "trapezoidal", target, selConst5, a, b, 9n);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "TRAP",
                explanation: "Composite trapezoidal rule is exact for constant functions regardless of interval length.",
                inputLabel: "f(x)=5, interval=[1,4], n=9",
                outputHex: out,
                expectedScaled: 15n * SCALE,
            });

            expect(result.absErr <= 1n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Method comparison on smooth polynomial benchmark
    // ------------------------------------------------------------

    describe("Section 2: Method comparison on smooth polynomial benchmark", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: Simpson methods outperform trapezoidal on f(x)=x^2`, async function () {
            const a = await qInt(0n);
            const b = await qInt(1n);
            const expectedScaled = SCALE / 3n;

            const outTrap = await runIntegration(harness, "trapezoidal", target, selSquare, a, b, 12n);
            const outS13 = await runIntegration(harness, "simpson13", target, selSquare, a, b, 12n);
            const outS38 = await runIntegration(harness, "simpson38", target, selSquare, a, b, 12n);

            const trapScaled = await outScaled(harness, outTrap);
            const s13Scaled = await outScaled(harness, outS13);
            const s38Scaled = await outScaled(harness, outS38);

            const trapErr = scaledAbsError(trapScaled, expectedScaled);
            const s13Err = scaledAbsError(s13Scaled, expectedScaled);
            const s38Err = scaledAbsError(s38Scaled, expectedScaled);

            printAccuracyBlock({
                t: `2.${testNo}.1`,
                method: "TRAP",
                explanation: "Trapezoidal result on quadratic benchmark.",
                input: "f(x)=x^2, interval=[0,1], n=12",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outTrap,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(trapScaled),
                absError: formatScaledInt(trapErr),
                relError: scaledRelError(trapScaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `2.${testNo}.2`,
                method: "S13",
                explanation: "Simpson 1/3 result on quadratic benchmark.",
                input: "f(x)=x^2, interval=[0,1], n=12",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS13,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s13Scaled),
                absError: formatScaledInt(s13Err),
                relError: scaledRelError(s13Scaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `2.${testNo}.3`,
                method: "S38",
                explanation: "Simpson 3/8 result on quadratic benchmark.",
                input: "f(x)=x^2, interval=[0,1], n=12",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS38,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s38Scaled),
                absError: formatScaledInt(s38Err),
                relError: scaledRelError(s38Scaled, expectedScaled),
            });

            expect(s13Err <= trapErr).to.equal(true);
            expect(s38Err <= trapErr).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Subinterval sensitivity
    // ------------------------------------------------------------

    describe("Section 3: Accuracy sensitivity to subinterval count", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: Trapezoidal accuracy improves as n increases for f(x)=x^2`, async function () {
            const a = await qInt(0n);
            const b = await qInt(1n);
            const expectedScaled = SCALE / 3n;

            const N_VALUES = [5n, 10n, 20n, 50n, 100n, 150n, 200n, 250n, 300n, 500n, 750n, 1000n, 1500n, 2000n];

            const errors: bigint[] = [];

            for (let i = 0; i < N_VALUES.length; i++) {
                const n = N_VALUES[i];

                const out = await runIntegration(harness, "trapezoidal", target, selSquare, a, b, n);
                const outScaledVal = await outScaled(harness, out);
                const err = scaledAbsError(outScaledVal, expectedScaled);

                errors.push(err);

                printAccuracyBlock({
                    t: `3.${testNo}.${i + 1}`,
                    method: "TRAP",
                    explanation: `Trapezoidal rule with n=${n.toString()}.`,
                    input: `f(x)=x^2, interval=[0,1], n=${n.toString()}`,
                    expectedHex: await qScaled(harness, expectedScaled),
                    outputHex: out,
                    expectedDec: formatScaledInt(expectedScaled),
                    outputDec: formatScaledInt(outScaledVal),
                    absError: formatScaledInt(err),
                    relError: scaledRelError(outScaledVal, expectedScaled),
                });
            }

            for (let i = 1; i < errors.length; i++) {
                expect(errors[i] < errors[i - 1]).to.equal(true);
            }
        });

        it(`Test 3.${++testNo}: Simpson 1/3 remains highly accurate as n increases for f(x)=sin(x)`, async function () {
            const a = await qInt(0n);
            const b = await harness.PI();

            const expectedScaled = 2n * SCALE;

            const outN6 = await runIntegration(harness, "simpson13", target, selSin, a, b, 6n);
            const outN30 = await runIntegration(harness, "simpson13", target, selSin, a, b, 30n);
            const outN120 = await runIntegration(harness, "simpson13", target, selSin, a, b, 120n);

            const n6Scaled = await outScaled(harness, outN6);
            const n30Scaled = await outScaled(harness, outN30);
            const n120Scaled = await outScaled(harness, outN120);

            const err6 = scaledAbsError(n6Scaled, expectedScaled);
            const err30 = scaledAbsError(n30Scaled, expectedScaled);
            const err120 = scaledAbsError(n120Scaled, expectedScaled);

            printAccuracyBlock({
                t: `3.${testNo}.1`,
                method: "S13",
                explanation: "Simpson 1/3 on sin(x) over [0,pi] with n=6.",
                input: "f(x)=sin(x), interval=[0,pi], n=6",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outN6,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(n6Scaled),
                absError: formatScaledInt(err6),
                relError: scaledRelError(n6Scaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `3.${testNo}.2`,
                method: "S13",
                explanation: "Simpson 1/3 on sin(x) over [0,pi] with n=30.",
                input: "f(x)=sin(x), interval=[0,pi], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outN30,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(n30Scaled),
                absError: formatScaledInt(err30),
                relError: scaledRelError(n30Scaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `3.${testNo}.3`,
                method: "S13",
                explanation: "Simpson 1/3 on sin(x) over [0,pi] with n=120.",
                input: "f(x)=sin(x), interval=[0,pi], n=120",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outN120,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(n120Scaled),
                absError: formatScaledInt(err120),
                relError: scaledRelError(n120Scaled, expectedScaled),
            });

            expect(err30 <= err6).to.equal(true);
            expect(err120 <= err30).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Smooth non-polynomial benchmarks
    // ------------------------------------------------------------

    describe("Section 4: Smooth non-polynomial benchmarks", function () {
        let testNo = 0;

        const t1 = `4.${++testNo}`;
        it(`Test ${t1}: Trapezoidal, Simpson 1/3, and Simpson 3/8 integrate sin(x) over [0,pi] with different accuracy`, async function () {
            const a = await qInt(0n);
            const b = await harness.PI();
            const expectedScaled = 2n * SCALE;

            const outTrap = await runIntegration(harness, "trapezoidal", target, selSin, a, b, 30n);
            const outS13 = await runIntegration(harness, "simpson13", target, selSin, a, b, 30n);
            const outS38 = await runIntegration(harness, "simpson38", target, selSin, a, b, 30n);

            const trapScaled = await outScaled(harness, outTrap);
            const s13Scaled = await outScaled(harness, outS13);
            const s38Scaled = await outScaled(harness, outS38);

            const trapErr = scaledAbsError(trapScaled, expectedScaled);
            const s13Err = scaledAbsError(s13Scaled, expectedScaled);
            const s38Err = scaledAbsError(s38Scaled, expectedScaled);

            printAccuracyBlock({
                t: `${t1}.1`,
                method: "TRAP",
                explanation: "The integral of sin(x) over [0,pi] equals 2, providing a smooth non-polynomial benchmark.",
                input: "f(x)=sin(x), interval=[0,pi], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outTrap,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(trapScaled),
                absError: formatScaledInt(trapErr),
                relError: scaledRelError(trapScaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `${t1}.2`,
                method: "S13",
                explanation: "The integral of sin(x) over [0,pi] equals 2, providing a smooth non-polynomial benchmark.",
                input: "f(x)=sin(x), interval=[0,pi], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS13,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s13Scaled),
                absError: formatScaledInt(s13Err),
                relError: scaledRelError(s13Scaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `${t1}.3`,
                method: "S38",
                explanation: "The integral of sin(x) over [0,pi] equals 2, providing a smooth non-polynomial benchmark.",
                input: "f(x)=sin(x), interval=[0,pi], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS38,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s38Scaled),
                absError: formatScaledInt(s38Err),
                relError: scaledRelError(s38Scaled, expectedScaled),
            });

            expect(s13Err < trapErr).to.equal(true);
            expect(s38Err < trapErr).to.equal(true);
            expect(s13Err < 100_000_000n).to.equal(true);
            expect(s38Err < 100_000_000n).to.equal(true);
        });

        const t2 = `4.${++testNo}`;
        it(`Test ${t2}: Trapezoidal, Simpson 1/3, and Simpson 3/8 integrate 1/x over [1,2] with different accuracy`, async function () {
            const a = await qInt(1n);
            const b = await qInt(2n);

            // ln(2) scaled approximately in the current decimal display domain
            const expectedScaled = 693147180559n;

            const outTrap = await runIntegration(harness, "trapezoidal", target, selInv, a, b, 30n);
            const outS13 = await runIntegration(harness, "simpson13", target, selInv, a, b, 30n);
            const outS38 = await runIntegration(harness, "simpson38", target, selInv, a, b, 30n);

            const trapScaled = await outScaled(harness, outTrap);
            const s13Scaled = await outScaled(harness, outS13);
            const s38Scaled = await outScaled(harness, outS38);

            const trapErr = scaledAbsError(trapScaled, expectedScaled);
            const s13Err = scaledAbsError(s13Scaled, expectedScaled);
            const s38Err = scaledAbsError(s38Scaled, expectedScaled);

            printAccuracyBlock({
                t: `${t2}.1`,
                method: "TRAP",
                explanation: "The integral of 1/x over [1,2] equals ln(2), serving as a smooth non-polynomial benchmark.",
                input: "f(x)=1/x, interval=[1,2], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outTrap,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(trapScaled),
                absError: formatScaledInt(trapErr),
                relError: scaledRelError(trapScaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `${t2}.2`,
                method: "S13",
                explanation: "The integral of 1/x over [1,2] equals ln(2), serving as a smooth non-polynomial benchmark.",
                input: "f(x)=1/x, interval=[1,2], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS13,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s13Scaled),
                absError: formatScaledInt(s13Err),
                relError: scaledRelError(s13Scaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `${t2}.3`,
                method: "S38",
                explanation: "The integral of 1/x over [1,2] equals ln(2), serving as a smooth non-polynomial benchmark.",
                input: "f(x)=1/x, interval=[1,2], n=30",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS38,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s38Scaled),
                absError: formatScaledInt(s38Err),
                relError: scaledRelError(s38Scaled, expectedScaled),
            });

            expect(s13Err < trapErr).to.equal(true);
            expect(s38Err < trapErr).to.equal(true);
            expect(s13Err < 100_000_000n).to.equal(true);
            expect(s38Err < 100_000_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 5: Piecewise / non-smooth benchmark
    // ------------------------------------------------------------

    describe("Section 5: Piecewise benchmark", function () {
        let testNo = 0;

        it(`Test 5.${++testNo}: Piecewise integrand is handled with bounded error`, async function () {
            const a = await qInt(0n);
            const b = await qInt(2n);

            const outTrap = await runIntegration(harness, "trapezoidal", target, selPiecewise, a, b, 12n);
            const outS13 = await runIntegration(harness, "simpson13", target, selPiecewise, a, b, 12n);
            const outS38 = await runIntegration(harness, "simpson38", target, selPiecewise, a, b, 12n);

            const expectedScaled = 4n * SCALE;

            const trapScaled = await outScaled(harness, outTrap);
            const s13Scaled = await outScaled(harness, outS13);
            const s38Scaled = await outScaled(harness, outS38);

            const trapErr = scaledAbsError(trapScaled, expectedScaled);
            const s13Err = scaledAbsError(s13Scaled, expectedScaled);
            const s38Err = scaledAbsError(s38Scaled, expectedScaled);

            printAccuracyBlock({
                t: `5.${testNo}.1`,
                method: "TRAP",
                explanation: "Trapezoidal result on piecewise benchmark.",
                input: "piecewise f(x), interval=[0,2], n=12",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outTrap,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(trapScaled),
                absError: formatScaledInt(trapErr),
                relError: scaledRelError(trapScaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `5.${testNo}.2`,
                method: "S13",
                explanation: "Simpson 1/3 result on piecewise benchmark.",
                input: "piecewise f(x), interval=[0,2], n=12",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS13,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s13Scaled),
                absError: formatScaledInt(s13Err),
                relError: scaledRelError(s13Scaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `5.${testNo}.3`,
                method: "S38",
                explanation: "Simpson 3/8 result on piecewise benchmark.",
                input: "piecewise f(x), interval=[0,2], n=12",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outS38,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(s38Scaled),
                absError: formatScaledInt(s38Err),
                relError: scaledRelError(s38Scaled, expectedScaled),
            });

            expect(trapErr < 1n * SCALE).to.equal(true);
            expect(s13Err < 1n * SCALE).to.equal(true);
            expect(s38Err < 1n * SCALE).to.equal(true);
        });
    });
});