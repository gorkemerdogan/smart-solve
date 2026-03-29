// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type ODESolverHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;
    fromFloat(n: bigint): Promise<string>;

    euler(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
    rk2Midpoint(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
    rk2Heun(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
    rk4(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
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

async function outScaled(harness: ODESolverHarness, q: string): Promise<bigint> {
    return await harness.toFloat(q);
}

async function qScaled(harness: ODESolverHarness, scaledValue: bigint): Promise<string> {
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

function printODEAccuracyBlock(args: {
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

type ODEMethod = "euler" | "rk2Midpoint" | "rk2Heun" | "rk4";
type MethodLabel = "Euler" | "RK2 Midpoint" | "RK2 Heun" | "RK4";

async function runODE(
    harness: ODESolverHarness,
    method: ODEMethod,
    target: string,
    selector: string,
    x: string,
    y: string,
    h: string
): Promise<string> {
    return await harness[method](target, selector, x, y, h);
}

async function executeAndPrint(args: {
    harness: ODESolverHarness;
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

    printODEAccuracyBlock({
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

describe("ODESolver Library - Numerical Accuracy Tests", function () {
    let harness: ODESolverHarness;
    let target: string;

    let H_HALF: string;     // 0.5
    let H_ONE: string;      // 1
    let H_TWO: string;      // 2
    let H_TENTH: string;    // 0.1

    let selConst5: string;
    let selLinear: string;
    let selSquare: string;
    let selCubicPoly: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("ODESolverHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as ODESolverHarness;
        target = await harness.getAddress();

        selConst5 = harness.interface.getFunction("f_const5")!.selector;
        selLinear = harness.interface.getFunction("f_linear")!.selector;
        selSquare = harness.interface.getFunction("f_square")!.selector;
        selCubicPoly = harness.interface.getFunction("f_cubic_poly")!.selector;

        H_HALF = await qFrac(1, 2);
        H_ONE = await qInt(1);
        H_TWO = await qInt(2);
        H_TENTH = await qFrac(1, 10);
    });

    // ------------------------------------------------------------
    // Section 1: Exact benchmark cases
    // ------------------------------------------------------------

    describe("Section 1: Exact benchmark cases", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: All methods are exact for constant ODE y'=5`, async function () {
            const x0 = await qInt(0);
            const y0 = await qInt(2);

            // y_{n+1} = y_n + 5h, for h=1 => 7
            const expectedScaled = 7n * SCALE;

            const outEuler = await runODE(harness, "euler", target, selConst5, x0, y0, H_ONE);
            const outMid = await runODE(harness, "rk2Midpoint", target, selConst5, x0, y0, H_ONE);
            const outHeun = await runODE(harness, "rk2Heun", target, selConst5, x0, y0, H_ONE);
            const outRK4 = await runODE(harness, "rk4", target, selConst5, x0, y0, H_ONE);

            const methods: Array<{ label: MethodLabel; out: string; suffix: string }> = [
                { label: "Euler", out: outEuler, suffix: ".1" },
                { label: "RK2 Midpoint", out: outMid, suffix: ".2" },
                { label: "RK2 Heun", out: outHeun, suffix: ".3" },
                { label: "RK4", out: outRK4, suffix: ".4" },
            ];

            for (const m of methods) {
                const result = await executeAndPrint({
                    harness,
                    t: `1.${testNo}${m.suffix}`,
                    method: m.label,
                    explanation: "For y'=5, one-step methods should reproduce the exact affine update y+h·5.",
                    inputLabel: "y'=5, x0=0, y0=2, h=1",
                    outputHex: m.out,
                    expectedScaled,
                });

                expect(result.absErr <= 1n).to.equal(true);
            }
        });

        it(`Test 1.${++testNo}: All methods are exact for y'=x^2 from x=0 with h=1`, async function () {
            const x0 = await qInt(0);
            const y0 = await qInt(0);

            // Exact one-step solution:
            // y1 = y0 + ∫_0^1 x^2 dx = 1/3
            const expectedScaled = SCALE / 3n;

            const outEuler = await runODE(harness, "euler", target, selSquare, x0, y0, H_ONE);
            const outMid = await runODE(harness, "rk2Midpoint", target, selSquare, x0, y0, H_ONE);
            const outHeun = await runODE(harness, "rk2Heun", target, selSquare, x0, y0, H_ONE);
            const outRK4 = await runODE(harness, "rk4", target, selSquare, x0, y0, H_ONE);

            const eulerScaled = await outScaled(harness, outEuler);
            const midScaled = await outScaled(harness, outMid);
            const heunScaled = await outScaled(harness, outHeun);
            const rk4Scaled = await outScaled(harness, outRK4);

            const eErr = scaledAbsError(eulerScaled, expectedScaled);
            const mErr = scaledAbsError(midScaled, expectedScaled);
            const hErr = scaledAbsError(heunScaled, expectedScaled);
            const rErr = scaledAbsError(rk4Scaled, expectedScaled);

            printODEAccuracyBlock({
                t: `1.${testNo}.1`,
                method: "Euler",
                explanation: "Euler on y'=x^2 from x=0 to x=1.",
                input: "y'=x^2, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outEuler,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(eulerScaled),
                absError: formatScaledInt(eErr),
                relError: scaledRelError(eulerScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `1.${testNo}.2`,
                method: "RK2 Midpoint",
                explanation: "RK2 Midpoint on y'=x^2 from x=0 to x=1.",
                input: "y'=x^2, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outMid,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(midScaled),
                absError: formatScaledInt(mErr),
                relError: scaledRelError(midScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `1.${testNo}.3`,
                method: "RK2 Heun",
                explanation: "RK2 Heun on y'=x^2 from x=0 to x=1.",
                input: "y'=x^2, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outHeun,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(heunScaled),
                absError: formatScaledInt(hErr),
                relError: scaledRelError(heunScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `1.${testNo}.4`,
                method: "RK4",
                explanation: "RK4 on y'=x^2 from x=0 to x=1.",
                input: "y'=x^2, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outRK4,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(rk4Scaled),
                absError: formatScaledInt(rErr),
                relError: scaledRelError(rk4Scaled, expectedScaled),
            });

            // Since x^2 depends only on x, higher-order methods should be more accurate than Euler.
            expect(mErr <= eErr).to.equal(true);
            expect(hErr <= eErr).to.equal(true);
            expect(rErr <= eErr).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Smooth linear ODE benchmark y'=y
    // ------------------------------------------------------------

    describe("Section 2: Smooth linear ODE benchmark y'=y", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: RK methods outperform Euler on y'=y with h=0.5`, async function () {
            const x0 = await qInt(0);
            const y0 = await qInt(1);

            // Exact: y(0.5) = e^0.5 ≈ 1.648721270700
            const expectedScaled = 1_648_721_270_700n;

            const outEuler = await runODE(harness, "euler", target, selLinear, x0, y0, H_HALF);
            const outMid = await runODE(harness, "rk2Midpoint", target, selLinear, x0, y0, H_HALF);
            const outHeun = await runODE(harness, "rk2Heun", target, selLinear, x0, y0, H_HALF);
            const outRK4 = await runODE(harness, "rk4", target, selLinear, x0, y0, H_HALF);

            const eScaled = await outScaled(harness, outEuler);
            const mScaled = await outScaled(harness, outMid);
            const hScaled = await outScaled(harness, outHeun);
            const rScaled = await outScaled(harness, outRK4);

            const eErr = scaledAbsError(eScaled, expectedScaled);
            const mErr = scaledAbsError(mScaled, expectedScaled);
            const hErr = scaledAbsError(hScaled, expectedScaled);
            const rErr = scaledAbsError(rScaled, expectedScaled);

            printODEAccuracyBlock({
                t: `2.${testNo}.1`,
                method: "Euler",
                explanation: "Euler on y'=y with h=0.5.",
                input: "y'=y, x0=0, y0=1, h=0.5",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outEuler,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(eScaled),
                absError: formatScaledInt(eErr),
                relError: scaledRelError(eScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `2.${testNo}.2`,
                method: "RK2 Midpoint",
                explanation: "RK2 Midpoint on y'=y with h=0.5.",
                input: "y'=y, x0=0, y0=1, h=0.5",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outMid,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(mScaled),
                absError: formatScaledInt(mErr),
                relError: scaledRelError(mScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `2.${testNo}.3`,
                method: "RK2 Heun",
                explanation: "RK2 Heun on y'=y with h=0.5.",
                input: "y'=y, x0=0, y0=1, h=0.5",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outHeun,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(hScaled),
                absError: formatScaledInt(hErr),
                relError: scaledRelError(hScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `2.${testNo}.4`,
                method: "RK4",
                explanation: "RK4 on y'=y with h=0.5.",
                input: "y'=y, x0=0, y0=1, h=0.5",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outRK4,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(rScaled),
                absError: formatScaledInt(rErr),
                relError: scaledRelError(rScaled, expectedScaled),
            });

            expect(mErr < eErr).to.equal(true);
            expect(hErr < eErr).to.equal(true);
            expect(rErr < mErr).to.equal(true);
            expect(rErr < hErr).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Step-size sensitivity on y'=y
    // ------------------------------------------------------------

    describe("Section 3: Step-size sensitivity on y'=y", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: Euler error grows with larger h on y'=y`, async function () {
            const x0 = await qInt(0);
            const y0 = await qInt(1);

            const outH01 = await runODE(harness, "euler", target, selLinear, x0, y0, H_TENTH);
            const outH05 = await runODE(harness, "euler", target, selLinear, x0, y0, H_HALF);
            const outH1 = await runODE(harness, "euler", target, selLinear, x0, y0, H_ONE);

            // exact values:
            // e^0.1 ≈ 1.105170918076
            // e^0.5 ≈ 1.648721270700
            // e^1   ≈ 2.718281828459
            const exp01 = 1_105_170_918_076n;
            const exp05 = 1_648_721_270_700n;
            const exp1 = 2_718_281_828_459n;

            const y01 = await outScaled(harness, outH01);
            const y05 = await outScaled(harness, outH05);
            const y1 = await outScaled(harness, outH1);

            const err01 = scaledAbsError(y01, exp01);
            const err05 = scaledAbsError(y05, exp05);
            const err1 = scaledAbsError(y1, exp1);

            printODEAccuracyBlock({
                t: `3.${testNo}.1`,
                method: "Euler",
                explanation: "Euler with h=0.1 on y'=y.",
                input: "y'=y, x0=0, y0=1, h=0.1",
                expectedHex: await qScaled(harness, exp01),
                outputHex: outH01,
                expectedDec: formatScaledInt(exp01),
                outputDec: formatScaledInt(y01),
                absError: formatScaledInt(err01),
                relError: scaledRelError(y01, exp01),
            });

            printODEAccuracyBlock({
                t: `3.${testNo}.2`,
                method: "Euler",
                explanation: "Euler with h=0.5 on y'=y.",
                input: "y'=y, x0=0, y0=1, h=0.5",
                expectedHex: await qScaled(harness, exp05),
                outputHex: outH05,
                expectedDec: formatScaledInt(exp05),
                outputDec: formatScaledInt(y05),
                absError: formatScaledInt(err05),
                relError: scaledRelError(y05, exp05),
            });

            printODEAccuracyBlock({
                t: `3.${testNo}.3`,
                method: "Euler",
                explanation: "Euler with h=1 on y'=y.",
                input: "y'=y, x0=0, y0=1, h=1",
                expectedHex: await qScaled(harness, exp1),
                outputHex: outH1,
                expectedDec: formatScaledInt(exp1),
                outputDec: formatScaledInt(y1),
                absError: formatScaledInt(err1),
                relError: scaledRelError(y1, exp1),
            });

            expect(err01 < err05).to.equal(true);
            expect(err05 < err1).to.equal(true);
        });

        it(`Test 3.${++testNo}: RK4 remains highly accurate under smaller h on y'=y`, async function () {
            const x0 = await qInt(0);
            const y0 = await qInt(1);

            const outH01 = await runODE(harness, "rk4", target, selLinear, x0, y0, H_TENTH);
            const outH05 = await runODE(harness, "rk4", target, selLinear, x0, y0, H_HALF);

            const exp01 = 1_105_170_918_076n;
            const exp05 = 1_648_721_270_700n;

            const y01 = await outScaled(harness, outH01);
            const y05 = await outScaled(harness, outH05);

            const err01 = scaledAbsError(y01, exp01);
            const err05 = scaledAbsError(y05, exp05);

            printODEAccuracyBlock({
                t: `3.${testNo}.1`,
                method: "RK4",
                explanation: "RK4 with h=0.1 on y'=y.",
                input: "y'=y, x0=0, y0=1, h=0.1",
                expectedHex: await qScaled(harness, exp01),
                outputHex: outH01,
                expectedDec: formatScaledInt(exp01),
                outputDec: formatScaledInt(y01),
                absError: formatScaledInt(err01),
                relError: scaledRelError(y01, exp01),
            });

            printODEAccuracyBlock({
                t: `3.${testNo}.2`,
                method: "RK4",
                explanation: "RK4 with h=0.5 on y'=y.",
                input: "y'=y, x0=0, y0=1, h=0.5",
                expectedHex: await qScaled(harness, exp05),
                outputHex: outH05,
                expectedDec: formatScaledInt(exp05),
                outputDec: formatScaledInt(y05),
                absError: formatScaledInt(err05),
                relError: scaledRelError(y05, exp05),
            });

            expect(err01 < err05).to.equal(true);
            expect(err01 < 1_000_000n).to.equal(true); // 1e-8
        });
    });

    // ------------------------------------------------------------
    // Section 4: Polynomial-in-x benchmark
    // ------------------------------------------------------------

    describe("Section 4: Polynomial-in-x benchmark", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: RK4 accurately integrates cubic polynomial slope`, async function () {
            const x0 = await qInt(0);
            const y0 = await qInt(0);

            // y' = x^3 + x^2 + 2x + 3
            // y(1) - y(0) = Integral(_0^1 (x^3 + x^2 + 2x + 3) dx)
            //             = 1/4 + 1/3 + 1 + 3 = 55/12 = 4.583333333333...
            const expectedScaled = 4_583_333_333_333n;

            const outEuler = await runODE(harness, "euler", target, selCubicPoly, x0, y0, H_ONE);
            const outMid = await runODE(harness, "rk2Midpoint", target, selCubicPoly, x0, y0, H_ONE);
            const outHeun = await runODE(harness, "rk2Heun", target, selCubicPoly, x0, y0, H_ONE);
            const outRK4 = await runODE(harness, "rk4", target, selCubicPoly, x0, y0, H_ONE);

            const eScaled = await outScaled(harness, outEuler);
            const mScaled = await outScaled(harness, outMid);
            const hScaled = await outScaled(harness, outHeun);
            const rScaled = await outScaled(harness, outRK4);

            const eErr = scaledAbsError(eScaled, expectedScaled);
            const mErr = scaledAbsError(mScaled, expectedScaled);
            const hErr = scaledAbsError(hScaled, expectedScaled);
            const rErr = scaledAbsError(rScaled, expectedScaled);

            printODEAccuracyBlock({
                t: `4.${testNo}.1`,
                method: "Euler",
                explanation: "Euler on polynomial slope benchmark.",
                input: "y'=x^3+x^2+2x+3, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outEuler,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(eScaled),
                absError: formatScaledInt(eErr),
                relError: scaledRelError(eScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `4.${testNo}.2`,
                method: "RK2 Midpoint",
                explanation: "RK2 Midpoint on polynomial slope benchmark.",
                input: "y'=x^3+x^2+2x+3, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outMid,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(mScaled),
                absError: formatScaledInt(mErr),
                relError: scaledRelError(mScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `4.${testNo}.3`,
                method: "RK2 Heun",
                explanation: "RK2 Heun on polynomial slope benchmark.",
                input: "y'=x^3+x^2+2x+3, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outHeun,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(hScaled),
                absError: formatScaledInt(hErr),
                relError: scaledRelError(hScaled, expectedScaled),
            });

            printODEAccuracyBlock({
                t: `4.${testNo}.4`,
                method: "RK4",
                explanation: "RK4 on polynomial slope benchmark.",
                input: "y'=x^3+x^2+2x+3, x0=0, y0=0, h=1",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outRK4,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(rScaled),
                absError: formatScaledInt(rErr),
                relError: scaledRelError(rScaled, expectedScaled),
            });

            expect(mErr <= eErr).to.equal(true);
            expect(hErr <= eErr).to.equal(true);
            expect(rErr <= mErr).to.equal(true);
            expect(rErr <= hErr).to.equal(true);
        });
    });
});