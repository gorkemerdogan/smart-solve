// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type DifferentiationHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;
    fromFloat(n: bigint): Promise<string>;

    forwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    backwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    centeredDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
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

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

async function qScaled(harness: DifferentiationHarness, scaledValue: bigint): Promise<string> {
    return await harness.fromFloat(asBigInt(scaledValue));
}

async function outScaled(harness: DifferentiationHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
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
    hLabel: string;
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
    console.log(`Step Size: ${args.hLabel}`);
    console.log(`Expected Output (hex): ${args.expectedHex}`);
    console.log(`Output (hex): ${args.outputHex}`);
    console.log(`Expected Output (dec): ${args.expectedDec}`);
    console.log(`Output (dec): ${args.outputDec}`);
    console.log(`Absolute Error: ${args.absError}`);
    console.log(`Relative Error: ${args.relError}`);
    console.log("------------------------------------------------------------");
}

type DiffMethodName = "forwardDiffHarness" | "backwardDiffHarness" | "centeredDiffHarness";
type DiffLabel = "FW" | "BW" | "CENT";

async function runDiff(
    harness: DifferentiationHarness,
    method: DiffMethodName,
    target: string,
    selector: string,
    x: string,
    h: string
): Promise<string> {
    return await harness[method](target, selector, x, h);
}

async function executeAndPrint(args: {
    harness: DifferentiationHarness;
    t: string;
    method: DiffLabel;
    explanation: string;
    inputLabel: string;
    hLabel: string;
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
        hLabel: args.hLabel,
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

describe("Differentiation Library - Numerical Accuracy Tests", function () {
    let harness: DifferentiationHarness;
    let target: string;

    let H_SMALL: string;    // 1e-3
    let H_ONE: string;      // 1
    let H_TEN: string;      // 10

    let selLinear: string;
    let selSquare: string;
    let selCube: string;
    let selAbs: string;
    let selConstFive: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("DifferentiationHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as DifferentiationHarness;
        target = await harness.getAddress();

        selLinear = harness.interface.getFunction("f_linear")!.selector;
        selSquare = harness.interface.getFunction("f_square")!.selector;
        selCube = harness.interface.getFunction("f_cube")!.selector;
        selAbs = harness.interface.getFunction("f_abs")!.selector;
        selConstFive = harness.interface.getFunction("f_constFive")!.selector;

        H_SMALL = await harness.qFromFrac(1n, 1000n); // 1e-3
        H_ONE = await harness.qFromInt(1n);
        H_TEN = await harness.qFromInt(10n);

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    // ------------------------------------------------------------
    // Section 1: Exact constant-derivative benchmark cases
    // ------------------------------------------------------------

    describe("Section 1: Exact constant-derivative benchmark cases", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: FW is exact for linear function f(x)=3x-2`, async function () {
            const x = await qInt(37n);
            const out = await runDiff(harness, "forwardDiffHarness", target, selLinear, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "FW",
                explanation: "For f(x)=3x-2, the derivative is exactly 3 for all x. Forward difference should recover it exactly.",
                inputLabel: "f(x)=3x-2, x=37",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 3n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: BW is exact for linear function f(x)=3x-2`, async function () {
            const x = await qInt(-12n);
            const out = await runDiff(harness, "backwardDiffHarness", target, selLinear, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "BW",
                explanation: "For f(x)=3x-2, the derivative is exactly 3 for all x. Backward difference should recover it exactly.",
                inputLabel: "f(x)=3x-2, x=-12",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 3n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: CENT is exact for linear function f(x)=3x-2`, async function () {
            const x = await qInt(0n);
            const out = await runDiff(harness, "centeredDiffHarness", target, selLinear, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "CENT",
                explanation: "For f(x)=3x-2, the derivative is exactly 3 for all x. Centered difference should recover it exactly.",
                inputLabel: "f(x)=3x-2, x=0",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 3n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 1.${++testNo}: CENT is exact for constant function`, async function () {
            const x = await qInt(25n);
            const out = await runDiff(harness, "centeredDiffHarness", target, selConstFive, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `1.${testNo}`,
                method: "CENT",
                explanation: "For the constant function f(x)=5, the derivative is exactly 0.",
                inputLabel: "f(x)=5, x=25",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 0n,
            });

            expect(result.absErr).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Exact / predictable polynomial cases
    // ------------------------------------------------------------

    describe("Section 2: Exact and predictable polynomial benchmark cases", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: CENT is exact for quadratic function`, async function () {
            const x = await qInt(7n);
            const out = await runDiff(harness, "centeredDiffHarness", target, selSquare, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `2.${testNo}`,
                method: "CENT",
                explanation: "For f(x)=x^2, centered difference is exact in symmetric form, giving f'(x)=2x.",
                inputLabel: "f(x)=x^2, x=7",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 14n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 2.${++testNo}: FW on quadratic gives predictable biased result`, async function () {
            const x = await qInt(7n);
            const out = await runDiff(harness, "forwardDiffHarness", target, selSquare, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `2.${testNo}`,
                method: "FW",
                explanation: "For f(x)=x^2 and h=1, forward difference returns 2x+h = 15.",
                inputLabel: "f(x)=x^2, x=7",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 15n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });

        it(`Test 2.${++testNo}: BW on quadratic gives predictable biased result`, async function () {
            const x = await qInt(7n);
            const out = await runDiff(harness, "backwardDiffHarness", target, selSquare, x, H_ONE);

            const result = await executeAndPrint({
                harness,
                t: `2.${testNo}`,
                method: "BW",
                explanation: "For f(x)=x^2 and h=1, backward difference returns 2x-h = 13.",
                inputLabel: "f(x)=x^2, x=7",
                hLabel: "h=1",
                outputHex: out,
                expectedScaled: 13n * SCALE,
            });

            expect(result.absErr).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Method comparison on smooth nonlinear function
    // ------------------------------------------------------------

    describe("Section 3: Method comparison on smooth nonlinear function", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: Centered difference is most accurate on cubic benchmark`, async function () {
            const x = await qInt(10n);

            const outFW = await runDiff(harness, "forwardDiffHarness", target, selCube, x, H_SMALL);
            const outBW = await runDiff(harness, "backwardDiffHarness", target, selCube, x, H_SMALL);
            const outCENT = await runDiff(harness, "centeredDiffHarness", target, selCube, x, H_SMALL);

            const expectedScaled = 300n * SCALE;

            const fwScaled = await outScaled(harness, outFW);
            const bwScaled = await outScaled(harness, outBW);
            const centScaled = await outScaled(harness, outCENT);

            const fwErr = scaledAbsError(fwScaled, expectedScaled);
            const bwErr = scaledAbsError(bwScaled, expectedScaled);
            const centErr = scaledAbsError(centScaled, expectedScaled);

            printAccuracyBlock({
                t: `3.${testNo}.1`,
                method: "FW",
                explanation: "Forward difference on f(x)=x^3 at x=10 with h=1e-3.",
                input: "f(x)=x^3, x=10",
                hLabel: "h=1e-3",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outFW,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(fwScaled),
                absError: formatScaledInt(fwErr),
                relError: scaledRelError(fwScaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `3.${testNo}.2`,
                method: "BW",
                explanation: "Backward difference on f(x)=x^3 at x=10 with h=1e-3.",
                input: "f(x)=x^3, x=10",
                hLabel: "h=1e-3",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outBW,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(bwScaled),
                absError: formatScaledInt(bwErr),
                relError: scaledRelError(bwScaled, expectedScaled),
            });

            printAccuracyBlock({
                t: `3.${testNo}.3`,
                method: "CENT",
                explanation: "Centered difference on f(x)=x^3 at x=10 with h=1e-3.",
                input: "f(x)=x^3, x=10",
                hLabel: "h=1e-3",
                expectedHex: await qScaled(harness, expectedScaled),
                outputHex: outCENT,
                expectedDec: formatScaledInt(expectedScaled),
                outputDec: formatScaledInt(centScaled),
                absError: formatScaledInt(centErr),
                relError: scaledRelError(centScaled, expectedScaled),
            });

            expect(centErr <= fwErr).to.equal(true);
            expect(centErr <= bwErr).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Step size sensitivity
    // ------------------------------------------------------------

    describe("Section 4: Step size sensitivity", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: Finite-difference error changes with step size on the cubic benchmark`, async function () {
            const x = await qInt(10n);
            const expectedScaled = 300n * SCALE;

            const STEP_CASES: Array<{ label: string; h: string }> = [
                { label: "h=1e-3", h: H_SMALL },
                { label: "h=1", h: H_ONE },
                { label: "h=10", h: H_TEN },
                { label: "h=20", h: await qInt(20n) },
                { label: "h=50", h: await qInt(50n) },
                { label: "h=100", h: await qInt(100n) },
                { label: "h=200", h: await qInt(200n) },
            ];

            const METHODS: Array<{
                method: DiffMethodName;
                label: DiffLabel;
                explanationPrefix: string;
            }> = [
                {
                    method: "forwardDiffHarness",
                    label: "FW",
                    explanationPrefix: "Forward difference"
                },
                {
                    method: "backwardDiffHarness",
                    label: "BW",
                    explanationPrefix: "Backward difference"
                },
                {
                    method: "centeredDiffHarness",
                    label: "CENT",
                    explanationPrefix: "Centered difference"
                },
            ];

            for (let m = 0; m < METHODS.length; m++) {
                const methodInfo = METHODS[m];
                const errors: bigint[] = [];

                for (let i = 0; i < STEP_CASES.length; i++) {
                    const s = STEP_CASES[i];

                    const out = await runDiff(
                        harness,
                        methodInfo.method,
                        target,
                        selCube,
                        x,
                        s.h
                    );

                    const outScaledVal = await outScaled(harness, out);
                    const err = scaledAbsError(outScaledVal, expectedScaled);
                    errors.push(err);

                    printAccuracyBlock({
                        t: `4.${testNo}.${m + 1}.${i + 1}`,
                        method: methodInfo.label,
                        explanation: `${methodInfo.explanationPrefix} with ${s.label} on the cubic benchmark.`,
                        input: "f(x)=x^3, x=10",
                        hLabel: s.label,
                        expectedHex: await qScaled(harness, expectedScaled),
                        outputHex: out,
                        expectedDec: formatScaledInt(expectedScaled),
                        outputDec: formatScaledInt(outScaledVal),
                        absError: formatScaledInt(err),
                        relError: scaledRelError(outScaledVal, expectedScaled),
                    });
                }

                for (let i = 1; i < errors.length; i++) {
                    expect(errors[i] >= errors[i - 1]).to.equal(true);
                }
            }
        });
    });

    // ------------------------------------------------------------
    // Section 5: Edge behavior at non-differentiable point
    // ------------------------------------------------------------

    describe("Section 5: Edge behavior at non-differentiable point", function () {
        let testNo = 0;

        it(`Test 5.${++testNo}: |x| at x=0 gives one-sided and symmetric finite-difference behavior`, async function () {
            const x = await qInt(0n);

            const outFW = await runDiff(harness, "forwardDiffHarness", target, selAbs, x, H_ONE);
            const outBW = await runDiff(harness, "backwardDiffHarness", target, selAbs, x, H_ONE);
            const outCENT = await runDiff(harness, "centeredDiffHarness", target, selAbs, x, H_ONE);

            const fwScaled = await outScaled(harness, outFW);
            const bwScaled = await outScaled(harness, outBW);
            const centScaled = await outScaled(harness, outCENT);

            printAccuracyBlock({
                t: `5.${testNo}.1`,
                method: "FW",
                explanation: "At x=0 for |x|, forward difference approximates the right-hand slope.",
                input: "f(x)=|x|, x=0",
                hLabel: "h=1",
                expectedHex: await qScaled(harness, 1n * SCALE),
                outputHex: outFW,
                expectedDec: "1",
                outputDec: formatScaledInt(fwScaled),
                absError: formatScaledInt(scaledAbsError(fwScaled, 1n * SCALE)),
                relError: scaledRelError(fwScaled, 1n * SCALE),
            });

            printAccuracyBlock({
                t: `5.${testNo}.2`,
                method: "BW",
                explanation: "At x=0 for |x|, backward difference approximates the left-hand slope.",
                input: "f(x)=|x|, x=0",
                hLabel: "h=1",
                expectedHex: await qScaled(harness, -1n * SCALE),
                outputHex: outBW,
                expectedDec: "-1",
                outputDec: formatScaledInt(bwScaled),
                absError: formatScaledInt(scaledAbsError(bwScaled, -1n * SCALE)),
                relError: scaledRelError(bwScaled, -1n * SCALE),
            });

            printAccuracyBlock({
                t: `5.${testNo}.3`,
                method: "CENT",
                explanation: "At x=0 for |x|, centered difference returns the symmetric secant slope.",
                input: "f(x)=|x|, x=0",
                hLabel: "h=1",
                expectedHex: await qScaled(harness, 0n),
                outputHex: outCENT,
                expectedDec: "0",
                outputDec: formatScaledInt(centScaled),
                absError: formatScaledInt(scaledAbsError(centScaled, 0n)),
                relError: scaledRelError(centScaled, 0n),
            });

            expect(fwScaled).to.equal(1n * SCALE);
            expect(bwScaled).to.equal(-1n * SCALE);
            expect(centScaled).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 6: Smooth-region behavior away from kink
    // ------------------------------------------------------------

    describe("Section 6: Smooth-region behavior away from kink", function () {
        let testNo = 0;

        it(`Test 6.${++testNo}: Centered difference recovers derivative of |x| away from x=0`, async function () {
            const xPos = await qInt(5n);
            const xNeg = await qInt(-5n);

            const outPos = await runDiff(harness, "centeredDiffHarness", target, selAbs, xPos, H_ONE);
            const outNeg = await runDiff(harness, "centeredDiffHarness", target, selAbs, xNeg, H_ONE);

            const posResult = await executeAndPrint({
                harness,
                t: `6.${testNo}.1`,
                method: "CENT",
                explanation: "Away from the kink, |x| behaves smoothly and centered difference should recover the local derivative.",
                inputLabel: "f(x)=|x|, x=5",
                hLabel: "h=1",
                outputHex: outPos,
                expectedScaled: 1n * SCALE,
            });

            const negResult = await executeAndPrint({
                harness,
                t: `6.${testNo}.2`,
                method: "CENT",
                explanation: "Away from the kink, |x| behaves smoothly and centered difference should recover the local derivative.",
                inputLabel: "f(x)=|x|, x=-5",
                hLabel: "h=1",
                outputHex: outNeg,
                expectedScaled: -1n * SCALE,
            });

            expect(posResult.absErr).to.equal(0n);
            expect(negResult.absErr).to.equal(0n);
        });
    });
});