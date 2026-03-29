// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type RootFindingHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;
    cmp(a: string, b: string): Promise<bigint>;
    evalHarness(target: string, sel: string, x: string): Promise<string>;

    rootFindingBisection(
        target: string,
        fSelector: string,
        a: string,
        b: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, boolean, string]>;

    rootFindingNewton(
        target: string,
        fSelector: string,
        dfTarget: string,
        dfSelector: string,
        x0: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, boolean, string]>;

    rootFindingSecant(
        target: string,
        fSelector: string,
        x0: string,
        x1: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, boolean, string]>;
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

async function outScaled(harness: RootFindingHarness, q: string): Promise<bigint> {
    return await harness.toFloat(q);
}

function printRootAccuracyBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expectedRootDec: string;
    outputRootDec: string;
    absRootError: string;
    relRootError: string;
    residualDec: string;
    iterations: string;
    converged: boolean;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected Root (dec): ${args.expectedRootDec}`);
    console.log(`Output Root (dec): ${args.outputRootDec}`);
    console.log(`Absolute Root Error: ${args.absRootError}`);
    console.log(`Relative Root Error: ${args.relRootError}`);
    console.log(`Residual |f(root)|: ${args.residualDec}`);
    console.log(`Iterations: ${args.iterations}`);
    console.log(`Converged: ${args.converged}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("RootFinding Library - Numerical Accuracy Tests", function () {
    let harness: RootFindingHarness;
    let target: string;

    let TOL_1E_9: string;

    let selX2Minus4: string;
    let selDf2x: string;
    let selCubic: string;
    let selDfCubic: string;
    let selQuartic: string;
    let selDfQuartic: string;

    let selShiftSmall: string;
    let selDfShiftSmall: string;
    let selShiftMedium: string;
    let selDfShiftMedium: string;
    let selShiftLarge: string;
    let selDfShiftLarge: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qUInt = async (x: number | bigint) => await harness.qFromUInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("RootFindingHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as RootFindingHarness;
        target = await harness.getAddress();

        selX2Minus4 = harness.interface.getFunction("f_x2_minus_4")!.selector;
        selDf2x = harness.interface.getFunction("df_2x")!.selector;

        selCubic = harness.interface.getFunction("f_cubic")!.selector;
        selDfCubic = harness.interface.getFunction("df_cubic")!.selector;

        selQuartic = harness.interface.getFunction("f_quartic")!.selector;
        selDfQuartic = harness.interface.getFunction("df_quartic")!.selector;

        selShiftSmall = harness.interface.getFunction("f_shift_small")!.selector;
        selDfShiftSmall = harness.interface.getFunction("df_shift_small")!.selector;

        selShiftMedium = harness.interface.getFunction("f_shift_medium")!.selector;
        selDfShiftMedium = harness.interface.getFunction("df_shift_medium")!.selector;

        selShiftLarge = harness.interface.getFunction("f_shift_large")!.selector;
        selDfShiftLarge = harness.interface.getFunction("df_shift_large")!.selector;

        TOL_1E_9 = await qFrac(1, 1_000_000_000);
    });

    // ------------------------------------------------------------
    // Section 1: Exact / near-exact benchmark roots
    // ------------------------------------------------------------

    describe("Section 1: Exact and near-exact benchmark roots", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: Bisection converges to root x=2 for x^2-4`, async function () {
            const a = await qInt(1);
            const b = await qInt(3);
            const expectedRoot = 2n * SCALE;

            const [root, iterations, converged, fAtRoot] =
                await harness.rootFindingBisection(target, selX2Minus4, a, b, TOL_1E_9, 100n);

            const rootScaled = await outScaled(harness, root);
            const residualScaled = absBigInt(await outScaled(harness, fAtRoot));
            const rootErr = scaledAbsError(rootScaled, expectedRoot);

            printRootAccuracyBlock({
                t: `1.${testNo}`,
                method: "Bisection",
                explanation: "Classic bracketing benchmark for f(x)=x^2-4 over [1,3].",
                input: "f(x)=x^2-4, interval=[1,3], tol=1e-9",
                expectedRootDec: "2",
                outputRootDec: formatScaledInt(rootScaled),
                absRootError: formatScaledInt(rootErr),
                relRootError: scaledRelError(rootScaled, expectedRoot),
                residualDec: formatScaledInt(residualScaled),
                iterations: iterations.toString(),
                converged,
            });

            expect(converged).to.equal(true);
            expect(rootErr < 10_000n).to.equal(true);      // 1e-8
            expect(residualScaled < 10_000n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Newton converges rapidly to root x=2 for x^2-4`, async function () {
            const x0 = await qInt(3);
            const expectedRoot = 2n * SCALE;

            const [root, iterations, converged, fAtRoot] =
                await harness.rootFindingNewton(target, selX2Minus4, target, selDf2x, x0, TOL_1E_9, 50n);

            const rootScaled = await outScaled(harness, root);
            const residualScaled = absBigInt(await outScaled(harness, fAtRoot));
            const rootErr = scaledAbsError(rootScaled, expectedRoot);

            printRootAccuracyBlock({
                t: `1.${testNo}`,
                method: "Newton",
                explanation: "Newton benchmark for f(x)=x^2-4 using x0=3.",
                input: "f(x)=x^2-4, df(x)=2x, x0=3, tol=1e-9",
                expectedRootDec: "2",
                outputRootDec: formatScaledInt(rootScaled),
                absRootError: formatScaledInt(rootErr),
                relRootError: scaledRelError(rootScaled, expectedRoot),
                residualDec: formatScaledInt(residualScaled),
                iterations: iterations.toString(),
                converged,
            });

            expect(converged).to.equal(true);
            expect(rootErr < 10_000n).to.equal(true);
            expect(residualScaled < 10_000n).to.equal(true);
            expect(iterations <= 10n).to.equal(true);
        });

        it(`Test 1.${++testNo}: Secant converges to root x=2 for x^2-4`, async function () {
            const x0 = await qInt(1);
            const x1 = await qInt(3);
            const expectedRoot = 2n * SCALE;

            const [root, iterations, converged, fAtRoot] =
                await harness.rootFindingSecant(target, selX2Minus4, x0, x1, TOL_1E_9, 50n);

            const rootScaled = await outScaled(harness, root);
            const residualScaled = absBigInt(await outScaled(harness, fAtRoot));
            const rootErr = scaledAbsError(rootScaled, expectedRoot);

            printRootAccuracyBlock({
                t: `1.${testNo}`,
                method: "Secant",
                explanation: "Secant benchmark for f(x)=x^2-4 using x0=1 and x1=3.",
                input: "f(x)=x^2-4, x0=1, x1=3, tol=1e-9",
                expectedRootDec: "2",
                outputRootDec: formatScaledInt(rootScaled),
                absRootError: formatScaledInt(rootErr),
                relRootError: scaledRelError(rootScaled, expectedRoot),
                residualDec: formatScaledInt(residualScaled),
                iterations: iterations.toString(),
                converged,
            });

            expect(converged).to.equal(true);
            expect(rootErr < 10_000n).to.equal(true);
            expect(residualScaled < 10_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Smooth nonlinear benchmark
    // ------------------------------------------------------------

    describe("Section 2: Smooth nonlinear benchmark", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: All methods converge on cubic benchmark x^3-x-2=0`, async function () {
            const a = await qInt(1);
            const b = await qInt(2);
            const x0Newton = await qInt(1);
            const x0Secant = await qInt(1);
            const x1Secant = await qInt(2);

            // Real root ≈ 1.521379706805
            const expectedRoot = 1_521_379_706_805n;

            const [rootB, iterB, convB, fB] =
                await harness.rootFindingBisection(target, selCubic, a, b, TOL_1E_9, 100n);

            const [rootN, iterN, convN, fN] =
                await harness.rootFindingNewton(target, selCubic, target, selDfCubic, x0Newton, TOL_1E_9, 50n);

            const [rootS, iterS, convS, fS] =
                await harness.rootFindingSecant(target, selCubic, x0Secant, x1Secant, TOL_1E_9, 50n);

            const rootBScaled = await outScaled(harness, rootB);
            const rootNScaled = await outScaled(harness, rootN);
            const rootSScaled = await outScaled(harness, rootS);

            const errB = scaledAbsError(rootBScaled, expectedRoot);
            const errN = scaledAbsError(rootNScaled, expectedRoot);
            const errS = scaledAbsError(rootSScaled, expectedRoot);

            const resB = absBigInt(await outScaled(harness, fB));
            const resN = absBigInt(await outScaled(harness, fN));
            const resS = absBigInt(await outScaled(harness, fS));

            printRootAccuracyBlock({
                t: `2.${testNo}.1`,
                method: "Bisection",
                explanation: "Bisection on cubic benchmark.",
                input: "f(x)=x^3-x-2, interval=[1,2], tol=1e-9",
                expectedRootDec: "1.521379706805",
                outputRootDec: formatScaledInt(rootBScaled),
                absRootError: formatScaledInt(errB),
                relRootError: scaledRelError(rootBScaled, expectedRoot),
                residualDec: formatScaledInt(resB),
                iterations: iterB.toString(),
                converged: convB,
            });

            printRootAccuracyBlock({
                t: `2.${testNo}.2`,
                method: "Newton",
                explanation: "Newton on cubic benchmark.",
                input: "f(x)=x^3-x-2, df(x)=3x^2-1, x0=1, tol=1e-9",
                expectedRootDec: "1.521379706805",
                outputRootDec: formatScaledInt(rootNScaled),
                absRootError: formatScaledInt(errN),
                relRootError: scaledRelError(rootNScaled, expectedRoot),
                residualDec: formatScaledInt(resN),
                iterations: iterN.toString(),
                converged: convN,
            });

            printRootAccuracyBlock({
                t: `2.${testNo}.3`,
                method: "Secant",
                explanation: "Secant on cubic benchmark.",
                input: "f(x)=x^3-x-2, x0=1, x1=2, tol=1e-9",
                expectedRootDec: "1.521379706805",
                outputRootDec: formatScaledInt(rootSScaled),
                absRootError: formatScaledInt(errS),
                relRootError: scaledRelError(rootSScaled, expectedRoot),
                residualDec: formatScaledInt(resS),
                iterations: iterS.toString(),
                converged: convS,
            });

            expect(convB).to.equal(true);
            expect(convN).to.equal(true);
            expect(convS).to.equal(true);

            expect(errB < 100_000n).to.equal(true); // 1e-7
            expect(errN < 100_000n).to.equal(true);
            expect(errS < 100_000n).to.equal(true);

            expect(iterN <= iterB).to.equal(true);
            expect(iterS <= iterB).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Exact shifted-linear roots across scales
    // ------------------------------------------------------------

    describe("Section 3: Exact shifted-linear roots across scales", function () {
        let testNo = 0;

        const CASES = [
            {
                label: "small root",
                expectedRoot: 2n * SCALE,
                fSel: "small",
                dfSel: "small",
                bisA: 0n,
                bisB: 4n,
                x0: 0n,
                s0: 0n,
                s1: 4n,
            },
            {
                label: "medium root",
                expectedRoot: 200n * SCALE,
                fSel: "medium",
                dfSel: "medium",
                bisA: 0n,
                bisB: 400n,
                x0: 0n,
                s0: 0n,
                s1: 400n,
            },
            {
                label: "large root",
                expectedRoot: 20_000n * SCALE,
                fSel: "large",
                dfSel: "large",
                bisA: 0n,
                bisB: 40_000n,
                x0: 0n,
                s0: 0n,
                s1: 40_000n,
            },
        ];

        for (const c of CASES) {
            it(`Test 3.${++testNo}: Newton exactly recovers ${c.label} shifted linear root`, async function () {
                const fSel =
                    c.fSel === "small" ? selShiftSmall :
                        c.fSel === "medium" ? selShiftMedium :
                            selShiftLarge;

                const dfSel =
                    c.dfSel === "small" ? selDfShiftSmall :
                        c.dfSel === "medium" ? selDfShiftMedium :
                            selDfShiftLarge;

                const x0 = await qUInt(c.x0);

                const [root, iterations, converged, fAtRoot] =
                    await harness.rootFindingNewton(target, fSel, target, dfSel, x0, TOL_1E_9, 20n);

                const rootScaled = await outScaled(harness, root);
                const residualScaled = absBigInt(await outScaled(harness, fAtRoot));
                const rootErr = scaledAbsError(rootScaled, c.expectedRoot);

                printRootAccuracyBlock({
                    t: `3.${testNo}`,
                    method: "Newton",
                    explanation: `Newton on ${c.label} shifted-linear benchmark with exact derivative 1.`,
                    input: `${c.label} shifted linear function, x0=0, tol=1e-9`,
                    expectedRootDec: formatScaledInt(c.expectedRoot),
                    outputRootDec: formatScaledInt(rootScaled),
                    absRootError: formatScaledInt(rootErr),
                    relRootError: scaledRelError(rootScaled, c.expectedRoot),
                    residualDec: formatScaledInt(residualScaled),
                    iterations: iterations.toString(),
                    converged,
                });

                expect(converged).to.equal(true);
                expect(rootErr).to.equal(0n);
                expect(residualScaled).to.equal(0n);
                expect(iterations <= 2n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 4: Quartic benchmark
    // ------------------------------------------------------------

    describe("Section 4: Quartic benchmark", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: Newton converges accurately on x^4-10=0`, async function () {
            const x0 = await qInt(2);

            // Positive root = 10^(1/4) ≈ 1.778279410038
            const expectedRoot = 1_778_279_410_038n;

            const [root, iterations, converged, fAtRoot] =
                await harness.rootFindingNewton(target, selQuartic, target, selDfQuartic, x0, TOL_1E_9, 50n);

            const rootScaled = await outScaled(harness, root);
            const residualScaled = absBigInt(await outScaled(harness, fAtRoot));
            const rootErr = scaledAbsError(rootScaled, expectedRoot);

            printRootAccuracyBlock({
                t: `4.${testNo}`,
                method: "Newton",
                explanation: "Quartic benchmark using Newton's method.",
                input: "f(x)=x^4-10, df(x)=4x^3, x0=2, tol=1e-9",
                expectedRootDec: "1.778279410038",
                outputRootDec: formatScaledInt(rootScaled),
                absRootError: formatScaledInt(rootErr),
                relRootError: scaledRelError(rootScaled, expectedRoot),
                residualDec: formatScaledInt(residualScaled),
                iterations: iterations.toString(),
                converged,
            });

            expect(converged).to.equal(true);
            expect(rootErr < 100_000n).to.equal(true);      // 1e-7
            expect(residualScaled < 100_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 5: Method comparison on same benchmark
    // ------------------------------------------------------------

    describe("Section 5: Direct method comparison on same benchmark", function () {
        let testNo = 0;

        it(`Test 5.1: All methods converge successfully on x^2-4 and iteration counts are compared`, async function () {
            const a = await qInt(1);
            const b = await qInt(3);
            const x0N = await qInt(3);
            const x0S = await qInt(1);
            const x1S = await qInt(3);

            const [, iterB, convB] =
                await harness.rootFindingBisection(target, selX2Minus4, a, b, TOL_1E_9, 100n);

            const [, iterN, convN] =
                await harness.rootFindingNewton(target, selX2Minus4, target, selDf2x, x0N, TOL_1E_9, 50n);

            const [, iterS, convS] =
                await harness.rootFindingSecant(target, selX2Minus4, x0S, x1S, TOL_1E_9, 50n);

            console.log("------------------------------------------------------------");
            console.log(`Test: 5.${testNo}`);
            console.log("Method comparison on f(x)=x^2-4");
            console.log(`Bisection iterations: ${iterB.toString()}, converged: ${convB}`);
            console.log(`Newton iterations: ${iterN.toString()}, converged: ${convN}`);
            console.log(`Secant iterations: ${iterS.toString()}, converged: ${convS}`);
            console.log("------------------------------------------------------------");

            expect(convB).to.equal(true);
            expect(convN).to.equal(true);
            expect(convS).to.equal(true);
        });
    });
});