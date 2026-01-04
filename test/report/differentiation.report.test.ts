// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type DifferentiationHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;

    toFloat(q: string): Promise<bigint>;

    forwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    backwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    centeredDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
};

let QZERO: string;

const TOL_APPROX = 10_000_000n; // Allow 1e-5 error for O(h) approximations
const TOL_EXACT = 100n;         // Allow 1e-10 error for Exact/O(h^2) cases (rounding noise)   

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, ""); // Trim trailing zeros
}

/// Helper to convert a Hex/Quad string to a formatted decimal string
async function fmt(harness: DifferentiationHarness, val: string): Promise<string> {
    const v = await harness.toFloat(val);
    return formatScaledInt(v);
}

function expectClose(actualHex: string, expectedHex: string, toleranceBits: bigint) {
    const a = BigInt(actualHex);
    const b = BigInt(expectedHex);
    let diff = a - b;
    if (diff < 0n) diff = -diff;

    if (diff > toleranceBits) {
        console.log(`    Mismatch > tolerance!`);
        console.log(`    Exp: ${expectedHex}`);
        console.log(`    Act: ${actualHex}`);
        console.log(`    Diff: ${diff}`);
        console.log(`    Tol:  ${toleranceBits}`);
    }
    expect(diff <= toleranceBits).to.be.true;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Differentiation Library - Extended Edge Cases", function () {
    let harness: DifferentiationHarness;
    let target: string;
    let t = 0;

    let H_STEP: string;
    let NEG_H_STEP: string

    let selSquare: string, selLinear: string, selConstFive: string, selAbs: string, selCube: string;
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

        selSquare = harness.interface.getFunction("f_square")!.selector;
        selLinear = harness.interface.getFunction("f_linear")!.selector;
        selConstFive = harness.interface.getFunction("f_constFive")!.selector;
        selAbs = harness.interface.getFunction("f_abs")!.selector;
        selCube = harness.interface.getFunction("f_cube")!.selector;

        H_STEP = await harness.qFromFrac(1, 100000000); // h = 1/100,000,000
        NEG_H_STEP = await harness.qFromFrac(-1, 100000000);
        QZERO = await harness.fromFloat(0n);
    });

    // ------------------------------------------------------------
    //  Forward Difference
    // ------------------------------------------------------------

    describe("Method 1: Forward Difference (Approximation)", function () {

        it("Test 1: Standard Positive Input (x^2 at x=3)", async function () {
            t++;
            const x = await qInt(3);
            const expected = await qInt(6);

            await touchGas(harness, "forwardDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selSquare, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "forwardDiff",
                explanation: "Forward diff of x^2 at x=3, derivative 2x gives expected slope 6.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 2: Negative Input (x^2 at x=-10)", async function () {
            t++;
            const x = await qInt(-10);
            const expected = await qInt(-20);

            await touchGas(harness, "forwardDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selSquare, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "forwardDiff",
                explanation: "Forward diff of x^2 at x=-10, analytical derivative 2x gives slope -20.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 3: Zero Slope/Const (f(x)=5 at x=100)", async function () {
            t++;
            const x = await qInt(100);
            const expected = await qInt(0);

            await touchGas(harness, "forwardDiffHarness", [target, selConstFive, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selConstFive, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selConstFive, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expect(out).to.equal(expected);
            printBlockRegular({
                t,
                method: "forwardDiff",
                explanation: "Forward diff of constant f(x)=5, derivative should be exactly zero everywhere.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 4: Explicit Step Size Logic", async function () {
            t++;
            const x = await qInt(4);
            const stepOne = await qInt(1);
            const expectedDistorted = await qInt(9); // 2(4) + 1

            await touchGas(harness, "forwardDiffHarness", [target, selSquare, x, stepOne]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selSquare, x, stepOne]);
            const out = await harness.forwardDiffHarness(target, selSquare, x, stepOne);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expectedDistorted);

            expectClose(out, expectedDistorted, TOL_EXACT);
            printBlockRegular({
                t,
                method: "forwardDiff",
                explanation: "Forward diff with explicit h=1 for x^2 at x=4, finite-diff slope is 9 (2x+h).",
                gas,
                inHex: x,
                expectedHex: expectedDistorted,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 5: Linear Function Exactness", async function () {
            t++;
            const x = await qInt(500);
            const expected = await qInt(3);

            await touchGas(harness, "forwardDiffHarness", [target, selLinear, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selLinear, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selLinear, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_EXACT);
            printBlockRegular({
                t,
                method: "forwardDiff",
                explanation: "Forward diff of linear f(x)=3x+1, constant analytical slope 3 should be matched.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 6: Revert on Invalid Selector", async function () {
            t++;
            const x = await qInt(1);
            await expect(harness.forwardDiffHarness(target, "0x12345678", x, QZERO)).to.be.reverted;
            printBlockRegular({
                t,
                method: "forwardDiff",
                explanation: "Forward diff with invalid function selector should revert staticcall to target.",
                gas: "N/A",
                inHex: x,
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted"
            });
        });
    });

    // ------------------------------------------------------------
    //  Backward Difference
    // ------------------------------------------------------------

    describe("Method 2: Backward Difference (Approximation)", function () {

        it("Test 7: Standard Positive Input (x^2 at x=4)", async function () {
            t++;
            const x = await qInt(4);
            const expected = await qInt(8);

            await touchGas(harness, "backwardDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selSquare, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "backwardDiff",
                explanation: "Backward diff of x^2 at x=4, derivative 2x gives expected slope 8.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 8: Zero Crossing (x^2 at x=0)", async function () {
            t++;
            const x = await qInt(0);
            const expected = NEG_H_STEP;

            await touchGas(harness, "backwardDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selSquare, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "backwardDiff",
                explanation: "Backward diff of x^2 at x=0, expects tiny negative slope ≈-h from (f(0)-f(-h))/h.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 9: Cubic Approximation (x^3 at x=-5)", async function () {
            t++;
            const x = await qInt(-5);
            const expected = await qInt(75);

            await touchGas(harness, "backwardDiffHarness", [target, selCube, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selCube, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selCube, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "backwardDiff",
                explanation: "Backward diff of x^3 at x=-5, analytical derivative 3x^2 gives slope 75.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 10: Discontinuous Derivative (Abs(x) at x=-1)", async function () {
            t++;
            const x = await qInt(-1);
            const expected = await qInt(-1);

            await touchGas(harness, "backwardDiffHarness", [target, selAbs, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selAbs, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selAbs, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "backwardDiff",
                explanation: "Backward diff of |x| at x=-1, left-sided derivative around kink is expected to be -1.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec,
            });
        });

        it("Test 11: Explicit Step Size Symmetry", async function () {
            t++;
            const x = await qInt(4);
            const stepOne = await qInt(1);
            const expectedDistorted = await qInt(7);

            await touchGas(harness, "backwardDiffHarness", [target, selSquare, x, stepOne]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selSquare, x, stepOne]);
            const out = await harness.backwardDiffHarness(target, selSquare, x, stepOne);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expectedDistorted);

            expectClose(out, expectedDistorted, TOL_EXACT);
            printBlockRegular({
                t,
                method: "backwardDiff",
                explanation: "Backward diff with h=1 for x^2 at x=4, finite-diff slope 7 from (f(4)-f(3))/1.",
                gas,
                inHex: x,
                expectedHex: expectedDistorted,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 12: Revert on Invalid Target", async function () {
            t++;
            const x = await qInt(1);
            await expect(harness.backwardDiffHarness(ethers.ZeroAddress, selSquare, x, QZERO)).to.be.reverted;
            printBlockRegular({
                t,
                method: "backwardDiff",
                explanation: "Backward diff with zero target address should revert due to invalid staticcall target.",
                gas: "N/A",
                inHex: x,
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted"
            });
        });
    });

    // ------------------------------------------------------------
    //  Centered Difference
    // ------------------------------------------------------------

    describe("Method 3: Centered Difference", function () {

        it("Test 13: Exact Quadratic (x^2 at x=10)", async function () {
            t++;
            const x = await qInt(10);
            const expected = await qInt(20);

            await touchGas(harness, "centeredDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selSquare, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_EXACT);
            printBlockRegular({
                t,
                method: "centeredDiff",
                explanation: "Centered diff of x^2 at x=10, symmetric stencil should match exact 2x=20.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 14: Negative Input Exactness (x^2 at x=-10)", async function () {
            t++;
            const x = await qInt(-10);
            const expected = await qInt(-20);

            await touchGas(harness, "centeredDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selSquare, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_EXACT);
            printBlockRegular({
                t,
                method: "centeredDiff",
                explanation: "Centered diff of x^2 at x=-10, symmetric stencil should recover exact 2x=-20.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 15: Singularity Handling (Abs(x) at x=0)", async function () {
            t++;
            const x = await qInt(0);
            const expected = await qInt(0);

            await touchGas(harness, "centeredDiffHarness", [target, selAbs, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selAbs, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selAbs, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expect(out).to.equal(expected);
            printBlockRegular({
                t,
                method: "centeredDiff",
                explanation: "Centered diff of |x| at x=0, symmetric limit around kink averages to zero.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 16: Cubic Approximation (x^3 at x=2)", async function () {
            t++;
            const x = await qInt(2);
            const expected = await qInt(12);

            await touchGas(harness, "centeredDiffHarness", [target, selCube, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selCube, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selCube, x, QZERO);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_APPROX);
            printBlockRegular({
                t,
                method: "centeredDiff",
                explanation: "Centered diff of x^3 at x=2, derivative 3x^2=12 with small higher-order error.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 17: Explicit Step Size (Robustness)", async function () {
            t++;
            const x = await qInt(123);
            const stepOne = await qInt(1);
            const expected = await qInt(246);

            await touchGas(harness, "centeredDiffHarness", [target, selSquare, x, stepOne]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selSquare, x, stepOne]);
            const out = await harness.centeredDiffHarness(target, selSquare, x, stepOne);

            const outDec = await fmt(harness, out);
            const expDec = await fmt(harness, expected);

            expectClose(out, expected, TOL_EXACT);
            printBlockRegular({
                t,
                method: "centeredDiff",
                explanation: "Centered diff with h=1 for x^2 at x=123, (f(x+h)-f(x-h))/(2h) gives 2x=246.",
                gas,
                inHex: x,
                expectedHex: expected,
                outHex: out,
                expectedDec: expDec,
                outDec: outDec
            });
        });

        it("Test 18: Revert on staticcall failure", async function () {
            t++;
            const x = await qInt(5);
            await expect(harness.centeredDiffHarness(target, "0xdeadbeef", x, QZERO)).to.be.reverted;
            printBlockRegular({
                t,
                method: "centeredDiff",
                explanation: "Centered diff with bogus selector should revert, covering staticcall failure path.",
                gas: "N/A",
                inHex: x,
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted"
            });
        });
    });
});