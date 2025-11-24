// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type DifferentiationHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;

    forwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    backwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    centeredDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
};

const QZERO = "0x00000000000000000000000000000000";

// TOLERANCES
const TOL_APPROX = 39614081257132168796771975168n; // For Forward/Backward diff (O(h) error). 2^95
const TOL_EXACT = 18446744073709551616n;           // For Centered diff (O(h^2) error) and linear functions. 2^64

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

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

async function touchGas(harness: DifferentiationHarness, method: string, args: any[]) {
    const data = harness.interface.encodeFunctionData(method, args);
    const [signer] = await ethers.getSigners();
    const to = await harness.getAddress();
    const tx = await signer.sendTransaction({ to, data });
    await tx.wait();
}

async function estimateGas(harness: DifferentiationHarness, method: string, args: any[]) {
    const anyH = harness as any;
    if (anyH[method]?.estimateGas) {
        return (await anyH[method].estimateGas(...args)).toString();
    }
    const data = harness.interface.encodeFunctionData(method, args);
    const [signer] = await ethers.getSigners();
    const to = await harness.getAddress();
    const gas = await signer.estimateGas({ to, data });
    return gas.toString();
}

function printBlock({ t, method, explanation, gas, inHex, outHex, outDec }: any) {
    const sep = "-".repeat(60);
    const decLine = outDec ? `Output: ${outDec}` : "";
    
    console.log(`
        ${sep}
        Test ${t}
        Method: ${method}
        Explanation: ${explanation}
        Gas Usage: ${gas}
        Input (x): ${inHex}
        Output (hex): ${outHex}
        ${decLine}
`.trim());
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

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "forwardDiff", explanation: "x^2 at x=3 -> 6", gas, inHex: x, outHex: out, outDec: "6" });
        });

        it("Test 2: Negative Input (x^2 at x=-10)", async function () {
            t++;
            const x = await qInt(-10);
            const expected = await qInt(-20);

            await touchGas(harness, "forwardDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selSquare, x, QZERO);

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "forwardDiff", explanation: "x^2 at x=-10 -> -20", gas, inHex: x, outHex: out, outDec: "-20" });
        });

        it("Test 3: Zero Slope/Const (f(x)=5 at x=100)", async function () {
            t++;
            const x = await qInt(100);
            const expected = await qInt(0);

            await touchGas(harness, "forwardDiffHarness", [target, selConstFive, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selConstFive, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selConstFive, x, QZERO);

            expect(out).to.equal(expected);
            printBlock({ t, method: "forwardDiff", explanation: "Constant -> 0", gas, inHex: x, outHex: out, outDec: "0" });
        });

        it("Test 4: Explicit Step Size Logic", async function () {
            t++;
            const x = await qInt(4);
            const stepOne = await qInt(1);
            const expectedDistorted = await qInt(9); // 2(4) + 1

            await touchGas(harness, "forwardDiffHarness", [target, selSquare, x, stepOne]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selSquare, x, stepOne]);
            const out = await harness.forwardDiffHarness(target, selSquare, x, stepOne);

            expectClose(out, expectedDistorted, TOL_EXACT);
            printBlock({ t, method: "forwardDiff", explanation: "Explicit h=1, x^2 at 4 -> 9", gas, inHex: x, outHex: out, outDec: "9" });
        });

        it("Test 5: Linear Function Exactness", async function () {
            t++;
            const x = await qInt(500);
            const expected = await qInt(3);

            await touchGas(harness, "forwardDiffHarness", [target, selLinear, x, QZERO]);
            const gas = await estimateGas(harness, "forwardDiffHarness", [target, selLinear, x, QZERO]);
            const out = await harness.forwardDiffHarness(target, selLinear, x, QZERO);

            expectClose(out, expected, TOL_EXACT);
            printBlock({ t, method: "forwardDiff", explanation: "Linear slope -> 3", gas, inHex: x, outHex: out, outDec: "3" });
        });

        it("Test 6: Revert on Invalid Selector", async function () {
            t++;
            const x = await qInt(1);
            await expect(harness.forwardDiffHarness(target, "0x12345678", x, QZERO)).to.be.reverted;
            printBlock({ t, method: "forwardDiff", explanation: "Revert", gas: "N/A", inHex: x, outHex: "Reverted", outDec: "Reverted" });
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

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "backwardDiff", explanation: "x^2 at x=4 -> 8", gas, inHex: x, outHex: out, outDec: "8" });
        });

        it("Test 8: Zero Crossing (x^2 at x=0)", async function () {
            t++;
            const x = await qInt(0);
            const expected = NEG_H_STEP;

            await touchGas(harness, "backwardDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selSquare, x, QZERO);

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "backwardDiff", explanation: "x^2 at x=0 -> -h", gas, inHex: x, outHex: out, outDec: "-1e-8" });
        });

        it("Test 9: Cubic Approximation (x^3 at x=-5)", async function () {
            t++;
            const x = await qInt(-5);
            const expected = await qInt(75);

            await touchGas(harness, "backwardDiffHarness", [target, selCube, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selCube, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selCube, x, QZERO);

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "backwardDiff", explanation: "x^3 at x=-5 -> 75", gas, inHex: x, outHex: out, outDec: "75" });
        });

        it("Test 10: Discontinuous Derivative (Abs(x) at x=-1)", async function () {
            t++;
            const x = await qInt(-1);
            const expected = await qInt(-1);

            await touchGas(harness, "backwardDiffHarness", [target, selAbs, x, QZERO]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selAbs, x, QZERO]);
            const out = await harness.backwardDiffHarness(target, selAbs, x, QZERO);

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "backwardDiff", explanation: "|x| at x=-1 -> -1", gas, inHex: x, outHex: out, outDec: "-1" });
        });

        it("Test 11: Explicit Step Size Symmetry", async function () {
            t++;
            const x = await qInt(4);
            const stepOne = await qInt(1);
            const expectedDistorted = await qInt(7);

            await touchGas(harness, "backwardDiffHarness", [target, selSquare, x, stepOne]);
            const gas = await estimateGas(harness, "backwardDiffHarness", [target, selSquare, x, stepOne]);
            const out = await harness.backwardDiffHarness(target, selSquare, x, stepOne);

            expectClose(out, expectedDistorted, TOL_EXACT);
            printBlock({ t, method: "backwardDiff", explanation: "Explicit h=1, x^2 at 4 -> 7", gas, inHex: x, outHex: out, outDec: "7" });
        });

        it("Test 12: Revert on Invalid Target", async function () {
            t++;
            const x = await qInt(1);
            await expect(harness.backwardDiffHarness(ethers.ZeroAddress, selSquare, x, QZERO)).to.be.reverted;
            printBlock({ t, method: "backwardDiff", explanation: "Revert on zero address", gas: "N/A", inHex: x, outHex: "Reverted", outDec: "Reverted" });
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

            expectClose(out, expected, TOL_EXACT);
            printBlock({ t, method: "centeredDiff", explanation: "x^2 at x=10 -> 20", gas, inHex: x, outHex: out, outDec: "20" });
        });

        it("Test 14: Negative Input Exactness (x^2 at x=-10)", async function () {
            t++;
            const x = await qInt(-10);
            const expected = await qInt(-20);

            await touchGas(harness, "centeredDiffHarness", [target, selSquare, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selSquare, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selSquare, x, QZERO);

            expectClose(out, expected, TOL_EXACT);
            printBlock({ t, method: "centeredDiff", explanation: "x^2 at x=-10 -> -20", gas, inHex: x, outHex: out, outDec: "-20" });
        });

        it("Test 15: Singularity Handling (Abs(x) at x=0)", async function () {
            t++;
            const x = await qInt(0);
            const expected = await qInt(0);

            await touchGas(harness, "centeredDiffHarness", [target, selAbs, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selAbs, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selAbs, x, QZERO);

            expect(out).to.equal(expected);
            printBlock({ t, method: "centeredDiff", explanation: "|x| at x=0 -> 0", gas, inHex: x, outHex: out, outDec: "0" });
        });

        it("Test 16: Cubic Approximation (x^3 at x=2)", async function () {
            t++;
            const x = await qInt(2);
            const expected = await qInt(12);

            await touchGas(harness, "centeredDiffHarness", [target, selCube, x, QZERO]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selCube, x, QZERO]);
            const out = await harness.centeredDiffHarness(target, selCube, x, QZERO);

            expectClose(out, expected, TOL_APPROX);
            printBlock({ t, method: "centeredDiff", explanation: "x^3 at x=2 -> 12", gas, inHex: x, outHex: out, outDec: "12" });
        });

        it("Test 17: Explicit Step Size (Robustness)", async function () {
            t++;
            const x = await qInt(123);
            const stepOne = await qInt(1);
            const expected = await qInt(246);

            await touchGas(harness, "centeredDiffHarness", [target, selSquare, x, stepOne]);
            const gas = await estimateGas(harness, "centeredDiffHarness", [target, selSquare, x, stepOne]);
            const out = await harness.centeredDiffHarness(target, selSquare, x, stepOne);

            expectClose(out, expected, TOL_EXACT);
            printBlock({ t, method: "centeredDiff", explanation: "Explicit h=1, x^2 at 123 -> 246", gas, inHex: x, outHex: out, outDec: "246" });
        });

        it("Test 18: Revert on staticcall failure", async function () {
            t++;
            const x = await qInt(5);
            await expect(harness.centeredDiffHarness(target, "0xdeadbeef", x, QZERO)).to.be.reverted;
            printBlock({ t, method: "centeredDiff", explanation: "Revert check", gas: "N/A", inHex: x, outHex: "Reverted", outDec: "Reverted" });
        });
    });
});