// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular, fmt } from "../test-utils";

// ------------------------------------------------------------
// Types & Constants
// ------------------------------------------------------------

type ODESolverHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(n: number | bigint, d: number | bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

    euler(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
    rk2Midpoint(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
    rk2Heun(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
    rk4(target: string, selector: string, x: string, y: string, h: string): Promise<string>;
};

let QZERO: string;
const TOL_EXACT = 18446744073709551616n;           // 2^64
const TOL_APPROX = 39614081257132168796771975168n; // 2^95

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

/**
 * @notice    Compares two numerical values and reverts if the absolute difference exceeds a tolerance.
 * @param h   The ODESolverHarness instance used for state access and conversion.
 * @param a   The string identifier or raw value of the first operand.
 * @param b   The string identifier or raw value of the second operand.
 * @param tol The maximum allowable absolute difference (tolerance) between a and b.
 */
async function expectClose(h: ODESolverHarness, a: string, b: string, tol: bigint) {
    const ai = await h.toFloat(a);
    const bi = await h.toFloat(b);
    let d = ai - bi;
    if (d < 0n) d = -d;
    expect(d <= tol).to.be.true;
}

/**
 * @notice  Retrieves a value from the harness and formats it as a decimal string.
 * @param h The ODESolverHarness instance used for state access and conversion.
 * @param q The string identifier or key for the value to be retrieved.
 * @return  A promise that resolves to a formatted decimal string representation of the value.
 */
async function getDec(h: ODESolverHarness, q: string): Promise<string> {
    const val = await h.toFloat(q);
    return fmt(val);
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("ODESolverFacet – Single-Step ODE Solvers", function () {
    let h: ODESolverHarness;
    let target: string;
    let t = 0;

    let H: string;
    let HNEG: string;

    let selConst5: string;
    let selLinear: string;
    let selSquare: string;

    const qi = async (x: number | bigint) => h.qFromInt(x);
    const qf = async (n: number | bigint, d: number | bigint) => h.qFromFrac(n, d);

    before(async () => {

        // Deploy MathLib
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathLib = await MathLibFactory.deploy();
        await mathLib.waitForDeployment();

        // Deploy ODESolverHarness with linked MathLib
        const HF = await ethers.getContractFactory("ODESolverHarness", {
            libraries: {
                MathLib: await mathLib.getAddress(),
            },
        });

        h = (await HF.deploy()) as unknown as ODESolverHarness;
        target = await h.getAddress();

        selConst5 = h.interface.getFunction("f_const5")!.selector;
        selLinear = h.interface.getFunction("f_linear")!.selector;
        selSquare = h.interface.getFunction("f_square")!.selector;

        QZERO = await harness.fromFloat(0n);
        H = await h.qFromFrac(1, 10);     // h = 0.1
        HNEG = await h.qFromFrac(-1, 10); // h = -0.1
    });

    // ---------------------------------------------------------===
    // Section 1: Euler
    // ---------------------------------------------------------===

    describe("Section 1: Euler Method", function () {

        it("Test 1: Constant ODE y' = 5", async function () {
            t++;
            const y0 = await qi(2);
            const expected = await qf(5, 2); // 2.5

            await touchGas(h, "euler", [target, selConst5, QZERO, y0, H]);
            const gas = await estimateGas(h, "euler", [target, selConst5, QZERO, y0, H]);
            const out = await h.euler(target, selConst5, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "Euler integrates constant slope exactly: y + h·5.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 2: Linear ODE y' = y", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qi(11);

            await touchGas(h, "euler", [target, selLinear, QZERO, y0, H]);
            const gas = await estimateGas(h, "euler", [target, selLinear, QZERO, y0, H]);
            const out = await h.euler(target, selLinear, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "Euler on y'=y gives y+h·y (first-order approximation).",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 3: Quadratic slope y'=x²", async function () {
            t++;
            const x = await qi(2);
            const y0 = await qi(1);
            const expected = await qf(7, 5); // 1.4

            await touchGas(h, "euler", [target, selSquare, x, y0, H]);
            const gas = await estimateGas(h, "euler", [target, selSquare, x, y0, H]);
            const out = await h.euler(target, selSquare, x, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "Euler with y'=x² uses slope at x only: y + h·x².",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 4: Zero step size", async function () {
            t++;
            const y0 = await qi(7);

            await touchGas(h, "euler", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "euler", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.euler(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "With h=0, Euler must return y unchanged.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 5: Negative step", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(19, 2); // 9.5

            await touchGas(h, "euler", [target, selConst5, QZERO, y0, HNEG]);
            const gas = await estimateGas(h, "euler", [target, selConst5, QZERO, y0, HNEG]);
            const out = await h.euler(target, selConst5, QZERO, y0, HNEG);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "Negative h integrates backward in time.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 6: Zero step size (h=0)", async function () {
            t++;
            const y0 = await qi(3);

            await touchGas(h, "euler", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "euler", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.euler(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "If step size h is 0, the state remains identical regardless of the slope.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 7: Invalid selector reverts", async function () {
            t++;
            await expect(
                h.euler(target, "0xdeadbeef", QZERO, await qi(1), H)
            ).to.be.reverted;

            printBlockRegular({
                t,
                method: "Euler",
                explanation: "Invalid selector must revert due to staticcall failure.",
                gas: "N/A",
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted",
            });
        });
    });

    // ---------------------------------------------------------===
    // Section 2: RK2 Midpoint
    // ---------------------------------------------------------===

    describe("Section 2: RK2 Midpoint Method", function () {

        it("Test 8: Constant ODE y' = 5 (exact)", async function () {
            t++;
            const y0 = await qi(2);
            const expected = await qf(5, 2); // 2.5;

            await touchGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, H]);
            const gas = await estimateGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, H]);
            const out = await h.rk2Midpoint(target, selConst5, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Midpoint RK2 integrates constant slope exactly.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 9: Linear ODE y' = y", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(221, 20); // 11.05

            await touchGas(h, "rk2Midpoint", [target, selLinear, QZERO, y0, H]);
            const gas = await estimateGas(h, "rk2Midpoint", [target, selLinear, QZERO, y0, H]);
            const out = await h.rk2Midpoint(target, selLinear, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Midpoint RK2 improves Euler accuracy for y'=y.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 10: Quadratic slope y'=x²", async function () {
            t++;
            const x = await qi(2);
            const y0 = await qi(1);
            const expected = await qf(29, 20); // 1.45

            await touchGas(h, "rk2Midpoint", [target, selSquare, x, y0, H]);
            const gas = await estimateGas(h, "rk2Midpoint", [target, selSquare, x, y0, H]);
            const out = await h.rk2Midpoint(target, selSquare, x, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Midpoint samples slope at x+h/2, improving quadratic integration.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 11: Zero step size", async function () {
            t++;
            const y0 = await qi(7);

            await touchGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.rk2Midpoint(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Zero step must preserve y exactly.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 12: Negative step", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(19, 2);  // 9.5

            await touchGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, HNEG]);
            const gas = await estimateGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, HNEG]);
            const out = await h.rk2Midpoint(target, selConst5, QZERO, y0, HNEG);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Negative step integrates backward in time.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 13: Zero step size (h=0)", async function () {
            t++;
            const y0 = await qi(3);

            await touchGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "rk2Midpoint", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.rk2Midpoint(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Midpoint method evaluates slope at x + h/2; with h=0.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 14: Invalid selector reverts", async function () {
            t++;
            await expect(
                h.rk2Midpoint(target, "0xdeadbeef", QZERO, await qi(1), H)
            ).to.be.reverted;

            printBlockRegular({
                t,
                method: "RK2 Midpoint",
                explanation: "Invalid selector must revert due to staticcall failure.",
                gas: "N/A",
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted",
            });
        });
    });

    // ---------------------------------------------------------===
    // Section 3: RK2 Heun
    // ---------------------------------------------------------===

    describe("Section 3: RK2 Heun Method", function () {

        it("Test 15: Constant ODE exact", async function () {
            t++;
            const y0 = await qi(2);
            const expected = await qf(5, 2); // 2.5;

            await touchGas(h, "rk2Heun", [target, selConst5, QZERO, y0, H]);
            const gas = await estimateGas(h, "rk2Heun", [target, selConst5, QZERO, y0, H]);
            const out = await h.rk2Heun(target, selConst5, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "Heun integrates constant slope exactly via trapezoidal averaging.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 16: Linear ODE y'=y", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(11051709, 1_000_000); // 11.051709

            await touchGas(h, "rk2Heun", [target, selLinear, QZERO, y0, H]);
            const gas = await estimateGas(h, "rk2Heun", [target, selLinear, QZERO, y0, H]);
            const out = await h.rk2Heun(target, selLinear, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "Heun averages initial and predicted slopes for better linear accuracy.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 17: Quadratic slope y'=x²", async function () {
            t++;
            const x = await qi(2);
            const y0 = await qi(1);
            const expected = await qf(7, 5); // 1.4

            await touchGas(h, "rk2Heun", [target, selSquare, x, y0, H]);
            const gas = await estimateGas(h, "rk2Heun", [target, selSquare, x, y0, H]);
            const out = await h.rk2Heun(target, selSquare, x, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "Heun uses forward prediction and trapezoidal correction.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 18: Zero step", async function () {
            t++;
            const y0 = await qi(6);

            await touchGas(h, "rk2Heun", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "rk2Heun", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.rk2Heun(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "Zero step leaves solution unchanged.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 19: Negative step", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(19, 2); // 9.5

            await touchGas(h, "rk2Heun", [target, selConst5, QZERO, y0, HNEG]);
            const gas = await estimateGas(h, "rk2Heun", [target, selConst5, QZERO, y0, HNEG]);
            const out = await h.rk2Heun(target, selConst5, QZERO, y0, HNEG);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "Negative step integrates backward.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 20: Zero step size (h=0)", async function () {
            t++;
            const y0 = await qi(4);

            await touchGas(h, "rk2Heun", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "rk2Heun", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.rk2Heun(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "When h=0, the predictor step is zero, nullifying the trapezoidal correction.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 21: Invalid selector reverts", async function () {
            t++;
            await expect(
                h.rk2Heun(target, "0xdeadbeef", QZERO, await qi(1), H)
            ).to.be.reverted;

            printBlockRegular({
                t,
                method: "RK2 Heun",
                explanation: "Invalid selector must revert.",
                gas: "N/A",
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted",
            });
        });
    });

    // ---------------------------------------------------------===
    // Section 4: RK4
    // ---------------------------------------------------------===

    describe("Section 4: RK4 Method", function () {

        it("Test 22: Constant ODE exact", async function () {
            t++;
            const y0 = await qi(2);
            const expected = await qf(5, 2); // 2.5;

            await touchGas(h, "rk4", [target, selConst5, QZERO, y0, H]);
            const gas = await estimateGas(h, "rk4", [target, selConst5, QZERO, y0, H]);
            const out = await h.rk4(target, selConst5, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "RK4 integrates constant slope exactly.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 23: Linear ODE y'=y", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(11051709, 1_000_000); // 11.051709

            await touchGas(h, "rk4", [target, selLinear, QZERO, y0, H]);
            const gas = await estimateGas(h, "rk4", [target, selLinear, QZERO, y0, H]);
            const out = await h.rk4(target, selLinear, QZERO, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "RK4 closely matches exponential growth for y'=y.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 24: Quadratic slope y'=x²", async function () {
            t++;
            const x = await qi(2);
            const y0 = await qi(1);
            const expected = await qf(7, 5); // 1.4

            await touchGas(h, "rk4", [target, selSquare, x, y0, H]);
            const gas = await estimateGas(h, "rk4", [target, selSquare, x, y0, H]);
            const out = await h.rk4(target, selSquare, x, y0, H);

            await expectClose(h, out, expected, TOL_APPROX);

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "RK4 samples slope at four points for high accuracy.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 25: Zero step", async function () {
            t++;
            const y0 = await qi(5);

            await touchGas(h, "rk4", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "rk4", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.rk4(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "Zero step must preserve state.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 26: Negative step", async function () {
            t++;
            const y0 = await qi(10);
            const expected = await qf(19, 2);  // 9.5

            await touchGas(h, "rk4", [target, selConst5, QZERO, y0, HNEG]);
            const gas = await estimateGas(h, "rk4", [target, selConst5, QZERO, y0, HNEG]);
            const out = await h.rk4(target, selConst5, QZERO, y0, HNEG);

            await expectClose(h, out, expected, TOL_EXACT);

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "Negative step integrates backward exactly for constant slope.",
                gas,
                inHex: y0,
                expectedHex: expected,
                outHex: out,
                expectedDec: await getDec(h, expected),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 27: Zero step size (h=0)", async function () {
            t++;
            const y0 = await qi(3);

            await touchGas(h, "rk4", [target, selConst5, QZERO, y0, QZERO]);
            const gas = await estimateGas(h, "rk4", [target, selConst5, QZERO, y0, QZERO]);
            const out = await h.rk4(target, selConst5, QZERO, y0, QZERO);

            expect(out).to.equal(y0);

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "Zero step size ensures all internal k-values are weighted against a zero multiplier.",
                gas,
                inHex: y0,
                expectedHex: y0,
                outHex: out,
                expectedDec: await getDec(h, y0),
                outDec: fmt(await h.toFloat(out)),
            });
        });

        it("Test 28: Invalid selector reverts", async function () {
            t++;
            await expect(
                h.rk4(target, "0xdeadbeef", QZERO, await qi(1), H)
            ).to.be.reverted;

            printBlockRegular({
                t,
                method: "RK4",
                explanation: "Invalid selector must revert.",
                gas: "N/A",
                expectedHex: "Reverted",
                outHex: "Reverted",
                expectedDec: "Reverted",
                outDec: "Reverted",
            });
        });
    });
});