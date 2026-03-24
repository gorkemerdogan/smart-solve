// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular, fmt } from "../test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type ODESolverHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(n: number | bigint, d: number | bigint): Promise<string>;
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

type ODEMethod = "euler" | "rk2Midpoint" | "rk2Heun" | "rk4";

async function getDec(h: ODESolverHarness, q: string): Promise<string> {
    const val = await h.toFloat(q);
    return fmt(val);
}

function toGasBigInt(gas: unknown): bigint {
    if (typeof gas === "bigint") return gas;
    return BigInt(gas as string);
}

async function runODECase(
    h: ODESolverHarness,
    method: ODEMethod,
    target: string,
    selector: string,
    x: string,
    y: string,
    step: string
) {
    await touchGas(h, method, [target, selector, x, y, step]);
    const gas = await estimateGas(h, method, [target, selector, x, y, step]);
    const out = await h[method](target, selector, x, y, step);
    return { gas, out };
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("ODESolverHarness - Gas Growth Tests", function () {
    let h: ODESolverHarness;
    let target: string;
    let t = 0;

    let QZERO: string;

    // Steps
    let H_001: string;
    let H_01: string;
    let H_05: string;
    let H_1: string;
    let H_NEG_01: string;

    // Selectors
    let selConst5: string;
    let selLinear: string;
    let selSquare: string;
    let selPoly: string;

    const qi = async (x: number | bigint) => h.qFromInt(x);
    const qf = async (n: number | bigint, d: number | bigint) => h.qFromFrac(n, d);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathLib = await MathLibFactory.deploy();
        await mathLib.waitForDeployment();

        const HF = await ethers.getContractFactory("ODESolverHarness", {
            libraries: {
                MathLib: await mathLib.getAddress(),
            },
        });

        h = (await HF.deploy()) as unknown as ODESolverHarness;
        await h.waitForDeployment();
        target = await h.getAddress();

        selConst5 = h.interface.getFunction("f_const5")!.selector;
        selLinear = h.interface.getFunction("f_linear")!.selector;
        selSquare = h.interface.getFunction("f_square")!.selector;
        selPoly = h.interface.getFunction("f_cubic_poly")!.selector;

        QZERO = await h.fromFloat(0n);

        H_001 = await qf(1, 100);   // 0.01
        H_01 = await qf(1, 10);     // 0.1
        H_05 = await qf(1, 2);      // 0.5
        H_1 = await qi(1);          // 1
        H_NEG_01 = await qf(-1, 10); // -0.1
    });

    // ------------------------------------------------------------
    // Section 1: Gas Sensitivity to RHS Function Structure
    // ------------------------------------------------------------

    describe("Section 1: Gas Sensitivity to RHS Function Structure", function () {
        const METHODS: Array<{ method: ODEMethod; label: string }> = [
            { method: "euler", label: "Euler" },
            { method: "rk2Midpoint", label: "RK2 Midpoint" },
            { method: "rk2Heun", label: "RK2 Heun" },
            { method: "rk4", label: "RK4" },
        ];

        const FUNCTIONS: Array<{ selector: () => string; label: string }> = [
            { selector: () => selConst5, label: "f_const5" },
            { selector: () => selLinear, label: "f_linear" },
            { selector: () => selSquare, label: "f_square" },
            { selector: () => selPoly, label: "f_cubic_poly" },
        ];

        for (const m of METHODS) {
            for (const f of FUNCTIONS) {
                it(`Test ${++t}: ${m.label} gas sensitivity for ${f.label}`, async function () {
                    const x0 = QZERO;
                    const y0 = await qi(10);

                    const { gas, out } = await runODECase(
                        h,
                        m.method,
                        target,
                        f.selector(),
                        x0,
                        y0,
                        H_01
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to RHS function structure using ${f.label} with x0=0, y0=10, and h=0.1.`,
                        gas,
                        inHex: `x0=0, y0=10, h=0.1`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await getDec(h, out),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 2: Gas Sensitivity to Step Size
    // ------------------------------------------------------------

    describe("Section 2: Gas Sensitivity to Step Size", function () {
        const METHODS: Array<{ method: ODEMethod; label: string }> = [
            { method: "euler", label: "Euler" },
            { method: "rk2Midpoint", label: "RK2 Midpoint" },
            { method: "rk2Heun", label: "RK2 Heun" },
            { method: "rk4", label: "RK4" },
        ];

        const STEP_CASES: Array<{ step: () => string; label: string }> = [
            { step: () => H_001, label: "0.01" },
            { step: () => H_01, label: "0.1" },
            { step: () => H_05, label: "0.5" },
            { step: () => H_1, label: "1" },
            { step: () => H_NEG_01, label: "-0.1" },
        ];

        for (const m of METHODS) {
            for (const s of STEP_CASES) {
                it(`Test ${++t}: ${m.label} gas sensitivity for h=${s.label}`, async function () {
                    const x0 = QZERO;
                    const y0 = await qi(10);

                    const { gas, out } = await runODECase(
                        h,
                        m.method,
                        target,
                        selLinear,
                        x0,
                        y0,
                        s.step()
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to step size using y'=y with x0=0, y0=10, and h=${s.label}.`,
                        gas,
                        inHex: `x0=0, y0=10, h=${s.label}`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await getDec(h, out),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 3: Gas Sensitivity to Initial y Value
    // ------------------------------------------------------------

    describe("Section 3: Gas Sensitivity to Initial y Value", function () {
        const METHODS: Array<{ method: ODEMethod; label: string }> = [
            { method: "euler", label: "Euler" },
            { method: "rk2Midpoint", label: "RK2 Midpoint" },
            { method: "rk2Heun", label: "RK2 Heun" },
            { method: "rk4", label: "RK4" },
        ];

        const Y_CASES: Array<{ y: number; label: string }> = [
            { y: -1000, label: "-1000" },
            { y: -1, label: "-1" },
            { y: 0, label: "0" },
            { y: 1, label: "1" },
            { y: 1000, label: "1000" },
        ];

        for (const m of METHODS) {
            for (const yc of Y_CASES) {
                it(`Test ${++t}: ${m.label} gas sensitivity for y0=${yc.label}`, async function () {
                    const x0 = QZERO;
                    const y0 = await qi(yc.y);

                    const { gas, out } = await runODECase(
                        h,
                        m.method,
                        target,
                        selLinear,
                        x0,
                        y0,
                        H_01
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to initial y value using y'=y with x0=0, h=0.1, and y0=${yc.label}.`,
                        gas,
                        inHex: `x0=0, y0=${yc.label}, h=0.1`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await getDec(h, out),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 4: Gas Sensitivity to Initial x Position
    // ------------------------------------------------------------

    describe("Section 4: Gas Sensitivity to Initial x Position", function () {
        const METHODS: Array<{ method: ODEMethod; label: string }> = [
            { method: "euler", label: "Euler" },
            { method: "rk2Midpoint", label: "RK2 Midpoint" },
            { method: "rk2Heun", label: "RK2 Heun" },
            { method: "rk4", label: "RK4" },
        ];

        const X_CASES: Array<{ x: number; label: string }> = [
            { x: -10, label: "-10" },
            { x: 0, label: "0" },
            { x: 1, label: "1" },
            { x: 10, label: "10" },
            { x: 100, label: "100" },
        ];

        for (const m of METHODS) {
            for (const xc of X_CASES) {
                it(`Test ${++t}: ${m.label} gas sensitivity for x0=${xc.label}`, async function () {
                    const x0 = await qi(xc.x);
                    const y0 = await qi(1);

                    const { gas, out } = await runODECase(
                        h,
                        m.method,
                        target,
                        selSquare,
                        x0,
                        y0,
                        H_01
                    );

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to initial x position using y'=x² with y0=1, h=0.1, and x0=${xc.label}.`,
                        gas,
                        inHex: `x0=${xc.label}, y0=1, h=0.1`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await getDec(h, out),
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });
});