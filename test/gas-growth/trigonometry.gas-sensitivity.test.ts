// SPDX-License-Identifier: MIT
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
//  Types
// ------------------------------------------------------------

type TrigHarness = Contract & {
    fromFloat(x: bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

    sin(x: string): Promise<string>;
    cos(x: string): Promise<string>;
    tan(x: string): Promise<string>;
    cot(x: string): Promise<string>;

    asin(x: string): Promise<string>;
    acos(x: string): Promise<string>;
    atan(x: string): Promise<string>;

    add(a: string, b: string): Promise<string>;

    QPI(): Promise<string>;
    QHALF_PI(): Promise<string>;
    isNaN(x: string): Promise<boolean>;
};

// ------------------------------------------------------------
//  Constants & Helpers
// ------------------------------------------------------------

const SCALE = 1e12;

async function toQuad(h: TrigHarness, x: number): Promise<string> {
    return h.fromFloat(BigInt(Math.round(x * SCALE)));
}

async function fromQuad(h: TrigHarness, q: string): Promise<number> {
    const scaled = await h.toFloat(q);
    return Number(scaled) / SCALE;
}

function fmt(x: number): string {
    if (Number.isNaN(x)) return "NaN";
    if (!Number.isFinite(x)) return String(x);
    return x.toFixed(12);
}

describe("Trigonometry - Gas Growth Tests", function () {
    let harness: TrigHarness;
    let t = 0;

    let QPI: string;
    let QHALF_PI: string;

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const math = await MathLibFactory.deploy();
        await math.waitForDeployment();

        const HF = await ethers.getContractFactory("TrigonometryHarness", {
            libraries: {
                "contracts/libraries/MathLib.sol:MathLib": await math.getAddress(),
            },
        });

        harness = (await HF.deploy()) as unknown as TrigHarness;
        await harness.waitForDeployment();

        QPI = await harness.QPI();
        QHALF_PI = await harness.QHALF_PI();
    });

    // ------------------------------------------------------------
    //  Section 1: Gas Sensitivity to Input Magnitude
    // ------------------------------------------------------------

    describe("Section 1: Gas Sensitivity to Input Magnitude", function () {
        const DIRECT_CASES = [-1000, -1, 0, 1, 1000];
        const INV_CASES = [-1, -0.5, 0, 0.5, 1];
        const ATAN_CASES = [-1000, -1, 0, 1, 1000];

        const DIRECT_METHODS: Array<{ method: "sin" | "cos" | "tan" | "cot"; label: string }> = [
            { method: "sin", label: "sin" },
            { method: "cos", label: "cos" },
            { method: "tan", label: "tan" },
            { method: "cot", label: "cot" },
        ];

        const INV_METHODS: Array<{ method: "asin" | "acos"; label: string }> = [
            { method: "asin", label: "asin" },
            { method: "acos", label: "acos" },
        ];

        for (const m of DIRECT_METHODS) {
            for (const x of DIRECT_CASES) {
                it(`Test ${++t}: ${m.label} gas sensitivity for x=${x}`, async function () {
                    const qx = await toQuad(harness, x);

                    await touchGas(harness, m.method, [qx]);
                    const gas = await estimateGas(harness, m.method, [qx]);

                    const out =
                        m.method === "sin" ? await harness.sin(qx) :
                        m.method === "cos" ? await harness.cos(qx) :
                        m.method === "tan" ? await harness.tan(qx) :
                        await harness.cot(qx);

                    const isNan = await harness.isNaN(out);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to input magnitude using x=${x}.`,
                        gas,
                        inHex: `x=${x}`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                    });
                });
            }
        }

        for (const m of INV_METHODS) {
            for (const x of INV_CASES) {
                it(`Test ${++t}: ${m.label} gas sensitivity for x=${x}`, async function () {
                    const qx = await toQuad(harness, x);

                    await touchGas(harness, m.method, [qx]);
                    const gas = await estimateGas(harness, m.method, [qx]);

                    const out =
                        m.method === "asin"
                            ? await harness.asin(qx)
                            : await harness.acos(qx);

                    const isNan = await harness.isNaN(out);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to input magnitude using x=${x}.`,
                        gas,
                        inHex: `x=${x}`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                    });
                });
            }
        }

        for (const x of ATAN_CASES) {
            it(`Test ${++t}: atan gas sensitivity for x=${x}`, async function () {
                const qx = await toQuad(harness, x);

                await touchGas(harness, "atan", [qx]);
                const gas = await estimateGas(harness, "atan", [qx]);

                const out = await harness.atan(qx);
                const isNan = await harness.isNaN(out);

                printBlockRegular({
                    t,
                    method: "atan",
                    explanation: `Gas sensitivity to input magnitude using x=${x}.`,
                    gas,
                    inHex: `x=${x}`,
                    expectedHex: "N/A",
                    outHex: out,
                    expectedDec: "N/A",
                    outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                });
            });
        }
    });

    // ------------------------------------------------------------
    //  Section 2: Gas Sensitivity to Branch / Critical Region
    // ------------------------------------------------------------

    describe("Section 2: Gas Sensitivity to Branch / Critical Region", function () {
        const EPS = 1e-6;

        const SIN_COS_CASES: Array<{ label: string; value: number | "PI_OVER_4" | "HALF_PI" | "PI" | "THREE_HALF_PI" | "TWO_PI" }> = [
            { label: "0", value: 0 },
            { label: "π/4", value: "PI_OVER_4" },
            { label: "π/2", value: "HALF_PI" },
            { label: "π", value: "PI" },
            { label: "3π/2", value: "THREE_HALF_PI" },
            { label: "2π", value: "TWO_PI" },
        ];

        const TAN_CASES: Array<{ label: string; value: number | "PI_OVER_4" | "PI_OVER_3" | "HALF_PI_MINUS_EPS" }> = [
            { label: "0", value: 0 },
            { label: "π/4", value: "PI_OVER_4" },
            { label: "π/3", value: "PI_OVER_3" },
            { label: "π/2-ε", value: "HALF_PI_MINUS_EPS" },
        ];

        const COT_CASES: Array<{ label: string; value: number | "EPS" | "PI_OVER_4" | "PI_OVER_3" | "PI_MINUS_EPS" }> = [
            { label: "ε", value: "EPS" },
            { label: "π/4", value: "PI_OVER_4" },
            { label: "π/3", value: "PI_OVER_3" },
            { label: "π-ε", value: "PI_MINUS_EPS" },
        ];

        const ASIN_ACOS_CASES = [-1, -0.999, 0, 0.999, 1];
        const ATAN_CASES = [-1000, -1, 0, 1, 1000];

        async function resolveSpecial(value: number | string): Promise<string> {
            if (typeof value === "number") return toQuad(harness, value);

            const pi = await fromQuad(harness, QPI);
            const halfPi = await fromQuad(harness, QHALF_PI);

            switch (value) {
                case "PI_OVER_4":
                    return toQuad(harness, Math.PI / 4);
                case "HALF_PI":
                    return QHALF_PI;
                case "PI":
                    return QPI;
                case "THREE_HALF_PI":
                    return toQuad(harness, 3 * Math.PI / 2);
                case "TWO_PI":
                    return toQuad(harness, 2 * Math.PI);
                case "PI_OVER_3":
                    return toQuad(harness, Math.PI / 3);
                case "HALF_PI_MINUS_EPS":
                    return toQuad(harness, halfPi - EPS);
                case "EPS":
                    return toQuad(harness, EPS);
                case "PI_MINUS_EPS":
                    return toQuad(harness, pi - EPS);
                default:
                    return toQuad(harness, 0);
            }
        }

        for (const method of ["sin", "cos"] as const) {
            for (const c of SIN_COS_CASES) {
                it(`Test ${++t}: ${method} gas sensitivity for critical point ${c.label}`, async function () {
                    const qx = await resolveSpecial(c.value);

                    await touchGas(harness, method, [qx]);
                    const gas = await estimateGas(harness, method, [qx]);

                    const out = method === "sin" ? await harness.sin(qx) : await harness.cos(qx);
                    const isNan = await harness.isNaN(out);

                    printBlockRegular({
                        t,
                        method,
                        explanation: `Gas sensitivity to branch / critical region using x=${c.label}.`,
                        gas,
                        inHex: `x=${c.label}`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                    });
                });
            }
        }

        for (const c of TAN_CASES) {
            it(`Test ${++t}: tan gas sensitivity for critical point ${c.label}`, async function () {
                const qx = await resolveSpecial(c.value);

                await touchGas(harness, "tan", [qx]);
                const gas = await estimateGas(harness, "tan", [qx]);

                const out = await harness.tan(qx);
                const isNan = await harness.isNaN(out);

                printBlockRegular({
                    t,
                    method: "tan",
                    explanation: `Gas sensitivity to branch / critical region using x=${c.label}.`,
                    gas,
                    inHex: `x=${c.label}`,
                    expectedHex: "N/A",
                    outHex: out,
                    expectedDec: "N/A",
                    outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                });
            });
        }

        for (const c of COT_CASES) {
            it(`Test ${++t}: cot gas sensitivity for critical point ${c.label}`, async function () {
                const qx = await resolveSpecial(c.value);

                await touchGas(harness, "cot", [qx]);
                const gas = await estimateGas(harness, "cot", [qx]);

                const out = await harness.cot(qx);
                const isNan = await harness.isNaN(out);

                printBlockRegular({
                    t,
                    method: "cot",
                    explanation: `Gas sensitivity to branch / critical region using x=${c.label}.`,
                    gas,
                    inHex: `x=${c.label}`,
                    expectedHex: "N/A",
                    outHex: out,
                    expectedDec: "N/A",
                    outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                });
            });
        }

        for (const method of ["asin", "acos"] as const) {
            for (const x of ASIN_ACOS_CASES) {
                it(`Test ${++t}: ${method} gas sensitivity for critical point ${x}`, async function () {
                    const qx = await toQuad(harness, x);

                    await touchGas(harness, method, [qx]);
                    const gas = await estimateGas(harness, method, [qx]);

                    const out = method === "asin" ? await harness.asin(qx) : await harness.acos(qx);
                    const isNan = await harness.isNaN(out);

                    printBlockRegular({
                        t,
                        method,
                        explanation: `Gas sensitivity to branch / critical region using x=${x}.`,
                        gas,
                        inHex: `x=${x}`,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                    });
                });
            }
        }

        for (const x of ATAN_CASES) {
            it(`Test ${++t}: atan gas sensitivity for critical point ${x}`, async function () {
                const qx = await toQuad(harness, x);

                await touchGas(harness, "atan", [qx]);
                const gas = await estimateGas(harness, "atan", [qx]);

                const out = await harness.atan(qx);
                const isNan = await harness.isNaN(out);

                printBlockRegular({
                    t,
                    method: "atan",
                    explanation: `Gas sensitivity to branch / critical region using x=${x}.`,
                    gas,
                    inHex: `x=${x}`,
                    expectedHex: "N/A",
                    outHex: out,
                    expectedDec: "N/A",
                    outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                });
            });
        }
    });
});