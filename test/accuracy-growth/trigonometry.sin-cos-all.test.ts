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

/**
 * Uses Math.PI for general degree-to-radian conversion, but injects exact
 * quad constants at critical angles so that special-case fast paths can be observed.
 */
async function degreeToQuadWithExactCriticalAngles(
    h: TrigHarness,
    deg: number,
    qpi: string,
    qhalfPi: string
): Promise<string> {
    if (deg === 0) return await toQuad(h, 0);
    if (deg === 90) return qhalfPi;
    if (deg === 180) return qpi;
    if (deg === 270) return await h.add(qpi, qhalfPi);
    if (deg === 360) return await h.add(qpi, qpi);

    const rad = deg * Math.PI / 180;
    return await toQuad(h, rad);
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Trigonometry - Gas Growth Tests", function () {
    let harness: TrigHarness;

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
        let testNo = 0;

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
                const t = `1.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for x=${x}`, async function () {
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
                        t: `${t}`,
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
                const t = `1.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for x=${x}`, async function () {
                    const qx = await toQuad(harness, x);

                    await touchGas(harness, m.method, [qx]);
                    const gas = await estimateGas(harness, m.method, [qx]);

                    const out =
                        m.method === "asin"
                            ? await harness.asin(qx)
                            : await harness.acos(qx);

                    const isNan = await harness.isNaN(out);

                    printBlockRegular({
                        t: `${t}`,
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
            const t = `1.${++testNo}`;

            it(`Test ${t}: atan gas sensitivity for x=${x}`, async function () {
                const qx = await toQuad(harness, x);

                await touchGas(harness, "atan", [qx]);
                const gas = await estimateGas(harness, "atan", [qx]);

                const out = await harness.atan(qx);
                const isNan = await harness.isNaN(out);

                printBlockRegular({
                    t: `${t}`,
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
    //  Section 3: Gas Consumption over Full Domain (1° resolution)
    // ------------------------------------------------------------

    describe("Section 3: Gas Consumption over Full Domain", function () {
        let testNo = 0;

        it("Test 3.1: sin & cos gas over [0°, 360°] with 1° resolution", async function () {
            const startDeg = 0;
            const endDeg = 360;

            let totalSin = 0n;
            let totalCos = 0n;

            let minSin = 10n ** 18n;
            let maxSin = 0n;

            let minCos = 10n ** 18n;
            let maxCos = 0n;

            for (let deg = startDeg; deg <= endDeg; deg++) {
                const qx = await degreeToQuadWithExactCriticalAngles(harness, deg, QPI, QHALF_PI);

                // ---- sin ----
                await touchGas(harness, "sin", [qx]);
                const sinGas = await estimateGas(harness, "sin", [qx]);

                const sinOut = await harness.sin(qx);
                const sinVal = await fromQuad(harness, sinOut);

                // ---- cos ----
                await touchGas(harness, "cos", [qx]);
                const cosGas = await estimateGas(harness, "cos", [qx]);

                const cosOut = await harness.cos(qx);
                const cosVal = await fromQuad(harness, cosOut);

                const sinGasBig = BigInt(sinGas.toString());
                const cosGasBig = BigInt(cosGas.toString());

                totalSin += sinGasBig;
                totalCos += cosGasBig;

                if (sinGasBig < minSin) minSin = sinGasBig;
                if (sinGasBig > maxSin) maxSin = sinGasBig;

                if (cosGasBig < minCos) minCos = cosGasBig;
                if (cosGasBig > maxCos) maxCos = cosGasBig;

                const t = `3.${++testNo}`;

                printBlockRegular({
                    t,
                    method: "sin",
                    explanation: `Gas measurement at ${deg}° (full-domain sweep; exact critical angles injected).`,
                    gas: `${sinGasBig}`,
                    inHex: `deg=${deg}`,
                    expectedHex: "N/A",
                    outHex: sinOut,
                    expectedDec: "N/A",
                    outDec: fmt(sinVal),
                });

                printBlockRegular({
                    t: `${t}-cos`,
                    method: "cos",
                    explanation: `Gas measurement at ${deg}° (full-domain sweep; exact critical angles injected).`,
                    gas: `${cosGasBig}`,
                    inHex: `deg=${deg}`,
                    expectedHex: "N/A",
                    outHex: cosOut,
                    expectedDec: "N/A",
                    outDec: fmt(cosVal),
                });
            }

            const sampleCount = BigInt(endDeg - startDeg + 1);
            const avgSin = totalSin / sampleCount;
            const avgCos = totalCos / sampleCount;

            console.log("------------------------------------------------------------");
            console.log("FULL DOMAIN SUMMARY");
            console.log("------------------------------------------------------------");
            console.log("Input generation: Math.PI for general angles, exact quad constants for 0°, 90°, 180°, 270°, 360°.");
            console.log(`sin -> avg: ${avgSin} | min: ${minSin} | max: ${maxSin}`);
            console.log(`cos -> avg: ${avgCos} | min: ${minCos} | max: ${maxCos}`);
        });
    });
});