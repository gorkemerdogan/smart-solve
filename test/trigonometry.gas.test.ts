// SPDX-License-Identifier: MIT
import {ethers} from "hardhat";
import type {Contract} from "ethers";
import {touchGas, estimateGas, printBlockRegular} from "./test-utils";

// ------------------------------------------------------------
// Types
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
// Constants & Helpers
// ------------------------------------------------------------

const SCALE = 1e12;
const REPEAT_COUNT = 10;

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

function avgBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => a + b, 0n) / BigInt(values.length);
}

function minBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => (a < b ? a : b));
}

function maxBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => (a > b ? a : b));
}

async function qHalfPiMinusEpsilon(
    h: TrigHarness,
    qhalfPi: string,
    epsilonRad: number
): Promise<string> {
    const qNegEps = await h.fromFloat(BigInt(Math.round(-epsilonRad * SCALE)));
    return await h.add(qhalfPi, qNegEps);
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

    const rad = (deg * Math.PI) / 180;
    return await toQuad(h, rad);
}

// ------------------------------------------------------------
// Test Suite
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
    // Section 1: Gas Sensitivity to Input Magnitude
    // ------------------------------------------------------------

    describe("Section 1: Gas Sensitivity to Input Magnitude", function () {
        let testNo = 0;

        const INV_CASES: number[] = Array.from({ length: 2001 }, (_, i) =>
            Number((-1 + i * 0.001).toFixed(3))
        );

        function buildPiecewiseRange(
            start: number,
            end: number,
            step: number
        ): number[] {
            const values: number[] = [];
            for (let x = start; x <= end + 1e-12; x += step) {
                values.push(Number(x.toFixed(6)));
            }
            return values;
        }

        function uniqueSorted(values: number[]): number[] {
            return Array.from(new Set(values.map(v => Number(v.toFixed(6))))).sort((a, b) => a - b);
        }

        const DIRECT_CASES: number[] = uniqueSorted([
            ...buildPiecewiseRange(-1000, -100, 10),
            ...buildPiecewiseRange(-99, -10, 1),
            ...buildPiecewiseRange(-9.9, -1, 0.1),
            ...buildPiecewiseRange(-0.99, 1, 0.01),
            ...buildPiecewiseRange(1.01, 10, 0.1),
            ...buildPiecewiseRange(11, 100, 1),
            ...buildPiecewiseRange(110, 1000, 10),
            0
        ]);

        const ATAN_CASES: number[] = uniqueSorted([
            ...buildPiecewiseRange(-1000, -100, 10),
            ...buildPiecewiseRange(-99, -10, 1),
            ...buildPiecewiseRange(-9.9, -1, 0.1),
            ...buildPiecewiseRange(-0.99, 1, 0.01),
            ...buildPiecewiseRange(1.01, 10, 0.1),
            ...buildPiecewiseRange(11, 100, 1),
            ...buildPiecewiseRange(110, 1000, 10),
            0
        ]);

        const DIRECT_METHODS: Array<{ method: "sin" | "cos" | "tan" | "cot"; label: string }> = [
            {method: "sin", label: "sin"},
            {method: "cos", label: "cos"},
            {method: "tan", label: "tan"},
            {method: "cot", label: "cot"},
        ];

        const INV_METHODS: Array<{ method: "asin" | "acos"; label: string }> = [
            {method: "asin", label: "asin"},
            {method: "acos", label: "acos"},
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
    // Section 2: Gas Sensitivity to Critical Region for Tangent
    // ------------------------------------------------------------

    describe("Section 2: Gas Sensitivity to Critical Region for Tangent", function () {
        let testNo = 0;

        const EPSILON = 1e-6;

        const TAN_CRITICAL_CASES: Array<{
            label: string;
            buildInput: () => Promise<string>;
        }> = [
            {
                label: "0",
                buildInput: async () => await toQuad(harness, 0),
            },
            {
                label: "π/4",
                buildInput: async () => await toQuad(harness, Math.PI / 4),
            },
            {
                label: "π/3",
                buildInput: async () => await toQuad(harness, Math.PI / 3),
            },
            {
                label: "π/2-ε",
                buildInput: async () => await qHalfPiMinusEpsilon(harness, QHALF_PI, EPSILON),
            },
        ];

        for (const c of TAN_CRITICAL_CASES) {
            const t = `2.${++testNo}`;

            it(`Test ${t}: tan gas sensitivity at critical region ${c.label}`, async function () {
                const qx = await c.buildInput();

                await touchGas(harness, "tan", [qx]);
                const gas = await estimateGas(harness, "tan", [qx]);

                const out = await harness.tan(qx);
                const isNan = await harness.isNaN(out);

                printBlockRegular({
                    t,
                    method: "tan",
                    explanation: `Gas sensitivity to critical region using x=${c.label}.`,
                    gas,
                    inHex: `x=${c.label}`,
                    expectedHex: "N/A",
                    outHex: out,
                    expectedDec: "N/A",
                    outDec: isNan ? "NaN" : fmt(await fromQuad(harness, out)),
                });
            });
        }
    });

    // ------------------------------------------------------------
    // Section 3: Gas Consumption over Full Domain (1 degree resolution, repeated)
    // ------------------------------------------------------------

    describe("Section 3: Gas Consumption over Full Domain", function () {
        let testNo = 0;

        const FULL_DOMAIN_REPEAT_COUNT = 5;

        it("Test 3.1: sin & cos gas over [0°, 360°] with 1° resolution and repeated measurements", async function () {
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

                const sinGasRuns: bigint[] = [];
                const cosGasRuns: bigint[] = [];

                let lastSinOut = "";
                let lastCosOut = "";
                let lastSinVal = 0;
                let lastCosVal = 0;

                for (let run = 1; run <= FULL_DOMAIN_REPEAT_COUNT; run++) {
                    // ---- sin ----
                    await touchGas(harness, "sin", [qx]);
                    const sinGas = await estimateGas(harness, "sin", [qx]);
                    const sinOut = await harness.sin(qx);
                    const sinVal = await fromQuad(harness, sinOut);

                    sinGasRuns.push(BigInt(sinGas.toString()));
                    lastSinOut = sinOut;
                    lastSinVal = sinVal;

                    // ---- cos ----
                    await touchGas(harness, "cos", [qx]);
                    const cosGas = await estimateGas(harness, "cos", [qx]);
                    const cosOut = await harness.cos(qx);
                    const cosVal = await fromQuad(harness, cosOut);

                    cosGasRuns.push(BigInt(cosGas.toString()));
                    lastCosOut = cosOut;
                    lastCosVal = cosVal;
                }

                const sinAvgGas = avgBigInt(sinGasRuns);
                const cosAvgGas = avgBigInt(cosGasRuns);

                totalSin += sinAvgGas;
                totalCos += cosAvgGas;

                if (sinAvgGas < minSin) minSin = sinAvgGas;
                if (sinAvgGas > maxSin) maxSin = sinAvgGas;

                if (cosAvgGas < minCos) minCos = cosAvgGas;
                if (cosAvgGas > maxCos) maxCos = cosAvgGas;

                const t = `3.${++testNo}`;

                printBlockRegular({
                    t,
                    method: "sin",
                    explanation: `Average gas measurement at ${deg}° over ${FULL_DOMAIN_REPEAT_COUNT} repeated runs (full-domain sweep; exact critical angles injected).`,
                    gas: `${sinAvgGas}`,
                    inHex: `deg=${deg}`,
                    expectedHex: "N/A",
                    outHex: lastSinOut,
                    expectedDec: "N/A",
                    outDec: fmt(lastSinVal),
                });

                printBlockRegular({
                    t: `${t}-cos`,
                    method: "cos",
                    explanation: `Average gas measurement at ${deg}° over ${FULL_DOMAIN_REPEAT_COUNT} repeated runs (full-domain sweep; exact critical angles injected).`,
                    gas: `${cosAvgGas}`,
                    inHex: `deg=${deg}`,
                    expectedHex: "N/A",
                    outHex: lastCosOut,
                    expectedDec: "N/A",
                    outDec: fmt(lastCosVal),
                });
            }

            const sampleCount = BigInt(endDeg - startDeg + 1);
            const avgSin = totalSin / sampleCount;
            const avgCos = totalCos / sampleCount;

            console.log("------------------------------------------------------------");
            console.log("FULL DOMAIN SUMMARY");
            console.log("------------------------------------------------------------");
            console.log(
                `Input generation: Math.PI for general angles, exact quad constants for 0°, 90°, 180°, 270°, 360°. Each angle was evaluated ${FULL_DOMAIN_REPEAT_COUNT} times and average gas is reported.`
            );
            console.log(`sin -> avg: ${avgSin} | min: ${minSin} | max: ${maxSin}`);
            console.log(`cos -> avg: ${avgCos} | min: ${minCos} | max: ${maxCos}`);
        });
    });
});