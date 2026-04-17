// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type TrigonometryHarness = Contract & {
    fromDouble(x: number | bigint): Promise<string>;
    fromUInt(x: number | bigint): Promise<string>;
    toDouble(x: string): Promise<unknown>;

    fromFloat(x: bigint): Promise<string>;
    toFloat(x: string): Promise<unknown>;

    abs(x: string): Promise<string>;
    neg(x: string): Promise<string>;
    cmp(a: string, b: string): Promise<bigint>;
    isNaN(x: string): Promise<boolean>;
    add(a: string, b: string): Promise<string>;
    mul(a: string, b: string): Promise<string>;

    sin(x: string): Promise<string>;
    cos(x: string): Promise<string>;
    tan(x: string): Promise<string>;
    cot(x: string): Promise<string>;
    asin(x: string): Promise<string>;
    acos(x: string): Promise<string>;
    atan(x: string): Promise<string>;

    QPI(): Promise<string>;
    QHALF_PI(): Promise<string>;
    QQUARTER_PI(): Promise<string>;
    QTWO_PI(): Promise<string>;
    QNAN(): Promise<string>;
};

// ------------------------------------------------------------
// Global Constants
// ------------------------------------------------------------

const NUM_CASES = 30;
const FIXED_SEED = "TRIG_MULTI_CASE_SEED_V1";

// Contract-side scaling.
// The harness uses SCALE = 1e18 for fromFloat / toFloat.
const SCALE = 10n ** 12n;

// Useful numeric constants in the same scaled representation.
const PI_SCALED = BigInt(Math.round(Math.PI * 1e18));
const HALF_PI_SCALED = PI_SCALED / 2n;
const QUARTER_PI_SCALED = PI_SCALED / 4n;
const TWO_PI_SCALED = PI_SCALED * 2n;

// ------------------------------------------------------------
// Helper Types
// ------------------------------------------------------------

type NumericRecord = {
    absError: bigint;
    relErrorScaled?: bigint;
    gas: bigint;
};

type BoolRecord = {
    ok: boolean;
    gas: bigint;
};

// ------------------------------------------------------------
// Generic Helpers
// ------------------------------------------------------------

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

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

function avgBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => a + b, 0n) / BigInt(values.length);
}

function minBigInt(values: bigint[]): bigint {
    return values.reduce((a, b) => (a < b ? a : b));
}

function maxBigInt(values: bigint[]): bigint {
    return values.reduce((a, b) => (a > b ? a : b));
}

function formatScaledInt(v: bigint, decimals = 18): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const s = 10n ** BigInt(decimals);

    const intPart = abs / s;
    const fracPart = abs % s;
    const fracStr = fracPart.toString().padStart(decimals, "0");

    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function toScaledFromNumber(x: number): bigint {
    return BigInt(Math.round(x * 1e18));
}

function scaledAbsError(actual: bigint, expected: bigint): bigint {
    return absBigInt(actual - expected);
}

/**
 * Relative error in SCALE units:
 *   rel = |actual - expected| / |expected|
 * If expected == 0, returns undefined.
 */
function scaledRelError(actual: bigint, expected: bigint): bigint | undefined {
    const den = absBigInt(expected);
    if (den === 0n) return undefined;

    const num = absBigInt(actual - expected);
    return (num * SCALE) / den;
}

// ------------------------------------------------------------
// Deterministic Pseudo-Random Helpers
// ------------------------------------------------------------

/**
 * Returns a deterministic pseudo-random bigint in [0, 2^256-1].
 */
function pseudoRandomBigInt(label: string, index: number): bigint {
    const digest = ethers.keccak256(
        ethers.solidityPacked(
            ["string", "string", "uint256"],
            [FIXED_SEED, label, BigInt(index)]
        )
    );
    return BigInt(digest);
}

/**
 * Maps a pseudo-random integer to the closed interval [min, max].
 */
function pseudoRandomScaledInRange(label: string, index: number, min: bigint, max: bigint): bigint {
    const lo = min < max ? min : max;
    const hi = min < max ? max : min;

    const span = hi - lo;
    if (span === 0n) return lo;

    const r = pseudoRandomBigInt(label, index) % (span + 1n);
    return lo + r;
}

/**
 * Generates a deterministic pseudo-random test value in [-10pi, 10pi].
 */
function randomAngleWideScaled(label: string, index: number): bigint {
    return pseudoRandomScaledInRange(label, index, -10n * PI_SCALED, 10n * PI_SCALED);
}

/**
 * Generates a deterministic pseudo-random x in [-1, 1].
 */
function randomUnitIntervalScaled(label: string, index: number): bigint {
    return pseudoRandomScaledInRange(label, index, -1n * SCALE, 1n * SCALE);
}

/**
 * Generates a deterministic pseudo-random x in [-100, 100].
 */
function randomAtanInputScaled(label: string, index: number): bigint {
    return pseudoRandomScaledInRange(label, index, -100n * SCALE, 100n * SCALE);
}

/**
 * Generates deterministic valid inputs for tan, staying away from singularities.
 * Chosen domain: [-3pi/2 + margin, 3pi/2 - margin], excluding points near pi/2 + k*pi.
 */
function randomValidTanInputScaled(index: number): bigint {
    const margin = 50_000_000n;
    while (true) {
        const x = pseudoRandomScaledInRange(
            "tan-valid",
            index,
            -3n * HALF_PI_SCALED + margin,
            3n * HALF_PI_SCALED - margin
        );

        let nearSingularity = false;
        for (let k = -3; k <= 3; k++) {
            const singular = HALF_PI_SCALED + BigInt(k) * PI_SCALED;
            if (absBigInt(x - singular) < 1_000_000_000n) {
                nearSingularity = true;
                break;
            }
        }

        if (!nearSingularity) return x;
        index += 1000;
    }
}

/**
 * Generates deterministic valid inputs for cot, staying away from k*pi.
 */
function randomValidCotInputScaled(index: number): bigint {
    const margin = 50_000_000n;
    while (true) {
        const x = pseudoRandomScaledInRange(
            "cot-valid",
            index,
            -3n * PI_SCALED + margin,
            3n * PI_SCALED - margin
        );

        let nearSingularity = false;
        for (let k = -3; k <= 3; k++) {
            const singular = BigInt(k) * PI_SCALED;
            if (absBigInt(x - singular) < 1_000_000_000n) {
                nearSingularity = true;
                break;
            }
        }

        if (!nearSingularity) return x;
        index += 1000;
    }
}

// ------------------------------------------------------------
// Contract Interaction Helpers
// ------------------------------------------------------------

async function qScaled(harness: TrigonometryHarness, scaledValue: bigint): Promise<string> {
    return await harness.fromFloat(scaledValue);
}

async function outScaled(harness: TrigonometryHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

async function estimateGasFor(
    harness: TrigonometryHarness,
    method: string,
    args: unknown[]
): Promise<bigint> {
    const fn = harness.getFunction(method) as unknown as {
        estimateGas: (...innerArgs: unknown[]) => Promise<bigint>;
    };
    return await fn.estimateGas(...args);
}

async function buildKPi(
    harness: TrigonometryHarness,
    k: bigint,
    QPI: string
): Promise<string> {
    const QZERO = await harness.fromDouble(0n);
    if (k === 0n) return QZERO;

    let acc = QZERO;

    if (k > 0n) {
        for (let i = 0n; i < k; i++) {
            acc = await harness.add(acc, QPI);
        }
        return acc;
    }

    const negQPI = await harness.neg(QPI);
    for (let i = 0n; i < -k; i++) {
        acc = await harness.add(acc, negQPI);
    }
    return acc;
}

// ------------------------------------------------------------
// Reporting Helpers
// ------------------------------------------------------------

function printNumericSummary(title: string, method: string, records: NumericRecord[]) {
    const absErrors = records.map((r) => r.absError);
    const relErrors = records
        .map((r) => r.relErrorScaled)
        .filter((v): v is bigint => v !== undefined);
    const gasValues = records.map((r) => r.gas);

    console.log("============================================================");
    console.log(`${title}`);
    console.log("============================================================");
    console.log(`Method             : ${method}`);
    console.log(`Number of Tests    : ${records.length}`);
    console.log(`Average Abs. Error : ${formatScaledInt(avgBigInt(absErrors))}`);
    console.log(`Max Abs. Error     : ${formatScaledInt(maxBigInt(absErrors))}`);
    console.log(
        `Average Rel. Error : ${relErrors.length > 0 ? formatScaledInt(avgBigInt(relErrors)) : "N/A"}`
    );
    console.log(`Min Gas            : ${minBigInt(gasValues).toString()}`);
    console.log(`Average Gas        : ${avgBigInt(gasValues).toString()}`);
    console.log(`Max Gas            : ${maxBigInt(gasValues).toString()}`);
    console.log("============================================================");
}

function printBooleanSummary(title: string, method: string, records: BoolRecord[]) {
    const gasValues = records.map((r) => r.gas);
    const okCount = records.filter((r) => r.ok).length;

    console.log("============================================================");
    console.log(`${title}`);
    console.log("============================================================");
    console.log(`Method             : ${method}`);
    console.log(`Number of Tests    : ${records.length}`);
    console.log(`Correct Cases      : ${okCount}`);
    console.log(`Accuracy           : ${okCount}/${records.length}`);
    console.log(`Min Gas            : ${minBigInt(gasValues).toString()}`);
    console.log(`Average Gas        : ${avgBigInt(gasValues).toString()}`);
    console.log(`Max Gas            : ${maxBigInt(gasValues).toString()}`);
    console.log("============================================================");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("Trigonometry Library - Multi-Case Accuracy Benchmarks", function () {
    let harness: TrigonometryHarness;

    let QPI: string;
    let QHALF_PI: string;
    let QQUARTER_PI: string;
    let QTWO_PI: string;

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("TrigonometryHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as TrigonometryHarness;

        QPI = await harness.QPI();
        QHALF_PI = await harness.QHALF_PI();
        QQUARTER_PI = await harness.QQUARTER_PI();
        QTWO_PI = await harness.QTWO_PI();
    });

    // ------------------------------------------------------------
    // 1) Symmetry Tests (30 each)
    // ------------------------------------------------------------

    it("should evaluate symmetry properties over 30 deterministic pseudo-random cases each", async function () {
        const sinRecords: NumericRecord[] = [];
        const cosRecords: NumericRecord[] = [];
        const atanRecords: NumericRecord[] = [];

        for (let i = 0; i < NUM_CASES; i++) {
            const xScaled = pseudoRandomScaledInRange("symmetry-x", i, 1n, 10n * PI_SCALED);
            const qx = await qScaled(harness, xScaled);
            const qNegX = await qScaled(harness, -xScaled);

            {
                const gas1 = await estimateGasFor(harness, "sin", [qx]);
                const gas2 = await estimateGasFor(harness, "sin", [qNegX]);

                const sPos = await harness.sin(qx);
                const sNeg = await harness.sin(qNegX);

                const sPosScaled = await outScaled(harness, sPos);
                const sNegScaled = await outScaled(harness, sNeg);

                const propertyActual = sNegScaled + sPosScaled;
                const absErr = absBigInt(propertyActual);
                const relErr = scaledRelError(propertyActual, 0n);

                sinRecords.push({
                    absError: absErr,
                    relErrorScaled: relErr,
                    gas: gas1 + gas2,
                });
            }

            {
                const gas1 = await estimateGasFor(harness, "cos", [qx]);
                const gas2 = await estimateGasFor(harness, "cos", [qNegX]);

                const cPos = await harness.cos(qx);
                const cNeg = await harness.cos(qNegX);

                const cPosScaled = await outScaled(harness, cPos);
                const cNegScaled = await outScaled(harness, cNeg);

                const propertyActual = cNegScaled - cPosScaled;
                const absErr = absBigInt(propertyActual);
                const relErr = scaledRelError(propertyActual, 0n);

                cosRecords.push({
                    absError: absErr,
                    relErrorScaled: relErr,
                    gas: gas1 + gas2,
                });
            }

            {
                const xAtanScaled = pseudoRandomScaledInRange("atan-symmetry-x", i, 1n, 100n * SCALE);
                const qxa = await qScaled(harness, xAtanScaled);
                const qNegXa = await qScaled(harness, -xAtanScaled);

                const gas1 = await estimateGasFor(harness, "atan", [qxa]);
                const gas2 = await estimateGasFor(harness, "atan", [qNegXa]);

                const aPos = await harness.atan(qxa);
                const aNeg = await harness.atan(qNegXa);

                const aPosScaled = await outScaled(harness, aPos);
                const aNegScaled = await outScaled(harness, aNeg);

                const propertyActual = aNegScaled + aPosScaled;
                const absErr = absBigInt(propertyActual);
                const relErr = scaledRelError(propertyActual, 0n);

                atanRecords.push({
                    absError: absErr,
                    relErrorScaled: relErr,
                    gas: gas1 + gas2,
                });
            }
        }

        printNumericSummary("Symmetry Accuracy Results", "sin(-x) = -sin(x)", sinRecords);
        printNumericSummary("Symmetry Accuracy Results", "cos(-x) = cos(x)", cosRecords);
        printNumericSummary("Symmetry Accuracy Results", "atan(-x) = -atan(x)", atanRecords);

        expect(sinRecords.length).to.equal(NUM_CASES);
        expect(cosRecords.length).to.equal(NUM_CASES);
        expect(atanRecords.length).to.equal(NUM_CASES);
    });

    // ------------------------------------------------------------
    // 2) Periodicity Tests (30 each)
    // ------------------------------------------------------------

    it("should evaluate periodicity properties over 30 deterministic pseudo-random cases each", async function () {
        const sinRecords: NumericRecord[] = [];
        const cosRecords: NumericRecord[] = [];

        for (let i = 0; i < NUM_CASES; i++) {
            const xScaled = randomAngleWideScaled("periodicity-x", i);
            const xPlusTwoPiScaled = xScaled + TWO_PI_SCALED;

            const qx = await qScaled(harness, xScaled);
            const qxPlusTwoPi = await qScaled(harness, xPlusTwoPiScaled);

            {
                const gas1 = await estimateGasFor(harness, "sin", [qx]);
                const gas2 = await estimateGasFor(harness, "sin", [qxPlusTwoPi]);

                const s1 = await harness.sin(qx);
                const s2 = await harness.sin(qxPlusTwoPi);

                const s1Scaled = await outScaled(harness, s1);
                const s2Scaled = await outScaled(harness, s2);

                const absErr = absBigInt(s2Scaled - s1Scaled);
                const relErr = scaledRelError(s2Scaled, s1Scaled);

                sinRecords.push({
                    absError: absErr,
                    relErrorScaled: relErr,
                    gas: gas1 + gas2,
                });
            }

            {
                const gas1 = await estimateGasFor(harness, "cos", [qx]);
                const gas2 = await estimateGasFor(harness, "cos", [qxPlusTwoPi]);

                const c1 = await harness.cos(qx);
                const c2 = await harness.cos(qxPlusTwoPi);

                const c1Scaled = await outScaled(harness, c1);
                const c2Scaled = await outScaled(harness, c2);

                const absErr = absBigInt(c2Scaled - c1Scaled);
                const relErr = scaledRelError(c2Scaled, c1Scaled);

                cosRecords.push({
                    absError: absErr,
                    relErrorScaled: relErr,
                    gas: gas1 + gas2,
                });
            }
        }

        printNumericSummary("Periodicity Accuracy Results", "sin(x + 2pi) = sin(x)", sinRecords);
        printNumericSummary("Periodicity Accuracy Results", "cos(x + 2pi) = cos(x)", cosRecords);

        expect(sinRecords.length).to.equal(NUM_CASES);
        expect(cosRecords.length).to.equal(NUM_CASES);
    });

    // ------------------------------------------------------------
    // 3) Difference Between sin(pi/4) and cos(pi/4) (30 repetitions)
    // ------------------------------------------------------------

    it("should evaluate the difference between sin(pi/4) and cos(pi/4) over 30 repeated runs", async function () {
        const records: NumericRecord[] = [];

        for (let i = 0; i < NUM_CASES; i++) {
            const gasSin = await estimateGasFor(harness, "sin", [QQUARTER_PI]);
            const gasCos = await estimateGasFor(harness, "cos", [QQUARTER_PI]);

            const s = await harness.sin(QQUARTER_PI);
            const c = await harness.cos(QQUARTER_PI);

            const sScaled = await outScaled(harness, s);
            const cScaled = await outScaled(harness, c);

            const propertyActual = sScaled - cScaled;
            const absErr = absBigInt(propertyActual);
            const relErr = scaledRelError(propertyActual, 0n);

            records.push({
                absError: absErr,
                relErrorScaled: relErr,
                gas: gasSin + gasCos,
            });
        }

        printNumericSummary(
            "Difference Accuracy Results at pi/4",
            "|sin(pi/4) - cos(pi/4)|",
            records
        );

        expect(records.length).to.equal(NUM_CASES);
    });

    // ------------------------------------------------------------
    // 4) Random Direct / Inverse Function Accuracy Tests (30 each)
    // ------------------------------------------------------------

    it("should evaluate direct and inverse trigonometric functions on 30 deterministic pseudo-random inputs each", async function () {
        const sinRecords: NumericRecord[] = [];
        const cosRecords: NumericRecord[] = [];
        const tanRecords: NumericRecord[] = [];
        const cotRecords: NumericRecord[] = [];
        const asinRecords: NumericRecord[] = [];
        const acosRecords: NumericRecord[] = [];
        const atanRecords: NumericRecord[] = [];

        for (let i = 0; i < NUM_CASES; i++) {
            {
                const xScaled = randomAngleWideScaled("sin-random", i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "sin", [qx]);
                const out = await harness.sin(qx);

                const actualScaled = await outScaled(harness, out);
                const expectedScaled = toScaledFromNumber(Math.sin(Number(xScaled) / 1e18));

                sinRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }

            {
                const xScaled = randomAngleWideScaled("cos-random", i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "cos", [qx]);
                const out = await harness.cos(qx);

                const actualScaled = await outScaled(harness, out);
                const expectedScaled = toScaledFromNumber(Math.cos(Number(xScaled) / 1e18));

                cosRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }

            {
                const xScaled = randomValidTanInputScaled(i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "tan", [qx]);
                const out = await harness.tan(qx);
                const isNaN = await harness.isNaN(out);

                expect(isNaN).to.equal(false);

                const actualScaled = await outScaled(harness, out);
                const expectedScaled = toScaledFromNumber(Math.tan(Number(xScaled) / 1e18));

                tanRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }

            {
                const xScaled = randomValidCotInputScaled(i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "cot", [qx]);
                const out = await harness.cot(qx);
                const isNaN = await harness.isNaN(out);

                expect(isNaN).to.equal(false);

                const actualScaled = await outScaled(harness, out);
                const expected = 1 / Math.tan(Number(xScaled) / 1e18);
                const expectedScaled = toScaledFromNumber(expected);

                cotRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }

            {
                const xScaled = randomUnitIntervalScaled("asin-random", i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "asin", [qx]);
                const out = await harness.asin(qx);
                const isNaN = await harness.isNaN(out);

                expect(isNaN).to.equal(false);

                const actualScaled = await outScaled(harness, out);
                const expectedScaled = toScaledFromNumber(Math.asin(Number(xScaled) / 1e18));

                asinRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }

            {
                const xScaled = randomUnitIntervalScaled("acos-random", i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "acos", [qx]);
                const out = await harness.acos(qx);
                const isNaN = await harness.isNaN(out);

                expect(isNaN).to.equal(false);

                const actualScaled = await outScaled(harness, out);
                const expectedScaled = toScaledFromNumber(Math.acos(Number(xScaled) / 1e18));

                acosRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }

            {
                const xScaled = randomAtanInputScaled("atan-random", i);
                const qx = await qScaled(harness, xScaled);

                const gas = await estimateGasFor(harness, "atan", [qx]);
                const out = await harness.atan(qx);

                const actualScaled = await outScaled(harness, out);
                const expectedScaled = toScaledFromNumber(Math.atan(Number(xScaled) / 1e18));

                atanRecords.push({
                    absError: scaledAbsError(actualScaled, expectedScaled),
                    relErrorScaled: scaledRelError(actualScaled, expectedScaled),
                    gas,
                });
            }
        }

        printNumericSummary("Random Accuracy Results", "sin(x)", sinRecords);
        printNumericSummary("Random Accuracy Results", "cos(x)", cosRecords);
        printNumericSummary("Random Accuracy Results", "tan(x)", tanRecords);
        printNumericSummary("Random Accuracy Results", "cot(x)", cotRecords);
        printNumericSummary("Random Accuracy Results", "asin(x)", asinRecords);
        printNumericSummary("Random Accuracy Results", "acos(x)", acosRecords);
        printNumericSummary("Random Accuracy Results", "atan(x)", atanRecords);

        expect(sinRecords.length).to.equal(NUM_CASES);
        expect(cosRecords.length).to.equal(NUM_CASES);
        expect(tanRecords.length).to.equal(NUM_CASES);
        expect(cotRecords.length).to.equal(NUM_CASES);
        expect(asinRecords.length).to.equal(NUM_CASES);
        expect(acosRecords.length).to.equal(NUM_CASES);
        expect(atanRecords.length).to.equal(NUM_CASES);
    });

    // ------------------------------------------------------------
    // 5) Direct / Inverse Consistency Tests (30 each)
    // ------------------------------------------------------------

    it("should evaluate direct and inverse consistency relations over 30 deterministic pseudo-random cases each", async function () {
        const asinSinRecords: NumericRecord[] = [];
        const acosCosRecords: NumericRecord[] = [];
        const tanAtanRecords: NumericRecord[] = [];

        for (let i = 0; i < NUM_CASES; i++) {
            {
                const xScaled = pseudoRandomScaledInRange(
                    "asin-sin-x",
                    i,
                    -HALF_PI_SCALED,
                    HALF_PI_SCALED
                );
                const qx = await qScaled(harness, xScaled);

                const gas1 = await estimateGasFor(harness, "sin", [qx]);
                const s = await harness.sin(qx);

                const gas2 = await estimateGasFor(harness, "asin", [s]);
                const recovered = await harness.asin(s);

                const recoveredScaled = await outScaled(harness, recovered);

                asinSinRecords.push({
                    absError: scaledAbsError(recoveredScaled, xScaled),
                    relErrorScaled: scaledRelError(recoveredScaled, xScaled),
                    gas: gas1 + gas2,
                });
            }

            {
                const xScaled = pseudoRandomScaledInRange(
                    "acos-cos-x",
                    i,
                    0n,
                    PI_SCALED
                );
                const qx = await qScaled(harness, xScaled);

                const gas1 = await estimateGasFor(harness, "cos", [qx]);
                const c = await harness.cos(qx);

                const gas2 = await estimateGasFor(harness, "acos", [c]);
                const recovered = await harness.acos(c);

                const recoveredScaled = await outScaled(harness, recovered);

                acosCosRecords.push({
                    absError: scaledAbsError(recoveredScaled, xScaled),
                    relErrorScaled: scaledRelError(recoveredScaled, xScaled),
                    gas: gas1 + gas2,
                });
            }

            {
                const xScaled = randomAtanInputScaled("tan-atan-x", i);
                const qx = await qScaled(harness, xScaled);

                const gas1 = await estimateGasFor(harness, "atan", [qx]);
                const a = await harness.atan(qx);

                const gas2 = await estimateGasFor(harness, "tan", [a]);
                const recovered = await harness.tan(a);

                const recoveredScaled = await outScaled(harness, recovered);

                tanAtanRecords.push({
                    absError: scaledAbsError(recoveredScaled, xScaled),
                    relErrorScaled: scaledRelError(recoveredScaled, xScaled),
                    gas: gas1 + gas2,
                });
            }
        }

        printNumericSummary("Consistency Accuracy Results", "asin(sin(x)) ≈ x", asinSinRecords);
        printNumericSummary("Consistency Accuracy Results", "acos(cos(x)) ≈ x", acosCosRecords);
        printNumericSummary("Consistency Accuracy Results", "tan(atan(x)) ≈ x", tanAtanRecords);

        expect(asinSinRecords.length).to.equal(NUM_CASES);
        expect(acosCosRecords.length).to.equal(NUM_CASES);
        expect(tanAtanRecords.length).to.equal(NUM_CASES);
    });

    // ------------------------------------------------------------
    // 6) Undefined Regions for Direct Functions (30 each)
    // ------------------------------------------------------------

    it("should evaluate undefined-region behavior for tan and cot over 30 cases each", async function () {
        const tanRecords: BoolRecord[] = [];
        const cotRecords: BoolRecord[] = [];

        const QZERO = await harness.fromDouble(0n);

        for (let i = 0; i < NUM_CASES; i++) {
            // tan(pi/2) -> NaN
            {
                const gas = await estimateGasFor(harness, "tan", [QHALF_PI]);
                const out = await harness.tan(QHALF_PI);
                const isNaN = await harness.isNaN(out);

                tanRecords.push({
                    ok: isNaN,
                    gas,
                });
            }

            // cot(0) -> NaN
            {
                const gas = await estimateGasFor(harness, "cot", [QZERO]);
                const out = await harness.cot(QZERO);
                const isNaN = await harness.isNaN(out);

                cotRecords.push({
                    ok: isNaN,
                    gas,
                });
            }
        }

        printBooleanSummary("Undefined-Region Results", "tan(pi/2) -> NaN", tanRecords);
        printBooleanSummary("Undefined-Region Results", "cot(0) -> NaN", cotRecords);

        expect(tanRecords.every((r) => r.ok)).to.equal(true);
        expect(cotRecords.every((r) => r.ok)).to.equal(true);
    });
});