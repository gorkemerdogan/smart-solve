// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "./test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type PolynomialHarness = Contract & {
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(num: bigint, den: bigint): Promise<string>;
    toFloat(q: string): Promise<[boolean, bigint]>;
    fromFloat(n: bigint): Promise<string>;

    evaluateHorners(coeffs: string[], x: string): Promise<string>;
    evaluateWithDerivative(coeffs: string[], x: string): Promise<[string, string] & { px: string; dpx: string }>;
    evalHornerMonic(lowerCoeffs: string[], x: string): Promise<string>;
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

type HornerMethod = "evaluateHorners" | "evaluateWithDerivative" | "evalHornerMonic";

function fmtScaled(v: bigint): string {
    const SCALE = 10n ** 12n;
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(12, "0");
    return `${neg ? "-" : ""}${intPart}.${fracStr}`.replace(/\.?0+$/, "");
}

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

function toGasBigInt(gas: unknown): bigint {
    if (typeof gas === "bigint") return gas;
    return BigInt(gas as string);
}

async function safeFromQuad(h: PolynomialHarness, hex: string): Promise<string> {
    const [ok, raw] = await h.toFloat(hex);
    if (!ok) return "overflow during decimal conversion";
    return fmtScaled(asBigInt(raw));
}

async function qScaledArrayFromInts(
    h: PolynomialHarness,
    arr: bigint[],
    factorNum: bigint,
    factorDen: bigint
): Promise<string[]> {
    return Promise.all(
        arr.map(v => {
            if (factorDen === 1n) return h.qFromInt(v * factorNum);
            return h.qFromFrac(v * factorNum, factorDen);
        })
    );
}

function buildIncreasingCoeffs(n: number): bigint[] {
    return Array.from({ length: n }, (_, i) => BigInt(i + 1));
}

function buildConstantCoeffs(n: number, c: bigint = 3n): bigint[] {
    return Array.from({ length: n }, () => c);
}

function buildAlternatingCoeffs(n: number): bigint[] {
    return Array.from({ length: n }, (_, i) => (i % 2 === 0 ? BigInt(i + 1) : -BigInt(i + 1)));
}

function buildSparseCoeffs(n: number): bigint[] {
    return Array.from({ length: n }, (_, i) => (i % 4 === 0 ? BigInt(i + 1) : 0n));
}

function buildDenseMixedCoeffs(n: number): bigint[] {
    return Array.from({ length: n }, (_, i) => {
        if (i % 3 === 0) return BigInt(i + 2);
        if (i % 3 === 1) return -BigInt(i + 1);
        return BigInt((i + 1) * 2);
    });
}

async function runHornerCase(
    h: PolynomialHarness,
    method: HornerMethod,
    coeffs: string[],
    x: string
) {
    if (method === "evaluateHorners") {
        await touchGas(h, method, [coeffs, x]);
        const gas = await estimateGas(h, method, [coeffs, x]);
        const out = await h.evaluateHorners(coeffs, x);
        return { gas, out };
    }

    if (method === "evaluateWithDerivative") {
        await touchGas(h, method, [coeffs, x]);
        const gas = await estimateGas(h, method, [coeffs, x]);
        const out = await h.evaluateWithDerivative(coeffs, x);
        return { gas, out };
    }

    await touchGas(h, method, [coeffs, x]);
    const gas = await estimateGas(h, method, [coeffs, x]);
    const out = await h.evalHornerMonic(coeffs, x);
    return { gas, out };
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("PolynomialHarness - Horner Gas Growth Tests", function () {
    let harness: PolynomialHarness;

    let X_NEG_1000: string;
    let X_NEG_1: string;
    let X_0: string;
    let X_1: string;
    let X_1000: string;

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const math = await MathLibFactory.deploy();
        await math.waitForDeployment();

        const HarnessFactory = await ethers.getContractFactory("PolynomialHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": await math.getAddress() }
        });

        harness = (await HarnessFactory.deploy()) as unknown as PolynomialHarness;
        await harness.waitForDeployment();

        X_NEG_1000 = await harness.qFromInt(-1000n);
        X_NEG_1 = await harness.qFromInt(-1n);
        X_0 = await harness.qFromInt(0n);
        X_1 = await harness.qFromInt(1n);
        X_1000 = await harness.qFromInt(1000n);
    });

    // ------------------------------------------------------------
    // Section 1: Gas Growth with Respect to Polynomial Length
    // ------------------------------------------------------------

    describe("Section 1: Gas Growth with Respect to Polynomial Length", function () {
        let testNo = 0;

        const SIZES = [16, ...Array.from({ length: 4096 / 128 }, (_, i) => (i + 1) * 128)];

        const METHODS: Array<{ method: HornerMethod; label: string }> = [
            { method: "evaluateHorners", label: "evaluateHorners" },
            { method: "evaluateWithDerivative", label: "evaluateWithDerivative" },
            { method: "evalHornerMonic", label: "evalHornerMonic" },
        ];

        for (const m of METHODS) {
            for (const n of SIZES) {
                const t = `1.${++testNo}`;

                it(`Test ${t}: ${m.label} gas growth with polynomial length n=${n}`, async function () {
                    const base = buildIncreasingCoeffs(n);
                    const coeffs =
                        m.method === "evalHornerMonic"
                            ? await qScaledArrayFromInts(harness, base.slice(0, n - 1), 1n, 1n)
                            : await qScaledArrayFromInts(harness, base, 1n, 1n);

                    const x = await harness.qFromFrac(5n, 4n); // 1.25

                    const { gas, out } = await runHornerCase(harness, m.method, coeffs, x);

                    let outHex: string;
                    let outDec: string;

                    if (m.method === "evaluateWithDerivative") {
                        const { px, dpx } = out as [string, string] & { px: string; dpx: string };
                        outHex = `p=${px}, p'=${dpx}`;
                        outDec = `p=${await safeFromQuad(harness, px)}, p'=${await safeFromQuad(harness, dpx)}`;
                    } else {
                        outHex = out as string;
                        outDec = await safeFromQuad(harness, out as string);
                    }

                    printBlockRegular({
                        t: `${t}`,
                        method: m.label,
                        explanation: `Gas growth with respect to polynomial length using increasing coefficients of length n=${n}.`,
                        gas,
                        inHex: `coeffs=[1..${m.method === "evalHornerMonic" ? n - 1 : n}], x=1.25`,
                        expectedHex: "N/A",
                        outHex,
                        expectedDec: "N/A",
                        outDec,
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 2: Gas Sensitivity to Evaluation Point Magnitude
    // ------------------------------------------------------------

    describe("Section 2: Gas Sensitivity to Evaluation Point Magnitude", function () {
        let testNo = 0;

        const METHODS: Array<{ method: HornerMethod; label: string }> = [
            { method: "evaluateHorners", label: "evaluateHorners" },
            { method: "evaluateWithDerivative", label: "evaluateWithDerivative" },
            { method: "evalHornerMonic", label: "evalHornerMonic" },
        ];

        const X_CASES: Array<{ x: () => string; label: string }> = [
            { x: () => X_NEG_1000, label: "-1000" },
            { x: () => X_NEG_1, label: "-1" },
            { x: () => X_0, label: "0" },
            { x: () => X_1, label: "1" },
            { x: () => X_1000, label: "1000" },
        ];

        for (const m of METHODS) {
            for (const xc of X_CASES) {
                const t = `2.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for x=${xc.label}`, async function () {
                    const base = buildIncreasingCoeffs(16);
                    const coeffs =
                        m.method === "evalHornerMonic"
                            ? await qScaledArrayFromInts(harness, base.slice(0, 15), 1n, 1n)
                            : await qScaledArrayFromInts(harness, base, 1n, 1n);

                    const { gas, out } = await runHornerCase(harness, m.method, coeffs, xc.x());

                    let outHex: string;
                    let outDec: string;

                    if (m.method === "evaluateWithDerivative") {
                        const { px, dpx } = out as [string, string] & { px: string; dpx: string };
                        outHex = `p=${px}, p'=${dpx}`;
                        outDec = `p=${await safeFromQuad(harness, px)}, p'=${await safeFromQuad(harness, dpx)}`;
                    } else {
                        outHex = out as string;
                        outDec = await safeFromQuad(harness, out as string);
                    }

                    printBlockRegular({
                        t: `${t}`,
                        method: m.label,
                        explanation: `Gas sensitivity to evaluation point magnitude using increasing coefficients and x=${xc.label}.`,
                        gas,
                        inHex: `len=${m.method === "evalHornerMonic" ? 15 : 16}, x=${xc.label}`,
                        expectedHex: "N/A",
                        outHex,
                        expectedDec: "N/A",
                        outDec,
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 3: Gas Sensitivity to Coefficient Scale
    // ------------------------------------------------------------

    describe("Section 3: Gas Sensitivity to Coefficient Scale", function () {
        let testNo = 0;

        const METHODS: Array<{ method: HornerMethod; label: string }> = [
            { method: "evaluateHorners", label: "evaluateHorners" },
            { method: "evaluateWithDerivative", label: "evaluateWithDerivative" },
            { method: "evalHornerMonic", label: "evalHornerMonic" },
        ];

        const SCALE_CASES: Array<{ label: string; factorNum: bigint; factorDen: bigint }> = [
            { label: "verySmall", factorNum: 1n, factorDen: 1_000_000n },
            { label: "small", factorNum: 1n, factorDen: 1_000n },
            { label: "normal", factorNum: 1n, factorDen: 1n },
            { label: "large", factorNum: 1_000n, factorDen: 1n },
            { label: "huge", factorNum: 1_000_000n, factorDen: 1n },
        ];

        for (const m of METHODS) {
            for (const s of SCALE_CASES) {
                const t = `3.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for coefficient scale ${s.label}`, async function () {
                    const base = buildIncreasingCoeffs(16);
                    const coeffs =
                        m.method === "evalHornerMonic"
                            ? await qScaledArrayFromInts(harness, base.slice(0, 15), s.factorNum, s.factorDen)
                            : await qScaledArrayFromInts(harness, base, s.factorNum, s.factorDen);

                    const { gas, out } = await runHornerCase(harness, m.method, coeffs, X_1);

                    let outHex: string;
                    let outDec: string;

                    if (m.method === "evaluateWithDerivative") {
                        const { px, dpx } = out as [string, string] & { px: string; dpx: string };
                        outHex = `p=${px}, p'=${dpx}`;
                        outDec = `p=${await safeFromQuad(harness, px)}, p'=${await safeFromQuad(harness, dpx)}`;
                    } else {
                        outHex = out as string;
                        outDec = await safeFromQuad(harness, out as string);
                    }

                    printBlockRegular({
                        t: `${t}`,
                        method: m.label,
                        explanation: `Gas sensitivity to coefficient scale using x=1 and ${s.label} scaled increasing coefficients.`,
                        gas,
                        inHex: `scale=${s.label}, x=1`,
                        expectedHex: "N/A",
                        outHex,
                        expectedDec: "N/A",
                        outDec,
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 4: Gas Sensitivity to Coefficient Pattern
    // ------------------------------------------------------------

    describe("Section 4: Gas Sensitivity to Coefficient Pattern", function () {
        let testNo = 0;

        const METHODS: Array<{ method: HornerMethod; label: string }> = [
            { method: "evaluateHorners", label: "evaluateHorners" },
            { method: "evaluateWithDerivative", label: "evaluateWithDerivative" },
            { method: "evalHornerMonic", label: "evalHornerMonic" },
        ];

        const PATTERN_CASES: Array<{ label: string; builder: (n: number) => bigint[] }> = [
            { label: "constant coefficients", builder: buildConstantCoeffs },
            { label: "increasing coefficients", builder: buildIncreasingCoeffs },
            { label: "alternating sign coefficients", builder: buildAlternatingCoeffs },
            { label: "sparse coefficients", builder: buildSparseCoeffs },
            { label: "dense mixed coefficients", builder: buildDenseMixedCoeffs },
        ];

        for (const m of METHODS) {
            for (const p of PATTERN_CASES) {
                const t = `4.${++testNo}`;

                it(`Test ${t}: ${m.label} gas sensitivity for ${p.label}`, async function () {
                    const base = p.builder(16);
                    const coeffs =
                        m.method === "evalHornerMonic"
                            ? await qScaledArrayFromInts(harness, base.slice(0, 15), 1n, 1n)
                            : await qScaledArrayFromInts(harness, base, 1n, 1n);

                    const { gas, out } = await runHornerCase(harness, m.method, coeffs, X_1);

                    let outHex: string;
                    let outDec: string;

                    if (m.method === "evaluateWithDerivative") {
                        const { px, dpx } = out as [string, string] & { px: string; dpx: string };
                        outHex = `p=${px}, p'=${dpx}`;
                        outDec = `p=${await safeFromQuad(harness, px)}, p'=${await safeFromQuad(harness, dpx)}`;
                    } else {
                        outHex = out as string;
                        outDec = await safeFromQuad(harness, out as string);
                    }

                    printBlockRegular({
                        t: `${t}`,
                        method: m.label,
                        explanation: `Gas sensitivity to coefficient pattern using ${p.label} with fixed length and x=1.`,
                        gas,
                        inHex: `pattern=${p.label}, len=${m.method === "evalHornerMonic" ? 15 : 16}, x=1`,
                        expectedHex: "N/A",
                        outHex,
                        expectedDec: "N/A",
                        outDec,
                    });

                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });
});