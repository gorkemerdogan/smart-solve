// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type DifferentiationHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;
    fromFloat(n: bigint): Promise<string>;

    forwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    backwardDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
    centeredDiffHarness(target: string, selector: string, x: string, h: string): Promise<string>;
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

async function fmt(harness: DifferentiationHarness, val: string): Promise<string> {
    const v = await harness.toFloat(val);
    return formatScaledInt(v);
}

type DiffMethodName = "forwardDiffHarness" | "backwardDiffHarness" | "centeredDiffHarness";
type DiffLabel = "FW" | "BW" | "CENT";

async function runGasCase(
    harness: DifferentiationHarness,
    method: DiffMethodName,
    target: string,
    selector: string,
    x: string,
    h: string
) {
    await touchGas(harness, method, [target, selector, x, h]);
    const gas = await estimateGas(harness, method, [target, selector, x, h]);
    const out = await harness[method](target, selector, x, h);
    return { gas, out };
}

function toGasBigInt(gas: unknown): bigint {
    if (typeof gas === "bigint") return gas;
    return BigInt(gas as string);
}

function gasSpread(values: bigint[]): bigint {
    let min = values[0];
    let max = values[0];
    for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
    }
    return max - min;
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("Differentiation Library - Gas Growth Tests", function () {
    let harness: DifferentiationHarness;
    let target: string;
    let t = 0;

    let QZERO: string;

    let H_TINY: string;
    let H_SMALL: string;
    let H_ONE: string;
    let H_TEN: string;
    let H_NEG_ONE: string;

    let selLinear: string;
    let selSquare: string;
    let selCube: string;
    let selAbs: string;

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

        selLinear = harness.interface.getFunction("f_linear")!.selector;
        selSquare = harness.interface.getFunction("f_square")!.selector;
        selCube = harness.interface.getFunction("f_cube")!.selector;
        selAbs = harness.interface.getFunction("f_abs")!.selector;

        QZERO = await harness.fromFloat(0n);

        H_TINY = await harness.qFromFrac(1, 100000000); // 1e-8
        H_SMALL = await harness.qFromFrac(1, 1000);     // 1e-3
        H_ONE = await harness.qFromInt(1);
        H_TEN = await harness.qFromInt(10);
        H_NEG_ONE = await harness.qFromInt(-1);
    });

    // ------------------------------------------------------------
    // Section 1: Input Magnitude Sensitivity
    // ------------------------------------------------------------

    describe("Section 1: Gas vs Input Magnitude", function () {
        const CASES = [
            { label: "x=-1e6", value: -1_000_000 },
            { label: "x=-1", value: -1 },
            { label: "x=0", value: 0 },
            { label: "x=1", value: 1 },
            { label: "x=1e6", value: 1_000_000 },
        ];

        const METHODS: Array<{ method: DiffMethodName; label: DiffLabel }> = [
            { method: "forwardDiffHarness", label: "FW" },
            { method: "backwardDiffHarness", label: "BW" },
            { method: "centeredDiffHarness", label: "CENT" },
        ];

        for (const m of METHODS) {
            it(`Test ${++t}: ${m.label} gas remains approximately stable as x magnitude changes`, async function () {
                const gasValues: bigint[] = [];

                for (const c of CASES) {
                    const x = await qInt(c.value);
                    const { gas, out } = await runGasCase(harness, m.method, target, selSquare, x, QZERO);
                    const gasBI = toGasBigInt(gas);
                    gasValues.push(gasBI);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to input magnitude using f(x)=x^2 and ${c.label}.`,
                        gas,
                        inHex: x,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await fmt(harness, out),
                    });
                }

                const spread = gasSpread(gasValues);

                console.log(`\n[${m.label}] gas values: ${gasValues.map(v => v.toString()).join(", ")}`);
                console.log(`[${m.label}] spread: ${spread.toString()}\n`);

                // Soft sanity assertion:
                // Input magnitude should not create large gas divergence.
                expect(spread < 5000n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 2: Step Size Sensitivity
    // ------------------------------------------------------------

    describe("Section 2: Gas vs Step Selection", function () {
        const STEP_CASES: Array<{ label: string; hKey: "QZERO" | "H_TINY" | "H_SMALL" | "H_ONE" | "H_TEN" | "H_NEG_ONE" }> = [
            { label: "default h (QZERO branch)", hKey: "QZERO" },
            { label: "h=1e-8", hKey: "H_TINY" },
            { label: "h=1e-3", hKey: "H_SMALL" },
            { label: "h=1", hKey: "H_ONE" },
            { label: "h=10", hKey: "H_TEN" },
            { label: "h=-1", hKey: "H_NEG_ONE" },
        ];

        const METHODS: Array<{ method: DiffMethodName; label: DiffLabel }> = [
            { method: "forwardDiffHarness", label: "FW" },
            { method: "backwardDiffHarness", label: "BW" },
            { method: "centeredDiffHarness", label: "CENT" },
        ];

        for (const m of METHODS) {
            it(`Test ${++t}: ${m.label} gas under different h selections`, async function () {
                const x = await qInt(100);
                const gasValues: bigint[] = [];

                for (const s of STEP_CASES) {
                    const h =
                        s.hKey === "QZERO" ? QZERO :
                        s.hKey === "H_TINY" ? H_TINY :
                        s.hKey === "H_SMALL" ? H_SMALL :
                        s.hKey === "H_ONE" ? H_ONE :
                        s.hKey === "H_TEN" ? H_TEN :
                        H_NEG_ONE;

                    const { gas, out } = await runGasCase(harness, m.method, target, selCube, x, h);
                    const gasBI = toGasBigInt(gas);
                    gasValues.push(gasBI);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas sensitivity to step size branch/selection with f(x)=x^3 and ${s.label}.`,
                        gas,
                        inHex: x,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await fmt(harness, out),
                    });
                }

                const spread = gasSpread(gasValues);

                console.log(`\n[${m.label}] step gas values: ${gasValues.map(v => v.toString()).join(", ")}`);
                console.log(`[${m.label}] step spread: ${spread.toString()}\n`);

                // Some branch variation is acceptable, but it should still stay limited.
                expect(spread < 8000n).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 3: Target Function Complexity Sensitivity
    // ------------------------------------------------------------

    describe("Section 3: Gas vs Target Function Complexity", function () {
        const FUNCTIONS: Array<{ label: string; selectorName: "linear" | "square" | "cube" | "abs" }> = [
            { label: "f_linear", selectorName: "linear" },
            { label: "f_square", selectorName: "square" },
            { label: "f_cube", selectorName: "cube" },
            { label: "f_abs", selectorName: "abs" },
        ];

        const METHODS: Array<{ method: DiffMethodName; label: DiffLabel }> = [
            { method: "forwardDiffHarness", label: "FW" },
            { method: "backwardDiffHarness", label: "BW" },
            { method: "centeredDiffHarness", label: "CENT" },
        ];

        for (const m of METHODS) {
            it(`Test ${++t}: ${m.label} gas across target function complexity`, async function () {
                const x = await qInt(7);

                for (const f of FUNCTIONS) {
                    const selector =
                        f.selectorName === "linear" ? selLinear :
                        f.selectorName === "square" ? selSquare :
                        f.selectorName === "cube" ? selCube :
                        selAbs;

                    const { gas, out } = await runGasCase(harness, m.method, target, selector, x, QZERO);

                    printBlockRegular({
                        t,
                        method: m.label,
                        explanation: `Gas comparison across target functions using ${f.label}.`,
                        gas,
                        inHex: x,
                        expectedHex: "N/A",
                        outHex: out,
                        expectedDec: "N/A",
                        outDec: await fmt(harness, out),
                    });
                }

                expect(true).to.equal(true);
            });
        }
    });

    // ------------------------------------------------------------
    // Section 4: Direct Method Comparison
    // ------------------------------------------------------------

    describe("Section 4: Direct Method Comparison", function () {
        it(`Test ${++t}: Compare FW, BW, CENT on same input and same target`, async function () {
            const x = await qInt(25);

            const fw = await runGasCase(harness, "forwardDiffHarness", target, selSquare, x, QZERO);
            const bw = await runGasCase(harness, "backwardDiffHarness", target, selSquare, x, QZERO);
            const cent = await runGasCase(harness, "centeredDiffHarness", target, selSquare, x, QZERO);

            printBlockRegular({
                t,
                method: "FW",
                explanation: "Direct comparison baseline for f(x)=x^2 at x=25.",
                gas: fw.gas,
                inHex: x,
                expectedHex: "N/A",
                outHex: fw.out,
                expectedDec: "N/A",
                outDec: await fmt(harness, fw.out),
            });

            printBlockRegular({
                t,
                method: "BW",
                explanation: "Direct comparison baseline for f(x)=x^2 at x=25.",
                gas: bw.gas,
                inHex: x,
                expectedHex: "N/A",
                outHex: bw.out,
                expectedDec: "N/A",
                outDec: await fmt(harness, bw.out),
            });

            printBlockRegular({
                t,
                method: "CENT",
                explanation: "Direct comparison baseline for f(x)=x^2 at x=25.",
                gas: cent.gas,
                inHex: x,
                expectedHex: "N/A",
                outHex: cent.out,
                expectedDec: "N/A",
                outDec: await fmt(harness, cent.out),
            });

            expect(toGasBigInt(cent.gas) >= toGasBigInt(fw.gas)).to.equal(true);
            expect(toGasBigInt(cent.gas) >= toGasBigInt(bw.gas)).to.equal(true);
        });
    });
});