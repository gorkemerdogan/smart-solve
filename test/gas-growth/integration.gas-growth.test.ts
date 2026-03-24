// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
//  Types
// ------------------------------------------------------------

type IntegrationHarness = Contract & {
  qFromInt(n: bigint | number): Promise<string>;
  qFromFrac(n: bigint | number, d: bigint | number): Promise<string>;
  PI(): Promise<string>;

  // Helpers
  toFloat(q: string): Promise<bigint>;
  fromFloat(n: bigint): Promise<string>;

  // Methods
  trapezoidal(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson13(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson38(target: string, selector: string, a: string, b: string, n: number): Promise<string>;

  // Base functions
  f_linear(x: string): Promise<string>;
  f_square(x: string): Promise<string>;
  f_cube(x: string): Promise<string>;
  f_const5(x: string): Promise<string>;

  // Scale functions
  f_verySmall(x: string): Promise<string>;
  f_small(x: string): Promise<string>;
  f_normal(x: string): Promise<string>;
  f_large(x: string): Promise<string>;
  f_huge(x: string): Promise<string>;
};

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
  return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

async function fmt(harness: IntegrationHarness, val: string): Promise<string> {
  const v = await harness.toFloat(val);
  return formatScaledInt(v);
}

type MethodName = "trapezoidal" | "simpson13" | "simpson38";
type MethodLabel = "trapezoidal" | "simpson13" | "simpson38";

async function runGasCase(
  harness: IntegrationHarness,
  method: MethodName,
  target: string,
  selector: string,
  a: string,
  b: string,
  n: number
) {
  await touchGas(harness, method, [target, selector, a, b, n]);
  const gas = await estimateGas(harness, method, [target, selector, a, b, n]);
  const out = await harness[method](target, selector, a, b, n);
  return { gas, out };
}

function toGasBigInt(gas: unknown): bigint {
  if (typeof gas === "bigint") return gas;
  return BigInt(gas as string);
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Integration Library - Gas Growth Tests", function () {
  let harness: IntegrationHarness;
  let target: string;
  let t = 0;

  // Constants
  let q0: string;
  let q1: string;
  let q10: string;
  let q100: string;
  let q1000: string;
  let q10000: string;

  // Selectors
  let selConst5: string;
  let selLinear: string;
  let selSquare: string;
  let selCube: string;

  let selVerySmall: string;
  let selSmall: string;
  let selNormal: string;
  let selLarge: string;
  let selHuge: string;

  before(async () => {
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const mathlib = await MathLibFactory.deploy();
    await mathlib.waitForDeployment();

    const HF = await ethers.getContractFactory("IntegrationHarness", {
      libraries: { MathLib: await mathlib.getAddress() },
    });

    harness = (await HF.deploy()) as unknown as IntegrationHarness;
    await harness.waitForDeployment();
    target = await harness.getAddress();

    q0 = await harness.qFromInt(0);
    q1 = await harness.qFromInt(1);
    q10 = await harness.qFromInt(10);
    q100 = await harness.qFromInt(100);
    q1000 = await harness.qFromInt(1000);
    q10000 = await harness.qFromInt(10000);

    selConst5 = harness.interface.getFunction("f_const5")!.selector;
    selLinear = harness.interface.getFunction("f_linear")!.selector;
    selSquare = harness.interface.getFunction("f_square")!.selector;
    selCube = harness.interface.getFunction("f_cube")!.selector;

    selVerySmall = harness.interface.getFunction("f_verySmall")!.selector;
    selSmall = harness.interface.getFunction("f_small")!.selector;
    selNormal = harness.interface.getFunction("f_normal")!.selector;
    selLarge = harness.interface.getFunction("f_large")!.selector;
    selHuge = harness.interface.getFunction("f_huge")!.selector;
  });

  // ------------------------------------------------------------
  //  Section 1: Gas Growth with Respect to n
  // ------------------------------------------------------------

  describe("Section 1: Gas Growth with Respect to n", function () {
    const N_CASES = [12, 30, 120, 300, 1200, 3000];

    const METHODS: Array<{ method: MethodName; label: MethodLabel }> = [
      { method: "trapezoidal", label: "trapezoidal" },
      { method: "simpson13", label: "simpson13" },
      { method: "simpson38", label: "simpson38" },
    ];

    for (const m of METHODS) {
      for (const n of N_CASES) {
        it(`Test ${++t}: ${m.label} gas growth with n=${n}`, async function () {
          const { gas, out } = await runGasCase(harness, m.method, target, selSquare, q0, q1, n);

          printBlockRegular({
            t,
            method: m.label,
            explanation: `Gas growth with respect to partition count n for f(x)=x^2 on [0,1], n=${n}.`,
            gas,
            inHex: `[0,1], n=${n}`,
            expectedHex: "N/A",
            outHex: out,
            expectedDec: "N/A",
            outDec: await fmt(harness, out),
          });

          expect(toGasBigInt(gas) > 0n).to.equal(true);
        });
      }
    }
  });

  // ------------------------------------------------------------
  //  Section 2: Gas Sensitivity to Interval Width
  // ------------------------------------------------------------

  describe("Section 2: Gas Sensitivity to Interval Width", function () {
    const INTERVAL_CASES: Array<{ label: string; b: () => Promise<string> }> = [
      { label: "[0,1]", b: async () => q1 },
      { label: "[0,10]", b: async () => q10 },
      { label: "[0,100]", b: async () => q100 },
      { label: "[0,1000]", b: async () => q1000 },
      { label: "[0,10000]", b: async () => q10000 },
    ];

    const METHODS: Array<{ method: MethodName; label: MethodLabel }> = [
      { method: "trapezoidal", label: "trapezoidal" },
      { method: "simpson13", label: "simpson13" },
      { method: "simpson38", label: "simpson38" },
    ];

    const FIXED_N = 300;

    for (const m of METHODS) {
      for (const intervalCase of INTERVAL_CASES) {
        it(`Test ${++t}: ${m.label} gas sensitivity on interval ${intervalCase.label}`, async function () {
          const b = await intervalCase.b();
          const { gas, out } = await runGasCase(harness, m.method, target, selSquare, q0, b, FIXED_N);

          printBlockRegular({
            t,
            method: m.label,
            explanation: `Gas sensitivity to interval width for f(x)=x^2 over ${intervalCase.label} with fixed n=${FIXED_N}.`,
            gas,
            inHex: `${intervalCase.label}, n=${FIXED_N}`,
            expectedHex: "N/A",
            outHex: out,
            expectedDec: "N/A",
            outDec: await fmt(harness, out),
          });

          expect(toGasBigInt(gas) > 0n).to.equal(true);
        });
      }
    }
  });

  // ------------------------------------------------------------
  //  Section 3: Gas Sensitivity to Function Degree
  // ------------------------------------------------------------

  describe("Section 3: Gas Sensitivity to Function Degree", function () {
    const FUNCTION_CASES: Array<{ label: string; selector: () => string }> = [
      { label: "f(x)=5", selector: () => selConst5 },
      { label: "f(x)=x", selector: () => selLinear },
      { label: "f(x)=x^2", selector: () => selSquare },
      { label: "f(x)=x^3", selector: () => selCube },
    ];

    const METHODS: Array<{ method: MethodName; label: MethodLabel }> = [
      { method: "trapezoidal", label: "trapezoidal" },
      { method: "simpson13", label: "simpson13" },
      { method: "simpson38", label: "simpson38" },
    ];

    const FIXED_N = 300;

    for (const m of METHODS) {
      for (const f of FUNCTION_CASES) {
        it(`Test ${++t}: ${m.label} gas sensitivity for ${f.label}`, async function () {
          const { gas, out } = await runGasCase(harness, m.method, target, f.selector(), q0, q1, FIXED_N);

          printBlockRegular({
            t,
            method: m.label,
            explanation: `Gas sensitivity to function degree/structure using ${f.label} on [0,1] with fixed n=${FIXED_N}.`,
            gas,
            inHex: `${f.label}, [0,1], n=${FIXED_N}`,
            expectedHex: "N/A",
            outHex: out,
            expectedDec: "N/A",
            outDec: await fmt(harness, out),
          });

          expect(toGasBigInt(gas) > 0n).to.equal(true);
        });
      }
    }
  });

  // ------------------------------------------------------------
  //  Section 4: Gas Sensitivity to Numeric Scale
  // ------------------------------------------------------------

  describe("Section 4: Gas Sensitivity to Numeric Scale", function () {
    const SCALE_CASES: Array<{ label: string; selector: () => string }> = [
      { label: "f_verySmall", selector: () => selVerySmall },
      { label: "f_small", selector: () => selSmall },
      { label: "f_normal", selector: () => selNormal },
      { label: "f_large", selector: () => selLarge },
      { label: "f_huge", selector: () => selHuge },
    ];

    const METHODS: Array<{ method: MethodName; label: MethodLabel }> = [
      { method: "trapezoidal", label: "trapezoidal" },
      { method: "simpson13", label: "simpson13" },
      { method: "simpson38", label: "simpson38" },
    ];

    const FIXED_N = 300;

    for (const m of METHODS) {
      for (const s of SCALE_CASES) {
        it(`Test ${++t}: ${m.label} gas sensitivity for ${s.label}`, async function () {
          const { gas, out } = await runGasCase(harness, m.method, target, s.selector(), q0, q1, FIXED_N);

          printBlockRegular({
            t,
            method: m.label,
            explanation: `Gas sensitivity to numeric scale using ${s.label} on [0,1] with fixed n=${FIXED_N}.`,
            gas,
            inHex: `${s.label}, [0,1], n=${FIXED_N}`,
            expectedHex: "N/A",
            outHex: out,
            expectedDec: "N/A",
            outDec: await fmt(harness, out),
          });

          expect(toGasBigInt(gas) > 0n).to.equal(true);
        });
      }
    }
  });
});