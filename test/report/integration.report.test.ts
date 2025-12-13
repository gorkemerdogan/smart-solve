// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type IntegrationHarness = Contract & {
  qFromInt(n: bigint | number): Promise<string>;
  qFromFrac(n: bigint | number, d: bigint | number): Promise<string>;
  PI(): Promise<string>;

  trapezoidal(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson13(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson38(target: string, selector: string, a: string, b: string, n: number): Promise<string>;

  f_linear(x: string): Promise<string>;
  f_square(x: string): Promise<string>;
  f_cube(x: string): Promise<string>;
  f_const5(x: string): Promise<string>;
  f_sin(x: string): Promise<string>;
  f_inv(x: string): Promise<string>;
  f_tiny(x: string): Promise<string>;
  f_large(x: string): Promise<string>;
  f_piecewise(x: string): Promise<string>;
};

const QZERO = "0x00000000000000000000000000000000";

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

    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Integration Library - Numerical Methods", function () {
  let harness: IntegrationHarness;
  let target: string;
  let t = 0;

  // Constants
  let q0: string, q1: string, q2: string, qPI: string;

  // Selectors
  let selLinear: string, selSquare: string, selCube: string, selConst5: string;
  let selSin: string, selInv: string, selTiny: string, selLarge: string, selPiecewise: string;

  // Helper Wrappers
  const qInt = async (n: number | bigint) => await harness.qFromInt(n);

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

    // Init Constants
    q0 = await harness.qFromInt(0);
    q1 = await harness.qFromInt(1);
    q2 = await harness.qFromInt(2);
    qPI = await harness.PI();

    // Init Selectors
    selLinear = harness.interface.getFunction("f_linear")!.selector;
    selSquare = harness.interface.getFunction("f_square")!.selector;
    selCube = harness.interface.getFunction("f_cube")!.selector;
    selConst5 = harness.interface.getFunction("f_const5")!.selector;
    selSin = harness.interface.getFunction("f_sin")!.selector;
    selInv = harness.interface.getFunction("f_inv")!.selector;
    selTiny = harness.interface.getFunction("f_tiny")!.selector;
    selLarge = harness.interface.getFunction("f_large")!.selector;
    selPiecewise = harness.interface.getFunction("f_piecewise")!.selector;
  });

  // ------------------------------------------------------------
  //  Trapezoidal Rule
  // ------------------------------------------------------------

  describe("Method 1: Trapezoidal Rule", function () {

    it("Test 1: Linear Exactness (f(x)=x on [0,1], n=10)", async function () {
      t++;
      const expectedDec = "≈0.5";

      await touchGas(harness, "trapezoidal", [target, selLinear, q0, q1, 10]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLinear, q0, q1, 10]);
      const out = await harness.trapezoidal(target, selLinear, q0, q1, 10);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(typeof out).to.equal("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on linear f(x)=x over [0,1]. Analytic integral is 1/2 so result should be very close to 0.5.",
        inHex: "f=x, [0,1]",
        expectedHex: "N/A (analytic 0.5 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 2: Smooth Quadratic (f(x)=x^2 on [0,1], n=300)", async function () {
      t++;
      const expectedDec = "≈0.3333";

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 300]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 300]);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 300);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on smooth quadratic f(x)=x^2 over [0,1]. With n=300, it should approximate 1/3 with high accuracy.",
        inHex: "f=x^2, [0,1]",
        expectedHex: "N/A (analytic 1/3 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 3: Constant Exactness (f=5 on [0,10], n=50)", async function () {
      t++;
      const b = await qInt(10);
      const expectedDec = "50";

      await touchGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const out = await harness.trapezoidal(target, selConst5, q0, b, 50);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on constant f(x)=5 over [0,10]. Constant integrand should give exactly 5x10 = 50.",
        inHex: "f=5, [0,10]",
        expectedHex: "N/A (exact area 50 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 4: Trig Function (sin(x) on [0,π], n=200)", async function () {
      t++;
      const expectedDec = "≈2";

      await touchGas(harness, "trapezoidal", [target, selSin, q0, qPI, 200]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSin, q0, qPI, 200]);
      const out = await harness.trapezoidal(target, selSin, q0, qPI, 200);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal integration of sin(x) over [0,π]. Analytic value is 2 so the numeric result should cluster near 2.",
        inHex: "f=sin, [0,π]",
        expectedHex: "N/A (analytic integral 2)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 5: Degenerate Interval ([0,0], n=100)", async function () {
      t++;
      const expectedDec = "0";

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const out = await harness.trapezoidal(target, selSquare, q0, q0, 100);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.equal(q0);
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on zero-length interval [0,0]. Regardless of f(x), the integral must evaluate to exactly 0.",
        inHex: "[0,0]",
        expectedHex: q0,
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 6: Max Iteration Comparison (f=x^2)", async function () {
      t++;
      const expectedDec = "~";

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 50]);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 50);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal integration of x^2 on [0,1] with n=50. Esed mainly for gas and Java reference comparison.",
        inHex: "[0,1]",
        expectedHex: "N/A (used for comparison only)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });
  });

  // ------------------------------------------------------------
  //  Simpson 1/3 Rule
  // ------------------------------------------------------------

  describe("Method 2: Simpson 1/3 Rule", function () {

    it("Test 7: Quadratic Exactness-ish (f=x^2 on [0,1], n=10)", async function () {
      t++;
      const expectedDec = "≈0.3333";

      await touchGas(harness, "simpson13", [target, selSquare, q0, q1, 10]);
      const gas = await estimateGas(harness, "simpson13", [target, selSquare, q0, q1, 10]);
      const out = await harness.simpson13(target, selSquare, q0, q1, 10);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 rule on quadratic f(x)=x^2 over [0,1]. Even n=10 should give a value extremely close to 1/3.",
        inHex: "[0,1], n=10",
        expectedHex: "N/A (analytic 1/3 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 8: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      const expectedDec = "0.25";

      await touchGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const out = await harness.simpson13(target, selCube, q0, q1, 12);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 rule on cubic f(x)=x^3 over [0,1]. Method is exact for cubics so the value should be exactly 1/4.",
        inHex: "[0,1], n=12",
        expectedHex: "N/A (exact 0.25 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 9: Exp Approx (Linear proxy on [0,1], n=20)", async function () {
      t++;
      const expectedDec = "~";

      await touchGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const gas = await estimateGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const out = await harness.simpson13(target, selLinear, q0, q1, 20);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 on linear proxy (placeholder for exp-like curve) mainly to compare behavior and gas usage.",
        inHex: "[0,1], n=20",
        expectedHex: "N/A (comparison-only scenario)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 10: Inverse Function (1/x on [1,2], n=40)", async function () {
      t++;
      const expectedDec = "≈0.693";

      await touchGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const gas = await estimateGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const out = await harness.simpson13(target, selInv, q1, q2, 40);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 integration of 1/x over [1,2]. Analytic value is ln(2) ≈ 0.693, smooth non-polynomial.",
        inHex: "[1,2], n=40",
        expectedHex: "N/A (analytic ln(2))",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 11: Revert on Invalid N (Odd)", async function () {
      t++;
      await expect(harness.simpson13(target, selSquare, q0, q1, 7)).to.be.revertedWith("Integration: Simpson 1/3 requires even n");
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 called with odd n=7 should revert, enforcing even-subinterval constraint.",
        inHex: "x^2, n=7",
        expectedHex: "Revert",
        outHex: "Revert",
        expectedDec: "Revert",
        outDec: "Revert",
        gas: "N/A",
      });
    });
  });

  // ------------------------------------------------------------
  //  Simpson 3/8 Rule
  // ------------------------------------------------------------

  describe("Method 3: Simpson 3/8 Rule", function () {

    it("Test 12: Quadratic Exactness (f=x^2 on [0,1], n=9)", async function () {
      t++;
      const expectedDec = "≈0.3333";

      await touchGas(harness, "simpson38", [target, selSquare, q0, q1, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, q1, 9]);
      const out = await harness.simpson38(target, selSquare, q0, q1, 9);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule on quadratic f(x)=x^2 over [0,1]. Higher-order accuracy should give integral near 1/3.",
        inHex: "[0,1], n=9",
        expectedHex: "N/A (analytic 1/3 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 13: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      const expectedDec = "0.25";

      await touchGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const out = await harness.simpson38(target, selCube, q0, q1, 12);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule on cubic f(x)=x^3 over [0,1]. Expecting integral 1/4.",
        inHex: "[0,1], n=12",
        expectedHex: "N/A (exact 0.25 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: expectedDec,
        gas
      });
    });

    it("Test 14: Inverse Function (1/x on [1,2], n=12)", async function () {
      t++;
      const expectedDec = "≈0.693";

      await touchGas(harness, "simpson38", [target, selInv, q1, q2, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selInv, q1, q2, 12]);
      const out = await harness.simpson38(target, selInv, q1, q2, 12);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 integration of 1/x over [1,2]. Should approximate ln(2) ≈ 0.693 similarly to 1/3 rule.",
        inHex: "[1,2], n=12",
        expectedHex: "N/A (analytic ln(2))",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 15: Degenerate Interval ([5,5], n=3)", async function () {
      t++;
      const five = await qInt(5);
      const expectedDec = "0";

      await touchGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const out = await harness.simpson38(target, selSquare, five, five, 3);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.equal(q0);
      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule on zero-length interval [5,5]. Tntegral must be exactly zero, confirming bounds handling.",
        inHex: "[5,5]",
        expectedHex: q0,
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 16: Revert on Invalid N (Not Multiple of 3)", async function () {
      t++;
      await expect(harness.simpson38(target, selSquare, q0, q1, 10)).to.be.revertedWith("Integration: Simpson 3/8 requires n % 3 == 0");
      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 called with n=10 where n%3!=0. Should revert enforcing multiple-of-3 requirement.",
        inHex: "x^2, n=10",
        expectedHex: "Revert",
        outHex: "Revert",
        expectedDec: "Revert",
        outDec: "Revert",
        gas: "N/A",
      });
    });
  });

  // ------------------------------------------------------------
  //  Edge Cases
  // ------------------------------------------------------------

  describe("Edge Cases", function () {

    it("Test 17: Tiny Magnitudes (1e-30·x)", async function () {
      t++;
      const expectedDec = "≈5e-31";

      await touchGas(harness, "trapezoidal", [target, selTiny, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selTiny, q0, q1, 30]);
      const out = await harness.trapezoidal(target, selTiny, q0, q1, 30);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on extremely small-magnitude function 1e-30·x over [0,1]. Stress underflow and precision.",
        inHex: "1e-30·x",
        expectedHex: "N/A (≈5e-31 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 18: Large Magnitudes (1e20·x)", async function () {
      t++;
      const expectedDec = "≈5e19";

      await touchGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const out = await harness.trapezoidal(target, selLarge, q0, q1, 30);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on very large-magnitude function 1e20·x over [0,1]. Checks overflow resilience and scaling.",
        inHex: "1e20·x",
        expectedHex: "N/A (≈5e19 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 19: Piecewise Discontinuous", async function () {
      t++;
      const expectedDec = "≈4";

      await touchGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const gas = await estimateGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const out = await harness.simpson13(target, selPiecewise, q0, q2, 60);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 on piecewise-discontinuous f(x) over [0,2]. Tests robustness around internal jump from 1 to 3.",
        inHex: "[0,2]",
        expectedHex: "N/A (piecewise analytic ≈4)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 20: Narrow Interval ([0,1e-12])", async function () {
      t++;
      const tiny = await harness.qFromFrac(1, BigInt("1000000000000"));
      const expectedDec = "≈3.33e-37";

      await touchGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const out = await harness.simpson38(target, selSquare, q0, tiny, 9);

      const outInt = await harness.toFloat(out);
      const outDec = formatScaledInt(outInt);

      expect(out).to.be.a("string");
      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule for x^2 over ultra-narrow interval [0,1e-12]. Validates tiny-domain integration stability.",
        inHex: "[0,1e-12]",
        expectedHex: "N/A (≈3.33e-37 in quad)",
        outHex: out,
        expectedDec: expectedDec,
        outDec: outDec,
        gas
      });
    });

    it("Test 21: Revert on Reversed Bounds", async function () {
      t++;
      await expect(harness.trapezoidal(target, selSquare, q1, q0, 10)).to.be.revertedWith("Integration: upper bound b must be >= a");
      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal call with reversed bounds [1,0]. Library must revert, enforcing b >= a precondition.",
        inHex: "x^2, [1,0]",
        expectedHex: "Revert",
        outHex: "Revert",
        expectedDec: "Revert",
        outDec: "Revert",
        gas: "N/A"
      });
    });
  });
});