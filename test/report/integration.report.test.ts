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

  // Helpers
  toFloat(q: string): Promise<bigint>;
  fromFloat(n: bigint): Promise<string>;

  // Methods
  trapezoidal(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson13(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson38(target: string, selector: string, a: string, b: string, n: number): Promise<string>;

  // Functions
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

let QZERO: string;

const TOL_EXACT = 50n; // ~1e-11 error (100n) for rounding noise in exact methods
const TOL_APPROX = 100_000n; // ~1e-7 error (100,000,000n) for simple Trapezoidal tests

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

// Check if values are close
async function expectClose(h: IntegrationHarness, actualHex: string, expectedHex: string, tolerance: bigint) {
  const a = await h.toFloat(actualHex);
  const b = await h.toFloat(expectedHex);
  let diff = a - b;
  if (diff < 0n) diff = -diff;

  if (diff > tolerance) {
    console.log(`    Mismatch > tolerance!`);
    console.log(`    Exp: ${b} (scaled)`);
    console.log(`    Act: ${a} (scaled)`);
    console.log(`    Diff:${diff}`);
  }
  expect(diff <= tolerance).to.be.true;
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
  let selLinear: string, selSquare: string, selCube: string, selConst5: string;
  let selSin: string, selInv: string, selTiny: string, selLarge: string, selPiecewise: string;

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

    q0 = await harness.qFromInt(0);
    q1 = await harness.qFromInt(1);
    q2 = await harness.qFromInt(2);
    qPI = await harness.PI();
    QZERO = q0;

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

  describe("Section 1: Trapezoidal Rule", function () {

    it("Test 1: Linear Exactness (f(x)=x on [0,1], n=10)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 2);
      const out = await harness.trapezoidal(target, selLinear, q0, q1, 10);
      await expectClose(harness, out, expected, TOL_EXACT); // Exact within rounding noise

      await touchGas(harness, "trapezoidal", [target, selLinear, q0, q1, 10]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLinear, q0, q1, 10]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Linear f(x)=x. Exact for degree 1.",
        gas,
        inHex: "f=x, [0,1]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 2: Smooth Quadratic (f(x)=x^2 on [0,1], n=3000)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 3);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 3000);
      await expectClose(harness, out, expected, TOL_APPROX);

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 3000]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 3000]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal approx of x^2.",
        gas,
        inHex: "f=x^2, [0,1]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 3: Constant Exactness (f=5 on [0,10], n=50)", async function () {
      t++;
      const b = await qInt(10);
      const expected = await qInt(50);
      const out = await harness.trapezoidal(target, selConst5, q0, b, 50);
      await expectClose(harness, out, expected, TOL_EXACT);

      await touchGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Constant function f=5. Exact area 50.",
        gas,
        inHex: "f=5, [0,10]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 4: Trig Function (sin(x) on [0,π], n=300)", async function () {
      t++;
      const expected = await qInt(2);
      const out = await harness.trapezoidal(target, selSin, q0, qPI, 300);
      await expectClose(harness, out, expected, 50_000_000n);

      await touchGas(harness, "trapezoidal", [target, selSin, q0, qPI, 300]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSin, q0, qPI, 300]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal integration of sin(x) over [0,π]. Clusters near analytic 2.",
        gas,
        inHex: "f=sin, [0,π]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 5: Degenerate Interval ([0,0], n=100)", async function () {
      t++;
      const expected = q0;
      const out = await harness.trapezoidal(target, selSquare, q0, q0, 100);
      await expectClose(harness, out, expected, 0n); // Must be 0

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Zero-length interval [0,0] must evaluate to exactly 0.",
        gas,
        inHex: "[0,0]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 6: Max Iteration Comparison (f=x^2)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 3);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 3000);
      await expectClose(harness, out, expected, TOL_APPROX);

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 3000]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 3000]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Benchmarking gas for n=50.",
        gas,
        inHex: "[0,1]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });


    it("Test 7: Tiny Magnitudes (1e-30·x)", async function () {
      t++;
      const expected = await harness.qFromFrac(5, BigInt("1" + "0".repeat(31)));
      const out = await harness.trapezoidal(target, selTiny, q0, q1, 30);
      await expectClose(harness, out, expected, TOL_EXACT); // Scaled value should be 0

      await touchGas(harness, "trapezoidal", [target, selTiny, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selTiny, q0, q1, 30]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Tiny magnitude 1e-30·x. Integral 5e-31. Checks underflow/precision limits.",
        gas,
        inHex: "1e-30·x",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 8: Large Magnitudes (1e20·x)", async function () {
      t++;
      const largeNum = BigInt("1" + "0".repeat(20));
      const expected = await harness.qFromFrac(largeNum, 2);
      const out = await harness.trapezoidal(target, selLarge, q0, q1, 30);
      await expectClose(harness, out, expected, TOL_APPROX);

      await touchGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Large magnitude 1e20·x. Integral 5e19. Checks overflow/scaling.",
        gas,
        inHex: "1e20·x",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 9: Revert on Reversed Bounds", async function () {
      t++;
      await expect(harness.trapezoidal(target, selSquare, q1, q0, 10)).to.be.revertedWith("Integration: upper bound b must be >= a");

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal call with reversed bounds [1,0]. Should revert.",
        gas: "N/A",
        inHex: "x^2, [1,0]",
        expectedHex: "Revert",
        outHex: "Revert",
        expectedDec: "Revert",
        outDec: "Revert"
      });
    });
  });

  // ------------------------------------------------------------
  //  Simpson 1/3 Rule
  // ------------------------------------------------------------

  describe("Section 2: Simpson 1/3 Rule", function () {

    it("Test 10: Quadratic Exactness (f=x^2 on [0,1], n=10)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 3);
      const out = await harness.simpson13(target, selSquare, q0, q1, 10);
      await expectClose(harness, out, expected, TOL_EXACT);

      await touchGas(harness, "simpson13", [target, selSquare, q0, q1, 10]);
      const gas = await estimateGas(harness, "simpson13", [target, selSquare, q0, q1, 10]);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 is exact for polynomials deg<=3. Matches 1/3 exactly.",
        gas,
        inHex: "[0,1], n=10",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 11: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 4);
      const out = await harness.simpson13(target, selCube, q0, q1, 12);
      await expectClose(harness, out, expected, TOL_EXACT);

      await touchGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson13", [target, selCube, q0, q1, 12]);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 exactness for cubics. Result must be exactly 0.25.",
        gas,
        inHex: "[0,1], n=12",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 12: Exp Approx (Linear proxy on [0,1], n=20)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 2);
      const out = await harness.simpson13(target, selLinear, q0, q1, 20);
      await expectClose(harness, out, expected, TOL_EXACT);

      await touchGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const gas = await estimateGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 on linear function. Should also be exact (deg 1 < 3).",
        gas,
        inHex: "[0,1], n=20",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 13: Inverse Function (1/x on [1,2], n=40)", async function () {
      t++;
      const ln2_js = Math.log(2);
      const expected = await harness.fromFloat(BigInt(Math.round(ln2_js * 1e12)));
      const out = await harness.simpson13(target, selInv, q1, q2, 40);
      await expectClose(harness, out, expected, TOL_APPROX);

      await touchGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const gas = await estimateGas(harness, "simpson13", [target, selInv, q1, q2, 40]);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 of 1/x over [1,2]. Approximates ln(2).",
        gas,
        inHex: "[1,2], n=40",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 14: Revert on Invalid N (Odd)", async function () {
      t++;
      await expect(harness.simpson13(target, selSquare, q0, q1, 7)).to.be.revertedWith("Integration: Simpson 1/3 requires even n");

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Simpson 1/3 called with odd n=7 should revert.",
        gas: "N/A",
        inHex: "x^2, n=7",
        expectedHex: "Revert",
        outHex: "Revert",
        expectedDec: "Revert",
        outDec: "Revert"
      });
    });

    it("Test 15: Piecewise Discontinuous (Split Domain)", async function () {
      t++;
      const expected = await qInt(4); // 1 + 3 = 4

      // Split the integral at the discontinuity x=1
      // Part 1: [0, 1]
      const out1 = await harness.simpson13(target, selPiecewise, q0, q1, 30);
      // Part 2: [1, 2]
      const out2 = await harness.simpson13(target, selPiecewise, q1, q2, 30);

      // Sum the parts
      const total = await harness.qAdd(out1, out2);

      // TOLERANCE EXPLANATION:
      // When integrating [0,1], Simpson's rule samples f(1).
      // The contract returns 3 at x=1 (because x>=1 ? 3 : 1).
      // This high endpoint distorts the left integral, adding ~0.022 of error.
      // This is unavoidable with closed Newton-Cotes methods on jump discontinuities.
      const PIECEWISE_TOL = 30_000_000_000n; // ~0.03

      await expectClose(harness, total, expected, PIECEWISE_TOL);

      // Gas estimate for reporting (approximation)
      await touchGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const gas = await estimateGas(harness, "simpson13", [target, selPiecewise, q0, q1, 60]);

      // Report the combined result to show correctness
      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Piecewise jump. Domain split [0,1] + [1,2] for correct convergence.",
        gas: gas,
        inHex: "[0,2] split",
        expectedHex: expected,
        outHex: total,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, total)
      });
    });
  });

  // ------------------------------------------------------------
  //  Simpson 3/8 Rule
  // ------------------------------------------------------------

  describe("Section 3: Simpson 3/8 Rule", function () {

    it("Test 16: Quadratic Exactness (f=x^2 on [0,1], n=9)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 3);
      const out = await harness.simpson38(target, selSquare, q0, q1, 9);
      await expectClose(harness, out, expected, TOL_EXACT);

      await touchGas(harness, "simpson38", [target, selSquare, q0, q1, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, q1, 9]);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule on quadratic f(x)=x^2. Exact result 1/3.",
        gas,
        inHex: "[0,1], n=9",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 17: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      const expected = await harness.qFromFrac(1, 4);
      const out = await harness.simpson38(target, selCube, q0, q1, 12);
      await expectClose(harness, out, expected, TOL_EXACT);

      await touchGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selCube, q0, q1, 12]);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule on cubic f(x)=x^3. Exact result 1/4.",
        gas,
        inHex: "[0,1], n=12",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 18: Inverse Function (1/x on [1,2], n=99)", async function () {
      t++;
      const ln2_js = Math.log(2);
      const expected = await harness.fromFloat(BigInt(Math.round(ln2_js * 1e12)));
      const out = await harness.simpson38(target, selInv, q1, q2, 99);
      await expectClose(harness, out, expected, TOL_APPROX);

      await touchGas(harness, "simpson38", [target, selInv, q1, q2, 99]);
      const gas = await estimateGas(harness, "simpson38", [target, selInv, q1, q2, 99]);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 integration of 1/x over [1,2]. Approximates ln(2).",
        gas,
        inHex: "[1,2], n=99",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 19: Degenerate Interval ([5,5], n=3)", async function () {
      t++;
      const five = await qInt(5);
      const expected = QZERO;
      const out = await harness.simpson38(target, selSquare, five, five, 3);
      await expectClose(harness, out, expected, 0n);

      await touchGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, five, five, 3]);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 rule on zero-length interval [5,5]. Integral must be zero.",
        gas,
        inHex: "[5,5]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 20: Revert on Invalid N (Not Multiple of 3)", async function () {
      t++;
      await expect(harness.simpson38(target, selSquare, q0, q1, 10)).to.be.revertedWith("Integration: Simpson 3/8 requires n % 3 == 0");

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 called with n=10 where n%3!=0. Should revert.",
        gas: "N/A",
        inHex: "x^2, n=10",
        expectedHex: "Revert",
        outHex: "Revert",
        expectedDec: "Revert",
        outDec: "Revert"
      });
    });

    it("Test 21: Narrow Interval ([0,1e-12])", async function () {
      t++;
      const tiny = await harness.qFromFrac(1, BigInt("1000000000000"));
      const expected = await harness.qFromFrac(1, BigInt("3" + "0".repeat(36)));
      const out = await harness.simpson38(target, selSquare, q0, tiny, 9);

      await touchGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);

      await expectClose(harness, out, expected, TOL_EXACT);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Ultra-narrow interval [0,1e-12]. Result 3.33e-37.",
        gas,
        inHex: "[0,1e-12]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });
  });

  // ------------------------------------------------------------
  //  Comparison
  // ------------------------------------------------------------

  describe("Section 4: Comparison", function () {

    it("Test 22: Shared Low-Gas Comparison (sin(x) on [0,π], n=30)", async function () {
      t++;
      const expected = await qInt(2);

      await touchGas(harness, "trapezoidal", [target, selSin, q0, qPI, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSin, q0, qPI, 30]);
      const out = await harness.trapezoidal(target, selSin, q0, qPI, 30);

      await expectClose(harness, out, expected, 5_000_000_000n);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Shared low-gas comparison case: trapezoidal integration of sin(x) over [0,π] with n=30.",
        gas,
        inHex: "f=sin, [0,π], n=30",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 23: Shared Low-Gas Comparison (sin(x) on [0,π], n=30)", async function () {
      t++;
      const expected = await qInt(2);

      await touchGas(harness, "simpson13", [target, selSin, q0, qPI, 30]);
      const gas = await estimateGas(harness, "simpson13", [target, selSin, q0, qPI, 30]);
      const out = await harness.simpson13(target, selSin, q0, qPI, 30);

      await expectClose(harness, out, expected, 100_000_000n);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Shared low-gas comparison case: Simpson 1/3 integration of sin(x) over [0,π] with n=30.",
        gas,
        inHex: "f=sin, [0,π], n=30",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 24: Shared Low-Gas Comparison (sin(x) on [0,π], n=30)", async function () {
      t++;
      const expected = await qInt(2);

      await touchGas(harness, "simpson38", [target, selSin, q0, qPI, 30]);
      const gas = await estimateGas(harness, "simpson38", [target, selSin, q0, qPI, 30]);
      const out = await harness.simpson38(target, selSin, q0, qPI, 30);

      await expectClose(harness, out, expected, 100_000_000n);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Shared low-gas comparison case: Simpson 3/8 integration of sin(x) over [0,π] with n=30.",
        gas,
        inHex: "f=sin, [0,π], n=30",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 25: Shared Comparison Scenario (sin(x) on [0,π], n=300)", async function () {
      t++;
      const expected = await qInt(2);

      await touchGas(harness, "trapezoidal", [target, selSin, q0, qPI, 300]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSin, q0, qPI, 300]);
      const out = await harness.trapezoidal(target, selSin, q0, qPI, 300);

      await expectClose(harness, out, expected, 50_000_000n);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Shared comparison case: trapezoidal integration of sin(x) over [0,π] with n=300. Exact integral is 2.",
        gas,
        inHex: "f=sin, [0,π], n=300",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 26: Shared Comparison Scenario (sin(x) on [0,π], n=300)", async function () {
      t++;
      const expected = await qInt(2);

      await touchGas(harness, "simpson13", [target, selSin, q0, qPI, 300]);
      const gas = await estimateGas(harness, "simpson13", [target, selSin, q0, qPI, 300]);
      const out = await harness.simpson13(target, selSin, q0, qPI, 300);

      await expectClose(harness, out, expected, 50_000_000n);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Shared comparison case: Simpson 1/3 integration of sin(x) over [0,π] with n=300. Exact integral is 2.",
        gas,
        inHex: "f=sin, [0,π], n=300",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 27: Shared Comparison Scenario (sin(x) on [0,π], n=300)", async function () {
      t++;
      const expected = await qInt(2);

      await touchGas(harness, "simpson38", [target, selSin, q0, qPI, 300]);
      const gas = await estimateGas(harness, "simpson38", [target, selSin, q0, qPI, 300]);
      const out = await harness.simpson38(target, selSin, q0, qPI, 300);

      await expectClose(harness, out, expected, 50_000_000n);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Shared comparison case: Simpson 3/8 integration of sin(x) over [0,π] with n=300. Exact integral is 2.",
        gas,
        inHex: "f=sin, [0,π], n=300",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });
  });
});