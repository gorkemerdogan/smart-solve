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
  qAdd(a: string, b: string): Promise<string>;
  PI(): Promise<string>;
  
  // Helpers for conversions
  toFloat(q: string): Promise<bigint>;
  fromFloat(n: bigint): Promise<string>;

  // Methods
  trapezoidal(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson13(target: string, selector: string, a: string, b: string, n: number): Promise<string>;
  simpson38(target: string, selector: string, a: string, b: string, n: number): Promise<string>;

  // Functions to integrate
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

const TOL_APPROX = 1_000_000n; // Allow ~1e-6 for approximations.
const TOL_EXACT = 100n;        // For exact methods (like Simpson on cubic), allow tiny rounding noise.

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

/// Check if values are close (for approximations)
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

/// Check if values are identical (for exact polynomials)
async function expectEqual(actualHex: string, expectedHex: string) {
    expect(actualHex).to.equal(expectedHex);
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
    QZERO = q0;

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
      
      const expected = await harness.qFromFrac(1, 2); // Exact integral of x from 0 to 1 is 0.5. Trapezoidal is exact for linear.

      await touchGas(harness, "trapezoidal", [target, selLinear, q0, q1, 10]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLinear, q0, q1, 10]);
      const out = await harness.trapezoidal(target, selLinear, q0, q1, 10);

      await expectClose(harness, out, expected, TOL_EXACT); // Should be very close to exact.

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on linear f(x)=x over [0,1]. Analytic integral is 0.5.",
        gas,
        inHex: "f=x, [0,1]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 2: Smooth Quadratic (f(x)=x^2 on [0,1], n=300)", async function () {
      t++;
      
      const expected = await harness.qFromFrac(1, 3); // Analytic integral of x^2 from 0 to 1 is 1/3.

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 300]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 300]);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 300);

      await expectClose(harness, out, expected, TOL_APPROX);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on f(x)=x^2 over [0,1]. n=300 approximates 1/3.",
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
      const expected = await qInt(50); // 5 * 10

      await touchGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const out = await harness.trapezoidal(target, selConst5, q0, b, 50);

      await expectEqual(out, expected);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Trapezoidal rule on constant f(x)=5 over [0,10]. Result is exactly 50.",
        gas,
        inHex: "f=5, [0,10]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 4: Trig Function (sin(x) on [0,π], n=200)", async function () {
      t++;
      
      const expected = await qInt(2); // Integral of sin(x) from 0 to pi is -cos(pi) - (-cos(0)) = 1 - (-1) = 2.

      await touchGas(harness, "trapezoidal", [target, selSin, q0, qPI, 200]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSin, q0, qPI, 200]);
      const out = await harness.trapezoidal(target, selSin, q0, qPI, 200);

      await expectClose(harness, out, expected, TOL_APPROX);

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

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const out = await harness.trapezoidal(target, selSquare, q0, q0, 100);

      await expectEqual(out, expected);

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

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 50]);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 50);

      await expectClose(harness, out, expected, TOL_APPROX);

      printBlockRegular({
        t,
        method: "trapezoidal",
        explanation: "Low iteration (n=50) check for x^2. Used for gas benchmarking.",
        gas,
        inHex: "[0,1]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });
  });

  // ------------------------------------------------------------
  //  Simpson 1/3 Rule
  // ------------------------------------------------------------

  describe("Method 2: Simpson 1/3 Rule", function () {

    it("Test 7: Quadratic Exactness (f=x^2 on [0,1], n=10)", async function () {
      t++;
      
      const expected = await harness.qFromFrac(1, 3); // Simpson 1/3 is exact for degree <= 3. x^2 integral is 1/3.

      await touchGas(harness, "simpson13", [target, selSquare, q0, q1, 10]);
      const gas = await estimateGas(harness, "simpson13", [target, selSquare, q0, q1, 10]);
      const out = await harness.simpson13(target, selSquare, q0, q1, 10);

      await expectEqual(out, expected);

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

    it("Test 8: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      
      const expected = await harness.qFromFrac(1, 4); // Analytic integral x^3 on [0,1] is 1/4 = 0.25.

      await touchGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const out = await harness.simpson13(target, selCube, q0, q1, 12);

      await expectEqual(out, expected);

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

    it("Test 9: Exp Approx (Linear proxy on [0,1], n=20)", async function () {
      t++;
      
      const expected = await harness.qFromFrac(1, 2); // f_linear integral [0,1] is 0.5.

      await touchGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const gas = await estimateGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const out = await harness.simpson13(target, selLinear, q0, q1, 20);

      await expectEqual(out, expected); 

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

    it("Test 10: Inverse Function (1/x on [1,2], n=40)", async function () {
      t++;

      const ln2_js = Math.log(2);
      const expected = await harness.fromFloat(BigInt(Math.round(ln2_js * 1e12)));

      await touchGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const gas = await estimateGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const out = await harness.simpson13(target, selInv, q1, q2, 40);

      await expectClose(harness, out, expected, TOL_APPROX);

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

    it("Test 11: Revert on Invalid N (Odd)", async function () {
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
  });

  // ------------------------------------------------------------
  //  Simpson 3/8 Rule
  // ------------------------------------------------------------

  describe("Method 3: Simpson 3/8 Rule", function () {

    it("Test 12: Quadratic Exactness (f=x^2 on [0,1], n=9)", async function () {
      t++;

      const expected = await harness.qFromFrac(1, 3);

      await touchGas(harness, "simpson38", [target, selSquare, q0, q1, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, q1, 9]);
      const out = await harness.simpson38(target, selSquare, q0, q1, 9);

      await expectEqual(out, expected);

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

    it("Test 13: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;

      const expected = await harness.qFromFrac(1, 4);

      await touchGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const out = await harness.simpson38(target, selCube, q0, q1, 12);

      await expectEqual(out, expected); 

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

    it("Test 14: Inverse Function (1/x on [1,2], n=12)", async function () {
      t++;
      const ln2_js = Math.log(2);
      const expected = await harness.fromFloat(BigInt(Math.round(ln2_js * 1e12)));

      await touchGas(harness, "simpson38", [target, selInv, q1, q2, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selInv, q1, q2, 12]);
      const out = await harness.simpson38(target, selInv, q1, q2, 12);

      await expectClose(harness, out, expected, TOL_APPROX);

      printBlockRegular({
        t,
        method: "simpson38",
        explanation: "Simpson 3/8 integration of 1/x over [1,2]. Approximates ln(2).",
        gas,
        inHex: "[1,2], n=12",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 15: Degenerate Interval ([5,5], n=3)", async function () {
      t++;
      const five = await qInt(5);
      const expected = QZERO;

      await touchGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const out = await harness.simpson38(target, selSquare, five, five, 3);

      await expectEqual(out, expected);

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

    it("Test 16: Revert on Invalid N (Not Multiple of 3)", async function () {
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
  });

  // ------------------------------------------------------------
  //  Edge Cases
  // ------------------------------------------------------------

  describe("Edge Cases", function () {

    it("Test 17: Tiny Magnitudes (1e-30·x)", async function () {
      t++;
      // If f(x) = 1e-30 * x, Integral 0->1 is 1e-30 * (1/2) = 5e-31
      const expected = await harness.qFromFrac(5, BigInt("1" + "0".repeat(31))); // 5/1e31

      await touchGas(harness, "trapezoidal", [target, selTiny, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selTiny, q0, q1, 30]);
      const out = await harness.trapezoidal(target, selTiny, q0, q1, 30);

      await expectClose(harness, out, expected, TOL_EXACT); // Expect 0 (or close to 0)

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

    it("Test 18: Large Magnitudes (1e20·x)", async function () {
      t++;
      // f(x) = 1e20 * x. Integral 0->1 is 0.5e20 = 5e19.
      const largeNum = BigInt("1" + "0".repeat(20));
      const expected = await harness.qFromFrac(largeNum, 2);

      await touchGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const out = await harness.trapezoidal(target, selLarge, q0, q1, 30);

      await expectClose(harness, out, expected, TOL_APPROX);

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

    it("Test 19: Piecewise Discontinuous", async function () {
      t++;
      // f_piecewise: x < 1 ? 1 : 3.
      // Integral 0->2 = Integral(0->1 of 1) + Integral(1->2 of 3) = 1 + 3 = 4.
      const expected = await qInt(4);

      await touchGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const gas = await estimateGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const out = await harness.simpson13(target, selPiecewise, q0, q2, 60);

      await expectClose(harness, out, expected, TOL_APPROX);

      printBlockRegular({
        t,
        method: "simpson13",
        explanation: "Piecewise jump. Integral 0->2 is 4. Tests robustness at discontinuity.",
        gas,
        inHex: "[0,2]",
        expectedHex: expected,
        outHex: out,
        expectedDec: await fmt(harness, expected),
        outDec: await fmt(harness, out)
      });
    });

    it("Test 20: Narrow Interval ([0,1e-12])", async function () {
      t++;
      const tiny = await harness.qFromFrac(1, BigInt("1000000000000"));
      // Integral x^2 from 0 to 1e-12 is (1e-12)^3 / 3 = 1e-36 / 3.
      const expected = await harness.qFromFrac(1, BigInt("3" + "0".repeat(36)));

      await touchGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const out = await harness.simpson38(target, selSquare, q0, tiny, 9);

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

    it("Test 21: Revert on Reversed Bounds", async function () {
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
});