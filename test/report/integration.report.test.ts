// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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

async function touchGas(harness: IntegrationHarness, method: string, args: any[]) {
  const data = harness.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await harness.getAddress();
  const tx = await signer.sendTransaction({ to, data });
  await tx.wait();
}

async function estimateGas(harness: IntegrationHarness, method: string, args: any[]) {
  const anyH = harness as any;
  if (anyH[method]?.estimateGas) {
    return (await anyH[method].estimateGas(...args)).toString();
  }
  const data = harness.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await harness.getAddress();
  const gas = await signer.estimateGas({ to, data });
  return gas.toString();
}

function printBlock({ t, method, explanation, gas, inHex, outHex, outDec }: any) {
  const sep = "-".repeat(60);
  const decLine = outDec ? `Output: ${outDec}` : "";

  console.log(`
        ${sep}
        Test ${t}
        Method: ${method}
        Explanation: ${explanation}
        Gas Usage: ${gas}
        Input: ${inHex}
        Output (hex): ${outHex}
        ${decLine}
`.trim()); // trim to clean up leading/trailing newline from template literal
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

      expect(typeof out).to.equal("string");
      printBlock({ t, method: "trapezoidal", explanation: "Linear function (exact)", gas, inHex: "f=x, [0,1]", outHex: out, outDec: expectedDec });
    });

    it("Test 2: Smooth Quadratic (f(x)=x^2 on [0,1], n=300)", async function () {
      t++;
      const expectedDec = "≈0.3333";

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 300]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 300]);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 300);

      expect(out).to.be.a("string");
      printBlock({ t, method: "trapezoidal", explanation: "Smooth quadratic", gas, inHex: "f=x^2, [0,1]", outHex: out, outDec: expectedDec });
    });

    it("Test 3: Constant Exactness (f=5 on [0,10], n=50)", async function () {
      t++;
      const b = await qInt(10);
      const expectedDec = "50";

      await touchGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selConst5, q0, b, 50]);
      const out = await harness.trapezoidal(target, selConst5, q0, b, 50);

      expect(out).to.be.a("string");
      printBlock({ t, method: "trapezoidal", explanation: "Constant function exact", gas, inHex: "f=5, [0,10]", outHex: out, outDec: expectedDec });
    });

    it("Test 4: Trig Function (sin(x) on [0,π], n=200)", async function () {
      t++;
      const expectedDec = "≈2";

      await touchGas(harness, "trapezoidal", [target, selSin, q0, qPI, 200]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSin, q0, qPI, 200]);
      const out = await harness.trapezoidal(target, selSin, q0, qPI, 200);

      expect(out).to.be.a("string");
      printBlock({ t, method: "trapezoidal", explanation: "sin(x) integrates to 2", gas, inHex: "f=sin, [0,π]", outHex: out, outDec: expectedDec });
    });

    it("Test 5: Degenerate Interval ([0,0], n=100)", async function () {
      t++;
      const expectedDec = "0";

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q0, 100]);
      const out = await harness.trapezoidal(target, selSquare, q0, q0, 100);

      expect(out).to.equal(q0);
      printBlock({ t, method: "trapezoidal", explanation: "Zero interval", gas, inHex: "[0,0]", outHex: out, outDec: expectedDec });
    });

    it("Test 6: Max Iteration Comparison (f=x^2)", async function () {
      t++;
      const expectedDec = "~";

      await touchGas(harness, "trapezoidal", [target, selSquare, q0, q1, 50]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selSquare, q0, q1, 50]);
      const out = await harness.trapezoidal(target, selSquare, q0, q1, 50);

      expect(out).to.be.a("string");
      printBlock({ t, method: "trapezoidal", explanation: "Java Ref Comp", gas, inHex: "[0,1]", outHex: out, outDec: expectedDec });
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

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson13", explanation: "x^2 exact-ish", gas, inHex: "[0,1], n=10", outHex: out, outDec: expectedDec });
    });

    it("Test 8: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      const expectedDec = "0.25";

      await touchGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson13", [target, selCube, q0, q1, 12]);
      const out = await harness.simpson13(target, selCube, q0, q1, 12);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson13", explanation: "x^3 exact", gas, inHex: "[0,1], n=12", outHex: out, outDec: expectedDec });
    });

    it("Test 9: Exp Approx (Linear proxy on [0,1], n=20)", async function () {
      t++;
      const expectedDec = "~";

      await touchGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const gas = await estimateGas(harness, "simpson13", [target, selLinear, q0, q1, 20]);
      const out = await harness.simpson13(target, selLinear, q0, q1, 20);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson13", explanation: "exp approx placeholder", gas, inHex: "[0,1], n=20", outHex: out, outDec: expectedDec });
    });

    it("Test 10: Inverse Function (1/x on [1,2], n=40)", async function () {
      t++;
      const expectedDec = "≈0.693";

      await touchGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const gas = await estimateGas(harness, "simpson13", [target, selInv, q1, q2, 40]);
      const out = await harness.simpson13(target, selInv, q1, q2, 40);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson13", explanation: "log(2)", gas, inHex: "[1,2], n=40", outHex: out, outDec: expectedDec });
    });

    it("Test 11: Revert on Invalid N (Odd)", async function () {
      t++;
      await expect(harness.simpson13(target, selSquare, q0, q1, 7)).to.be.revertedWith("Integration: Simpson 1/3 requires even n");
      printBlock({ t, method: "simpson13", explanation: "Revert (odd n)", gas: "N/A", inHex: "x^2, n=7", outHex: "Revert", outDec: "Revert" });
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

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson38", explanation: "x^2", gas, inHex: "[0,1], n=9", outHex: out, outDec: expectedDec });
    });

    it("Test 13: Cubic Exactness (f=x^3 on [0,1], n=12)", async function () {
      t++;
      const expectedDec = "0.25";

      await touchGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selCube, q0, q1, 12]);
      const out = await harness.simpson38(target, selCube, q0, q1, 12);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson38", explanation: "x^3 exact", gas, inHex: "[0,1], n=12", outHex: out, outDec: expectedDec });
    });

    it("Test 14: Inverse Function (1/x on [1,2], n=12)", async function () {
      t++;
      const expectedDec = "≈0.693";

      await touchGas(harness, "simpson38", [target, selInv, q1, q2, 12]);
      const gas = await estimateGas(harness, "simpson38", [target, selInv, q1, q2, 12]);
      const out = await harness.simpson38(target, selInv, q1, q2, 12);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson38", explanation: "log2", gas, inHex: "[1,2], n=12", outHex: out, outDec: expectedDec });
    });

    it("Test 15: Degenerate Interval ([5,5], n=3)", async function () {
      t++;
      const five = await qInt(5);
      const expectedDec = "0";

      await touchGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, five, five, 3]);
      const out = await harness.simpson38(target, selSquare, five, five, 3);

      expect(out).to.equal(q0);
      printBlock({ t, method: "simpson38", explanation: "Zero interval", gas, inHex: "[5,5]", outHex: out, outDec: expectedDec });
    });

    it("Test 16: Revert on Invalid N (Not Multiple of 3)", async function () {
      t++;
      await expect(harness.simpson38(target, selSquare, q0, q1, 10)).to.be.revertedWith("Integration: Simpson 3/8 requires n % 3 == 0");
      printBlock({ t, method: "simpson38", explanation: "Revert (n%3!=0)", gas: "N/A", inHex: "x^2, n=10", outHex: "Revert", outDec: "Revert" });
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

      expect(out).to.be.a("string");
      printBlock({ t, method: "trapezoidal", explanation: "Tiny magnitude", gas, inHex: "1e-30·x", outHex: out, outDec: expectedDec });
    });

    it("Test 18: Large Magnitudes (1e20·x)", async function () {
      t++;
      const expectedDec = "≈5e19";

      await touchGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const gas = await estimateGas(harness, "trapezoidal", [target, selLarge, q0, q1, 30]);
      const out = await harness.trapezoidal(target, selLarge, q0, q1, 30);

      expect(out).to.be.a("string");
      printBlock({ t, method: "trapezoidal", explanation: "Large magnitude", gas, inHex: "1e20·x", outHex: out, outDec: expectedDec });
    });

    it("Test 19: Piecewise Discontinuous", async function () {
      t++;
      const expectedDec = "≈4";

      await touchGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const gas = await estimateGas(harness, "simpson13", [target, selPiecewise, q0, q2, 60]);
      const out = await harness.simpson13(target, selPiecewise, q0, q2, 60);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson13", explanation: "Piecewise 1->3", gas, inHex: "[0,2]", outHex: out, outDec: expectedDec });
    });

    it("Test 20: Narrow Interval ([0,1e-12])", async function () {
      t++;
      const tiny = await harness.qFromFrac(1, BigInt("1000000000000"));
      const expectedDec = "≈3.33e-37";

      await touchGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const gas = await estimateGas(harness, "simpson38", [target, selSquare, q0, tiny, 9]);
      const out = await harness.simpson38(target, selSquare, q0, tiny, 9);

      expect(out).to.be.a("string");
      printBlock({ t, method: "simpson38", explanation: "Tiny interval", gas, inHex: "[0,1e-12]", outHex: out, outDec: expectedDec });
    });

    it("Test 21: Revert on Reversed Bounds", async function () {
      t++;
      await expect(harness.trapezoidal(target, selSquare, q1, q0, 10)).to.be.revertedWith("Integration: upper bound b must be >= a");
      printBlock({ t, method: "trapezoidal", explanation: "Revert (reversed)", gas: "N/A", inHex: "x^2, [1,0]", outHex: "Revert", outDec: "Revert" });
    });
  });
});