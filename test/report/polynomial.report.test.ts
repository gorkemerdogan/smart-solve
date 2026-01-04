// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";
import { string } from "hardhat/internal/core/params/argumentTypes";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type PolynomialHarness = Contract & {
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;

  evaluateHorners(coeffs: string[], x: string): Promise<string>;
  evaluateWithDerivative(coeffs: string[], x: string): Promise<[string, string] & { px: string; dpx: string }>;
  derivative(coeffs: string[]): Promise<string[]>;

  add(a: string[], b: string[]): Promise<string[]>;
  sub(a: string[], b: string[]): Promise<string[]>;
  mulScalar(a: string[], k: string): Promise<string[]>;
  mul(a: string[], b: string[]): Promise<string[]>;

  integral(a: string[], C: string): Promise<string[]>;
  degree(a: string[]): Promise<bigint>;
  trimTrailingZeros(a: string[]): Promise<string[]>;
  syntheticDivide(coeffs: string[], root: string): Promise<[string[], string]>;
  evalHornerMonic(lowerCoeffs: string[], x: string): Promise<string>;

  getAddress(): Promise<string>;
};

// ------------------------------------------------------------
//  JS Numeric Mirrors (Verification)
// ------------------------------------------------------------

function evalHornerNum(c: number[], x: number) { let y = c[c.length - 1]; for (let i = c.length - 2; i >= 0; i--) y = y * x + c[i]; return y; }
function evalHornerWithDerivNum(c: number[], x: number) { let p = c[c.length - 1], dp = 0; for (let i = c.length - 2; i >= 0; i--) { dp = dp * x + p; p = p * x + c[i]; } return [p, dp]; }
function derivativeCoeffsNum(c: number[]) { if (c.length <= 1) return [0]; const d = new Array(c.length - 1); for (let i = 1; i < c.length; i++)d[i - 1] = c[i] * i; return d; }
function addNum(a: number[], b: number[]) { const n = Math.max(a.length, b.length); const o = new Array(n).fill(0); for (let i = 0; i < n; i++)o[i] = (a[i] ?? 0) + (b[i] ?? 0); return o; }
function subNum(a: number[], b: number[]) { const n = Math.max(a.length, b.length); const o = new Array(n).fill(0); for (let i = 0; i < n; i++)o[i] = (a[i] ?? 0) - (b[i] ?? 0); return o; }
function mulScalarNum(a: number[], k: number) { return a.map(v => v * k); }
function mulNum(a: number[], b: number[]) { const o = new Array(a.length + b.length - 1).fill(0); for (let i = 0; i < a.length; i++)for (let j = 0; j < b.length; j++)o[i + j] += a[i] * b[j]; return o; }
function integralNum(a: number[], C: number) { const o = new Array(a.length + 1).fill(0); o[0] = C; for (let i = 0; i < a.length; i++)o[i + 1] = a[i] / (i + 1); return o; }
function degreeNum(a: number[]) { for (let i = a.length - 1; i >= 0; i--)if (a[i] !== 0) return i; return 0; }
function trimTrailingZerosNum(a: number[]) { return a.slice(0, degreeNum(a) + 1); }
function syntheticDivideNum(c: number[], r: number) { const n = c.length; if (n === 0) return { q: [0], r: 0 }; if (n === 1) return { q: [0], r: c[0] }; const q = new Array(n - 1).fill(0); q[n - 2] = c[n - 1]; for (let i = n - 1; i > 1; i--)q[i - 2] = c[i - 1] + r * q[i - 1]; const rem = c[0] + r * q[0]; return { q, r: rem }; }
function evalHornerMonicNum(lower: number[], x: number) { let acc = 1; for (let i = lower.length - 1; i >= 0; i--)acc = acc * x + lower[i]; return acc; }

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

/**
 * @notice Formats an array into a readable string showing the first and last N elements.
 * @param  arr   Array of any type to be formatted.
 * @param  limit The number of elements to show at the start and the end. Default: 3.
 * @return A string representation of the array: "[head..., ..., tail] (len=x)".
 */
function headTail(arr: any[], limit = 3) {
  if (arr.length <= limit * 2) return `[${arr.join(", ")}] (len=${arr.length})`;
  const head = arr.slice(0, limit).join(", ");
  const tail = arr.slice(-limit).join(", ");
  return `[${head}, …, ${tail}] (len=${arr.length})`;
}

/// Converts a 128-bit quad-precision hex string from the contract into a JS number by dividing the raw integer value by the scale. 
async function fromQuad(h: PolynomialHarness, hex: string): Promise<number> {
  const raw: bigint = await h.toFloat(hex);
  return Number(raw) / 1e12;
}

async function fromQuadArray(h: PolynomialHarness, hexArr: string[]): Promise<number[]> {
  const decArr: number[] = [];
  for (const hex of hexArr) {
    decArr.push(await fromQuad(h, hex));
  }
  return decArr;
}

/**
 * @notice Formats an array of hex strings into a single comma-separated string.
 * @param  arr Array of strings.
 * @return A string wrapped in square brackets: "[0x1, 0x2]".
 */
const fmtHexArr = (arr: string[]) => `[${arr.join(", ")}]`;

/**
 * @notice Formats an array of numbers or strings into a single comma-separated string.
 * @param  arr Array of values.
 * @return A string wrapped in square brackets: "[100, 200]".
 */
const fmtDecArr = (arr: (number | string)[]) => `[${arr.join(", ")}]`;

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Polynomial Library - Operations & Calculus", function () {
  let harness: PolynomialHarness;
  let t = 0;

  let QZERO: string;

  // Helper Wrappers
  const qInt = async (n: number | string) => await harness.qFromInt(BigInt(n));
  const qFrac = async (n: number | string, d: number | string) => await harness.qFromFrac(BigInt(n), BigInt(d));

  before(async () => {
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();

    const HarnessFactory = await ethers.getContractFactory("PolynomialHarness", {
      libraries: { "contracts/libraries/MathLib.sol:MathLib": await math.getAddress() }
    });

    harness = (await HarnessFactory.deploy()) as unknown as PolynomialHarness;
    await harness.waitForDeployment();

    QZERO = await harness.fromFloat(0n);
  });

  // ------------------------------------------------------------
  //  Core Evaluation
  // ------------------------------------------------------------

  describe("Section 1: Evaluation Methods", function () {

    it("Test 1: Horner Evaluation (3x^2 + 2x + 5 at x=5)", async function () {
      t++;
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const x = await qInt(5);
      const expectedDec = evalHornerNum([5, 2, 3], 5);

      await touchGas(harness, "evaluateHorners", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateHorners", [coeffs, x]);
      const out = await harness.evaluateHorners(coeffs, x);

      const outDec = await fromQuad(harness, out);

      printBlockRegular({
        t,
        method: "evaluateHorners",
        explanation: "Horner evaluation of 3x^2 + 2x + 5 at x=5. Verifies basic polynomial pipeline against JS numeric mirror.",
        inHex: "f=[5,2,3], x=5",
        expectedHex: "N/A (analytic double result)",
        outHex: out,
        expectedDec: `${expectedDec}`,
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 2: Eval + Derivative (3x^2 + 2x + 5 at x=1.2)", async function () {
      t++;
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const x = await qFrac(6, 5); // 1.2
      const [pDec, dpDec] = evalHornerWithDerivNum([5, 2, 3], 1.2);

      await touchGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const { px, dpx } = await harness.evaluateWithDerivative(coeffs, x);

      const pOut = await fromQuad(harness, px);
      const dpOut = await fromQuad(harness, dpx);

      printBlockRegular({
        t,
        method: "evaluateWithDerivative",
        explanation: "One-pass Horner evaluation and derivative for 3x^2 + 2x + 5 at x=1.2. Checks coupled (p, p') output.",
        inHex: "x=1.2",
        expectedHex: "N/A (p, p' in quad form)",
        outHex: `p=${px}, p'=${dpx}`,
        expectedDec: `p=${pDec}, p'=${dpDec}`,
        outDec: `p=${pOut}, p'=${dpOut}`,
        gas,
      });
    });

    it("Test 3: Horner Monic (x^3 + 4x^2 + 0.5x + 2 at x=5)", async function () {
      t++;
      const lower = [await qInt(2), await qFrac(1, 2), await qInt(4)];
      const x = await qInt(5);
      const expectedDec = evalHornerMonicNum([2, 0.5, 4], 5);

      await touchGas(harness, "evalHornerMonic", [lower, x]);
      const gas = await estimateGas(harness, "evalHornerMonic", [lower, x]);
      const out = await harness.evalHornerMonic(lower, x);

      const outDec = await fromQuad(harness, out);

      printBlockRegular({
        t,
        method: "evalHornerMonic",
        explanation: "Horner evaluation with implicit leading coefficient 1 (monic) using lower=[2,0.5,4] at x=5.",
        inHex: "lower=[2, 0.5, 4], x=5",
        expectedHex: "N/A (monic analytic result)",
        outHex: out,
        expectedDec: `${expectedDec}`,
        outDec: `${outDec}`,
        gas,
      });
    });
  });

  // ------------------------------------------------------------
  //  Arithmetic Operations
  // ------------------------------------------------------------

  describe("Section 2: Arithmetic", function () {

    it("Test 4: Addition ((x+2) + (3x+4))", async function () {
      t++;
      const p1 = [await qInt(2), await qInt(1)];
      const p2 = [await qInt(4), await qInt(3)];
      const expected = fmtDecArr(addNum([2, 1], [4, 3]));

      await touchGas(harness, "add", [p1, p2]);
      const gas = await estimateGas(harness, "add", [p1, p2]);
      const out = await harness.add(p1, p2);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "add",
        explanation: "Coefficient-wise polynomial addition: (x+2)+(3x+4) → (4x+6) in coefficient form.",
        inHex: "[2,1] + [4,3]",
        expectedHex: "N/A (computed via JS mirror)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 5: Subtraction ((x+2) - (3x+4))", async function () {
      t++;
      const p1 = [await qInt(2), await qInt(1)];
      const p2 = [await qInt(4), await qInt(3)];
      const expected = fmtDecArr(subNum([2, 1], [4, 3]));

      await touchGas(harness, "sub", [p1, p2]);
      const gas = await estimateGas(harness, "sub", [p1, p2]);
      const out = await harness.sub(p1, p2);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "sub",
        explanation: "Coefficient-wise subtraction: (x+2)-(3x+4) → (-2x-2) with proper length handling.",
        inHex: "[2,1] - [4,3]",
        expectedHex: "N/A (computed via JS mirror)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 6: Scalar Multiplication (2 * (4x^2 + 3x + 1))", async function () {
      t++;
      const p = [await qInt(1), await qInt(3), await qInt(4)];
      const k = await qInt(2);
      const expected = fmtDecArr(mulScalarNum([1, 3, 4], 2));

      await touchGas(harness, "mulScalar", [p, k]);
      const gas = await estimateGas(harness, "mulScalar", [p, k]);
      const out = await harness.mulScalar(p, k);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "mulScalar",
        explanation: "Scalar multiplication of coefficient vector by 2, checking each entry is doubled.",
        inHex: "2 * [1,3,4]",
        expectedHex: "N/A (coeffs doubled in quad)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 7: Multiplication (Convolution) ((x+1)*(x+1))", async function () {
      t++;
      const p1 = [await qInt(1), await qInt(1)];
      const p2 = [await qInt(1), await qInt(1)];
      const expected = fmtDecArr(mulNum([1, 1], [1, 1]));

      await touchGas(harness, "mul", [p1, p2]);
      const gas = await estimateGas(harness, "mul", [p1, p2]);
      const out = await harness.mul(p1, p2);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "mul",
        explanation: "Polynomial convolution for (x+1)^2; ensures classic [1,2,1] coefficient pattern.",
        inHex: "(x+1)^2",
        expectedHex: "N/A (convolution via JS mirror)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });
  });

  // ------------------------------------------------------------
  //  Calculus & Utils
  // ------------------------------------------------------------

  describe("Section 3: Calculus & Utilities", function () {

    it("Test 8: Derivative (of 3x^2 + 2x + 5)", async function () {
      t++;
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const expected = fmtDecArr(derivativeCoeffsNum([5, 2, 3]));

      await touchGas(harness, "derivative", [coeffs]);
      const gas = await estimateGas(harness, "derivative", [coeffs]);
      const out = await harness.derivative(coeffs);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "derivative",
        explanation: "Coefficient-wise derivative using power rule on 3x^2+2x+5 → 6x+2.",
        inHex: "[5,2,3]",
        expectedHex: "N/A (derived in JS mirror)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 9: Integral (of 6x + 12, C=5)", async function () {
      t++;
      const p = [await qInt(12), await qInt(6)];
      const C = await qInt(5);
      const expected = fmtDecArr(integralNum([12, 6], 5));

      await touchGas(harness, "integral", [p, C]);
      const gas = await estimateGas(harness, "integral", [p, C]);
      const out = await harness.integral(p, C);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "integral",
        explanation: "Indefinite integral of 6x+12 with constant C=5 to verify term-by-term integration.",
        inHex: "6x+12, C=5",
        expectedHex: "N/A (integrated form in quad)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 10: Synthetic Division (P / (x - 1.5))", async function () {
      t++;
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const root = await qFrac(3, 2);
      const { q: rQ, r: rR } = syntheticDivideNum([5, 2, 3], 1.5);

      await touchGas(harness, "syntheticDivide", [coeffs, root]);
      const gas = await estimateGas(harness, "syntheticDivide", [coeffs, root]);
      const [qHex, rHex] = await harness.syntheticDivide(coeffs, root);

      const expectedDec = `Q=${fmtDecArr(rQ)}, R=${rR}`;

      const qOutDec = await fromQuadArray(harness, qHex);
      const rOutDec = await fromQuad(harness, rHex);

      printBlockRegular({
        t,
        method: "syntheticDivide",
        explanation: "Ruffini-style synthetic division of P=[5,2,3] by (x-1.5). Checks quotient and remainder pair.",
        inHex: "P=[5,2,3], r=1.5",
        expectedHex: "N/A (Q,R in quad form)",
        outHex: `Q=${fmtHexArr(qHex)}, R=${rHex}`,
        expectedDec,
        outDec: `Q=${fmtDecArr(qOutDec)}, R=${rOutDec}`,
        gas,
      });
    });

    it("Test 11: Degree Check", async function () {
      t++;
      const p = [await qInt(5), await qInt(2), await qInt(0), await qInt(0)];
      const expected = degreeNum([5, 2, 0, 0]);

      await touchGas(harness, "degree", [p]);
      const gas = await estimateGas(harness, "degree", [p]);
      const out = await harness.degree(p);

      printBlockRegular({
        t,
        method: "degree",
        explanation: "Degree computation: highest non-zero index of [5,2,0,0] should be 1.",
        inHex: "[5,2,0,0]",
        expectedHex: "N/A (scalar degree)",
        outHex: out.toString(),
        expectedDec: `${expected}`,
        outDec: `${Number(out)}`,
        gas,
      });
    });

    it("Test 12: Trim Trailing Zeros", async function () {
      t++;
      const p = [await qInt(5), await qInt(2), await qInt(0), await qInt(0)];
      const expected = fmtDecArr(trimTrailingZerosNum([5, 2, 0, 0]));

      await touchGas(harness, "trimTrailingZeros", [p]);
      const gas = await estimateGas(harness, "trimTrailingZeros", [p]);
      const out = await harness.trimTrailingZeros(p);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "trimTrailingZeros",
        explanation: "Trims superfluous high-degree zeros to canonicalize polynomial representation.",
        inHex: "[5,2,0,0]",
        expectedHex: "N/A (canonical form in quad)",
        outHex: fmtHexArr(out),
        expectedDec: expected,
        outDec: `${outDec}`,
        gas,
      });
    });
  });

  // ------------------------------------------------------------
  //  Edge Cases
  // ------------------------------------------------------------

  describe("Section 4: Edge Cases", function () {

    it("Test 13: Empty Horners (Eval -> 0)", async function () {
      t++;
      const coeffs: string[] = [];
      const x = await qInt(7);

      await touchGas(harness, "evaluateHorners", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateHorners", [coeffs, x]);
      const out = await harness.evaluateHorners(coeffs, x);

      const outDec = await fromQuad(harness, out);

      printBlockRegular({
        t,
        method: "evaluateHorners",
        explanation: "Evaluating an empty coefficient array should behave like polynomial 0 and return 0.",
        inHex: "[], x=7",
        expectedHex: `${QZERO}`,
        outHex: out,
        expectedDec: "0",
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 14: Constant Polynomial (Eval -> C)", async function () {
      t++;
      const coeffs = [await qInt(7)];
      const x = await qInt(123);

      await touchGas(harness, "evaluateHorners", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateHorners", [coeffs, x]);
      const out = await harness.evaluateHorners(coeffs, x);

      const outDec = await fromQuad(harness, out);

      printBlockRegular({
        t,
        method: "evaluateHorners",
        explanation: "Evaluating constant polynomial [7] at any x must return the same constant value.",
        inHex: "[7]",
        expectedHex: coeffs[0],
        outHex: out,
        expectedDec: "7",
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 15: Constant Derivative (p' -> 0)", async function () {
      t++;
      const coeffs = [await qInt(5)];
      const x = await qInt(42);

      await touchGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const { dpx } = await harness.evaluateWithDerivative(coeffs, x);

      const dpxDec = await fromQuad(harness, dpx);

      printBlockRegular({
        t,
        method: "evaluateWithDerivative",
        explanation: "Derivative of a constant polynomial is identically 0; checks dpx output only.",
        inHex: "[5]",
        expectedHex: `${QZERO}`,
        outHex: `p'=${dpx}`,
        expectedDec: "p'=0",
        outDec: `p'=${dpxDec}`,
        gas,
      });
    });

    it("Test 16: Zero Scalar Mul", async function () {
      t++;
      const p = [await qInt(5), await qInt(4)];
      const k = await qInt(0);

      await touchGas(harness, "mulScalar", [p, k]);
      const gas = await estimateGas(harness, "mulScalar", [p, k]);
      const out = await harness.mulScalar(p, k);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "mulScalar",
        explanation: "Scaling any polynomial by 0 should yield the all-zero polynomial.",
        inHex: "k=0",
        expectedHex: "N/A (all coeffs zero in quad)",
        outHex: fmtHexArr(out),
        expectedDec: "[0]",
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 17: Mul by Empty", async function () {
      t++;
      const a: string[] = [];
      const b = [await qInt(1), await qInt(2)];

      await touchGas(harness, "mul", [a, b]);
      const gas = await estimateGas(harness, "mul", [a, b]);
      const out = await harness.mul(a, b);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "mul",
        explanation: "Multiplication where one operand is empty should behave as multiplying by 0.",
        inHex: "[] * [1,2]",
        expectedHex: "N/A (zero polynomial)",
        outHex: fmtHexArr(out),
        expectedDec: "[0]",
        outDec: `${outDec}`,
        gas,
      });
    });

    it("Test 18: Integral of Empty", async function () {
      t++;
      const a: string[] = [];
      const C = await qInt(9);

      await touchGas(harness, "integral", [a, C]);
      const gas = await estimateGas(harness, "integral", [a, C]);
      const out = await harness.integral(a, C);

      const outDec = await fromQuadArray(harness, out);

      printBlockRegular({
        t,
        method: "integral",
        explanation: "Integrating an empty polynomial should yield pure constant C in the result.",
        inHex: "[] + C=9",
        expectedHex: "N/A (constant-only polynomial)",
        outHex: fmtHexArr(out),
        expectedDec: "[9]",
        outDec: `${outDec}`,
        gas,
      });
    });
  });

  // ------------------------------------------------------------
  //  Stress Testing
  // ------------------------------------------------------------

  describe("Section 5: Stress Tests", function () {
    const SIZES = [16, 32, 64];

    async function buildCoeffs(n: number): Promise<string[]> {
      const arr: string[] = [];
      for (let i = 1; i <= n; i++) arr.push(await qInt(i));
      return arr;
    }

    it("Test 19: Large Horner Evaluation", async function () {
      for (const n of SIZES) {
        t++;
        const coeffs = await buildCoeffs(n);
        const x = await qFrac(5, 4); // 1.25

        // Calculate JS Expected
        const jsCoeffs = Array.from({ length: n }, (_, i) => i + 1);
        const expected = evalHornerNum(jsCoeffs, 1.25);

        await touchGas(harness, "evaluateHorners", [coeffs, x]);
        const gas = await estimateGas(harness, "evaluateHorners", [coeffs, x]);
        const out = await harness.evaluateHorners(coeffs, x);

        const outDec = await fromQuad(harness, out);

        printBlockRegular({
          t,
          method: `evaluateHorners (n=${n})`,
          explanation: "Horner evaluation on growing degree n, used to observe linear scaling and gas trends.",
          inHex: `a=[1..${n}], x=1.25`,
          expectedHex: "N/A (large-n analytic double)",
          outHex: out,
          expectedDec: `${expected}`,
          outDec: `${outDec}`,
          gas,
        });
      }
    });

    it("Test 20: Large Addition", async function () {
      for (const n of SIZES) {
        t++;
        const a = await buildCoeffs(n);
        const b = await buildCoeffs(n);

        // Calculate JS Expected
        const jsArr = Array.from({ length: n }, (_, i) => i + 1);
        const expected = addNum(jsArr, jsArr);

        await touchGas(harness, "add", [a, b]);
        const gas = await estimateGas(harness, "add", [a, b]);
        const out = await harness.add(a, b);

        const outDec = await fromQuadArray(harness, out);

        printBlockRegular({
          t,
          method: `add (n=${n})`,
          explanation: "Wide-vector addition for length-n polynomials; tests linear complexity in coefficient count.",
          inHex: `a=[1..${n}], b=[1..${n}]`,
          expectedHex: "N/A (large coeff array)",
          outHex: headTail(out),
          expectedDec: headTail(expected),
          outDec: headTail(outDec),
          gas,
        });
      }
    });

    it("Test 21: Large Multiplication", async function () {
      for (const n of SIZES) {
        t++;
        const a = await buildCoeffs(n);
        const b = await buildCoeffs(n);

        // Calculate JS Expected
        const jsArr = Array.from({ length: n }, (_, i) => i + 1);
        const expected = mulNum(jsArr, jsArr);

        await touchGas(harness, "mul", [a, b]);
        const gas = await estimateGas(harness, "mul", [a, b]);
        const out = await harness.mul(a, b);

        const outDec = await fromQuadArray(harness, out);

        printBlockRegular({
          t,
          method: `mul (n=${n})`,
          explanation: "Stress test for O(n^2) polynomial multiplication with increasing degrees.",
          inHex: `a=[1..${n}], b=[1..${n}]`,
          expectedHex: "N/A (large convolution result)",
          outHex: headTail(out),
          expectedDec: headTail(expected),
          outDec: headTail(outDec),
          gas,
        });
      }
    });
  });
});