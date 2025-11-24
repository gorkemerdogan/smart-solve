// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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

const QZERO = "0x00000000000000000000000000000000";

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
//  Formatting Helpers
// ------------------------------------------------------------

function headTail(arr: any[], limit = 3) {
  if (arr.length <= limit * 2) return `[${arr.join(", ")}] (len=${arr.length})`;
  const head = arr.slice(0, limit).join(", ");
  const tail = arr.slice(-limit).join(", ");
  return `[${head}, …, ${tail}] (len=${arr.length})`;
}

const fmtHexArr = (arr: string[]) => `[${arr.join(", ")}]`;
const fmtDecArr = (arr: (number | string)[]) => `[${arr.join(", ")}]`;

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

async function touchGas(harness: PolynomialHarness, method: string, args: any[]) {
  const data = harness.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await harness.getAddress();
  const tx = await signer.sendTransaction({ to, data });
  await tx.wait();
}

async function estimateGas(harness: PolynomialHarness, method: string, args: any[]) {
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
  const decLine = outDec ? `Output: ${outDec}` : ""; // Conditional output string based on whether outDec is provided


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

describe("Polynomial Library - Operations & Calculus", function () {
  let harness: PolynomialHarness;
  let t = 0;

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

      printBlock({ t, method: "evaluateHorners", explanation: "O(n) Eval", gas, inHex: "f=[5,2,3], x=5", outHex: out, outDec: expectedDec });
    });

    it("Test 2: Eval + Derivative (3x^2 + 2x + 5 at x=1.2)", async function () {
      t++;
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const x = await qFrac(6, 5); // 1.2
      const [pDec, dpDec] = evalHornerWithDerivNum([5, 2, 3], 1.2);

      await touchGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const { px, dpx } = await harness.evaluateWithDerivative(coeffs, x);

      printBlock({ t, method: "evaluateWithDerivative", explanation: "One-pass eval", gas, inHex: "x=1.2", outHex: `p=${px}, p'=${dpx}`, outDec: `p=${pDec}, p'=${dpDec}` });
    });

    it("Test 3: Horner Monic (x^3 + 4x^2 + 0.5x + 2 at x=5)", async function () {
      t++;
      const lower = [await qInt(2), await qFrac(1, 2), await qInt(4)];
      const x = await qInt(5);
      const expectedDec = evalHornerMonicNum([2, 0.5, 4], 5);

      await touchGas(harness, "evalHornerMonic", [lower, x]);
      const gas = await estimateGas(harness, "evalHornerMonic", [lower, x]);
      const out = await harness.evalHornerMonic(lower, x);

      printBlock({ t, method: "evalHornerMonic", explanation: "Implicit leading 1", gas, inHex: "lower=[2, 0.5, 4], x=5", outHex: out, outDec: expectedDec });
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

      printBlock({ t, method: "add", explanation: "Coeff-wise add", gas, inHex: "[2,1] + [4,3]", outHex: fmtHexArr(out), outDec: expected });
    });

    it("Test 5: Subtraction ((x+2) - (3x+4))", async function () {
      t++;
      const p1 = [await qInt(2), await qInt(1)];
      const p2 = [await qInt(4), await qInt(3)];
      const expected = fmtDecArr(subNum([2, 1], [4, 3]));

      await touchGas(harness, "sub", [p1, p2]);
      const gas = await estimateGas(harness, "sub", [p1, p2]);
      const out = await harness.sub(p1, p2);

      printBlock({ t, method: "sub", explanation: "Coeff-wise sub", gas, inHex: "[2,1] - [4,3]", outHex: fmtHexArr(out), outDec: expected });
    });

    it("Test 6: Scalar Multiplication (2 * (4x^2 + 3x + 1))", async function () {
      t++;
      const p = [await qInt(1), await qInt(3), await qInt(4)];
      const k = await qInt(2);
      const expected = fmtDecArr(mulScalarNum([1, 3, 4], 2));

      await touchGas(harness, "mulScalar", [p, k]);
      const gas = await estimateGas(harness, "mulScalar", [p, k]);
      const out = await harness.mulScalar(p, k);

      printBlock({ t, method: "mulScalar", explanation: "Scale coeffs", gas, inHex: "2 * [1,3,4]", outHex: fmtHexArr(out), outDec: expected });
    });

    it("Test 7: Multiplication (Convolution) ((x+1)*(x+1))", async function () {
      t++;
      const p1 = [await qInt(1), await qInt(1)];
      const p2 = [await qInt(1), await qInt(1)];
      const expected = fmtDecArr(mulNum([1, 1], [1, 1]));

      await touchGas(harness, "mul", [p1, p2]);
      const gas = await estimateGas(harness, "mul", [p1, p2]);
      const out = await harness.mul(p1, p2);

      printBlock({ t, method: "mul", explanation: "Poly Convolution", gas, inHex: "(x+1)^2", outHex: fmtHexArr(out), outDec: expected });
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

      printBlock({ t, method: "derivative", explanation: "Power rule", gas, inHex: "[5,2,3]", outHex: fmtHexArr(out), outDec: expected });
    });

    it("Test 9: Integral (of 6x + 12, C=5)", async function () {
      t++;
      const p = [await qInt(12), await qInt(6)];
      const C = await qInt(5);
      const expected = fmtDecArr(integralNum([12, 6], 5));

      await touchGas(harness, "integral", [p, C]);
      const gas = await estimateGas(harness, "integral", [p, C]);
      const out = await harness.integral(p, C);

      printBlock({ t, method: "integral", explanation: "Indefinite Integral", gas, inHex: "6x+12, C=5", outHex: fmtHexArr(out), outDec: expected });
    });

    it("Test 10: Synthetic Division (P / (x - 1.5))", async function () {
      t++;
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const root = await qFrac(3, 2);
      const { q: rQ, r: rR } = syntheticDivideNum([5, 2, 3], 1.5);

      await touchGas(harness, "syntheticDivide", [coeffs, root]);
      const gas = await estimateGas(harness, "syntheticDivide", [coeffs, root]);
      const [qHex, rHex] = await harness.syntheticDivide(coeffs, root);

      printBlock({ t, method: "syntheticDivide", explanation: "Ruffini's Rule", gas, inHex: "P=[5,2,3], r=1.5", outHex: `Q=${fmtHexArr(qHex)}, R=${rHex}`, outDec: `Q=${fmtDecArr(rQ)}, R=${rR}` });
    });

    it("Test 11: Degree Check", async function () {
      t++;
      const p = [await qInt(5), await qInt(2), await qInt(0), await qInt(0)];
      const expected = degreeNum([5, 2, 0, 0]);

      await touchGas(harness, "degree", [p]);
      const gas = await estimateGas(harness, "degree", [p]);
      const out = await harness.degree(p);

      printBlock({ t, method: "degree", explanation: "Highest non-zero", gas, inHex: "[5,2,0,0]", outHex: out.toString(), outDec: expected });
    });

    it("Test 12: Trim Trailing Zeros", async function () {
      t++;
      const p = [await qInt(5), await qInt(2), await qInt(0), await qInt(0)];
      const expected = fmtDecArr(trimTrailingZerosNum([5, 2, 0, 0]));

      await touchGas(harness, "trimTrailingZeros", [p]);
      const gas = await estimateGas(harness, "trimTrailingZeros", [p]);
      const out = await harness.trimTrailingZeros(p);

      printBlock({ t, method: "trimTrailingZeros", explanation: "Canonicalize", gas, inHex: "[5,2,0,0]", outHex: fmtHexArr(out), outDec: expected });
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

      printBlock({ t, method: "evaluateHorners", explanation: "Empty is 0", gas, inHex: "[], x=7", outHex: out, outDec: "0" });
    });

    it("Test 14: Constant Polynomial (Eval -> C)", async function () {
      t++;
      const coeffs = [await qInt(7)];
      const x = await qInt(123);

      await touchGas(harness, "evaluateHorners", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateHorners", [coeffs, x]);
      const out = await harness.evaluateHorners(coeffs, x);

      printBlock({ t, method: "evaluateHorners", explanation: "Constant return", gas, inHex: "[7]", outHex: out, outDec: "7" });
    });

    it("Test 15: Constant Derivative (p' -> 0)", async function () {
      t++;
      const coeffs = [await qInt(5)];
      const x = await qInt(42);

      await touchGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const gas = await estimateGas(harness, "evaluateWithDerivative", [coeffs, x]);
      const { dpx } = await harness.evaluateWithDerivative(coeffs, x);

      printBlock({ t, method: "evaluateWithDerivative", explanation: "Deriv of const is 0", gas, inHex: "[5]", outHex: `p'=${dpx}`, outDec: "p'=0" });
    });

    it("Test 16: Zero Scalar Mul", async function () {
      t++;
      const p = [await qInt(5), await qInt(4)];
      const k = await qInt(0);

      await touchGas(harness, "mulScalar", [p, k]);
      const gas = await estimateGas(harness, "mulScalar", [p, k]);
      const out = await harness.mulScalar(p, k);

      printBlock({ t, method: "mulScalar", explanation: "Scale by 0", gas, inHex: "k=0", outHex: fmtHexArr(out), outDec: "[0]" });
    });

    it("Test 17: Mul by Empty", async function () {
      t++;
      const a: string[] = [];
      const b = [await qInt(1), await qInt(2)];

      await touchGas(harness, "mul", [a, b]);
      const gas = await estimateGas(harness, "mul", [a, b]);
      const out = await harness.mul(a, b);

      printBlock({ t, method: "mul", explanation: "Empty multiplicand", gas, inHex: "[] * [1,2]", outHex: fmtHexArr(out), outDec: "[0]" });
    });

    it("Test 18: Integral of Empty", async function () {
      t++;
      const a: string[] = [];
      const C = await qInt(9);

      await touchGas(harness, "integral", [a, C]);
      const gas = await estimateGas(harness, "integral", [a, C]);
      const out = await harness.integral(a, C);

      printBlock({ t, method: "integral", explanation: "Returns C", gas, inHex: "[] + C=9", outHex: fmtHexArr(out), outDec: "[9]" });
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

        printBlock({
          t,
          method: `evaluateHorners (n=${n})`,
          explanation: "Linear scaling check",
          gas,
          inHex: `a=[1..${n}], x=1.25`,
          outHex: out,
          outDec: expected
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

        printBlock({
          t,
          method: `add (n=${n})`,
          explanation: "Linear scaling check",
          gas,
          inHex: `a=[1..${n}], b=[1..${n}]`,
          outHex: headTail(out),
          outDec: headTail(expected)
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

        printBlock({
          t,
          method: `mul (n=${n})`,
          explanation: "O(n^2) scaling check",
          gas,
          inHex: `a=[1..${n}], b=[1..${n}]`,
          outHex: headTail(out),
          outDec: headTail(expected)
        });
      }
    });
  });
});