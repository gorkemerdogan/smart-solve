// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";

/**
 * ============================
 *  Fixed-point (SD59x18) utils
 * ============================
 */

const SCALE = 10n ** 18n;

import type { Contract } from "ethers";

// Minimal typing for the methods we call in tests
type PolynomialFixedHarness = Contract & {
  evaluateHorners(coeffs: bigint[], x: bigint): Promise<bigint>;
  derivative(coeffs: bigint[]): Promise<bigint[]>;
  getAddress(): Promise<string>;
};

async function newHarness(): Promise<PolynomialFixedHarness> {
  const Factory = await ethers.getContractFactory("PolynomialFixedHarness");
  const harness = await Factory.deploy();
  await harness.waitForDeployment();
  return harness as unknown as PolynomialFixedHarness;
}

/**
 * Convert a human-readable decimal (string/number) into SD59x18 bigint.
 * Uses ethers.parseUnits for safe decimal parsing.
 * @param v Value like "12.345" or 3 (treated as 3.0).
 */
function fp(v: string | number): bigint {
  if (typeof v === "number") {
    // Convert number to string to avoid FP rounding surprises
    return ethers.parseUnits(v.toString(), 18);
  }
  return ethers.parseUnits(v, 18);
}

/**
 * Fixed-point multiplication: (a * b) / 1e18
 * Solidity library likely truncates toward zero; BigInt division does that by default.
 */
function fpMul(a: bigint, b: bigint): bigint {
  return (a * b) / SCALE;
}

/**
 * Fixed-point addition (trivial; here for symmetry).
 */
function fpAdd(a: bigint, b: bigint): bigint {
  return a + b;
}

/**
 * Evaluate polynomial via naive method (for cross-check):
 *         f(x) = sum_i (coeffs[i] * x^i) using SD59x18 arithmetic.
 * @param coeffs Array where coeffs[i] is SD59x18 coefficient of x^i.
 * @param x SD59x18 input.
 */
function evalNaive(coeffs: bigint[], x: bigint): bigint {
  let pow = SCALE; // x^0 = 1.0 in SD59x18
  let acc = 0n;
  for (let i = 0; i < coeffs.length; i++) {
    // term = coeffs[i] * pow  (both SD59x18) => fpMul
    const term = fpMul(coeffs[i], pow);
    acc = fpAdd(acc, term);
    // pow *= x
    pow = fpMul(pow, x);
  }
  return acc;
}

/**
 * Compare two SD59x18 values with a tolerance (ulps).
 * Most operations are exact here (BigInt math), but allow small diff for future rounding changes.
 */
function expectAlmostEq(a: bigint, b: bigint, tol: bigint = 0n) {
  const diff = a >= b ? a - b : b - a;
  expect(diff, `|${a} - ${b}| > ${tol}`).to.be.lte(tol);
}

describe("PolynomialFixed (library) — Horner & Derivative", function () {
  it("deploys the harness", async function () {
    const harness = await newHarness();
    expect(await harness.getAddress()).to.be.properAddress;
  });

  describe("evaluateHorners", function () {
    it("returns 0 for empty coeffs", async function () {
      const harness = await newHarness();

      const out = await harness.evaluateHorners([], fp(2));
      expect(out).to.equal(0n);
    });

    it("evaluates constant polynomial f(x) = c", async function () {
      const harness = await newHarness();

      const c = fp("5.25");
      const out = await harness.evaluateHorners([c], fp("123.456")); // x unused
      expect(out).to.equal(c);
    });

    it("evaluates linear polynomial f(x) = a0 + a1*x", async function () {
      const harness = await newHarness();

      const a0 = fp("2.5");
      const a1 = fp("-3");
      const x = fp("1.75");

      // Expected via naive
      const expected = evalNaive([a0, a1], x);
      const out = await harness.evaluateHorners([a0, a1], x);

      expect(out).to.equal(expected);
    });

    it("evaluates quadratic f(x) = 5 + 2x + 3x^2", async function () {
      const harness = await newHarness();

      const coeffs = [fp("5"), fp("2"), fp("3")];
      const x = fp("1.2");

      const expected = evalNaive(coeffs, x);
      const out = await harness.evaluateHorners(coeffs, x);

      expect(out).to.equal(expected);
    });

    it("handles negative x and negative coefficients", async function () {
      const harness = await newHarness();

      // f(x) = -1.5 + 0.5x - 2x^2 + 0.25x^3
      const coeffs = [fp("-1.5"), fp("0.5"), fp("-2"), fp("0.25")];
      const x = fp("-0.8");

      const expected = evalNaive(coeffs, x);
      const out = await harness.evaluateHorners(coeffs, x);

      expect(out).to.equal(expected);
    });

    it("matches naive evaluation for a higher-degree polynomial", async function () {
      const harness = await newHarness();

      // f(x) = sum_{i=0..7} ( (i-3)/2 * x^i )
      const coeffs: bigint[] = [];
      for (let i = 0; i < 8; i++) {
        const val = (i - 3) / 2; // can be negative/positive/zero
        coeffs.push(fp(val.toString()));
      }
      const x = fp("0.75");

      const expected = evalNaive(coeffs, x);
      const out = await harness.evaluateHorners(coeffs, x);

      expect(out).to.equal(expected);
    });
  });

  describe("derivative", function () {
    it("derivative of empty or constant returns [0]", async function () {
      const harness = await newHarness();

      const d0 = await harness.derivative([]);        // []
      const d1 = await harness.derivative([fp("7")]); // [7] => [0]

      expect(d0.length).to.equal(1);
      expect(d0[0]).to.equal(0n);

      expect(d1.length).to.equal(1);
      expect(d1[0]).to.equal(0n);
    });

    it("derivative of f(x) = 5 + 2x + 3x^2 is [2, 6]", async function () {
      const harness = await newHarness();

      const coeffs = [fp("5"), fp("2"), fp("3")];
      const d = await harness.derivative(coeffs);

      // Expected: [2, 6] in SD59x18
      expect(d.length).to.equal(2);
      expect(d[0]).to.equal(fp("2"));
      expect(d[1]).to.equal(fp("6"));
    });

    it("derivative scales by SD59x18(i) exactly (checks internal scaling)", async function () {
      const harness = await newHarness();

      // f(x) = a0 + a1 x + a2 x^2 + a3 x^3
      const coeffs = [fp("1.1"), fp("-2.2"), fp("3.3"), fp("-4.4")];

      // Expected derivative:
      // [ 1*a1, 2*a2, 3*a3 ]  BUT note library uses SD59x18(i) then fpMul
      // i.e., d[i-1] = coeffs[i] * (i * 1e18) / 1e18 => coeffs[i] * i  (exact)
      const expected = [
        coeffs[1] * 1n, // 1 * a1
        coeffs[2] * 2n, // 2 * a2
        coeffs[3] * 3n, // 3 * a3
      ];

      const d = await harness.derivative(coeffs);

      expect(d.length).to.equal(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(d[i]).to.equal(expected[i]);
      }
    });

    it("numerical check: derivative matches finite difference at a point", async function () {
      const harness = await newHarness();

      // f(x) = 0.5 - 1.25x + 2.0x^2 - 0.75x^3
      const coeffs = [fp("0.5"), fp("-1.25"), fp("2.0"), fp("-0.75")];
      const x0 = fp("0.8");

      // d(x) from analytic derivative polynomial:
      const d = await harness.derivative(coeffs);

      // Evaluate d(x0)
      const d_plain = [...d];
      const dAtX0 = await harness.evaluateHorners(d_plain, x0);
      
      // Finite difference: (f(x0 + h) - f(x0 - h)) / (2h)
      const h = fp("0.000001"); // small step
      const fPlus: bigint = await harness.evaluateHorners(coeffs, fpAdd(x0, h));
      const fMinus: bigint = await harness.evaluateHorners(coeffs, fpAdd(x0, -h));
      const num: bigint = fPlus - fMinus;
      const den: bigint = 2n * h; // still SD59x18
      const fd: bigint = (num * SCALE) / den; // divide in SD59x18: (num / den) with scaling correction

      // Expect close (should be extremely close with BigInt arithmetic)
      expectAlmostEq(dAtX0, fd, 3000000n); // allow a few wei of tolerance
    });
  });

  describe("integration sanity: derivative and evaluate agree on identity", function () {
    it("f'(x0) *≈* (f(x0+eps) - f(x0-eps)) / (2eps)", async function () {
      const harness = await newHarness();

      // A bit longer polynomial
      const coeffs = [fp("1.0"), fp("0.5"), fp("-1.0"), fp("2.5"), fp("-0.1")]; // degree 4
      const x0 = fp("-0.3");
      const eps = fp("0.000001");

      const d = await harness.derivative(coeffs);
      const d_plain = [...d]; // Convert to a plain JS array
      const dAtX0 = await harness.evaluateHorners(d_plain, x0); // Pass the plain array

      const fPlus: bigint = await harness.evaluateHorners(coeffs, fpAdd(x0, eps));
      const fMinus: bigint = await harness.evaluateHorners(coeffs, fpAdd(x0, -eps));

      const num: bigint = fPlus - fMinus;
      const den: bigint = 2n * eps;
      const fd: bigint = (num * SCALE) / den;

      expectAlmostEq(dAtX0, fd, 3000000n);
    });
  });
});