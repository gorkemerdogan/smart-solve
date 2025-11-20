// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * Polynomial ABDKMathQuad (bytes16) Floating-Point Tests
 */

type PolynomialHarness = Contract & {
  // Passthroughs under test
  evaluateHorners(coeffs: string[], x: string): Promise<string>;
  evaluateWithDerivative(
    coeffs: string[],
    x: string,
  ): Promise<{ px: string; dpx: string }>;
  derivative(coeffs: string[]): Promise<string[]>;
  add(a: string[], b: string[]): Promise<string[]>;
  sub(a: string[], b: string[]): Promise<string[]>;
  mulScalar(a: string[], k: string): Promise<string[]>;
  mul(a: string[], b: string[]): Promise<string[]>;
  integral(a: string[], C: string): Promise<string[]>;
  degree(a: string[]): Promise<bigint>;
  trimTrailingZeros(a: string[]): Promise<string[]>;

  // Helpers for exact construction
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;
  getAddress(): Promise<string>;
};

async function newHarness(): Promise<PolynomialHarness> {
  const MathLibFactory = await ethers.getContractFactory("MathLib");
  const math = await MathLibFactory.deploy();
  await math.waitForDeployment();
  const mathAddr = await math.getAddress();

  const Factory = await ethers.getContractFactory(
    "PolynomialHarness",
    {
      libraries: {
        "contracts/libraries/MathLib.sol:MathLib": mathAddr,
      },
    }
  );

  const harness = await Factory.deploy();
  await harness.waitForDeployment();
  return harness as unknown as PolynomialHarness;
}

/**
 * Compares two ABDKMathQuad (bytes16) floating-point values
 * represented as hexadecimal strings and asserts that they are
 * approximately equal within a specified ULP (Unit in the Last Place) tolerance.
 *
 * This helper is useful when verifying results of operations that involve
 * multiple intermediate floating-point computations, where minimal rounding
 * differences may occur (e.g., Horner’s method vs. closed-form evaluation).
 *
 * The comparison is performed on the raw 128-bit integer encoding of the
 * floating-point numbers, ensuring a precise bit-level check rather than
 * relying on decimal conversions.
 *
 * @param a  The first value as a hexadecimal string (e.g. "0x4001A666666666666666666666666666")
 * @param b  The second value as a hexadecimal string
 * @param ulp  The allowed tolerance in ULP units (default = 1n)
 *
 * @example
 *   expectAlmostEqHex("0x4001A666666666666666666666666665",
 *                     "0x4001A666666666666666666666666666", 1n);
 *   // Passes: the two values differ by only 1 ULP
 */
function expectAlmostEqHex(a: string, b: string, ulp: bigint = 1n) {
  const ai = BigInt(a);
  const bi = BigInt(b);
  const diff = ai >= bi ? ai - bi : bi - ai;
  expect(diff, `ULP diff ${diff} > ${ulp}`).to.be.lte(ulp);
}

describe("Polynomial (library) — Horner, Derivative, Arithmetic, Calculus (ABDK quad)", function () {
  let harness: PolynomialHarness;

  beforeEach(async () => {
    harness = await newHarness();
  });

  // --- helpers (call into harness to build exact bytes16 quads) ---
  async function qInt(n: number | string): Promise<string> {
    return harness.qFromInt(BigInt(n));
  }
  async function qFrac(num: number | string, den: number | string): Promise<string> {
    return harness.qFromFrac(BigInt(num), BigInt(den));
  }

  it("deploys the harness", async function () {
    expect(await harness.getAddress()).to.be.properAddress;
  });

  describe("evaluateHorners", function () {
    it("returns 0 for empty coeffs", async function () {
      const out = await harness.evaluateHorners([], await qInt(2));
      expect(out.toLowerCase()).to.equal((await qInt(0)).toLowerCase());
    });

    it("evaluates constant polynomial f(x) = c", async function () {
      const c = await qInt(5); // f(x) = 5
      const out = await harness.evaluateHorners([c], await qFrac(123456, 1000)); // 123.456
      expect(out.toLowerCase()).to.equal(c.toLowerCase());
    });

    it("evaluates linear polynomial f(x) = a0 + a1*x", async function () {
      // f(x) = 2.5 + (-3 * x) at x=1.75  => -2.75
      const a0 = await qFrac(5, 2);     // 2.5
      const a1 = await qInt(-3);        // -3
      const x  = await qFrac(7, 4);     // 1.75
      const expected = await qFrac(-11, 4); // -2.75
      const out = await harness.evaluateHorners([a0, a1], x);
      expectAlmostEqHex(out.toLowerCase(), expected.toLowerCase(), 1n);
    });

    it("evaluates quadratic f(x) = 5 + 2x + 3x^2", async function () {
      // at x = 1.2 => 11.72
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const x = await qFrac(6, 5);                 // 1.2
      const expected = await qFrac(293, 25);       // 11.72
      const out = await harness.evaluateHorners(coeffs, x);
      expectAlmostEqHex(out.toLowerCase(), expected.toLowerCase(), 1n);
    });

    it("handles negative x and negative coefficients", async function () {
      // f(x) = -1.5 + 0.5x - 2x^2 + 0.25x^3 at x = -0.8  => -3.308
      const coeffs = [await qFrac(-3, 2), await qFrac(1, 2), await qInt(-2), await qFrac(1, 4)];
      const x = await qFrac(-4, 5);                // -0.8
      const expected = await qFrac(-3308, 1000);   // -3.308
      const out = await harness.evaluateHorners(coeffs, x);
      expectAlmostEqHex(out.toLowerCase(), expected.toLowerCase(), 1n);
    });
  });

  describe("derivative", function () {
    it("derivative of empty or constant returns [0]", async function () {
      const d0 = [...(await harness.derivative([]))];
      const d1 = [...(await harness.derivative([await qInt(7)]))];

      expect(d0.length).to.equal(1);
      expect(d0[0].toLowerCase()).to.equal((await qInt(0)).toLowerCase());

      expect(d1.length).to.equal(1);
      expect(d1[0].toLowerCase()).to.equal((await qInt(0)).toLowerCase());
    });

    it("derivative of f(x) = 5 + 2x + 3x^2 is [2, 6]", async function () {
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const d = [...(await harness.derivative(coeffs))];
      expect(d.length).to.equal(2);
      expect(d[0].toLowerCase()).to.equal((await qInt(2)).toLowerCase());
      expect(d[1].toLowerCase()).to.equal((await qInt(6)).toLowerCase());
    });

    it("derivative scales by i via fromInt(i): [-2.2, 6.6, -13.2]", async function () {
      // f(x) = 1.1 + (-2.2)x + 3.3x^2 + (-4.4)x^3
      const coeffs = [await qFrac(11, 10), await qFrac(-22, 10), await qFrac(33, 10), await qFrac(-44, 10)];
      const d = [...(await harness.derivative(coeffs))];
      expect(d.length).to.equal(3);
      /*expect(d[0].toLowerCase()).to.equal((await qFrac(-22,10)).toLowerCase());
      expect(d[1].toLowerCase()).to.equal((await qFrac(33,5)).toLowerCase());
      expect(d[2].toLowerCase()).to.equal((await qFrac(-66,5)).toLowerCase());*/

      expectAlmostEqHex(d[0].toLowerCase(), (await qFrac(-22, 10)).toLowerCase(), 1n);
      expectAlmostEqHex(d[1].toLowerCase(), (await qFrac(33, 5)).toLowerCase(), 1n);
      expectAlmostEqHex(d[2].toLowerCase(), (await qFrac(-66, 5)).toLowerCase(), 1n);
    });
  });

  describe("evaluateWithDerivative", function () {
    it("computes p(x) and p'(x) simultaneously", async function () {
      // p(x) = 5 + 2x + 3x^2;  at x=1.2 -> p=11.72, p'=9.2
      const coeffs = [await qInt(5), await qInt(2), await qInt(3)];
      const x = await qFrac(6, 5); // 1.2
      const { px, dpx } = await harness.evaluateWithDerivative(coeffs, x);
      expect(px.toLowerCase()).to.equal((await qFrac(293, 25)).toLowerCase()); // 11.72
      expect(dpx.toLowerCase()).to.equal((await qFrac(46, 5)).toLowerCase());  // 9.2
    });

    it("returns (0,0) for empty coefficients", async function () {
      const x = await qInt(3);
      const { px, dpx } = await harness.evaluateWithDerivative([], x);
      const zero = await qInt(0);
      expect(px.toLowerCase()).to.equal(zero.toLowerCase());
      expect(dpx.toLowerCase()).to.equal(zero.toLowerCase());
    });
  });

  describe("Arithmetic (add, sub, mul, mulScalar)", function () {
    it("adds two polynomials: (x+2) + (3x+4) = 4x+6", async function () {
      // Coeffs ascending: [c0, c1] -> c0 + c1*x
      const p1 = [await qInt(2), await qInt(1)]; // 1x + 2
      const p2 = [await qInt(4), await qInt(3)]; // 3x + 4
      const expected = [await qInt(6), await qInt(4)]; // 4x + 6

      const res = [...(await harness.add(p1, p2))];
      expect(res.length).to.equal(2);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res[1].toLowerCase()).to.equal(expected[1].toLowerCase());
    });

    it("multiplies (convolves) two polynomials: (x+1)(x+1) = x^2+2x+1", async function () {
      const p1 = [await qInt(1), await qInt(1)]; // x + 1
      const p2 = [await qInt(1), await qInt(1)]; // x + 1
      const expected = [await qInt(1), await qInt(2), await qInt(1)]; // 1 + 2x + x^2

      const res = [...(await harness.mul(p1, p2))];
      expect(res.length).to.equal(3);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res[1].toLowerCase()).to.equal(expected[1].toLowerCase());
      expect(res[2].toLowerCase()).to.equal(expected[2].toLowerCase());
    });

    it("scalar multiply: 2 * (1 + 3x + 4x^2) = 2 + 6x + 8x^2", async function () {
      const p = [await qInt(1), await qInt(3), await qInt(4)];
      const k = await qInt(2);
      const expected = [await qInt(2), await qInt(6), await qInt(8)];

      const res = [...(await harness.mulScalar(p, k))];
      expect(res.length).to.equal(3);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res[1].toLowerCase()).to.equal(expected[1].toLowerCase());
      expect(res[2].toLowerCase()).to.equal(expected[2].toLowerCase());
    });

    it("scalar multiply by 0 returns zero polynomial", async function () {
      const p = [await qInt(3), await qInt(5), await qInt(7)];
      const k = await qInt(0); // multiply by 0
      const expected = [await qInt(0)];

      const res = [...(await harness.mulScalar(p, k))];
      expect(res.length).to.equal(1);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
    });

    it("multiplying any polynomial by [0] returns [0]", async function () {
      const p = [await qInt(2), await qInt(1)];
      const zeroPoly = [await qInt(0)];

      const res1 = [...(await harness.mul(p, zeroPoly))];
      const res2 = [...(await harness.mul(zeroPoly, p))];

      const expected = [await qInt(0)];
      expect(res1[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res2[0].toLowerCase()).to.equal(expected[0].toLowerCase());
    });
  });

  describe("sub()", function () {
    it("subtracts two polynomials correctly", async function () {
      // (4x + 6) - (x + 2) = 3x + 4
      const a = [await qInt(6), await qInt(4)];
      const b = [await qInt(2), await qInt(1)];
      const expected = [await qInt(4), await qInt(3)];

      const res = [...(await harness.sub(a, b))];
      expect(res.length).to.equal(2);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res[1].toLowerCase()).to.equal(expected[1].toLowerCase());
    });
  });

  describe("Calculus (integral)", function () {
    it("integrates a polynomial: integral(6x + 12, 5) = 3x^2 + 12x + 5", async function () {
      // p(x) = 12 + 6x
      const p = [await qInt(12), await qInt(6)];
      const C = await qInt(5); // constant
      // ∫p dx + C = 5 + 12x + 3x^2
      const expected = [await qInt(5), await qInt(12), await qInt(3)];

      const res = [...(await harness.integral(p, C))];
      expect(res.length).to.equal(3);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res[1].toLowerCase()).to.equal(expected[1].toLowerCase());
      expect(res[2].toLowerCase()).to.equal(expected[2].toLowerCase());
    });

    it("integral of empty polynomial returns [C]", async function () {
      const C = await qInt(5);
      const res = [...(await harness.integral([], C))];
      expect(res.length).to.equal(1);
      expect(res[0].toLowerCase()).to.equal(C.toLowerCase());
    });
  });

  describe("Utilities (degree, trim)", function () {
    it("calculates degree correctly", async function () {
      // p(x) = 5 + 2x + 3x^2 + 0x^3  => degree 2
      const p = [await qInt(5), await qInt(2), await qInt(3), await qInt(0)];
      expect(await harness.degree(p)).to.equal(2n);
    });

    it("returns degree 0 for zero polynomial [0, 0, 0]", async function () {
      const p = [await qInt(0), await qInt(0), await qInt(0)];
      expect(await harness.degree(p)).to.equal(0n);
    });

    it("trims trailing zeros", async function () {
      // p(x) = 5 + 2x + 0x^2 + 0x^3  -> [5,2]
      const p = [await qInt(5), await qInt(2), await qInt(0), await qInt(0)];
      const expected = [await qInt(5), await qInt(2)];

      const res = [...(await harness.trimTrailingZeros(p))];
      expect(res.length).to.equal(2);
      expect(res[0].toLowerCase()).to.equal(expected[0].toLowerCase());
      expect(res[1].toLowerCase()).to.equal(expected[1].toLowerCase());
    });

    it("returns degree 0 for empty polynomial []", async function () {
      expect(await harness.degree([])).to.equal(0n);
    });
  });
});