// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";

/**
 * ABDKMathQuad (bytes16) Floating-Point Test
 */

import type { Contract } from "ethers";

// Minimal typing for the methods we call in tests
// This defines what we expect our test harness contract to have.
type PolynomialQuadHarness = Contract & {
  evaluateHorners(coeffs: string[], x: string): Promise<string>; // bytes16[] and bytes16
  derivative(coeffs: string[]): Promise<string[]>; // bytes16[]
  getAddress(): Promise<string>;
};

/**
 * Deploys a new instance of the PolynomialQuadHarness contract.
 * @returns A promise that resolves to the deployed contract instance.
 */
async function newHarness(): Promise<PolynomialQuadHarness> {
  // Deploy the new, updated test harness
  const Factory = await ethers.getContractFactory("PolynomialQuadHarness");
  const harness = await Factory.deploy();
  await harness.waitForDeployment();
  return harness as unknown as PolynomialQuadHarness;
}

/**
 * Helper lookup function for ABDKMathQuad (bytes16) values.
 * Converts human-readable test strings into their corresponding
 * 128-bit quadruple-precision floating-point hex string values.
 *
 * Note: These values are standard IEEE-754 128-bit (quad) format.
 * @param v The string to convert (e.g., "1", "0.5", "-3")
 */
function quad(v: string): string {
  // A lookup table for all the constants we need in our tests.
  const lookup: { [key: string]: string } = {
    "0": "0x00000000000000000000000000000000",
    "1": "0x3FFF0000000000000000000000000000",
    "2": "0x40000000000000000000000000000000",
    "3": "0x40018000000000000000000000000000",
    "5": "0x40024000000000000000000000000000",
    "6": "0x40028000000000000000000000000000",
    "7": "0x4002C000000000000000000000000000",
    "123.456": "0x4005EDD2F1A9FBE76000000000000000",
    "-3": "0xC0018000000000000000000000000000",
    "2.5": "0x40014000000000000000000000000000",
    "1.75": "0x3FFFE000000000000000000000000000",
    "-2.75": "0xC0016000000000000000000000000000", // f(x) = 2.5 + (-3 * 1.75) = 2.5 - 5.25 = -2.75
    "1.2": "0x3FFFA666666666666000000000000000",
    "11.32": "0x4003551EB851EB852000000000000000", // f(x) = 5 + 2*1.2 + 3*1.2^2 = 5 + 2.4 + 3*1.44 = 7.4 + 4.32 = 11.72
    // Correction: Should be 11.72
    "11.72": "0x400375C28F5C28F5C000000000000000",
    "-0.8": "0xBFFFCccccccccccd000000000000000",
    "-1.5": "0xBFFFC000000000000000000000000000",
    "0.5": "0x3FFE0000000000000000000000000000",
    "-2": "0xC0000000000000000000000000000000",
    "0.25": "0x3FFD0000000000000000000000000000",
    "-3.028": "0xC00183A6666666666000000000000000", // f(x) = -1.5 + 0.5*(-0.8) - 2*(-0.8)^2 + 0.25*(-0.8)^3
    //   = -1.5 - 0.4 - 2*(0.64) + 0.25*(-0.512)
    //   = -1.9 - 1.28 - 0.128 = -3.308
    // Correction: Should be -3.308
    "-3.308": "0xC001A78D6B49D2678000000000000000",
    // Values for other tests
    "1.1": "0x3FFF199999999999A000000000000000",
    "-2.2": "0xBFFF3333333333334000000000000000",
    "3.3": "0x4001A666666666666000000000000000",
    "6.6": "0x4002A666666666666000000000000000",
    "-4.4": "0xC0023333333333334000000000000000",
    "-13.2": "0xC003A666666666666000000000000000",
  };

  const val = lookup[v];
  if (!val) {
    throw new Error(`Test Error: quad() helper does not know the value for "${v}"`);
  }
  return val;
}

// Replicating ABDKMathQuad (floating-point) logic in JS is complex and
// would require a library that is not compatible with Ethers v6.

describe("PolynomialQuad (library) — Horner & Derivative", function () {
  it("deploys the harness", async function () {
    const harness = await newHarness();
    expect(await harness.getAddress()).to.be.properAddress;
  });

  describe("evaluateHorners", function () {
    it("returns 0 for empty coeffs", async function () {
      const harness = await newHarness();
      // f(x) = 0
      const out = await harness.evaluateHorners([], quad("2"));
      expect(out).to.equal(quad("0"));
    });

    it("evaluates constant polynomial f(x) = c", async function () {
      const harness = await newHarness();
      // f(x) = 5
      const c = quad("5");
      const out = await harness.evaluateHorners([c], quad("123.456")); // x is unused
      expect(out).to.equal(c);
    });

    it("evaluates linear polynomial f(x) = a0 + a1*x", async function () {
      const harness = await newHarness();
      // f(x) = 2.5 + (-3 * x)
      const a0 = quad("2.5");
      const a1 = quad("-3");
      const x = quad("1.75");

      // Expected = 2.5 + (-3 * 1.75) = 2.5 - 5.25 = -2.75
      const expected = quad("-2.75");
      const out = await harness.evaluateHorners([a0, a1], x);

      expect(out).to.equal(expected);
    });

    it("evaluates quadratic f(x) = 5 + 2x + 3x^2", async function () {
      const harness = await newHarness();
      // f(x) = 5 + 2x + 3x^2
      const coeffs = [quad("5"), quad("2"), quad("3")];
      const x = quad("1.2");

      // Expected = 5 + 2*(1.2) + 3*(1.2)^2 = 5 + 2.4 + 3*(1.44) = 7.4 + 4.32 = 11.72
      const expected = quad("11.72");
      const out = await harness.evaluateHorners(coeffs, x);

      expect(out).to.equal(expected);
    });

    it("handles negative x and negative coefficients", async function () {
      const harness = await newHarness();

      // f(x) = -1.5 + 0.5x - 2x^2 + 0.25x^3
      const coeffs = [quad("-1.5"), quad("0.5"), quad("-2"), quad("0.25")];
      const x = quad("-0.8");

      // Expected = -1.5 + 0.5*(-0.8) - 2*(-0.8)^2 + 0.25*(-0.8)^3
      //   = -1.5 - 0.4 - 2*(0.64) + 0.25*(-0.512)
      //   = -1.9 - 1.28 - 0.128 = -3.308
      const expected = quad("-3.308");
      const out = await harness.evaluateHorners(coeffs, x);

      expect(out).to.equal(expected);
    });

    // The "higher-degree" test that used evalNaive was removed.
  });

  describe("derivative", function () {
    it("derivative of empty or constant returns [0]", async function () {
      const harness = await newHarness();

      // f(x) = 0 -> f'(x) = 0
      const d0 = await harness.derivative([]); // []
      // f(x) = 7 -> f'(x) = 0
      const d1 = await harness.derivative([quad("7")]); // [7] => [0]

      // Note: Ethers v6 returns read-only `Result` objects.
      // We must copy them into a plain array before treating them as an array.
      const d0_plain = [...d0];
      const d1_plain = [...d1];

      // The derivative of a constant or empty poly is f'(x) = 0, represented as [0]
      expect(d0_plain.length).to.equal(1);
      expect(d0_plain[0]).to.equal(quad("0"));

      expect(d1_plain.length).to.equal(1);
      expect(d1_plain[0]).to.equal(quad("0"));
    });

    it("derivative of f(x) = 5 + 2x + 3x^2 is [2, 6]", async function () {
      const harness = await newHarness();

      // f(x) = 5 + 2x + 3x^2
      const coeffs = [quad("5"), quad("2"), quad("3")];
      // f'(x) = 2 + 6x
      const d = await harness.derivative(coeffs);
      const d_plain = [...d]; // Copy the Ethers v6 read-only result

      // Expected derivative: [2, 6]
      expect(d_plain.length).to.equal(2);
      expect(d_plain[0]).to.equal(quad("2"));
      expect(d_plain[1]).to.equal(quad("6"));
    });

    it("derivative scales by fromInt(i) (manual check)", async function () {
      const harness = await newHarness();

      // f(x) = a0 + a1*x + a2*x^2 + a3*x^3
      const coeffs = [quad("1.1"), quad("-2.2"), quad("3.3"), quad("-4.4")];

      // Expected derivative (from Solidity logic):
      // d[0] = mul(a1, fromInt(1)) = -2.2 * 1 = -2.2
      // d[1] = mul(a2, fromInt(2)) = 3.3 * 2 = 6.6
      // d[2] = mul(a3, fromInt(3)) = -4.4 * 3 = -13.2
      const expected = [quad("-2.2"), quad("6.6"), quad("-13.2")];

      const d = await harness.derivative(coeffs);
      const d_plain = [...d]; // Copy the read-only result

      expect(d_plain.length).to.equal(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(d_plain[i]).to.equal(expected[i]);
      }
    });
  });

});