// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * @title Integration Library Test Suite
 * @notice Tests trapezoidal, Simpson 1/3, and Simpson 3/8 rules
 *         via IntegrationHarness, using several analytic integrands
 *         and a bunch of edge cases.
 */

type IntegrationHarness = Contract & {
  // helpers
  qFromInt(x: number | bigint): Promise<string>;
  qFromUInt(x: number | bigint): Promise<string>;
  qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
  cmp(a: string, b: string): Promise<bigint>;
  absDiff(a: string, b: string): Promise<string>;

  // library wrappers
  trap(
    target: string,
    sel: string,
    a: string,
    b: string,
    n: number | bigint
  ): Promise<string>;

  simpson13(
    target: string,
    sel: string,
    a: string,
    b: string,
    n: number | bigint
  ): Promise<string>;

  simpson38(
    target: string,
    sel: string,
    a: string,
    b: string,
    n: number | bigint
  ): Promise<string>;
};

describe("Integration Library via IntegrationHarness", function () {
  let harness: IntegrationHarness;
  let target: string;

  // function selectors for our integrands
  const sel: Record<string, string> = {};

  // some cached quad constants
  const q: Record<string, string> = {};

  // tolerances for approximate checks
  let TOL_1E9: string; // 1e-9 as quad
  let TOL_TRAP: string; // looser tolerance for trapezoidal on curved functions

  async function expectApprox(
    actual: string,
    expected: string,
    tol: string,
    msg?: string
  ) {
    const diff = await harness.absDiff(actual, expected);
    const cmp = await harness.cmp(diff, tol);
    expect(Number(cmp), msg ?? "abs error too large").to.be.lessThanOrEqual(0);
  }

  before(async () => {
    // 1 — Deploy MathLib
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

    // 2 — Deploy IntegrationHarness
    const HarnessFactory = await ethers.getContractFactory(
      "IntegrationHarness",
      {
        libraries: {
          "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
      }
    );

    harness = (await HarnessFactory.deploy()) as unknown as IntegrationHarness;
    await harness.waitForDeployment();
    target = await harness.getAddress();

    // selectors
    const sigs = [
      "f_zero(bytes16)",
      "f_one(bytes16)",
      "f_linear(bytes16)",
      "f_square(bytes16)",
      "f_cube(bytes16)",
      "f_revert(bytes16)",
    ];
    for (const s of sigs) sel[s.split("(")[0]] = ethers.id(s).slice(0, 10);

    // quad constants
    q.zero = await harness.qFromInt(0);
    q.one = await harness.qFromInt(1);
    q.two = await harness.qFromInt(2);
    q.mOne = await harness.qFromInt(-1);
    q.mTwo = await harness.qFromInt(-2);

    TOL_1E9 = await harness.qFromFrac(1, 1_000_000_000n);
    TOL_TRAP = await harness.qFromFrac(1, 10_000n);
  });

  // ---------------------------------------------------------
  // Trapezoidal Rule Tests
  // ---------------------------------------------------------
  describe("Trapezoidal Rule", function () {
    it("1. f(x)=0 on [0,1], any n → integral = 0", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const I = await harness.trapezoidal(target, sel.f_zero, a, b, 10);
      const cmp = await harness.cmp(I, q.zero);
      expect(Number(cmp)).to.equal(0);
    });

    it("2. f(x)=1 on [0,1], n=1 → exact 1", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const I = await harness.trapezoidal(target, sel.f_one, a, b, 1);
      const cmp = await harness.cmp(I, q.one);
      expect(Number(cmp)).to.equal(0);
    });

    it("3. f(x)=1 on [0,5], n=10 → exact 5", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(5);
      const expected = await harness.qFromInt(5);
      const I = await harness.trapezoidal(target, sel.f_one, a, b, 10);
      const cmp = await harness.cmp(I, expected);
      expect(Number(cmp)).to.equal(0);
    });

    it("4. f(x)=x on [0,1], n=10 → ≈ 1/2", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 2); // 0.5
      const I = await harness.trapezoidal(target, sel.f_linear, a, b, 10);
      await expectApprox(I, expected, TOL_1E9, "trap: x on [0,1]");
    });

    it("5. f(x)=x on [0,2], n=20 → ≈ 2", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(2);
      const expected = await harness.qFromInt(2); // ∫0^2 x dx = 2
      const I = await harness.trapezoidal(target, sel.f_linear, a, b, 20);
      await expectApprox(I, expected, TOL_1E9, "trap: x on [0,2]");
    });

    it("6. f(x)=x^2 on [0,1], n=100 → ≈ 1/3", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 3);
      const I = await harness.trapezoidal(target, sel.f_square, a, b, 100);
      await expectApprox(I, expected, TOL_TRAP, "trap: x^2 on [0,1]");
    });

    it("7. a==b → integral = 0 and f(x) is never called (using reverting f)", async () => {
      const a = await harness.qFromInt(2);
      const b = a;
      const I = await harness.trapezoidal(target, sel.f_revert, a, b, 10);
      const cmp = await harness.cmp(I, q.zero);
      expect(Number(cmp)).to.equal(0);
    });

    it("8. n=0 should revert", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      await expect(
        harness.trapezoidal(target, sel.f_one, a, b, 0)
      ).to.be.revertedWith("Integration: n must be > 0");
    });

    it("9. b < a should revert", async () => {
      const a = await harness.qFromInt(2);
      const b = await harness.qFromInt(1);
      await expect(
        harness.trapezoidal(target, sel.f_one, a, b, 10)
      ).to.be.revertedWith("Integration: upper bound b must be >= a");
    });

    it("10. target == address(0) should revert", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      await expect(
        harness.trapezoidal(ethers.ZeroAddress, sel.f_one, a, b, 10)
      ).to.be.revertedWith("Integration: target is zero");
    });
  });

  // ---------------------------------------------------------
  // Simpson 1/3 Rule Tests
  // ---------------------------------------------------------
  describe("Simpson 1/3 Rule", function () {
    it("11. f(x)=1 on [0,1], n=2 (even) → ≈ 1", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = q.one;
      const I = await harness.simpson13(target, sel.f_one, a, b, 2);
      await expectApprox(I, expected, TOL_1E9, "simpson 1/3: 1 on [0,1]");
    });

    it("12. f(x)=x on [0,1], n=10 (even) → ≈ 1/2", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 2);
      const I = await harness.simpson13(target, sel.f_linear, a, b, 10);
      await expectApprox(I, expected, TOL_1E9, "simpson 1/3: x on [0,1]");
    });

    it("13. f(x)=x^2 on [0,1], n=10 → ≈ 1/3", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 3);
      const I = await harness.simpson13(target, sel.f_square, a, b, 10);
      await expectApprox(I, expected, TOL_1E9, "simpson 1/3: x^2 on [0,1]");
    });

    it("14. f(x)=x^3 on [0,1], n=10 → ≈ 1/4", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 4);
      const I = await harness.simpson13(target, sel.f_cube, a, b, 10);
      await expectApprox(I, expected, TOL_1E9, "simpson 1/3: x^3 on [0,1]");
    });

    it("15. a==b → integral = 0 and f(x) not called", async () => {
      const a = await harness.qFromInt(3);
      const b = a;
      const I = await harness.simpson13(target, sel.f_revert, a, b, 2);
      const cmp = await harness.cmp(I, q.zero);
      expect(Number(cmp)).to.equal(0);
    });

    it("16. odd n should revert", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      await expect(
        harness.simpson13(target, sel.f_one, a, b, 3)
      ).to.be.revertedWith("Integration: Simpson 1/3 requires even n");
    });

    it("17. b < a should revert", async () => {
      const a = await harness.qFromInt(2);
      const b = await harness.qFromInt(0);
      await expect(
        harness.simpson13(target, sel.f_one, a, b, 2)
      ).to.be.revertedWith("Integration: upper bound b must be >= a");
    });
  });

  // ---------------------------------------------------------
  // Simpson 3/8 Rule Tests
  // ---------------------------------------------------------
  describe("Simpson 3/8 Rule", function () {
    it("18. f(x)=1 on [0,1], n=3 (multiple of 3) → ≈ 1", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = q.one;
      const I = await harness.simpson38(target, sel.f_one, a, b, 3);
      await expectApprox(I, expected, TOL_1E9, "simpson 3/8: 1 on [0,1]");
    });

    it("19. f(x)=x on [0,1], n=3 → ≈ 1/2", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 2);
      const I = await harness.simpson38(target, sel.f_linear, a, b, 3);
      await expectApprox(I, expected, TOL_1E9, "simpson 3/8: x on [0,1]");
    });

    it("20. f(x)=x^3 on [0,1], n=3 → ≈ 1/4", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      const expected = await harness.qFromFrac(1, 4);
      const I = await harness.simpson38(target, sel.f_cube, a, b, 3);
      await expectApprox(I, expected, TOL_1E9, "simpson 3/8: x^3 on [0,1]");
    });

    it("21. n not multiple of 3 should revert", async () => {
      const a = await harness.qFromInt(0);
      const b = await harness.qFromInt(1);
      await expect(
        harness.simpson38(target, sel.f_one, a, b, 4)
      ).to.be.revertedWith("Integration: Simpson 3/8 requires n % 3 == 0");
    });

    it("22. a==b → integral = 0 and f(x) not called", async () => {
      const a = await harness.qFromInt(5);
      const b = a;
      const I = await harness.simpson38(target, sel.f_revert, a, b, 3);
      const cmp = await harness.cmp(I, q.zero);
      expect(Number(cmp)).to.equal(0);
    });
  });
});