// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * @title Trigonometric Library Test Suite
 * @notice Tests sin, cos, tan, cot, asin, acos, atan, acot using quad precision.
 */

// Extend a generic Contract type with our harness methods
type TrigonometricHarness = Contract & {
  // Quad helpers
  qFromInt(n: number | bigint): Promise<string>;
  qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
  ZERO(): Promise<string>;
  absDiff(a: string, b: string): Promise<string>;
  cmp(a: string, b: string): Promise<bigint>;

  qAdd(a: string, b: string): Promise<string>;
  qSub(a: string, b: string): Promise<string>;
  qMul(a: string, b: string): Promise<string>;
  qDiv(a: string, b: string): Promise<string>;

  // Trig functions
  sin(x: string): Promise<string>;
  cos(x: string): Promise<string>;
  tan(x: string): Promise<string>;
  cot(x: string): Promise<string>;
  asin(x: string): Promise<string>;
  acos(x: string): Promise<string>;
  atan(x: string): Promise<string>;
  acot(x: string): Promise<string>;
};

describe("Trigonometric Library Complete Test Suite", function () {
  let harness: TrigonometricHarness;
  const q: Record<string, string> = {};

  // tolerance for |a - b|
  let TOL: string;

  before(async () => {
    const Factory = await ethers.getContractFactory("TrigonometricHarness");
    // Cast via unknown to satisfy ethers v6 typings
    harness = (await Factory.deploy()) as unknown as TrigonometricHarness;
    await harness.waitForDeployment();

    // Basic constants
    q.ZERO = await harness.ZERO();
    q.ONE = await harness.qFromInt(1);
    q.MONE = await harness.qFromInt(-1);

    // π, π/2, 2π (all within Number.MAX_SAFE_INTEGER)
    q.PI = await harness.qFromFrac(
      Number(3141592653589793n),
      Number(1000000000000000n)
    ); // ≈ 3.141592653589793

    q.HALF_PI = await harness.qFromFrac(
      Number(1570796326794896n),
      Number(1000000000000000n)
    ); // ≈ π/2

    q.TWO_PI = await harness.qFromFrac(
      Number(6283185307179586n),
      Number(1000000000000000n)
    ); // ≈ 2π

    // tolerance = 1e-12
    TOL = await harness.qFromFrac(1, 1_000_000_000_000);
  });

  // Helper for |a-b| <= tolerance
  async function approx(a: string, b: string) {
    const diff = await harness.absDiff(a, b);
    const cmp = Number(await harness.cmp(diff, TOL));
    expect(cmp).to.be.lessThanOrEqual(0);
  }

  // ---------------------------------------------------------------
  // 1. sin(0) = 0
  // ---------------------------------------------------------------
  it("1. sin(0) == 0", async () => {
    const s = await harness.sin(q.ZERO);
    await approx(s, q.ZERO);
  });

  // ---------------------------------------------------------------
  // 2. cos(0) = 1
  // ---------------------------------------------------------------
  it("2. cos(0) == 1", async () => {
    const c = await harness.cos(q.ZERO);
    await approx(c, q.ONE);
  });

  // ---------------------------------------------------------------
  // 3. sin(π/2) = 1
  // ---------------------------------------------------------------
  it("3. sin(pi/2) ≈ 1", async () => {
    const s = await harness.sin(q.HALF_PI);
    await approx(s, q.ONE);
  });

  // ---------------------------------------------------------------
  // 4. cos(π/2) = 0
  // ---------------------------------------------------------------
  it("4. cos(pi/2) ≈ 0", async () => {
    const c = await harness.cos(q.HALF_PI);
    await approx(c, q.ZERO);
  });

  // ---------------------------------------------------------------
  // 5. sin(π) = 0
  // ---------------------------------------------------------------
  it("5. sin(pi) ≈ 0", async () => {
    const s = await harness.sin(q.PI);
    await approx(s, q.ZERO);
  });

  // ---------------------------------------------------------------
  // 6. cos(π) = -1
  // ---------------------------------------------------------------
  it("6. cos(pi) ≈ -1", async () => {
    const c = await harness.cos(q.PI);
    await approx(c, q.MONE);
  });

  // ---------------------------------------------------------------
  // 7. sin(-x) = -sin(x)
  // ---------------------------------------------------------------
  it("7. sin(-x) = -sin(x)", async () => {
    const x = await harness.qFromInt(1);
    const minusX = await harness.qFromInt(-1);

    const sx = await harness.sin(x);
    const sNeg = await harness.sin(minusX);

    // Flip sign bit of the IEEE-754 quad
    function qNeg(v: string): string {
      const asBig = BigInt(v); // v is hex string like "0x...."
      const flipped = asBig ^ (1n << 127n); // flip sign bit
      return ethers.toBeHex(flipped, 16); // 16 bytes
    }

    const expectedNeg = qNeg(sx);
    await approx(sNeg, expectedNeg);
  });

  // ---------------------------------------------------------------
  // 8. cos(-x) = cos(x)
  // ---------------------------------------------------------------
  it("8. cos(-x) = cos(x)", async () => {
    const x = await harness.qFromInt(1);
    const minusX = await harness.qFromInt(-1);

    const c1 = await harness.cos(x);
    const c2 = await harness.cos(minusX);
    await approx(c1, c2);
  });

  // ---------------------------------------------------------------
  // 9. Pythagorean identity: sin² + cos² ≈ 1
  // ---------------------------------------------------------------
  it("9. sin^2(x) + cos^2(x) ≈ 1 for samples", async () => {
    // rational samples in radians: -2, -1, -1/2, 0, 1/2, 1, 2
    const samples: Array<[number, number]> = [
      [-2, 1],
      [-1, 1],
      [-1, 2],
      [0, 1],
      [1, 2],
      [1, 1],
      [2, 1],
    ];

    for (const [num, den] of samples) {
      const qx = await harness.qFromFrac(num, den);
      const s = await harness.sin(qx);
      const c = await harness.cos(qx);

      const s2 = await harness.qMul(s, s);
      const c2 = await harness.qMul(c, c);
      const sum = await harness.qAdd(s2, c2);

      await approx(sum, q.ONE);
    }
  });

  // ---------------------------------------------------------------
  // 10. tan(x) = sin / cos
  // ---------------------------------------------------------------
  it("10. tan(x) ≈ sin(x)/cos(x)", async () => {
    const x = await harness.qFromFrac(1, 2); // 0.5
    const t = await harness.tan(x);
    const s = await harness.sin(x);
    const c = await harness.cos(x);
    const ratio = await harness.qDiv(s, c);

    await approx(t, ratio);
  });

  // ---------------------------------------------------------------
  // 11. cot(x) = cos / sin
  // ---------------------------------------------------------------
  it("11. cot(x) ≈ cos(x)/sin(x)", async () => {
    const x = await harness.qFromFrac(1, 3);
    const co = await harness.cot(x);
    const s = await harness.sin(x);
    const c = await harness.cos(x);
    const ratio = await harness.qDiv(c, s);

    await approx(co, ratio);
  });

  // ---------------------------------------------------------------
  // 12. tan(pi/2) should revert
  // ---------------------------------------------------------------
  it("12. tan(pi/2) reverts", async () => {
    await expect(harness.tan(q.HALF_PI)).to.be.reverted;
  });

  // ---------------------------------------------------------------
  // 13. cot(0) should revert
  // ---------------------------------------------------------------
  it("13. cot(0) reverts", async () => {
    await expect(harness.cot(q.ZERO)).to.be.reverted;
  });

  // ---------------------------------------------------------------
  // 14. asin(sin(x)) ≈ x  for x ∈ [-0.5, 0.5]
  // ---------------------------------------------------------------
  it("14. asin(sin(x)) ≈ x  small range", async () => {
    // represent as rationals
    const samples: Array<[number, number]> = [
      [-1, 2], // -0.5
      [-3, 10], // -0.3
      [0, 1],
      [2, 5], // 0.4
    ];

    for (const [num, den] of samples) {
      const qx = await harness.qFromFrac(num, den);
      const s = await harness.sin(qx);
      const a = await harness.asin(s);
      await approx(a, qx);
    }
  });

  // ---------------------------------------------------------------
  // 15. acos(cos(x)) ≈ x for x ∈ [0, 3]
  // ---------------------------------------------------------------
  it("15. acos(cos(x)) ≈ x for x ∈ [0, 3]", async () => {
    // simple rationals 0, 0.5, 1, 1.5, 2, 3
    const samples: Array<[number, number]> = [
      [0, 1],
      [1, 2],
      [1, 1],
      [3, 2],
      [2, 1],
      [3, 1],
    ];

    for (const [num, den] of samples) {
      const qx = await harness.qFromFrac(num, den);
      const c = await harness.cos(qx);
      const a = await harness.acos(c);
      await approx(a, qx);
    }
  });

  // ---------------------------------------------------------------
  // 16. atan(x) basic correctness via tan(atan(x)) ≈ x
  // ---------------------------------------------------------------
  it("16. atan(x) basic correctness", async () => {
    const samples: Array<[number, number]> = [
      [-2, 1],
      [-1, 1],
      [-1, 2],
      [0, 1],
      [3, 10],
      [1, 1],
      [2, 1],
    ];

    for (const [num, den] of samples) {
      const qx = await harness.qFromFrac(num, den);
      const t = await harness.atan(qx);
      const tt = await harness.tan(t);
      await approx(tt, qx);
    }
  });

  // ---------------------------------------------------------------
  // 17. acot(x) ≈ atan(1/x)
  // ---------------------------------------------------------------
  it("17. acot(x) ≈ atan(1/x)", async () => {
    const samples: Array<[number, number]> = [
      [-2, 1],
      [-1, 1],
      [1, 2],
      [1, 1],
      [2, 1],
    ];

    const qOne = await harness.qFromInt(1);

    for (const [num, den] of samples) {
      const qx = await harness.qFromFrac(num, den);
      const ac = await harness.acot(qx);
      const inv = await harness.qDiv(qOne, qx); // 1/x
      const atanInv = await harness.atan(inv);

      await approx(ac, atanInv);
    }
  });

  // ---------------------------------------------------------------
  // 18. very large angle reduction sin(2π * 1e6) ≈ 0
  // ---------------------------------------------------------------
  it("18. Large input reduction: sin(2π * 1e6) ≈ 0", async () => {
    const million = await harness.qFromInt(1_000_000);
    const x = await harness.qMul(q.TWO_PI, million);
    const s = await harness.sin(x);
    await approx(s, q.ZERO);
  });

  // ---------------------------------------------------------------
  // 19. very small angle sin(1e-12) ≈ x
  // ---------------------------------------------------------------
  it("19. small-angle sin(x) ≈ x", async () => {
    const tiny = await harness.qFromFrac(1, 1_000_000_000_000);
    const s = await harness.sin(tiny);
    await approx(s, tiny);
  });

  // ---------------------------------------------------------------
  // 20. Identity: tan(x) * cot(x) = 1 (where defined)
  // ---------------------------------------------------------------
  it("20. tan(x) * cot(x) ≈ 1", async () => {
    const qx = await harness.qFromFrac(1, 4); // 0.25
    const t = await harness.tan(qx);
    const c = await harness.cot(qx);
    const prod = await harness.qMul(t, c);

    await approx(prod, q.ONE);
  });
});