// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * @title Trigonometric Library Test Suite
 * @notice Tests sin, cos, tan, cot, asin, acos, atan, acot using quad precision.
 */

type TrigonometricHarness = Contract & {
  // helpers
  qFromInt(n: number | bigint): Promise<string>;
  qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
  qAdd(a: string, b: string): Promise<string>;
  qSub(a: string, b: string): Promise<string>;
  qMul(a: string, b: string): Promise<string>;
  qDiv(a: string, b: string): Promise<string>;
  cmp(a: string, b: string): Promise<bigint>;
  absDiff(a: string, b: string): Promise<string>;

  // constants from QuadConstants
  PI(): Promise<string>;
  HALF_PI(): Promise<string>;
  TWO_PI(): Promise<string>;
  DEFAULT_TOL(): Promise<string>;

  // trig functions
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

  // Tolerance for |a - b|
  let TOL: string;

  before(async () => {
    const Factory = await ethers.getContractFactory("TrigonometricHarness");
    const deployed = (await Factory.deploy()) as unknown as TrigonometricHarness;
    await deployed.waitForDeployment();
    harness = deployed;

    // Basic scalars
    q.ZERO = await harness.qFromInt(0);
    q.ONE = await harness.qFromInt(1);
    q.MONE = await harness.qFromInt(-1);

    // Constants from library
    q.PI = await harness.PI();
    q.HALF_PI = await harness.HALF_PI();
    q.TWO_PI = await harness.TWO_PI();

    TOL = await harness.TRIG_TOL();
  });

  // Helper: |a-b| <= TOL
  async function approx(a: string, b: string, msg?: string) {
    const diff = await harness.absDiff(a, b);
    const c = await harness.cmp(diff, TOL); // <= 0 => |a-b| <= tol
    expect(c <= 0n, msg ?? "approx check failed").to.equal(true);
  }

  // Helper: rational x = num/den
  async function qFrac(num: bigint, den: bigint): Promise<string> {
    return harness.qFromFrac(num, den);
  }

  // ---------------------------------------------------------------
  // 1. sin(0) = 0
  // ---------------------------------------------------------------
  it("1. sin(0) ≈ 0", async () => {
    const s = await harness.sin(q.ZERO);
    await approx(s, q.ZERO);
  });

  // ---------------------------------------------------------------
  // 2. cos(0) = 1
  // ---------------------------------------------------------------
  it("2. cos(0) ≈ 1", async () => {
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
  // 7. sin(-x) = -sin(x)  (odd symmetry)
  // ---------------------------------------------------------------
  it("7. sin(-x) = -sin(x)", async () => {
    const x = await harness.qFromInt(1);
    const sx = await harness.sin(x);
    const sxNeg = await harness.sin(await harness.qFromInt(-1));

    // -sin(x) via qMul
    const minusOne = q.MONE;
    const neg = await harness.qMul(minusOne, sx);
    await approx(sxNeg, neg);
  });

  // ---------------------------------------------------------------
  // 8. cos(-x) = cos(x)  (even symmetry)
  // ---------------------------------------------------------------
  it("8. cos(-x) = cos(x)", async () => {
    const x = await harness.qFromInt(1);
    const c1 = await harness.cos(x);
    const c2 = await harness.cos(await harness.qFromInt(-1));
    await approx(c1, c2);
  });

  // ---------------------------------------------------------------
  // 9. Pythagorean: sin²(x) + cos²(x) ≈ 1
  // ---------------------------------------------------------------
  it("9. sin^2(x) + cos^2(x) ≈ 1 (several samples)", async () => {
    // samples as (num, den)
    const samples: Array<[bigint, bigint]> = [
      [-2n, 1n],
      [-1n, 1n],
      [-1n, 2n],
      [7n, 10n],
      [1n, 1n],
      [2n, 1n],
      [3n, 1n],
    ];

    for (const [num, den] of samples) {
      const qx = await qFrac(num, den);
      const s = await harness.sin(qx);
      const c = await harness.cos(qx);

      const s2 = await harness.qMul(s, s);
      const c2 = await harness.qMul(c, c);
      const sum = await harness.qAdd(s2, c2);

      await approx(sum, q.ONE, `sin^2+cos^2 failed for ${num}/${den}`);
    }
  });

  // ---------------------------------------------------------------
  // 10. tan(x) ≈ sin(x)/cos(x)
  // ---------------------------------------------------------------
  it("10. tan(x) ≈ sin(x)/cos(x)", async () => {
    const x = await qFrac(1n, 2n); // 0.5
    const t = await harness.tan(x);
    const s = await harness.sin(x);
    const c = await harness.cos(x);
    const ratio = await harness.qDiv(s, c);
    await approx(t, ratio);
  });

  // ---------------------------------------------------------------
  // 11. cot(x) ≈ cos(x)/sin(x)
  // ---------------------------------------------------------------
  it("11. cot(x) ≈ cos(x)/sin(x)", async () => {
    const x = await qFrac(1n, 3n); // ~0.333...
    const co = await harness.cot(x);
    const s = await harness.sin(x);
    const c = await harness.cos(x);
    const ratio = await harness.qDiv(c, s);
    await approx(co, ratio);
  });

  // ---------------------------------------------------------------
  // 12. tan(pi/2) should revert (cos → 0)
  // ---------------------------------------------------------------
  it("12. tan(pi/2) reverts", async () => {
    await expect(harness.tan(q.HALF_PI)).to.be.revertedWith("tan: undefined");
  });

  // ---------------------------------------------------------------
  // 13. cot(0) should revert (sin → 0)
  // ---------------------------------------------------------------
  it("13. cot(0) reverts", async () => {
    await expect(harness.cot(q.ZERO)).to.be.revertedWith("cot: undefined");
  });

  // ---------------------------------------------------------------
  // 14. asin(sin(x)) ≈ x  for x ∈ [-0.5, 0.5] (principal branch)
  // ---------------------------------------------------------------
  it("14. asin(sin(x)) ≈ x on small range", async () => {
    const samples: Array<[bigint, bigint]> = [
      [-1n, 2n], // -0.5
      [-3n, 10n], // -0.3
      [0n, 1n],
      [2n, 5n], // 0.4
    ];

    for (const [num, den] of samples) {
      const qx = await qFrac(num, den);
      const s = await harness.sin(qx);
      const a = await harness.asin(s);
      await approx(a, qx, `asin(sin(x)) failed for ${num}/${den}`);
    }
  });

  // ---------------------------------------------------------------
  // 15. acos(cos(x)) ≈ x for x ∈ [0, π]
  // ---------------------------------------------------------------
  it("15. acos(cos(x)) ≈ x on [0, pi]", async () => {
    // Using rational approximations of some values in [0,3]
    const samples: Array<[bigint, bigint]> = [
      [0n, 1n],
      [1n, 2n],  // ~0.5
      [1n, 1n],  // 1
      [6n, 5n],  // ~1.2
      [2n, 1n],  // 2
      [3n, 1n],  // 3 (slightly < pi)
    ];

    for (const [num, den] of samples) {
      const qx = await qFrac(num, den);
      const c = await harness.cos(qx);
      const a = await harness.acos(c);
      await approx(a, qx, `acos(cos(x)) failed for ${num}/${den}`);
    }
  });

  // ---------------------------------------------------------------
  // 16. atan(x) → inverse via tan(atan(x)) ≈ x
  // ---------------------------------------------------------------
  it("16. atan(x) basic correctness via tan(atan(x))", async () => {
    const samples: Array<[bigint, bigint]> = [
      [-2n, 1n],
      [-1n, 1n],
      [-1n, 2n],
      [0n, 1n],
      [3n, 10n],
      [1n, 1n],
      [2n, 1n],
    ];

    for (const [num, den] of samples) {
      const qx = await qFrac(num, den);
      const t = await harness.atan(qx);
      const tt = await harness.tan(t);
      await approx(tt, qx, `tan(atan(x)) failed for ${num}/${den}`);
    }
  });

  // ---------------------------------------------------------------
  // 17. acot(x) ≈ atan(1/x)
  // ---------------------------------------------------------------
  it("17. acot(x) ≈ atan(1/x)", async () => {
    const samples: Array<[bigint, bigint]> = [
      [-2n, 1n],
      [-1n, 1n],
      [1n, 2n],
      [1n, 1n],
      [2n, 1n],
    ];

    const one = await harness.qFromInt(1);

    for (const [num, den] of samples) {
      const qx = await qFrac(num, den);
      const ac = await harness.acot(qx);
      const inv = await harness.qDiv(one, qx);
      const atanInv = await harness.atan(inv);
      await approx(ac, atanInv, `acot(x) vs atan(1/x) failed for ${num}/${den}`);
    }
  });

  // ---------------------------------------------------------------
  // 18. Large input reduction: sin(2π * 1e6) ≈ 0
  // ---------------------------------------------------------------
  it("18. Large input reduction: sin(2π * 1e6) ≈ 0", async () => {
    const million = await harness.qFromInt(1_000_000);
    const twoPi = q.TWO_PI;
    const x = await harness.qMul(twoPi, million);
    const s = await harness.sin(x);
    await approx(s, q.ZERO);
  });

  // ---------------------------------------------------------------
  // 19. Small-angle approximation: sin(x) ≈ x for x ≈ 0
  // ---------------------------------------------------------------
  it("19. small-angle sin(x) ≈ x", async () => {
    const tiny = await harness.qFromFrac(1n, 10n ** 12n); // 1e-12
    const s = await harness.sin(tiny);
    await approx(s, tiny);
  });

  // ---------------------------------------------------------------
  // 20. Identity: tan(x) * cot(x) ≈ 1 (where defined)
  // ---------------------------------------------------------------
  it("20. tan(x) * cot(x) ≈ 1", async () => {
    const qx = await qFrac(1n, 4n); // 0.25
    const t = await harness.tan(qx);
    const c = await harness.cot(qx);
    const prod = await harness.qMul(t, c);
    await approx(prod, q.ONE);
  });
});