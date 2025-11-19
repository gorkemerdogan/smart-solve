// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * @title RootFindingFacet Complete Test Suite
 * @notice Comprehensive verification of Bisection, Newton, and Secant
 *         as exposed by the RootFindingFacet.
 */

// ---------------------------------------------------------
// Type Definitions
// ---------------------------------------------------------

type RootFindingHarness = Contract & {
  qFromInt(n: bigint | number): Promise<string>;
  qFromFrac(n: bigint | number, d: bigint | number): Promise<string>;

  ZERO(): Promise<string>;
  ONE(): Promise<string>;
  TWO(): Promise<string>;
  THREE(): Promise<string>;
  FOUR(): Promise<string>;

  cmp(a: string, b: string): Promise<bigint>;

  f_x2_minus_4(x: string): Promise<string>;
  df_2x(x: string): Promise<string>;
  f_cubic(x: string): Promise<string>;
  df_cubic(x: string): Promise<string>;
};

type RootFindingFacet = Contract & {
  rootFindingBisection(
    target: string,
    fSelector: string,
    a: string,
    b: string
  ): Promise<[string, bigint, bigint, string]>;

  rootFindingNewton(
    target: string,
    fSelector: string,
    dfTarget: string,
    dfSelector: string,
    x0: string
  ): Promise<[string, bigint, bigint, string]>;

  rootFindingSecant(
    target: string,
    fSelector: string,
    x0: string,
    x1: string
  ): Promise<[string, bigint, bigint, string]>;
};

// Successful result returns error == 1.
const ROOT_ERROR_OK = 1;

// ---------------------------------------------------------
// Test Suite
// ---------------------------------------------------------

describe("RootFindingFacet Complete Test Suite", function () {
  let harness: RootFindingHarness;
  let rootFacet: RootFindingFacet;
  let target: string;

  // Function selectors collected at runtime
  const sel: Record<string, string> = {};

  // Cached quad-precision constants from harness
  const q: Record<string, string> = {};

  // Conceptual configuration values
  let TOL_DEFAULT: string;
  let TOL_MIN_DEFAULT: string;
  let TOL_LARGE: string;
  let TOL_TINY: string;
  let TOL_ZERO: string;

  // Planned storage slots for LibNumericConfig
  let CONFIG_SLOT_0: string;
  let CONFIG_SLOT_1: string;

  /**
   * @notice Directly writes numeric config values into storage.
   * @dev The facet currently does NOT read these values,
   *      but this helper is kept for forward compatibility.
   */
  async function setNumericConfig(
    minTol: string,
    tol: string,
    maxIter: bigint | number
  ) {
    const facetAddress = await rootFacet.getAddress();

    // Slot 0: [ tol (16 bytes) | minTol (16 bytes) ]
    const word0 = ethers.concat([tol, minTol]);
    await ethers.provider.send("hardhat_setStorageAt", [
      facetAddress,
      CONFIG_SLOT_0,
      word0,
    ]);

    // Slot 1: maxIter (uint256)
    const word1 = ethers.toBeHex(maxIter, 32);
    await ethers.provider.send("hardhat_setStorageAt", [
      facetAddress,
      CONFIG_SLOT_1,
      word1,
    ]);
  }

  // ---------------------------------------------------------
  // Deployment and Setup
  // ---------------------------------------------------------

  before(async () => {
    // Deploy MathLib
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

    // Deploy RootFindingHarness
    const HarnessFactory = await ethers.getContractFactory(
      "RootFindingHarness",
      {
        libraries: {
          "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
      }
    );

    harness = (await HarnessFactory.deploy()) as unknown as RootFindingHarness;
    await harness.waitForDeployment();
    target = await harness.getAddress();

    // Deploy RootFindingFacet
    const FacetFactory = await ethers.getContractFactory(
      "RootFindingFacet",
      {
        libraries: {
          "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
      }
    );

    rootFacet = (await FacetFactory.deploy()) as unknown as RootFindingFacet;
    await rootFacet.waitForDeployment();

    // Cache function selectors
    const funcs = [
      "f_x2_minus_4(bytes16)",
      "df_2x(bytes16)",
      "f_cubic(bytes16)",
      "df_cubic(bytes16)",
    ];
    for (const f of funcs) {
      sel[f.split("(")[0]] = ethers.id(f).slice(0, 10);
    }

    // Cache quad constants from harness
    q.q0 = await harness.ZERO();
    q.q1 = await harness.ONE();
    q.q2 = await harness.TWO();
    q.q3 = await harness.THREE();
    q.q4 = await harness.FOUR();
    q.qM1 = await harness.qFromInt(-1);
    q.qM3 = await harness.qFromInt(-3);
    q.qM10 = await harness.qFromInt(-10);
    q.q10 = await harness.qFromInt(10);

    // Pre-computed tolerances (for future config integration)
    TOL_DEFAULT     = await harness.qFromFrac(1, BigInt("1000000000000"));        // 1e12
    TOL_MIN_DEFAULT = await harness.qFromFrac(1, BigInt("1000000000000000"));     // 1e15
    TOL_LARGE       = await harness.qFromFrac(1, BigInt("1000000"));              // 1e6
    TOL_TINY        = await harness.qFromFrac(1, BigInt("1000000000000000000"));  // 1e18
    TOL_ZERO = q.q0;

    // Calculate planned storage layout for LibNumericConfig
    const baseSlot =
      ethers.toBigInt(
        ethers.keccak256(
          ethers.toUtf8Bytes("diamond.standard.numeric.config")
        )
      ) - 1n;

    CONFIG_SLOT_0 = ethers.toBeHex(baseSlot, 32);
    CONFIG_SLOT_1 = ethers.toBeHex(baseSlot + 1n, 32);
  });

  // Reset storage before each test
  beforeEach(async () => {
    await setNumericConfig(TOL_ZERO, TOL_ZERO, 0);
  });

  // ---------------------------------------------------------
  // Bisection Tests
  // ---------------------------------------------------------

  describe("Bisection", function () {
    it("1. standard case (x^2 - 4 on [1,3])", async () => {
      const [root, iterations, error] =
        await rootFacet.rootFindingBisection(target, sel.f_x2_minus_4, q.q1, q.q3);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.greaterThan(0);

      // Expected root ≈ 2
      const low = await harness.qFromFrac(19995, 10000);
      const high = await harness.qFromFrac(20005, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });

    it("2. reversed interval ([3,1]) is auto-normalized", async () => {
      const [root, , error] =
        await rootFacet.rootFindingBisection(target, sel.f_x2_minus_4, q.q3, q.q1);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);

      const low = await harness.qFromFrac(19995, 10000);
      const high = await harness.qFromFrac(20005, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });

    it("3. revert if no sign change", async () => {
      await expect(
        rootFacet.rootFindingBisection(target, sel.f_x2_minus_4, q.q3, q.q4)
      ).to.be.revertedWith("No sign change");
    });

    it("4. immediate convergence when f(a)=0", async () => {
      const [root, iterations, error, fAtRoot] =
        await rootFacet.rootFindingBisection(target, sel.f_x2_minus_4, q.q2, q.q3);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.equal(0);
      expect(root).to.equal(q.q2);
      expect(fAtRoot).to.equal(q.q0);
    });

    it("5. immediate convergence when f(b)=0", async () => {
      const [root, iterations, error, fAtRoot] =
        await rootFacet.rootFindingBisection(target, sel.f_x2_minus_4, q.q1, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.equal(0);
      expect(root).to.equal(q.q2);
      expect(fAtRoot).to.equal(q.q0);
    });

    it("6. bounded iterations even with tiny config maxIter", async () => {
      await setNumericConfig(TOL_ZERO, TOL_ZERO, 5);

      const [, iterations, error] =
        await rootFacet.rootFindingBisection(target, sel.f_cubic, q.q1, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.within(1, 200);
    });
  });

  // ---------------------------------------------------------
  // Newton Tests
  // ---------------------------------------------------------

  describe("Newton-Raphson", function () {
    it("7. quadratic, x0=3 → approx 2", async () => {
      const [root, iterations, error] =
        await rootFacet.rootFindingNewton(target, sel.f_x2_minus_4, target, sel.df_2x, q.q3);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.greaterThan(0);

      const low = await harness.qFromFrac(19995, 10000);
      const high = await harness.qFromFrac(20005, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });

    it("8. cubic, x0=1 → approx 1.521", async () => {
      const [root, iterations, error] =
        await rootFacet.rootFindingNewton(target, sel.f_cubic, target, sel.df_cubic, q.q1);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.greaterThan(0);

      const low = await harness.qFromFrac(15205, 10000);
      const high = await harness.qFromFrac(15215, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });

    it("9. zero derivative must revert", async () => {
      await expect(
        rootFacet.rootFindingNewton(target, sel.f_x2_minus_4, target, sel.df_2x, q.q0)
      ).to.be.revertedWith("Zero derivative");
    });

    it("10. immediate convergence when f(x0)=0", async () => {
      const [root, iterations, error, fAtRoot] =
        await rootFacet.rootFindingNewton(target, sel.f_x2_minus_4, target, sel.df_2x, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.equal(0);
      expect(root).to.equal(q.q2);
      expect(fAtRoot).to.equal(q.q0);
    });

    it("11. bounded iterations even with tiny maxIter config", async () => {
      await setNumericConfig(TOL_ZERO, TOL_ZERO, 3);

      const [, iterations, error] =
        await rootFacet.rootFindingNewton(target, sel.f_cubic, target, sel.df_cubic, q.qM10);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.within(1, 200);
    });

    it("12. negative root from x0=-3 → approx -2", async () => {
      const [root, , error] =
        await rootFacet.rootFindingNewton(target, sel.f_x2_minus_4, target, sel.df_2x, q.qM3);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);

      const low = await harness.qFromFrac(-20005, 10000);
      const high = await harness.qFromFrac(-19995, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });
  });

  // ---------------------------------------------------------
  // Secant Tests
  // ---------------------------------------------------------

  describe("Secant", function () {
    it("13. quadratic, x0=1, x1=3 → approx 2", async () => {
      const [root, iterations, error] =
        await rootFacet.rootFindingSecant(target, sel.f_x2_minus_4, q.q1, q.q3);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.greaterThan(0);

      const low = await harness.qFromFrac(19995, 10000);
      const high = await harness.qFromFrac(20005, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });

    it("14. zero slope case must revert", async () => {
      await expect(
        rootFacet.rootFindingSecant(target, sel.f_x2_minus_4, q.qM1, q.q1)
      ).to.be.revertedWith("Zero slope");
    });

    it("15. immediate convergence when x1 is a root", async () => {
      const [root, iterations, error, fAtRoot] =
        await rootFacet.rootFindingSecant(target, sel.f_x2_minus_4, q.q1, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.equal(0);
      expect(root).to.equal(q.q2);
      expect(fAtRoot).to.equal(q.q0);
    });

    it("16. bounded iterations with tiny maxIter config", async () => {
      await setNumericConfig(TOL_ZERO, TOL_ZERO, 2);

      const [, iterations, error] =
        await rootFacet.rootFindingSecant(target, sel.f_cubic, q.qM10, q.q10);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.within(1, 200);
    });

    it("17. negative root (x^2-4) from negative seeds", async () => {
      const [root, iterations, error] =
        await rootFacet.rootFindingSecant(target, sel.f_x2_minus_4, q.qM1, q.qM3);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.greaterThan(0);

      const low = await harness.qFromFrac(-20005, 10000);
      const high = await harness.qFromFrac(-19995, 10000);
      expect(Number(await harness.cmp(root, low))).to.be.at.least(0);
      expect(Number(await harness.cmp(root, high))).to.be.at.most(0);
    });
  });

  // ---------------------------------------------------------
  // Config / Internal
  // ---------------------------------------------------------

  describe("Internal Helpers & Config", function () {
    it("18. default config yields successful convergence (smoke test)", async () => {
      const [, iterations, error] =
        await rootFacet.rootFindingBisection(target, sel.f_cubic, q.q1, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.within(1, 200);
    });

    it("19. _eval should revert on bad selector", async () => {
      const BAD_SELECTOR = "0x12345678";

      await expect(
        rootFacet.rootFindingBisection(target, BAD_SELECTOR, q.q1, q.q3)
      ).to.be.reverted;
    });

    it("20. clamping tol (future compatibility)", async () => {
      await setNumericConfig(TOL_ZERO, TOL_TINY, 50);

      const [, iterations, error] =
        await rootFacet.rootFindingBisection(target, sel.f_cubic, q.q1, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.within(1, 200);
    });

    it("21. clamping tol with large minTol (future compatibility)", async () => {
      await setNumericConfig(TOL_LARGE, TOL_TINY, 50);

      const [, iterations, error] =
        await rootFacet.rootFindingBisection(target, sel.f_cubic, q.q1, q.q2);

      expect(Number(error)).to.equal(ROOT_ERROR_OK);
      expect(Number(iterations)).to.be.within(1, 200);
    });
  });
});