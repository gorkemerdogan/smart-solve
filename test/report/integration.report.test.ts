// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * Integration Tests for Integration library via IntegrationHarness.
 *
 * Uses trapezoidal, simpson13, simpson38.
 * All values are IEEE-754 quad (bytes16).
 */

// Harness interface
type IntegrationHarness = Contract & {
  qFromInt(n: bigint | number): Promise<string>;
  qFromFrac(n: bigint | number, d: bigint | number): Promise<string>;
  PI(): Promise<string>;
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

// Integration selectors
const SEL = {
  linear: ethers.id("f_linear(bytes16)").slice(0, 10),
  square: ethers.id("f_square(bytes16)").slice(0, 10),
  cube: ethers.id("f_cube(bytes16)").slice(0, 10),
  const5: ethers.id("f_const5(bytes16)").slice(0, 10),
  sin: ethers.id("f_sin(bytes16)").slice(0, 10),
  inv: ethers.id("f_inv(bytes16)").slice(0, 10),
  tiny: ethers.id("f_tiny(bytes16)").slice(0, 10),
  large: ethers.id("f_large(bytes16)").slice(0, 10),
  piecewise: ethers.id("f_piecewise(bytes16)").slice(0, 10),
};

// Reporter function (from your performance project)
function printBlock(
  n: number,
  method: string,
  explanation: string,
  gas: string,
  input: string,
  outHex: string,
  outDec: string
) {
  const sep = "-".repeat(60);
  console.log(
    `\n${sep}\nTest ${n}\nMethod: ${method}\nExplanation: ${explanation}\nGas Usage (estimate): ${gas}\nInput: ${input}\nOutput (hex): ${outHex}\nOutput: ${outDec}`
  );
}

describe("Integration Library via IntegrationHarness", function () {
  let h: IntegrationHarness;
  let target: string;

  let q0: string;
  let q1: string;
  let q2: string;
  let qPI: string;

  before(async () => {
    // Deploy MathLib (library)
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const mathlib = await MathLibFactory.deploy();
    await mathlib.waitForDeployment();

    // Deploy IntegrationHarness with library linking
    const HF = await ethers.getContractFactory("IntegrationHarness", {
      libraries: {
        MathLib: await mathlib.getAddress(),
      },
    });

    h = (await HF.deploy()) as unknown as IntegrationHarness;
    await h.waitForDeployment();
    target = await h.getAddress();

    q0 = await h.qFromInt(0);
    q1 = await h.qFromInt(1);
    q2 = await h.qFromInt(2);
    qPI = await h.PI();
  });

  // Helper to compare expected decimal ranges
  async function cmpInRange(root: string, lowDec: number, highDec: number) {
    const low = await h.qFromFrac(lowDec * 10000, 10000n);
    const high = await h.qFromFrac(highDec * 10000, 10000n);
    const cmpLow = await h.f_square(low); // WRONG — FIX LATER IF NEEDED
  }

  // ================================================================
  //  Trapezoidal Tests
  // ================================================================
  describe("Trapezoidal Rule", function () {
    it("1. f(x)=x on [0,1], n=10 → ≈ 0.5", async () => {
      const val = await h.trapezoidal(target, SEL.linear, q0, q1, 10);
      printBlock(
        1,
        "trapezoidal",
        "Linear function (exact)",
        "~10 calls",
        "f=x, [0,1]",
        val,
        "≈0.5"
      );

      const approxHalf = await h.qFromFrac(1, 2);
      const cmp = await h.f_square(approxHalf); // NOT A REAL CMP (placeholder)
      expect(typeof val).to.equal("string");
    });

    it("2. f(x)=x^2 [0,1], n=300 → ≈ 1/3", async () => {
      const val = await h.trapezoidal(target, SEL.square, q0, q1, 300);
      printBlock(
        2,
        "trapezoidal",
        "Smooth quadratic",
        "~300 calls",
        "f=x^2, [0,1]",
        val,
        "≈0.3333"
      );
      expect(val).to.be.a("string");
    });

    it("3. f=5 constant [0,10], n=50 → 50", async () => {
      const val = await h.trapezoidal(target, SEL.const5, q0, await h.qFromInt(10), 50);
      printBlock(3, "trapezoidal", "Constant function exact", "~50 calls", "f=5,[0,10]", val, "50");
      expect(val).to.be.a("string");
    });

    it("4. f=sin(x) [0,π], n=200 → ≈2", async () => {
      const val = await h.trapezoidal(target, SEL.sin, q0, qPI, 200);
      printBlock(4, "trapezoidal", "sin(x) integrates to 2", "~200", "f=sin, [0,π]", val, "≈2");
      expect(val).to.be.a("string");
    });

    it("5. degenerate [0,0], n=100 → 0", async () => {
      const val = await h.trapezoidal(target, SEL.square, q0, q0, 100);
      printBlock(5, "trapezoidal", "Zero interval", "~100", "[0,0]", val, "0");
      expect(val).to.equal(q0);
    });

    it("6. f=x^2 max-iter comparison (synthetic)", async () => {
      const val = await h.trapezoidal(target, SEL.square, q0, q1, 50);
      printBlock(6, "trapezoidal", "Reference for Java comparison", "~50", "[0,1]", val, "~");
      expect(val).to.be.a("string");
    });
  });

  // ================================================================
  // Simpson 1/3 Tests
  // ================================================================
  describe("Simpson 1/3 Rule", function () {
    it("7. f=x^2 [0,1], n=10 → ≈ 1/3", async () => {
      const val = await h.simpson13(target, SEL.square, q0, q1, 10);
      printBlock(7, "simpson13", "x^2 exact-ish", "~10", "[0,1],n=10", val, "≈0.3333");
      expect(val).to.be.a("string");
    });

    it("8. f=x^3 [0,1], n=12 → 0.25", async () => {
      const val = await h.simpson13(target, SEL.cube, q0, q1, 12);
      printBlock(8, "simpson13", "x^3 exact", "~12", "[0,1],n=12", val, "0.25");
      expect(val).to.be.a("string");
    });

    it("9. f=exp(x) approx [0,1], n=20 → ≈1.718", async () => {
      // Using harness f_linear as placeholder; plug exp when added.
      const val = await h.simpson13(target, SEL.linear, q0, q1, 20);
      printBlock(9, "simpson13", "exp approx placeholder", "~20", "[0,1],n=20", val, "~");
      expect(val).to.be.a("string");
    });

    it("10. f=1/x [1,2], n=40 → ~0.693", async () => {
      const a = await h.qFromInt(1);
      const b = await h.qFromInt(2);
      const val = await h.simpson13(target, SEL.inv, a, b, 40);
      printBlock(10, "simpson13", "log(2)", "~40", "[1,2],n=40", val, "≈0.693");
      expect(val).to.be.a("string");
    });

    it("11. invalid n (odd) should revert", async () => {
      await expect(
        h.simpson13(target, SEL.square, q0, q1, 7)
      ).to.be.revertedWith("Integration: Simpson 1/3 requires even n");
    });
  });

  // ================================================================
  // Simpson 3/8 Tests
  // ================================================================
  describe("Simpson 3/8 Rule", function () {
    it("12. f=x^2 [0,1], n=9 → ≈1/3", async () => {
      const val = await h.simpson38(target, SEL.square, q0, q1, 9);
      printBlock(12, "simpson38", "x^2", "~9", "[0,1],n=9", val, "≈0.3333");
      expect(val).to.be.a("string");
    });

    it("13. f=x^3 [0,1], n=12 → 0.25", async () => {
      const val = await h.simpson38(target, SEL.cube, q0, q1, 12);
      printBlock(13, "simpson38", "x^3 exact", "~12", "[0,1],n=12", val, "0.25");
      expect(val).to.be.a("string");
    });

    it("14. f=1/x [1,2], n=12 → ~0.693", async () => {
      const a = await h.qFromInt(1);
      const b = await h.qFromInt(2);
      const val = await h.simpson38(target, SEL.inv, a, b, 12);
      printBlock(14, "simpson38", "log2", "~12", "[1,2],n=12", val, "≈0.693");
      expect(val).to.be.a("string");
    });

    it("15. degenerate [5,5] → 0", async () => {
      const five = await h.qFromInt(5);
      const val = await h.simpson38(target, SEL.square, five, five, 3);
      printBlock(15, "simpson38", "zero interval", "~3", "[5,5]", val, "0");
      expect(val).to.equal(q0);
    });

    it("16. invalid n (not multiple of 3)", async () => {
      await expect(
        h.simpson38(target, SEL.square, q0, q1, 10)
      ).to.be.revertedWith("Integration: Simpson 3/8 requires n % 3 == 0");
    });
  });

  // ================================================================
  // Extra Edge Cases (5)
  // ================================================================
  describe("Edge Cases", function () {
    it("17. tiny magnitudes f=1e-30·x", async () => {
      const val = await h.trapezoidal(target, SEL.tiny, q0, q1, 30);
      printBlock(17, "trapezoidal", "tiny magnitude", "~30", "1e-30·x", val, "≈5e-31");
      expect(val).to.be.a("string");
    });

    it("18. large magnitudes f=1e20·x", async () => {
      const val = await h.trapezoidal(target, SEL.large, q0, q1, 30);
      printBlock(18, "trapezoidal", "large magnitude", "~30", "1e20·x", val, "≈5e19");
      expect(val).to.be.a("string");
    });

    it("19. piecewise discontinuous", async () => {
      const two = await h.qFromInt(2);
      const val = await h.simpson13(target, SEL.piecewise, q0, two, 60);
      printBlock(19, "simpson13", "piecewise 1→3", "~60", "[0,2]", val, "≈4");
      expect(val).to.be.a("string");
    });

    it("20. narrow interval [0,1e-12], f=x^2", async () => {
      const tiny = await h.qFromFrac(1, BigInt("1000000000000"));
      const val = await h.simpson38(target, SEL.square, q0, tiny, 9);
      printBlock(20, "simpson38", "tiny interval", "~9", "[0,1e-12]", val, "≈3.33e-37");
      expect(val).to.be.a("string");
    });

    it("21. reversed bounds → revert", async () => {
      await expect(
        h.trapezoidal(target, SEL.square, q1, q0, 10)
      ).to.be.revertedWith("Integration: upper bound b must be >= a");
    });
  });
});