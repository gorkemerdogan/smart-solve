// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type RootFindingHarness = Contract & {
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;
  toFloat(x: string): Promise<bigint>;
  getAddress(): Promise<string>;
};

type RootFindingFacet = Contract & {
  rootFindingBisection(target: string, fSelector: string, a: string, b: string): Promise<[string, bigint, boolean, string]>;
  rootFindingNewton(target: string, fSelector: string, dfTarget: string, dfSelector: string, x0: string): Promise<[string, bigint, boolean, string]>;
  rootFindingSecant(target: string, fSelector: string, x0: string, x1: string): Promise<[string, bigint, boolean, string]>;
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

type MethodKey = "bisection" | "newton" | "secant";

function trim(n: number) {
  if (!Number.isFinite(n)) return String(n);
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a >= 1e6 || a < 1e-6) return n.toExponential(6);
  return n.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
}

async function fromQuad(h: RootFindingHarness, hex: string): Promise<number> {
  const raw: bigint = await h.toFloat(hex);
  return Number(raw) / 1e12;
}

function toGasBigInt(gas: unknown): bigint {
  if (typeof gas === "bigint") return gas;
  return BigInt(gas as string);
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("RootFinding - Gas Growth Tests", function () {
  let harness: RootFindingHarness;
  let root: RootFindingFacet;
  let t = 0;

  let target: string;

  // Existing selectors from current harness
  let sel_fx2m4: string;
  let sel_df2x: string;
  let sel_fcubic: string;
  let sel_dfcubic: string;

  // Additional selectors expected in harness if you add them
  let sel_fShiftSmall: string;
  let sel_dfShiftSmall: string;
  let sel_fShiftMedium: string;
  let sel_dfShiftMedium: string;
  let sel_fShiftLarge: string;
  let sel_dfShiftLarge: string;

  const qInt = async (n: number | string) => harness.qFromInt(BigInt(n));
  const qFrac = async (n: number | string, d: number | string) => harness.qFromFrac(BigInt(n), BigInt(d));

  before(async () => {
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

    const HarnessFactory = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { "contracts/libraries/MathLib.sol:MathLib": mathAddr },
    });
    harness = (await HarnessFactory.deploy()) as unknown as RootFindingHarness;
    await harness.waitForDeployment();

    const FacetFactory = await ethers.getContractFactory("RootFindingFacet", {
      libraries: { "contracts/libraries/MathLib.sol:MathLib": mathAddr },
    });
    root = (await FacetFactory.deploy()) as unknown as RootFindingFacet;
    await root.waitForDeployment();

    target = await harness.getAddress();

    sel_fx2m4 = ethers.id("f_x2_minus_4(bytes16)").slice(0, 10);
    sel_df2x = ethers.id("df_2x(bytes16)").slice(0, 10);
    sel_fcubic = ethers.id("f_cubic(bytes16)").slice(0, 10);
    sel_dfcubic = ethers.id("df_cubic(bytes16)").slice(0, 10);

    sel_fShiftSmall = ethers.id("f_shift_small(bytes16)").slice(0, 10);
    sel_dfShiftSmall = ethers.id("df_shift_small(bytes16)").slice(0, 10);
    sel_fShiftMedium = ethers.id("f_shift_medium(bytes16)").slice(0, 10);
    sel_dfShiftMedium = ethers.id("df_shift_medium(bytes16)").slice(0, 10);
    sel_fShiftLarge = ethers.id("f_shift_large(bytes16)").slice(0, 10);
    sel_dfShiftLarge = ethers.id("df_shift_large(bytes16)").slice(0, 10);
  });

  // ------------------------------------------------------------
  // Section 1: Gas Sensitivity to Function Structure
  // ------------------------------------------------------------

  describe("Section 1: Gas Sensitivity to Function Structure", function () {
    const FUNCTION_CASES = [
      {
        label: "f(x)=x^2-4",
        fSelector: () => sel_fx2m4,
        dfSelector: () => sel_df2x,
        bisectionArgs: async () => [await qInt(1), await qInt(3)],
        newtonArgs: async () => [await qInt(3)],
        secantArgs: async () => [await qInt(1), await qInt(3)],
      },
      {
        label: "f(x)=x^3-x-2",
        fSelector: () => sel_fcubic,
        dfSelector: () => sel_dfcubic,
        bisectionArgs: async () => [await qInt(1), await qInt(2)],
        newtonArgs: async () => [await qInt(1)],
        secantArgs: async () => [await qInt(1), await qInt(2)],
      },
    ];

    for (const fc of FUNCTION_CASES) {
      it(`Test ${++t}: Bisection gas sensitivity for ${fc.label}`, async function () {
        const [a, b] = await fc.bisectionArgs();

        await touchGas(root, "rootFindingBisection", [target, fc.fSelector(), a, b]);
        const gas = await estimateGas(root, "rootFindingBisection", [target, fc.fSelector(), a, b]);
        const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, fc.fSelector(), a, b);

        printBlockRegular({
          t,
          method: "bisection",
          explanation: `Gas sensitivity to function structure using ${fc.label}.`,
          inHex: `${fc.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(toGasBigInt(gas) > 0n).to.eq(true);
      });

      it(`Test ${++t}: Newton gas sensitivity for ${fc.label}`, async function () {
        const [x0] = await fc.newtonArgs();

        await touchGas(root, "rootFindingNewton", [target, fc.fSelector(), target, fc.dfSelector(), x0]);
        const gas = await estimateGas(root, "rootFindingNewton", [target, fc.fSelector(), target, fc.dfSelector(), x0]);
        const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, fc.fSelector(), target, fc.dfSelector(), x0);

        printBlockRegular({
          t,
          method: "newton",
          explanation: `Gas sensitivity to function structure using ${fc.label}.`,
          inHex: `${fc.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(toGasBigInt(gas) > 0n).to.eq(true);
      });

      it(`Test ${++t}: Secant gas sensitivity for ${fc.label}`, async function () {
        const [x0, x1] = await fc.secantArgs();

        await touchGas(root, "rootFindingSecant", [target, fc.fSelector(), x0, x1]);
        const gas = await estimateGas(root, "rootFindingSecant", [target, fc.fSelector(), x0, x1]);
        const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, fc.fSelector(), x0, x1);

        printBlockRegular({
          t,
          method: "secant",
          explanation: `Gas sensitivity to function structure using ${fc.label}.`,
          inHex: `${fc.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(toGasBigInt(gas) > 0n).to.eq(true);
      });
    }
  });

  // ------------------------------------------------------------
  // Section 2: Gas Sensitivity to Initial Guess / Interval Width
  // ------------------------------------------------------------

  describe("Section 2: Gas Sensitivity to Initial Guess / Interval Width", function () {
    const BISECTION_CASES = [
      { label: "[1,3]", args: async () => [await qInt(1), await qInt(3)] },
      { label: "[1.5,3]", args: async () => [await qFrac(3, 2), await qInt(3)] },
      { label: "[1.9,3]", args: async () => [await qFrac(19, 10), await qInt(3)] },
      { label: "[0,10]", args: async () => [await qInt(0), await qInt(10)] },
      { label: "[-100,0]", args: async () => [await qInt(-100), await qInt(0)] },
    ];

    for (const c of BISECTION_CASES) {
      it(`Test ${++t}: Bisection gas sensitivity for interval ${c.label}`, async function () {
        const [a, b] = await c.args();

        await touchGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
        const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
        const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fx2m4, a, b);

        printBlockRegular({
          t,
          method: "bisection",
          explanation: `Gas sensitivity to interval width using f(x)=x^2-4 on ${c.label}.`,
          inHex: `f=x^2-4, ${c.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(ok).to.eq(true);
      });
    }

    const NEWTON_CASES = [
      { label: "x0=2.1", arg: async () => await qFrac(21, 10) },
      { label: "x0=3", arg: async () => await qInt(3) },
      { label: "x0=10", arg: async () => await qInt(10) },
      { label: "x0=-10", arg: async () => await qInt(-10) },
      { label: "x0=100", arg: async () => await qInt(100) },
    ];

    for (const c of NEWTON_CASES) {
      it(`Test ${++t}: Newton gas sensitivity for ${c.label}`, async function () {
        const x0 = await c.arg();

        await touchGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
        const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
        const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fx2m4, target, sel_df2x, x0);

        printBlockRegular({
          t,
          method: "newton",
          explanation: `Gas sensitivity to initial guess using f(x)=x^2-4 and ${c.label}.`,
          inHex: `f=x^2-4, df=2x, ${c.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(ok).to.eq(true);
      });
    }

    const SECANT_CASES = [
      { label: "(1,3)", args: async () => [await qInt(1), await qInt(3)] },
      { label: "(1.5,3)", args: async () => [await qFrac(3, 2), await qInt(3)] },
      { label: "(0,10)", args: async () => [await qInt(0), await qInt(10)] },
      { label: "(-10,9)", args: async () => [await qInt(-10), await qInt(9)] },
      { label: "(100,101)", args: async () => [await qInt(100), await qInt(101)] },
    ];

    for (const c of SECANT_CASES) {
      it(`Test ${++t}: Secant gas sensitivity for seeds ${c.label}`, async function () {
        const [x0, x1] = await c.args();

        await touchGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
        const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
        const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fx2m4, x0, x1);

        printBlockRegular({
          t,
          method: "secant",
          explanation: `Gas sensitivity to initial seeds using f(x)=x^2-4 and ${c.label}.`,
          inHex: `f=x^2-4, seeds=${c.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(ok).to.eq(true);
      });
    }
  });

  // ------------------------------------------------------------
  // Section 3: Gas Sensitivity to Root Scale
  // ------------------------------------------------------------

  describe("Section 3: Gas Sensitivity to Root Scale", function () {
    const SCALE_CASES = [
      {
        label: "small root",
        f: () => sel_fShiftSmall,
        df: () => sel_dfShiftSmall,
        bisectionArgs: async () => [await qInt(0), await qInt(4)],
        newtonArg: async () => await qInt(0),
        secantArgs: async () => [await qInt(0), await qInt(4)],
      },
      {
        label: "medium root",
        f: () => sel_fShiftMedium,
        df: () => sel_dfShiftMedium,
        bisectionArgs: async () => [await qInt(0), await qInt(400)],
        newtonArg: async () => await qInt(0),
        secantArgs: async () => [await qInt(0), await qInt(400)],
      },
      {
        label: "large root",
        f: () => sel_fShiftLarge,
        df: () => sel_dfShiftLarge,
        bisectionArgs: async () => [await qInt(0), await qInt(40000)],
        newtonArg: async () => await qInt(0),
        secantArgs: async () => [await qInt(0), await qInt(40000)],
      },
    ];

    for (const c of SCALE_CASES) {
      it(`Test ${++t}: Bisection gas sensitivity for ${c.label}`, async function () {
        const [a, b] = await c.bisectionArgs();

        await touchGas(root, "rootFindingBisection", [target, c.f(), a, b]);
        const gas = await estimateGas(root, "rootFindingBisection", [target, c.f(), a, b]);
        const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, c.f(), a, b);

        printBlockRegular({
          t,
          method: "bisection",
          explanation: `Gas sensitivity to root scale using ${c.label}.`,
          inHex: `${c.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(ok).to.eq(true);
      });

      it(`Test ${++t}: Newton gas sensitivity for ${c.label}`, async function () {
        const x0 = await c.newtonArg();

        await touchGas(root, "rootFindingNewton", [target, c.f(), target, c.df(), x0]);
        const gas = await estimateGas(root, "rootFindingNewton", [target, c.f(), target, c.df(), x0]);
        const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, c.f(), target, c.df(), x0);

        printBlockRegular({
          t,
          method: "newton",
          explanation: `Gas sensitivity to root scale using ${c.label}.`,
          inHex: `${c.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(ok).to.eq(true);
      });

      it(`Test ${++t}: Secant gas sensitivity for ${c.label}`, async function () {
        const [x0, x1] = await c.secantArgs();

        await touchGas(root, "rootFindingSecant", [target, c.f(), x0, x1]);
        const gas = await estimateGas(root, "rootFindingSecant", [target, c.f(), x0, x1]);
        const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, c.f(), x0, x1);

        printBlockRegular({
          t,
          method: "secant",
          explanation: `Gas sensitivity to root scale using ${c.label}.`,
          inHex: `${c.label}`,
          expectedHex: "N/A",
          outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
          expectedDec: "N/A",
          outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
          gas,
        });

        expect(ok).to.eq(true);
      });
    }
  });

  // ------------------------------------------------------------
  // Section 4: Direct Method Comparison
  // ------------------------------------------------------------

  describe("Section 4: Direct Method Comparison", function () {
    it(`Test ${++t}: Shared normal scenario (Bisection, x^2-4 on [1,3])`, async function () {
      const a = await qInt(1);
      const b = await qInt(3);

      await touchGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fx2m4, a, b);

      printBlockRegular({
        t,
        method: "bisection",
        explanation: "Shared normal scenario: bisection on x^2-4 over [1,3].",
        inHex: "f=x^2-4, [1,3]",
        expectedHex: "N/A",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        expectedDec: "N/A",
        outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
        gas,
      });
    });

    it(`Test ${++t}: Shared normal scenario (Newton, x^2-4, x0=3)`, async function () {
      const x0 = await qInt(3);

      await touchGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fx2m4, target, sel_df2x, x0);

      printBlockRegular({
        t,
        method: "newton",
        explanation: "Shared normal scenario: Newton-Raphson on x^2-4 from x0=3.",
        inHex: "f=x^2-4, df=2x, x0=3",
        expectedHex: "N/A",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        expectedDec: "N/A",
        outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
        gas,
      });
    });

    it(`Test ${++t}: Shared normal scenario (Secant, x^2-4, x0=1, x1=3)`, async function () {
      const x0 = await qInt(1);
      const x1 = await qInt(3);

      await touchGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
      const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
      const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fx2m4, x0, x1);

      printBlockRegular({
        t,
        method: "secant",
        explanation: "Shared normal scenario: secant on x^2-4 with seeds 1 and 3.",
        inHex: "f=x^2-4, x0=1, x1=3",
        expectedHex: "N/A",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        expectedDec: "N/A",
        outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
        gas,
      });
    });

    it(`Test ${++t}: Shared stress scenario (Bisection, x^3-x-2 on [1,2])`, async function () {
      const a = await qInt(1);
      const b = await qInt(2);

      await touchGas(root, "rootFindingBisection", [target, sel_fcubic, a, b]);
      const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fcubic, a, b]);
      const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fcubic, a, b);

      printBlockRegular({
        t,
        method: "bisection",
        explanation: "Shared stress scenario: bisection on x^3-x-2 over [1,2].",
        inHex: "f=x^3-x-2, [1,2]",
        expectedHex: "N/A",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        expectedDec: "N/A",
        outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
        gas,
      });
    });

    it(`Test ${++t}: Shared stress scenario (Newton, x^3-x-2, x0=1)`, async function () {
      const x0 = await qInt(1);

      await touchGas(root, "rootFindingNewton", [target, sel_fcubic, target, sel_dfcubic, x0]);
      const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fcubic, target, sel_dfcubic, x0]);
      const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fcubic, target, sel_dfcubic, x0);

      printBlockRegular({
        t,
        method: "newton",
        explanation: "Shared stress scenario: Newton-Raphson on x^3-x-2 from x0=1.",
        inHex: "f=x^3-x-2, df=3x^2-1, x0=1",
        expectedHex: "N/A",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        expectedDec: "N/A",
        outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
        gas,
      });
    });

    it(`Test ${++t}: Shared stress scenario (Secant, x^3-x-2, x0=1, x1=2)`, async function () {
      const x0 = await qInt(1);
      const x1 = await qInt(2);

      await touchGas(root, "rootFindingSecant", [target, sel_fcubic, x0, x1]);
      const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fcubic, x0, x1]);
      const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fcubic, x0, x1);

      printBlockRegular({
        t,
        method: "secant",
        explanation: "Shared stress scenario: secant on x^3-x-2 with seeds 1 and 2.",
        inHex: "f=x^3-x-2, x0=1, x1=2",
        expectedHex: "N/A",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        expectedDec: "N/A",
        outDec: `root=${trim(await fromQuad(harness, rHex))}, f=${trim(await fromQuad(harness, fHex))}, iter=${iters}`,
        gas,
      });
    });
  });
});