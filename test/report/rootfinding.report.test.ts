// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type RootFindingHarness = Contract & {
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;

  f_x2_minus_4(x: string): Promise<string>;
  df_2x(x: string): Promise<string>;
  f_cubic(x: string): Promise<string>;
  df_cubic(x: string): Promise<string>;
};

type RootFindingFacet = Contract & {
  rootFindingBisection(target: string, fSelector: string, a: string, b: string): Promise<[string, bigint, boolean, string]>;
  rootFindingNewton(target: string, fSelector: string, dfTarget: string, dfSelector: string, x0: string): Promise<[string, bigint, boolean, string]>;
  rootFindingSecant(target: string, fSelector: string, x0: string, x1: string): Promise<[string, bigint, boolean, string]>;
};

// ------------------------------------------------------------
//  JS Numeric Mirrors (Verification)
// ------------------------------------------------------------

function f_x2_minus_4(x: number) { return x * x - 4; }
function df_2x(x: number) { return 2 * x; }
function f_cubic(x: number) { return x * x * x - x - 2; }
function df_cubic(x: number) { return 3 * x * x - 1; }

function bisectionRef(f: (x: number) => number, a: number, b: number, eps = 1e-12, maxIter = 200) {
  if (a > b) [a, b] = [b, a];
  let fa = f(a), fb = f(b);
  if (fa === 0) return { root: a, iterations: 0, converged: true, fAtRoot: fa };
  if (fb === 0) return { root: b, iterations: 0, converged: true, fAtRoot: fb };
  if (fa * fb > 0) throw new Error("No sign change");
  let left = a, right = b, mid = 0, fm = 0;
  for (let k = 0; k < maxIter; k++) {
    mid = 0.5 * (left + right);
    fm = f(mid);
    if (Math.abs(fm) <= eps || 0.5 * (right - left) <= eps)
      return { root: mid, iterations: k + 1, converged: true, fAtRoot: fm };
    if (fa * fm < 0) { right = mid; fb = fm; } else { left = mid; fa = fm; }
  }
  return { root: mid, iterations: maxIter, converged: false, fAtRoot: fm };
}

function newtonRef(f: (x: number) => number, df: (x: number) => number, x0: number, eps = 1e-12, maxIter = 100) {
  let x = x0, fx = f(x);
  if (Math.abs(fx) <= eps) return { root: x, iterations: 0, converged: true, fAtRoot: fx };
  for (let k = 0; k < maxIter; k++) {
    const dfx = df(x);
    if (dfx === 0) throw new Error("Zero derivative");
    const xNext = x - fx / dfx;
    const fxNext = f(xNext);
    if (Math.abs(fxNext) <= eps || Math.abs(xNext - x) <= eps)
      return { root: xNext, iterations: k + 1, converged: true, fAtRoot: fxNext };
    x = xNext; fx = fxNext;
  }
  return { root: x, iterations: maxIter, converged: false, fAtRoot: fx };
}

function secantRef(f: (x: number) => number, x0: number, x1: number, eps = 1e-12, maxIter = 100) {
  let xPrev = x0, x = x1;
  let fPrev = f(xPrev), fx = f(x);
  if (Math.abs(fx) <= eps) return { root: x, iterations: 0, converged: true, fAtRoot: fx };
  for (let k = 0; k < maxIter; k++) {
    const denom = fx - fPrev;
    if (denom === 0) throw new Error("Zero slope");
    const xNext = x - fx * (x - xPrev) / denom;
    const fxNext = f(xNext);
    if (Math.abs(fxNext) <= eps || Math.abs(xNext - x) <= eps)
      return { root: xNext, iterations: k + 1, converged: true, fAtRoot: fxNext };
    xPrev = x; fPrev = fx;
    x = xNext; fx = fxNext;
  }
  return { root: x, iterations: maxIter, converged: false, fAtRoot: fx };
}

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

function trim(n: number) {
  if (!Number.isFinite(n)) return String(n);
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a >= 1e6 || a < 1e-6) return n.toExponential(6);
  return n.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
}

async function touchGas(contract: Contract, method: string, args: any[]) {
  const data = contract.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await contract.getAddress();
  const tx = await signer.sendTransaction({ to, data });
  await tx.wait();
}

async function estimateGas(contract: Contract, method: string, args: any[]) {
  const anyC = contract as any;
  if (anyC[method]?.estimateGas) {
    return (await anyC[method].estimateGas(...args)).toString();
  }
  const data = contract.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await contract.getAddress();
  const gas = await signer.estimateGas({ to, data });
  return gas.toString();
}

function printBlock({ t, method, explanation, gas, inHex, outHex, outDec }: any) {
  const sep = "-".repeat(60);
  const decLine = outDec ? `Output: ${outDec}` : "";

  console.log(`
        ${sep}
        Test ${t}
        Method: ${method}
        Explanation: ${explanation}
        Gas Usage: ${gas}
        Input: ${inHex}
        Output (hex): ${outHex}
        ${decLine}
`.trim());
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("RootFinding — Report (values + gas)", function () {
  let harness: RootFindingHarness;
  let root: RootFindingFacet;
  let t = 0;

  // Selectors
  let sel_fx2m4: string;
  let sel_df2x: string;
  let sel_fcubic: string;
  let sel_dfcubic: string;

  // Helper Wrappers
  const qInt = async (n: number | string) => await harness.qFromInt(BigInt(n));
  const qFrac = async (n: number | string, d: number | string) => await harness.qFromFrac(BigInt(n), BigInt(d));

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

    sel_fx2m4 = ethers.id("f_x2_minus_4(bytes16)").slice(0, 10);
    sel_df2x = ethers.id("df_2x(bytes16)").slice(0, 10);
    sel_fcubic = ethers.id("f_cubic(bytes16)").slice(0, 10);
    sel_dfcubic = ethers.id("df_cubic(bytes16)").slice(0, 10);
  });

  // ------------------------------------------------------------
  //  Bisection Method
  // ------------------------------------------------------------

  describe("Section 1: Bisection", function () {

    it("Test 1: Standard (x^2-4 on [1,3])", async function () {
      t++;
      const a = await qInt(1);
      const b = await qInt(3);
      const ref = bisectionRef(f_x2_minus_4, 1, 3);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fx2m4, a, b);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "bisection",
        explanation: "Standard convergence root~2",
        gas,
        inHex: "f=x^2-4, [1,3]",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 2: Reversed Interval Auto-swap (x^2-4 on [3,1])", async function () {
      t++;
      const a = await qInt(3);
      const b = await qInt(1);
      const ref = bisectionRef(f_x2_minus_4, 3, 1);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fx2m4, a, b);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "bisection",
        explanation: "Auto-swap a>b",
        gas,
        inHex: "f=x^2-4, [3,1]",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 3: Immediate Convergence (f(a)=0)", async function () {
      t++;
      const a = await qInt(2);
      const b = await qInt(3);
      const ref = bisectionRef(f_x2_minus_4, 2, 3);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fx2m4, a, b]);
      const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fx2m4, a, b);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "bisection",
        explanation: "Endpoint is root",
        gas,
        inHex: "f=x^2-4, [2,3]",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 4: Revert on No Sign Change", async function () {
      t++;
      const a = await qInt(3);
      const b = await qInt(4);
      const target = await harness.getAddress();

      await expect(root.rootFindingBisection(target, sel_fx2m4, a, b)).to.be.reverted;

      printBlock({
        t,
        method: "bisection",
        explanation: "Revert (no sign change)",
        gas: "N/A",
        inHex: "f=x^2-4, [3,4]",
        outHex: "Reverted",
        outDec: "Reverted"
      });
    });

    it("Test 5: Cubic Function (x^3-x-2 on [1,2])", async function () {
      t++;
      const a = await qInt(1);
      const b = await qInt(2);
      const ref = bisectionRef(f_cubic, 1, 2);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingBisection", [target, sel_fcubic, a, b]);
      const gas = await estimateGas(root, "rootFindingBisection", [target, sel_fcubic, a, b]);
      const [rHex, iters, ok, fHex] = await root.rootFindingBisection(target, sel_fcubic, a, b);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "bisection",
        explanation: "Cubic root ≈1.521",
        gas,
        inHex: "f=x^3-x-2, [1,2]",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });
  });

  // ------------------------------------------------------------
  //  Newton-Raphson Method
  // ------------------------------------------------------------

  describe("Section 2: Newton-Raphson", function () {

    it("Test 6: Quadratic Standard (x^2-4, x0=3)", async function () {
      t++;
      const x0 = await qInt(3);
      const ref = newtonRef(f_x2_minus_4, df_2x, 3);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fx2m4, target, sel_df2x, x0);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "newton",
        explanation: "Quadratic convergence",
        gas,
        inHex: "f=x^2-4, df=2x, x0=3",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 7: Cubic Standard (x^3-x-2, x0=1)", async function () {
      t++;
      const x0 = await qInt(1);
      const ref = newtonRef(f_cubic, df_cubic, 1);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingNewton", [target, sel_fcubic, target, sel_dfcubic, x0]);
      const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fcubic, target, sel_dfcubic, x0]);
      const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fcubic, target, sel_dfcubic, x0);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "newton",
        explanation: "Cubic root finding",
        gas,
        inHex: "f=x^3-x-2, df=3x^2-1, x0=1",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 8: Immediate Convergence (x0=2)", async function () {
      t++;
      const x0 = await qInt(2);
      const ref = newtonRef(f_x2_minus_4, df_2x, 2);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fx2m4, target, sel_df2x, x0);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "newton",
        explanation: "Guess is root",
        gas,
        inHex: "f=x^2-4, df=2x, x0=2",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 9: Revert on Zero Derivative (x0=0)", async function () {
      t++;
      const x0 = await qInt(0);
      const target = await harness.getAddress();

      await expect(root.rootFindingNewton(target, sel_fx2m4, target, sel_df2x, x0)).to.be.reverted;

      printBlock({
        t,
        method: "newton",
        explanation: "Revert (df=0)",
        gas: "N/A",
        inHex: "f=x^2-4, df=2x, x0=0",
        outHex: "Reverted",
        outDec: "Reverted"
      });
    });

    it("Test 10: Tight Tolerance (x0=3)", async function () {
      t++;
      const x0 = await qInt(3);
      const ref = newtonRef(f_x2_minus_4, df_2x, 3);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const gas = await estimateGas(root, "rootFindingNewton", [target, sel_fx2m4, target, sel_df2x, x0]);
      const [rHex, iters, ok, fHex] = await root.rootFindingNewton(target, sel_fx2m4, target, sel_df2x, x0);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "newton",
        explanation: "Check stop conditions",
        gas,
        inHex: "f=x^2-4, df=2x, x0=3",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });
  });

  // ------------------------------------------------------------
  //  Secant Method
  // ------------------------------------------------------------

  describe("Section 3: Secant", function () {

    it("Test 11: Standard (x^2-4, x0=1, x1=3)", async function () {
      t++;
      const x0 = await qInt(1);
      const x1 = await qInt(3);
      const ref = secantRef(f_x2_minus_4, 1, 3);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
      const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
      const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fx2m4, x0, x1);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "secant",
        explanation: "Derivative-free",
        gas,
        inHex: "f=x^2-4, x0=1, x1=3",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 12: Cubic Function (x0=1, x1=2)", async function () {
      t++;
      const x0 = await qInt(1);
      const x1 = await qInt(2);
      const ref = secantRef(f_cubic, 1, 2);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingSecant", [target, sel_fcubic, x0, x1]);
      const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fcubic, x0, x1]);
      const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fcubic, x0, x1);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "secant",
        explanation: "Cubic root",
        gas,
        inHex: "f=x^3-x-2, x0=1, x1=2",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 13: Immediate Convergence (x1=2)", async function () {
      t++;
      const x0 = await qInt(1);
      const x1 = await qInt(2); // Root
      const ref = secantRef(f_x2_minus_4, 1, 2);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
      const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fx2m4, x0, x1]);
      const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fx2m4, x0, x1);

      expect(ok).to.eq(true);
      printBlock({
        t,
        method: "secant",
        explanation: "x1 is root",
        gas,
        inHex: "f=x^2-4, x0=1, x1=2",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });

    it("Test 14: Revert on Zero Slope (x0=1, x1=-1)", async function () {
      t++;
      const x0 = await qInt(1);
      const x1 = await qInt(-1);
      const target = await harness.getAddress();

      await expect(root.rootFindingSecant(target, sel_fx2m4, x0, x1)).to.be.reverted;

      printBlock({
        t,
        method: "secant",
        explanation: "Revert (denom=0)",
        gas: "N/A",
        inHex: "f=x^2-4, x0=1, x1=-1",
        outHex: "Reverted",
        outDec: "Reverted"
      });
    });

    it("Test 15: Poor Start Points (Wide)", async function () {
      t++;
      const x0 = await qInt(-10);
      const x1 = await qInt(10);
      const ref = secantRef(f_cubic, -10, 10, 1e-12, 20);
      const target = await harness.getAddress();

      await touchGas(root, "rootFindingSecant", [target, sel_fcubic, x0, x1]);
      const gas = await estimateGas(root, "rootFindingSecant", [target, sel_fcubic, x0, x1]);
      const [rHex, iters, ok, fHex] = await root.rootFindingSecant(target, sel_fcubic, x0, x1);

      printBlock({
        t,
        method: "secant",
        explanation: "Slow progress / Max iter",
        gas,
        inHex: "f=x^3-x-2, x0=-10, x1=10",
        outHex: `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
        outDec: `ref~${trim(ref.root)}, f(ref)~${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
      });
    });
  });
});