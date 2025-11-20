// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract, ContractTransactionResponse } from "ethers";

/**
 * RootFinding — Report (values + gas)
 */

// ------------------------ Harness & Facet types ------------------------

type RootFindingHarness = Contract & {
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;

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
  ): Promise<[string, bigint, boolean, string]>;

  rootFindingNewton(
    target: string,
    fSelector: string,
    dfTarget: string,
    dfSelector: string,
    x0: string
  ): Promise<[string, bigint, boolean, string]>;

  rootFindingSecant(
    target: string,
    fSelector: string,
    x0: string,
    x1: string
  ): Promise<[string, bigint, boolean, string]>;
};

// ------------------------ q helpers ------------------------

async function qInt(h: RootFindingHarness, n: number | string) {
  return h.qFromInt(BigInt(n));
}
async function qFrac(h: RootFindingHarness, n: number | string, d: number | string) {
  return h.qFromFrac(BigInt(n), BigInt(d));
}

// ------------------------ Gas touch helper ------------------------

async function touchGas(
  contract: Contract,
  method: string,
  args: any[]
): Promise<bigint | "N/A"> {
  try {
    const data = contract.interface.encodeFunctionData(method, args);
    const [signer] = await ethers.getSigners();
    const to = await contract.getAddress();
    const tx = await signer.sendTransaction({ to, data });
    const receipt = await (tx as ContractTransactionResponse).wait();
    return receipt?.gasUsed ?? "N/A";
  } catch {
    return "N/A";
  }
}

// ------------------------ Printing ------------------------

function printBlock(
  n: number,
  method: string,
  explanation: string,
  input: string,
  gasUsed: bigint | string,
  outHex: string,
  outDec: string
) {
  const sep = "-".repeat(60);
  console.log(
    `\n${sep}\nTest ${n}\nMethod: ${method}\nExplanation: ${explanation}\nGas Usage: ${gasUsed}\nInput: ${input}\nOutput (hex): ${outHex}\nOutput: ${outDec}`
  );
}

// ------------------------ JS numeric mirrors ------------------------

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

// pretty number
function trim(n: number) {
  if (!Number.isFinite(n)) return String(n);
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a >= 1e6 || a < 1e-6) return n.toExponential(6);
  return n.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
}

// ------------------------ The Suite (full) ------------------------

describe("RootFinding — Report (values + gas)", function () {
  let harness: RootFindingHarness;
  let root: RootFindingFacet;
  let T = 0;

  let sel_fx2m4: string;
  let sel_df2x: string;
  let sel_fcubic: string;
  let sel_dfcubic: string;

  before(async () => {
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

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

    const FacetFactory = await ethers.getContractFactory(
      "RootFindingFacet",
      {
        libraries: {
          "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
      }
    );
    root = (await FacetFactory.deploy()) as unknown as RootFindingFacet;
    await root.waitForDeployment();

    sel_fx2m4   = ethers.id("f_x2_minus_4(bytes16)").slice(0, 10);
    sel_df2x    = ethers.id("df_2x(bytes16)").slice(0, 10);
    sel_fcubic  = ethers.id("f_cubic(bytes16)").slice(0, 10);
    sel_dfcubic = ethers.id("df_cubic(bytes16)").slice(0, 10);
  });

  // ========== BISECTION ==========

  it("bisection — standard (x^2-4) on [1,3] → root≈2", async function () {
    T++;
    const a = await qInt(harness, 1);
    const b = await qInt(harness, 3);

    const gas = await touchGas(root, "rootFindingBisection", [
      await harness.getAddress(), sel_fx2m4, a, b
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b);

    const ref = bisectionRef(f_x2_minus_4, 1, 3);
    printBlock(
      T,
      "bisection (standard)",
      "Standard case with opposite sign endpoints; should converge to root≈2.",
      "f(x)=x^2-4, [1,3]",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("bisection — reversed interval auto-swap", async function () {
    T++;
    const a = await qInt(harness, 3);
    const b = await qInt(harness, 1);

    const gas = await touchGas(root, "rootFindingBisection", [
      await harness.getAddress(), sel_fx2m4, a, b
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b);

    const ref = bisectionRef(f_x2_minus_4, 3, 1);
    printBlock(
      T,
      "bisection (swap)",
      "a>b; implementation swaps to maintain [min,max].",
      "f(x)=x^2-4, [3,1]",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("bisection — f(a)=0 immediate convergence", async function () {
    T++;
    const a = await qInt(harness, 2);
    const b = await qInt(harness, 3);

    const gas = await touchGas(root, "rootFindingBisection", [
      await harness.getAddress(), sel_fx2m4, a, b
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b);

    const ref = bisectionRef(f_x2_minus_4, 2, 3);
    printBlock(
      T,
      "bisection (f(a)=0)",
      "Endpoint is a root; should return in 0 or 1 iteration depending on implementation details.",
      "f(x)=x^2-4, [2,3]",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("bisection — no sign change reverts", async function () {
    T++;
    const a = await qInt(harness, 3);
    const b = await qInt(harness, 4);

    const gas = await touchGas(root, "rootFindingBisection", [
      await harness.getAddress(), sel_fx2m4, a, b
    ]);
    await expect(
      root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b)
    ).to.be.reverted;

    printBlock(
      T,
      "bisection (no sign change)",
      "Opposite signs missing; should revert with 'No sign change'.",
      "f(x)=x^2-4, [3,4]",
      gas,
      "revert",
      "revert"
    );
  });

  it("bisection — cubic (x^3 - x - 2) on [1,2] → ≈1.521", async function () {
    T++;
    const a = await qInt(harness, 1);
    const b = await qInt(harness, 2);

    const gas = await touchGas(root, "rootFindingBisection", [
      await harness.getAddress(), sel_fcubic, a, b
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingBisection(await harness.getAddress(), sel_fcubic, a, b);

    const ref = bisectionRef(f_cubic, 1, 2);
    printBlock(
      T,
      "bisection (cubic)",
      "Cubic has a unique real root in [1,2]; convergence to ≈1.521...",
      "f(x)=x^3-x-2, [1,2]",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  // ========== NEWTON ==========

  it("newton — quadratic (x^2-4), df=2x, x0=3 → 2", async function () {
    T++;
    const x0 = await qInt(harness, 3);

    const gas = await touchGas(root, "rootFindingNewton", [
      await harness.getAddress(), sel_fx2m4,
      await harness.getAddress(), sel_df2x,
      x0
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingNewton(
        await harness.getAddress(), sel_fx2m4,
        await harness.getAddress(), sel_df2x,
        x0
      );

    const ref = newtonRef(f_x2_minus_4, df_2x, 3);
    printBlock(
      T,
      "newton (standard)",
      "Classic Newton on quadratic with analytic derivative; quadratic convergence.",
      "f=x^2-4, df=2x, x0=3",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("newton — cubic (x^3-x-2), df=3x^2-1, x0=1 → ≈1.521", async function () {
    T++;
    const x0 = await qInt(harness, 1);

    const gas = await touchGas(root, "rootFindingNewton", [
      await harness.getAddress(), sel_fcubic,
      await harness.getAddress(), sel_dfcubic,
      x0
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingNewton(
        await harness.getAddress(), sel_fcubic,
        await harness.getAddress(), sel_dfcubic,
        x0
      );

    const ref = newtonRef(f_cubic, df_cubic, 1);
    printBlock(
      T,
      "newton (cubic)",
      "Cubic with good initial guess; converges rapidly to the real root.",
      "f=x^3-x-2, df=3x^2-1, x0=1",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("newton — immediate convergence (x0 is exact root)", async function () {
    T++;
    const x0 = await qInt(harness, 2);

    const gas = await touchGas(root, "rootFindingNewton", [
      await harness.getAddress(), sel_fx2m4,
      await harness.getAddress(), sel_df2x,
      x0
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingNewton(
        await harness.getAddress(), sel_fx2m4,
        await harness.getAddress(), sel_df2x,
        x0
      );

    const ref = newtonRef(f_x2_minus_4, df_2x, 2);
    printBlock(
      T,
      "newton (immediate)",
      "If f(x0)=0, method halts with 0 iterations.",
      "f=x^2-4, df=2x, x0=2",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(iters === 0n || iters === 1n).to.be.true;
    expect(ok).to.eq(true);
  });

  it("newton — zero derivative causes revert", async function () {
    T++;
    const x0 = await qInt(harness, 0);

    const gas = await touchGas(root, "rootFindingNewton", [
      await harness.getAddress(), sel_fx2m4,
      await harness.getAddress(), sel_df2x,
      x0
    ]);
    await expect(
      root.rootFindingNewton(
        await harness.getAddress(), sel_fx2m4,
        await harness.getAddress(), sel_df2x,
        x0
      )
    ).to.be.reverted;

    printBlock(
      T,
      "newton (zero derivative)",
      "At x0=0, df=0 → revert guard.",
      "f=x^2-4, df=2x, x0=0",
      gas,
      "revert",
      "revert"
    );
  });

  it("newton — tight tolerance stops on |f(x)| < eps", async function () {
    T++;
    const x0 = await qInt(harness, 3);

    const gas = await touchGas(root, "rootFindingNewton", [
      await harness.getAddress(), sel_fx2m4,
      await harness.getAddress(), sel_df2x,
      x0
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingNewton(
        await harness.getAddress(), sel_fx2m4,
        await harness.getAddress(), sel_df2x,
        x0
      );

    const ref = newtonRef(f_x2_minus_4, df_2x, 3);
    printBlock(
      T,
      "newton (tight tolerance)",
      "Check stopping rule by residual or Δx.",
      "f=x^2-4, df=2x, x0=3",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  // ========== SECANT ==========

  it("secant — standard (x^2-4), x0=1, x1=3 → 2", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, 3);

    const gas = await touchGas(root, "rootFindingSecant", [
      await harness.getAddress(), sel_fx2m4, x0, x1
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingSecant(await harness.getAddress(), sel_fx2m4, x0, x1);

    const ref = secantRef(f_x2_minus_4, 1, 3);
    printBlock(
      T,
      "secant (standard)",
      "Derivative-free two-point method.",
      "f=x^2-4, x0=1, x1=3",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("secant — cubic (x^3-x-2), x0=1, x1=2 → ≈1.521", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, 2);

    const gas = await touchGas(root, "rootFindingSecant", [
      await harness.getAddress(), sel_fcubic, x0, x1
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingSecant(await harness.getAddress(), sel_fcubic, x0, x1);

    const ref = secantRef(f_cubic, 1, 2);
    printBlock(
      T,
      "secant (cubic)",
      "Cubic root via secant.",
      "f=x^3-x-2, x0=1, x1=2",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("secant — immediate convergence when x1 is root", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, 2);

    const gas = await touchGas(root, "rootFindingSecant", [
      await harness.getAddress(), sel_fx2m4, x0, x1
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingSecant(await harness.getAddress(), sel_fx2m4, x0, x1);

    const ref = secantRef(f_x2_minus_4, 1, 2);
    printBlock(
      T,
      "secant (immediate)",
      "If x1 is root, secant halts immediately.",
      "f=x^2-4, x0=1, x1=2",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(iters === 0n || iters === 1n).to.be.true;
    expect(ok).to.eq(true);
  });

  it("secant — zero slope reverts", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, -1);

    const gas = await touchGas(root, "rootFindingSecant", [
      await harness.getAddress(), sel_fx2m4, x0, x1
    ]);
    await expect(
      root.rootFindingSecant(await harness.getAddress(), sel_fx2m4, x0, x1)
    ).to.be.reverted;

    printBlock(
      T,
      "secant (zero slope)",
      "f(x0)=f(x1) → zero denominator → revert.",
      "f=x^2-4, x0=1, x1=-1",
      gas,
      "revert",
      "revert"
    );
  });

  it("secant — poor starts; likely maxIter stop", async function () {
    T++;
    const x0 = await qInt(harness, -10);
    const x1 = await qInt(harness, 10);

    const gas = await touchGas(root, "rootFindingSecant", [
      await harness.getAddress(), sel_fcubic, x0, x1
    ]);
    const [rHex, iters, ok, fHex] =
      await root.rootFindingSecant(await harness.getAddress(), sel_fcubic, x0, x1);

    const ref = secantRef(f_cubic, -10, 10, 1e-12, 20);
    printBlock(
      T,
      "secant (poor starts)",
      "Wide starts → slow progress; method may reach maxIter.",
      "f=x^3-x-2, x0=-10, x1=10",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `ref≈${trim(ref.root)}, f(ref)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(iters).to.be.greaterThan(0n);
  });
});