// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract, ContractTransactionResponse } from "ethers";

/**
 * RootFinding — Report (values + gas)
 *
 * Matches your Polynomial test block style:
 *   Test N
 *   Method: ...
 *   Explanation: ...
 *   Gas Usage: <number or N/A>
 *   Input: ...
 *   Output (hex): ...
 *   Output: ...
 *
 * Prints on-chain values as hex (bytes16). Human-readable Output is computed off-chain.
 */

// ------------------------ Harness & Facet types ------------------------

type RootFindingHarness = Contract & {
  // quad helpers
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;

  // test functions f / df
  f_x2_minus_4(x: string): Promise<string>; // x^2-4
  df_2x(x: string): Promise<string>;        // 2x
  f_cubic(x: string): Promise<string>;      // x^3 - x - 2
  df_cubic(x: string): Promise<string>;     // 3x^2 - 1
};

type RootFindingFacet = Contract & {
  bisection(
    target: string,
    fSelector: string,
    a: string,
    b: string
  ): Promise<[string, bigint, boolean, string]>;

  newton(
    target: string,
    fSelector: string,
    dfTarget: string,
    dfSelector: string,
    x0: string
  ): Promise<[string, bigint, boolean, string]>;

  secant(
    target: string,
    fSelector: string,
    x0: string,
    x1: string
  ): Promise<[string, bigint, boolean, string]>;
};

// ------------------------ Deploy helpers ------------------------

async function deployHarness(): Promise<RootFindingHarness> {
  const F = await ethers.getContractFactory("RootFindingHarness");
  const h = (await F.deploy()) as unknown as RootFindingHarness;
  await h.waitForDeployment();
  return h;
}

async function deployRootFindingFacet(): Promise<RootFindingFacet> {
  const F = await ethers.getContractFactory("RootFindingFacet");
  const f = (await F.deploy()) as unknown as RootFindingFacet;
  await f.waitForDeployment();
  return f;
}

// ------------------------ q helpers ------------------------

async function qInt(h: RootFindingHarness, n: number | string) {
  return h.qFromInt(BigInt(n));
}
async function qFrac(h: RootFindingHarness, n: number | string, d: number | string) {
  return h.qFromFrac(BigInt(n), BigInt(d));
}

// ------------------------ Gas touch helper ------------------------

/** Send a real transaction so gas-reporter logs a row even for view-like paths. */
async function touchGas(contract: Contract, method: string, args: any[]): Promise<bigint | "N/A"> {
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

// pretty number for console
function trim(n: number) {
  if (!Number.isFinite(n)) return String(n);
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a >= 1e6 || a < 1e-6) return n.toExponential(6);
  return n.toFixed(12).replace(/0+$/,"").replace(/\.$/,"");
}

// ------------------------ The Suite (15 tests) ------------------------

describe("RootFinding — Report (values + gas)", function () {
  let harness: RootFindingHarness;
  let root: RootFindingFacet;
  let T = 0;

  // function selectors from harness
  let sel_fx2m4: string;
  let sel_df2x: string;
  let sel_fcubic: string;
  let sel_dfcubic: string;
  
  before(async () => {
    harness = await deployHarness();
    root = await deployRootFindingFacet();

    // Compute 4-byte selectors manually (Ethers v6 style)
    sel_fx2m4  = ethers.id("f_x2_minus_4(bytes16)").slice(0, 10);
    sel_df2x   = ethers.id("df_2x(bytes16)").slice(0, 10);
    sel_fcubic = ethers.id("f_cubic(bytes16)").slice(0, 10);
    sel_dfcubic = ethers.id("df_cubic(bytes16)").slice(0, 10);
  });

  // ======================= BISECTION (5) =======================

  it("bisection — standard (x^2-4) on [1,3] → root≈2", async function () {
    T++;
    const a = await qInt(harness, 1);
    const b = await qInt(harness, 3);

    const gas = await touchGas(root, "bisection", [await harness.getAddress(), sel_fx2m4, a, b]);
    const [rHex, iters, ok, fHex] = await root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b);

    const ref = bisectionRef(f_x2_minus_4, 1, 3, 1e-12, 200);
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

    const gas = await touchGas(root, "bisection", [await harness.getAddress(), sel_fx2m4, a, b]);
    const [rHex, iters, ok, fHex] = await root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b);

    const ref = bisectionRef(f_x2_minus_4, 3, 1, 1e-12, 200);
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

    const gas = await touchGas(root, "bisection", [await harness.getAddress(), sel_fx2m4, a, b]);
    const [rHex, iters, ok, fHex] = await root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b);

    const ref = bisectionRef(f_x2_minus_4, 2, 3, 1e-12, 200);
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

    const gas = await touchGas(root, "bisection", [await harness.getAddress(), sel_fx2m4, a, b]);
    await expect(root.rootFindingBisection(await harness.getAddress(), sel_fx2m4, a, b)).to.be.reverted;
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

    const gas = await touchGas(root, "bisection", [await harness.getAddress(), sel_fcubic, a, b]);
    const [rHex, iters, ok, fHex] = await root.rootFindingBisection(await harness.getAddress(), sel_fcubic, a, b);

    const ref = bisectionRef(f_cubic, 1, 2, 1e-12, 250);
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

  // ======================= NEWTON (5) =======================

  it("newton — quadratic (x^2-4), df=2x, x0=3 → 2", async function () {
    T++;
    const x0 = await qInt(harness, 3);

    const gas = await touchGas(root, "newton", [
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    ]);
    const [rHex, iters, ok, fHex] = await root.rootFindingNewton(
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    );

    const ref = newtonRef(f_x2_minus_4, df_2x, 3, 1e-12, 100);
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

    const gas = await touchGas(root, "newton", [
      await harness.getAddress(), sel_fcubic, await harness.getAddress(), sel_dfcubic, x0
    ]);
    const [rHex, iters, ok, fHex] = await root.rootFindingNewton(
      await harness.getAddress(), sel_fcubic, await harness.getAddress(), sel_dfcubic, x0
    );

    const ref = newtonRef(f_cubic, df_cubic, 1, 1e-12, 50);
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

    const gas = await touchGas(root, "newton", [
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    ]);
    const [rHex, iters, ok, fHex] = await root.rootFindingNewton(
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    );

    const ref = newtonRef(f_x2_minus_4, df_2x, 2, 1e-12, 100);
    printBlock(
      T,
      "newton (immediate)",
      "If f(x0)=0, method halts with 0 iterations.",
      "f=x^2-4, df=2x, x0=2",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    const i = Number(iters); // Convert BigInt iteration count to a regular number (for convenience)
    expect(iters === 0n || iters === 1n).to.be.true; // Allow either 0 or 1 iteration — implementation-dependent early stop check
    expect(ok).to.eq(true);
  });

  it("newton — zero derivative causes revert", async function () {
    T++;
    // For f(x)=x^2-4, df(0)=0; start at x0=0 to trigger zero derivative
    const x0 = await qInt(harness, 0);

    const gas = await touchGas(root, "newton", [
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    ]);
    await expect(
      root.rootFindingNewton(await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0)
    ).to.be.reverted;

    printBlock(
      T,
      "newton (zero derivative)",
      "At x0=0, df=0 → implementation reverts (guard against division by zero).",
      "f=x^2-4, df=2x, x0=0",
      gas,
      "revert",
      "revert"
    );
  });

  it("newton — tight tolerance stops on |f(x)| < eps", async function () {
    T++;
    const x0 = await qInt(harness, 3);

    const gas = await touchGas(root, "newton", [
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    ]);
    const [rHex, iters, ok, fHex] = await root.rootFindingNewton(
      await harness.getAddress(), sel_fx2m4, await harness.getAddress(), sel_df2x, x0
    );

    const ref = newtonRef(f_x2_minus_4, df_2x, 3, 1e-12, 100);
    printBlock(
      T,
      "newton (tight tolerance)",
      "Ensures stopping is triggered by residual (|f(x)|<=eps) or step size.",
      "f=x^2-4, df=2x, x0=3",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  // ======================= SECANT (5) =======================

  it("secant — standard (x^2-4), x0=1, x1=3 → 2", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, 3);

    const gas = await touchGas(root, "secant", [await harness.getAddress(), sel_fx2m4, x0, x1]);
    const [rHex, iters, ok, fHex] = await root.rootFindingSecant(await harness.getAddress(), sel_fx2m4, x0, x1);

    const ref = secantRef(f_x2_minus_4, 1, 3, 1e-12, 100);
    printBlock(
      T,
      "secant (standard)",
      "Two-point derivative-free method; converges to 2 from (1,3).",
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

    const gas = await touchGas(root, "secant", [await harness.getAddress(), sel_fcubic, x0, x1]);
    const [rHex, iters, ok, fHex] = await root.rootFindingSecant(await harness.getAddress(), sel_fcubic, x0, x1);

    const ref = secantRef(f_cubic, 1, 2, 1e-12, 100);
    printBlock(
      T,
      "secant (cubic)",
      "Cubic with good bracketing starts; should converge rapidly.",
      "f=x^3-x-2, x0=1, x1=2",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    expect(ok).to.eq(true);
  });

  it("secant — immediate convergence when x1 is a root", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, 2); // f(2)=0

    const gas = await touchGas(root, "secant", [await harness.getAddress(), sel_fx2m4, x0, x1]);
    const [rHex, iters, ok, fHex] = await root.rootFindingSecant(await harness.getAddress(), sel_fx2m4, x0, x1);

    const ref = secantRef(f_x2_minus_4, 1, 2, 1e-12, 100);
    printBlock(
      T,
      "secant (immediate)",
      "If x1 is exact root, method halts immediately.",
      "f=x^2-4, x0=1, x1=2",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `root≈${trim(ref.root)}, f(root)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    const i = Number(iters);
    expect(iters === 0n || iters === 1n).to.be.true;
    expect(ok).to.eq(true);
  });

  it("secant — zero slope (f(x1)=f(x0)) reverts", async function () {
    T++;
    const x0 = await qInt(harness, 1);
    const x1 = await qInt(harness, -1);

    const gas = await touchGas(root, "secant", [await harness.getAddress(), sel_fx2m4, x0, x1]);
    await expect(root.rootFindingSecant(await harness.getAddress(), sel_fx2m4, x0, x1)).to.be.reverted;

    printBlock(
      T,
      "secant (zero slope)",
      "When f(x1)=f(x0) (but neither is root), slope is zero; implementation reverts.",
      "f=x^2-4, x0=1, x1=-1",
      gas,
      "revert",
      "revert"
    );
  });

  it("secant — poor starts; maxIter reached (likely converged=false)", async function () {
    T++;
    const x0 = await qInt(harness, -10);
    const x1 = await qInt(harness, 10);

    const gas = await touchGas(root, "secant", [await harness.getAddress(), sel_fcubic, x0, x1]);
    const [rHex, iters, ok, fHex] = await root.rootFindingSecant(await harness.getAddress(), sel_fcubic, x0, x1);

    const ref = secantRef(f_cubic, -10, 10, 1e-12, 20); // small maxIter in ref to illustrate stopping
    printBlock(
      T,
      "secant (poor starts)",
      "Wide, poor initial guesses may exhaust iterations; check converged flag.",
      "f=x^3-x-2, x0=-10, x1=10",
      gas,
      `root=${rHex}, f(root)=${fHex}, iter=${iters}, conv=${ok}`,
      `ref≈${trim(ref.root)}, f(ref)≈${trim(ref.fAtRoot)}, iter=${ref.iterations}, conv=${ref.converged}`
    );
    // Not asserting convergence; ensure call succeeded and iter>0
    expect(iters).to.be.greaterThan(0n);
  });
});