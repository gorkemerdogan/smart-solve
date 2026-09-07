// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";

const PRE_OPTIMIZATION_OUTPUTS: Record<string, string[]> = {
  "rectangular-fixed-1": [
    "0x3ffd6666666666666666666666666665",
    "0xbffdcccccccccccccccccccccccccccc",
    "0x3ffe7fffffffffffffffffffffffffff",
  ],
  "rectangular-fixed-5": [
    "0x3ffece79e59f2ba9d1f601797cc39ffc",
    "0xbfff86e2339c0ebedfa43fe5c91d14e2",
    "0x40001d01b866e43aa79bbadc0980b241",
  ],
  "rectangular-fixed-12": [
    "0x3fff096af0cb4e413934a19e85cf5aa0",
    "0xbffff2e00c09161a5f4769b697629561",
    "0x40006e4a49e472e9c47d2a0916af265b",
  ],
};

describe("Gradient-descent least-squares temporary-allocation benchmark", function () {
  this.timeout(120_000);

  let solver: any;
  let rectangularA: string[];
  let rectangularB: string[];
  let rectangularX0: string[];
  let alpha: string;
  let strictTolerance: string;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    const mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();

    const Harness = await ethers.getContractFactory("LinearSolversHarness", {
      libraries: { MathLib: await mathLib.getAddress() },
    });
    solver = await Harness.deploy();
    await solver.waitForDeployment();

    // Deterministic 4x3 least-squares system with b = A * [1, -2, 3].
    rectangularA = await Promise.all([
      2n, 0n, 1n,
      1n, 3n, 0n,
      0n, 1n, 2n,
      1n, 1n, 1n,
    ].map(value => solver.qFromInt(value)));
    rectangularB = await Promise.all([5n, -5n, 4n, 2n].map(value => solver.qFromInt(value)));
    rectangularX0 = await Promise.all([0n, 0n, 0n].map(value => solver.qFromInt(value)));
    alpha = await solver.qFromFrac(1n, 20n);
    strictTolerance = await solver.qFromFrac(1n, 10n ** 30n);

    console.log("============================================================");
    console.log("GRADIENT-DESCENT LEAST-SQUARES TEMPORARY-ALLOCATION BENCHMARK");
    console.log("Execution model: LinearSolversHarness estimateGas + eth_call result");
    console.log("Cases: deterministic fixed-budget progress and exact early convergence");
    console.log("============================================================");
  });

  async function runRectangularCase(maxIter: bigint): Promise<void> {
    const callable = solver.getFunction("gradientDescentLeastSquares");
    const args = [4n, 3n, rectangularA, rectangularB, rectangularX0, alpha, maxIter, strictTolerance];
    const gas = await callable.estimateGas(...args);
    const [output, iterations] = await callable.staticCall(...args);

    expect(iterations).to.equal(maxIter);
    expect([...output], `fixed-${maxIter}: binary128 output changed from pre-optimization baseline`)
      .to.deep.equal(PRE_OPTIMIZATION_OUTPUTS[`rectangular-fixed-${maxIter}`]);
    console.log(
      `GD_ALLOCATION_GAS | case=rectangular-fixed-${maxIter} | gas=${gas} | ` +
      `iterations=${iterations} | output=${output.join(",")}`
    );
  }

  it("measures fixed-budget progress at several iteration counts", async function () {
    await runRectangularCase(1n);
    await runRectangularCase(5n);
    await runRectangularCase(12n);
  });

  it("preserves exact early-convergence behavior", async function () {
    const identity = await Promise.all([
      1n, 0n, 0n,
      0n, 1n, 0n,
      0n, 0n, 1n,
    ].map(value => solver.qFromInt(value)));
    const target = await Promise.all([2n, -1n, 4n].map(value => solver.qFromInt(value)));
    const x0 = await Promise.all([0n, 0n, 0n].map(value => solver.qFromInt(value)));
    const alphaOne = await solver.qFromInt(1n);
    const callable = solver.getFunction("gradientDescentLeastSquares");
    const args = [3n, 3n, identity, target, x0, alphaOne, 10n, strictTolerance];
    const gas = await callable.estimateGas(...args);
    const [output, iterations] = await callable.staticCall(...args);

    expect(iterations).to.equal(1n);
    expect([...output]).to.deep.equal(target);
    console.log(
      `GD_ALLOCATION_GAS | case=identity-early-convergence | gas=${gas} | ` +
      `iterations=${iterations} | output=${output.join(",")}`
    );
  });
});
