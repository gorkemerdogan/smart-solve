// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";

const BASELINE_OUTPUTS: Record<string, string[]> = {
  "jacobi-converged": [
    "0x3ffefffffff211c1d6db72c9b53fd471",
    "0x3ffffffffff211c1d6db72c9b53fd470",
    "0x40007ffffff97e1a84a18362fbe7ec97",
    "0x4000fffffff9f3541dd54d611d2feef5",
  ],
  "gauss-seidel-converged": [
    "0x3fff000000085a8aee81b92b24323427",
    "0x3ffffffffffa02636fb37d3a3ef0eacf",
    "0x40008000000166328ad0e917e177a402",
    "0x4000ffffffff8899d1ba5cf80a2d73fe",
  ],
  "jacobi-exhausted-8": [
    "0x3ffeff38e38e38e38e38e38e38e38e39",
    "0x3fffff38e38e38e38e38e38e38e38e38",
    "0x40007fa2f684bda12f684bda12f684bd",
    "0x4000ffa97b425ed097b425ed097b425e",
  ],
  "gauss-seidel-exhausted-8": [
    "0x3fff0000b02c3f35ba781948b0fcd6ea",
    "0x3fffffff81a88f469598c1d7f7926fab",
    "0x400080001d825393d743c668a5c01679",
    "0x4000fffff629e4240d941332736aa32c",
  ],
};

describe("Jacobi/Gauss-Seidel iterate-change allocation benchmark", function () {
  this.timeout(120_000);

  let solver: any;
  let A: string[];
  let b: string[];
  let x0: string[];
  let convergenceTolerance: string;
  let exhaustionTolerance: string;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    const mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();

    const Harness = await ethers.getContractFactory("LinearSolversHarness", {
      libraries: { MathLib: await mathLib.getAddress() },
    });
    solver = await Harness.deploy();
    await solver.waitForDeployment();

    const integers = [
      4n, 1n, 0n, 0n,
      1n, 4n, 1n, 0n,
      0n, 1n, 4n, 1n,
      0n, 0n, 1n, 3n,
    ];
    A = await Promise.all(integers.map(value => solver.qFromInt(value)));
    b = await Promise.all([6n, 12n, 18n, 15n].map(value => solver.qFromInt(value)));
    x0 = await Promise.all([0n, 0n, 0n, 0n].map(value => solver.qFromInt(value)));
    convergenceTolerance = await solver.qFromFrac(1n, 10n ** 15n);
    exhaustionTolerance = await solver.qFromFrac(1n, 10n ** 30n);

    console.log("============================================================");
    console.log("ITERATIVE SOLVER TEMPORARY-ALLOCATION GAS BENCHMARK");
    console.log("System: deterministic 4x4 strictly diagonally dominant system");
    console.log("Execution model: LinearSolversHarness estimateGas; status-bearing APIs");
    console.log("============================================================");
  });

  async function runCase(
    method: "jacobiWithStatus" | "gaussSeidelWithStatus",
    label: string,
    maxIter: bigint,
    tolerance: string,
    expectedConverged: boolean
  ): Promise<void> {
    const callable = solver.getFunction(method);
    const gas = await callable.estimateGas(4n, A, b, x0, maxIter, tolerance);
    const [output, iterations, converged] = await callable.staticCall(
      4n,
      A,
      b,
      x0,
      maxIter,
      tolerance
    );

    expect(converged).to.equal(expectedConverged);
    if (!expectedConverged) expect(iterations).to.equal(maxIter);
    expect([...output], `${label}: binary128 output changed from pre-optimization baseline`).to.deep.equal(
      BASELINE_OUTPUTS[label]
    );
    console.log(
      `ITERATIVE_ALLOCATION_GAS | case=${label} | gas=${gas} | iterations=${iterations} | ` +
      `converged=${converged} | output=${output.join(",")}`
    );
  }

  it("measures converged Jacobi and Gauss-Seidel calls", async function () {
    await runCase("jacobiWithStatus", "jacobi-converged", 100n, convergenceTolerance, true);
    await runCase("gaussSeidelWithStatus", "gauss-seidel-converged", 100n, convergenceTolerance, true);
  });

  it("measures fixed-budget non-converged Jacobi and Gauss-Seidel calls", async function () {
    await runCase("jacobiWithStatus", "jacobi-exhausted-8", 8n, exhaustionTolerance, false);
    await runCase("gaussSeidelWithStatus", "gauss-seidel-exhausted-8", 8n, exhaustionTolerance, false);
  });
});
