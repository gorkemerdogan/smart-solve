// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { deployFullSmartSolve } from "../scripts/deploy";
import {
  DIAMOND_ESTIMATE_CALL,
  formatCallbackBenchmark,
  formatBenchmarkExecution,
} from "./test-utils";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";

/**
 * Production-path benchmark: calls enter SmartSolve through its fallback,
 * resolve RootFindingFacet via the selector table, and read proxy-held numeric
 * configuration. RootFindingHarness is retained only as a deterministic
 * callback target and quadruple-precision value converter.
 */
describe("SmartSolve Diamond RootFinding production-path benchmark", function () {
  let resultWriter: BenchmarkResultWriter;

  before(async function () {
    resultWriter = await BenchmarkResultWriter.create({
      suite: "rootfinding.diamond",
    });
  });

  after(async function () {
    await resultWriter.flush();
  });

  it("measures bisection through the fully deployed Diamond", async function () {
    const deployment = await deployFullSmartSolve();

    const rootFinding = await ethers.getContractAt(
      "RootFindingFacet",
      deployment.diamondAddress
    );
    const numericConfig = await ethers.getContractAt(
      "NumericConfigFacet",
      deployment.diamondAddress
    );
    const loupe = await ethers.getContractAt("IDiamondLoupe", deployment.diamondAddress);

    const Harness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    const callback = await Harness.deploy();
    await callback.waitForDeployment();

    const tolerance = await callback.qFromFrac(1n, 1_000_000_000_000n);
    const maxIter = 100n;
    await (await numericConfig.setTol(tolerance)).wait();
    await (await numericConfig.setMaxIter(maxIter)).wait();

    const callbackAddress = await callback.getAddress();
    const selector = callback.interface.getFunction("f_x2_minus_4")!.selector;
    const a = await callback.qFromInt(1n);
    const b = await callback.qFromInt(4n);
    const rootFindingSelector = rootFinding.interface.getFunction("rootFindingBisection")!.selector;

    expect(await loupe.facetAddress(rootFindingSelector)).to.equal(
      deployment.facetAddresses.RootFindingFacet
    );
    expect(await numericConfig.getTol()).to.equal(tolerance);
    expect(await numericConfig.getMaxIter()).to.equal(maxIter);

    const gas = await rootFinding
      .getFunction("rootFindingBisection")
      .estimateGas(callbackAddress, selector, a, b);
    const result = await rootFinding.rootFindingBisection(callbackAddress, selector, a, b);
    const rootScaled = await callback.toFloat(result.root);

    console.log("Production Diamond root-finding benchmark");
    console.log("  execution model:", formatBenchmarkExecution(DIAMOND_ESTIMATE_CALL));
    console.log("  full deployment + installation gas:", deployment.deploymentGasUsed.toString());
    console.log("  Diamond fallback bisection gas:", gas.toString());
    console.log("  iterations:", result.iterations.toString());
    console.log(
      "  callback model:",
      formatCallbackBenchmark({ staticCalls: 2n + result.iterations, detail: "f target" })
    );

    expect(result.converged).to.equal(true);
    expect(rootScaled >= 1_999_999_000_000_000_000n).to.equal(true);
    expect(rootScaled <= 2_000_001_000_000_000_000n).to.equal(true);

    resultWriter.record({
      benchmark: "bisection x^2 - 4 on [1, 4]",
      category: "root-finding",
      operation: "rootFindingBisection",
      execution: benchmarkExecutionRecord(DIAMOND_ESTIMATE_CALL),
      callback: {
        model: "target_staticcall",
        functionEvaluations: (2n + result.iterations).toString(),
        detail: "f target",
        gasIncludesCallback: true,
      },
      input: {
        intervalStart: 1,
        intervalEnd: 4,
        tolerance: "1e-12",
        iterationCap: 100,
      },
      gas: gas.toString(),
      status: "success",
      errorMetrics: { rootScaled: rootScaled.toString() },
      convergence: {
        converged: result.converged,
        iterations: result.iterations.toString(),
      },
    });
  });
});
