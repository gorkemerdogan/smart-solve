// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { deployFullSmartSolve } from "../scripts/deploy";
import { binary128ToScaledInt } from "./precision-utils";

const QZERO = "0x00000000000000000000000000000000";
const SCALE = 10n ** 33n;

function abs(value: bigint): bigint {
  return value < 0n ? -value : value;
}

describe("SmartSolve Diamond numeric configuration", function () {
  it("exposes effective defaults, configured values, and reset semantics through the proxy", async function () {
    const deployment = await deployFullSmartSolve();
    const config: any = await ethers.getContractAt("NumericConfigFacet", deployment.diamondAddress);

    const Harness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    const converter = await Harness.deploy();
    await converter.waitForDeployment();

    const defaultTol = await converter.qFromFrac(1n, 10n ** 12n);
    const defaultMinTol = await converter.qFromFrac(1n, 10n ** 15n);
    const defaultDiffStep = await converter.qFromFrac(1n, 10n ** 8n);
    const defaultRootMinTol = await converter.qFromFrac(1n, 10n ** 36n);
    const configuredTol = await converter.qFromFrac(1n, 10n ** 30n);
    const configuredMinTol = await converter.qFromFrac(1n, 10n ** 36n);
    const configuredDiffStep = await converter.qFromFrac(1n, 10n ** 10n);

    expect(await config.getTol()).to.equal(defaultTol);
    expect(await config.getMinTol()).to.equal(defaultMinTol);
    expect(await config.getMaxIter()).to.equal(100n);
    expect(await config.getDiffStep()).to.equal(defaultDiffStep);
    expect(await config.getRootFindingConfig()).to.deep.equal([
      defaultTol,
      defaultRootMinTol,
      200n,
    ]);

    await (await config.setTol(configuredTol)).wait();
    await (await config.setMinTol(configuredMinTol)).wait();
    await (await config.setMaxIter(1130n)).wait();
    await (await config.setDiffStep(configuredDiffStep)).wait();

    expect(await config.getTol()).to.equal(configuredTol);
    expect(await config.getMinTol()).to.equal(configuredMinTol);
    expect(await config.getMaxIter()).to.equal(1130n);
    expect(await config.getDiffStep()).to.equal(configuredDiffStep);
    expect(await config.getRootFindingConfig()).to.deep.equal([
      configuredTol,
      configuredMinTol,
      1130n,
    ]);

    // Zero remains the established "unset" sentinel, so reads reveal the
    // effective defaults rather than the raw zero storage values.
    await (await config.setTol(QZERO)).wait();
    await (await config.setMinTol(QZERO)).wait();
    await (await config.setMaxIter(0n)).wait();
    await (await config.setDiffStep(QZERO)).wait();

    expect(await config.getTol()).to.equal(defaultTol);
    expect(await config.getMinTol()).to.equal(defaultMinTol);
    expect(await config.getMaxIter()).to.equal(100n);
    expect(await config.getDiffStep()).to.equal(defaultDiffStep);
    expect(await config.getRootFindingConfig()).to.deep.equal([
      defaultTol,
      defaultRootMinTol,
      200n,
    ]);
  });

  it("rejects negative floating-point configuration values through the proxy", async function () {
    const deployment = await deployFullSmartSolve();
    const config: any = await ethers.getContractAt("NumericConfigFacet", deployment.diamondAddress);
    const Harness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    const converter = await Harness.deploy();
    await converter.waitForDeployment();
    const negativeOne = await converter.qFromInt(-1n);

    await expect(config.setTol(negativeOne)).to.be.revertedWith(
      "NumericConfig: tol must be non-negative"
    );
    await expect(config.setMinTol(negativeOne)).to.be.revertedWith(
      "NumericConfig: minTol must be non-negative"
    );
    await expect(config.setDiffStep(negativeOne)).to.be.revertedWith(
      "NumericConfig: diffStep must be non-negative"
    );
  });

  it("uses the benchmark precision regime and enforces minTol through the Diamond root-finding path", async function () {
    const deployment = await deployFullSmartSolve();
    const config: any = await ethers.getContractAt("NumericConfigFacet", deployment.diamondAddress);
    const rootFinding = await ethers.getContractAt("RootFindingFacet", deployment.diamondAddress);
    const Harness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    const callback = await Harness.deploy();
    await callback.waitForDeployment();

    const minTol = await callback.qFromFrac(1n, 10n ** 36n);
    const benchmarkTol = await callback.qFromFrac(1n, 10n ** 30n);
    const belowConfiguredMinTol = await callback.qFromFrac(1n, 10n ** 36n);
    const enforcedMinTol = await callback.qFromFrac(1n, 10n ** 30n);
    const a = await callback.qFromInt(1n);
    const b = await callback.qFromInt(4n);
    const callbackAddress = await callback.getAddress();
    const selector = callback.interface.getFunction("f_x2_minus_4")!.selector;

    await (await config.setMinTol(minTol)).wait();
    await (await config.setTol(benchmarkTol)).wait();
    await (await config.setMaxIter(1130n)).wait();

    const highPrecision = await rootFinding.rootFindingBisection(callbackAddress, selector, a, b);
    const highPrecisionRoot = binary128ToScaledInt(highPrecision.root, 33);
    expect(highPrecision.converged).to.equal(true);
    expect(highPrecision.iterations).to.be.greaterThan(90n);
    expect(abs(highPrecisionRoot - 2n * SCALE)).to.be.at.most(SCALE / (10n ** 28n));

    // A configured tolerance below minTol is accepted for compatibility but
    // clamped by RootFindingFacet before it reaches the library.
    await (await config.setMinTol(enforcedMinTol)).wait();
    await (await config.setTol(belowConfiguredMinTol)).wait();
    const clamped = await rootFinding.rootFindingBisection(callbackAddress, selector, a, b);
    expect(clamped.converged).to.equal(true);
    expect(clamped.iterations).to.be.greaterThan(90n);
  });
});
