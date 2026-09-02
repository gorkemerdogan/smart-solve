// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { deployFullSmartSolve } from "../scripts/deploy";

const QZERO = "0x00000000000000000000000000000000";
const QPOSITIVE_INFINITY = "0x7fff0000000000000000000000000000";
const SEED = ethers.keccak256(ethers.toUtf8Bytes("power-iteration-status-seed"));

describe("MatrixMaster power iteration status", function () {
  let harness: any;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    const mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();

    const Harness = await ethers.getContractFactory("MatrixMasterHarness", {
      libraries: { MathLib: await mathLib.getAddress() },
    });
    harness = await Harness.deploy();
    await harness.waitForDeployment();
  });

  async function diagonalTwoOne(contract: any): Promise<string[]> {
    const zero = await contract.qFromInt(0n);
    return [await contract.qFromInt(2n), zero, zero, await contract.qFromInt(1n)];
  }

  it("uses the effective default tolerance when tol is zero", async function () {
    const A = await diagonalTwoOne(harness);
    const explicitDefault = await harness.qFromFrac(1n, 10n ** 12n);

    const fromZero = await harness.powerIterationWithStatusHarness(2n, 2n, A, SEED, QZERO);
    const fromExplicitDefault = await harness.powerIterationWithStatusHarness(
      2n,
      2n,
      A,
      SEED,
      explicitDefault
    );

    expect(fromZero[0]).to.equal(fromExplicitDefault[0]);
    expect(fromZero[3]).to.deep.equal(fromExplicitDefault[3]);
    expect(fromZero[4]).to.equal(fromExplicitDefault[4]);
    expect(fromZero[5]).to.equal(true);
  });

  it("rejects negative and non-finite tolerances", async function () {
    const A = await diagonalTwoOne(harness);
    const negativeOne = await harness.qFromInt(-1n);

    await expect(
      harness.powerIterationWithStatusHarness(2n, 2n, A, SEED, negativeOne)
    ).to.be.revertedWith("MatrixMaster: tol must be non-negative");
    await expect(
      harness.powerIterationWithStatusHarness(2n, 2n, A, SEED, QPOSITIVE_INFINITY)
    ).to.be.revertedWith("MatrixMaster: tol must be finite");
  });

  it("reports successful convergence and a meaningful iteration count", async function () {
    const A = await diagonalTwoOne(harness);
    const tol = await harness.qFromFrac(1n, 1_000_000n);

    const result = await harness.powerIterationWithStatusAndMaxIterHarness(
      2n,
      2n,
      A,
      SEED,
      tol,
      100n
    );

    expect(result[4]).to.be.greaterThan(0n);
    expect(result[4]).to.be.lessThan(100n);
    expect(result[5]).to.equal(true);
  });

  it("reports max-iteration exhaustion without claiming convergence", async function () {
    const A = await diagonalTwoOne(harness);
    const tol = await harness.qFromFrac(1n, 10n ** 15n);

    const result = await harness.powerIterationWithStatusAndMaxIterHarness(
      2n,
      2n,
      A,
      SEED,
      tol,
      1n
    );

    expect(result[4]).to.equal(1n);
    expect(result[5]).to.equal(false);
  });

  it("keeps the legacy powerIteration result ABI and values", async function () {
    const A = await diagonalTwoOne(harness);
    const tol = await harness.qFromFrac(1n, 1_000_000n);

    const legacy = await harness.powerIterationHarness(2n, 2n, A, SEED, tol);
    const withStatus = await harness.powerIterationWithStatusHarness(2n, 2n, A, SEED, tol);

    expect(legacy[0]).to.equal(withStatus[0]);
    expect(legacy[1]).to.equal(withStatus[1]);
    expect(legacy[2]).to.equal(withStatus[2]);
    expect(legacy[3]).to.deep.equal(withStatus[3]);
  });

  it("routes status-bearing production calls through the Diamond and reports configured exhaustion", async function () {
    const deployment = await deployFullSmartSolve();
    const matrix: any = await ethers.getContractAt("MatrixMasterFacet", deployment.diamondAddress);
    const config: any = await ethers.getContractAt("NumericConfigFacet", deployment.diamondAddress);
    const loupe: any = await ethers.getContractAt("IDiamondLoupe", deployment.diamondAddress);
    const Converter = await ethers.getContractFactory("MatrixMasterHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    const converter: any = await Converter.deploy();
    await converter.waitForDeployment();
    const A = await diagonalTwoOne(converter);

    const selector = matrix.interface.getFunction("powerIterationWithStatus")!.selector;
    expect(await loupe.facetAddress(selector)).to.equal(deployment.facetAddresses.MatrixMasterFacet);

    const configuredTol = await converter.qFromFrac(1n, 1_000_000n);
    await (await config.setTol(configuredTol)).wait();
    const fromZero = await matrix.powerIterationWithStatus(2n, 2n, A, SEED, QZERO);
    const fromConfiguredTol = await matrix.powerIterationWithStatus(
      2n,
      2n,
      A,
      SEED,
      configuredTol
    );
    expect(fromZero[0]).to.equal(fromConfiguredTol[0]);
    expect(fromZero[3]).to.deep.equal(fromConfiguredTol[3]);
    expect(fromZero[5]).to.equal(true);

    await (await config.setMaxIter(1n)).wait();
    const tol = await converter.qFromFrac(1n, 10n ** 15n);
    const result = await matrix.powerIterationWithStatus(2n, 2n, A, SEED, tol);

    expect(result[4]).to.equal(1n);
    expect(result[5]).to.equal(false);
  });
});
