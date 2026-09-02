// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { deployFullSmartSolve } from "../scripts/deploy";

const QZERO = "0x00000000000000000000000000000000";
const QPOSITIVE_INFINITY = "0x7fff0000000000000000000000000000";
const QNAN = "0x7fff8000000000000000000000000000";

describe("Numerical input-validation regressions", function () {
  let matrix: any;
  let linear: any;
  let differentiation: any;
  let mathLib: any;
  let converter: any;
  let callbackAddress: string;
  let callbackSelector: string;
  let zero: string;
  let one: string;
  let two: string;
  let three: string;

  before(async function () {
    const deployment = await deployFullSmartSolve();
    matrix = await ethers.getContractAt("MatrixMasterFacet", deployment.diamondAddress);
    linear = await ethers.getContractAt("LinearSolversFacet", deployment.diamondAddress);
    differentiation = await ethers.getContractAt("DifferentiationFacet", deployment.diamondAddress);
    mathLib = await ethers.getContractAt("contracts/libraries/MathLib.sol:MathLib", deployment.mathLibAddress);

    const Harness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    converter = await Harness.deploy();
    await converter.waitForDeployment();
    callbackAddress = await converter.getAddress();
    callbackSelector = converter.interface.getFunction("f_x2_minus_4")!.selector;

    [zero, one, two, three] = await Promise.all([
      converter.qFromInt(0n),
      converter.qFromInt(1n),
      converter.qFromInt(2n),
      converter.qFromInt(3n),
    ]);
  });

  it("rejects short and extra linear-solver matrix/vector arrays explicitly", async function () {
    await expect(
      linear.gradientDescentLeastSquares(2n, 1n, [one], [one, one], [zero], one, 1n, one)
    ).to.be.revertedWith("LinearSolversFacet: matrix length mismatch");

    await expect(
      linear.gradientDescentLeastSquares(2n, 1n, [one, one], [one], [zero], one, 1n, one)
    ).to.be.revertedWith("LinearSolversFacet: vector length mismatch");

    await expect(
      linear.jacobiWithStatus(1n, [one], [one], [zero, zero], 1n, one)
    ).to.be.revertedWith("LinearSolversFacet: vector length mismatch");

    await expect(linear.gaussianElimination(1n, [one, one], [one])).to.be.revertedWith(
      "LinearSolversFacet: matrix length mismatch"
    );
    await expect(linear.luDecomposition(1n, [one, one])).to.be.revertedWith(
      "LinearSolversFacet: matrix length mismatch"
    );
    await expect(
      linear.gaussSeidelWithStatus(1n, [one], [one, one], [zero], 1n, one)
    ).to.be.revertedWith("LinearSolversFacet: vector length mismatch");
  });

  it("rejects malformed CSR structures and vector lengths explicitly", async function () {
    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [0n, 1n], [0n], [one], 2n, [one, one])
    ).to.be.revertedWith("MatrixMaster: invalid CSR row pointer length");

    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [1n, 1n, 1n], [0n], [one], 2n, [one, one])
    ).to.be.revertedWith("MatrixMaster: CSR row pointer must start at zero");

    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [0n, 1n, 0n], [0n], [one], 2n, [one, one])
    ).to.be.revertedWith("MatrixMaster: CSR row pointer not monotonic");

    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [0n, 0n, 0n], [0n], [one], 2n, [one, one])
    ).to.be.revertedWith("MatrixMaster: CSR terminal pointer mismatch");

    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [0n, 1n, 1n], [], [one], 2n, [one, one])
    ).to.be.revertedWith("MatrixMaster: CSR nnz length mismatch");

    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [0n, 1n, 1n], [2n], [one], 2n, [one, one])
    ).to.be.revertedWith("MatrixMaster: CSR column out of bounds");

    await expect(
      matrix.mulSparseMatrixVector(2n, 2n, [0n, 1n, 1n], [0n], [one], 2n, [one])
    ).to.be.revertedWith("MatrixMaster: vector length mismatch");
  });

  it("validates COO bounds and emits column-sorted CSR rows", async function () {
    await expect(
      matrix.createSparseFromTriplets(2n, 2n, [2n], [0n], [one])
    ).to.be.revertedWith("MatrixMaster: COO row out of bounds");
    await expect(
      matrix.createSparseFromTriplets(2n, 2n, [0n], [2n], [one])
    ).to.be.revertedWith("MatrixMaster: COO column out of bounds");

    const result = await matrix.createSparseFromTriplets(
      2n,
      3n,
      [0n, 0n, 1n],
      [2n, 0n, 1n],
      [three, one, two]
    );
    expect(result.rowPtr).to.deep.equal([0n, 2n, 3n]);
    expect(result.outColInd).to.deep.equal([0n, 2n, 1n]);
    expect(result.outValues).to.deep.equal([one, three, two]);
  });

  it("rejects zero-dimensional matrix inputs instead of treating an empty square as valid", async function () {
    await expect(matrix.det(0n, 0n, [])).to.be.revertedWith(
      "MatrixFacet: dimensions must be > 0"
    );
    await expect(matrix.inverse(0n, 0n, [])).to.be.revertedWith(
      "MatrixFacet: dimensions must be > 0"
    );
  });

  it("rejects negative and non-finite differentiation steps while preserving zero defaulting", async function () {
    const negativeOne = await converter.qFromInt(-1n);
    await expect(
      differentiation.forwardDiff(callbackAddress, callbackSelector, one, negativeOne)
    ).to.be.revertedWith("Differentiation: step must be positive");
    await expect(
      differentiation.backwardDiff(callbackAddress, callbackSelector, one, QPOSITIVE_INFINITY)
    ).to.be.revertedWith("Differentiation: step must be finite");
    await expect(
      differentiation.centeredDiff(callbackAddress, callbackSelector, one, QNAN)
    ).to.be.revertedWith("Differentiation: step must be finite");

    const defaultStepResult = await differentiation.forwardDiff(
      callbackAddress,
      callbackSelector,
      one,
      QZERO
    );
    expect(defaultStepResult).to.match(/^0x[0-9a-f]{32}$/i);
  });

  it("documents and preserves ABDK IEEE-style division special values", async function () {
    const finiteOverZero = await mathLib.div(one, zero);
    const zeroOverZero = await mathLib.div(zero, zero);
    expect(await mathLib.isInfinity(finiteOverZero)).to.equal(true);
    expect(await mathLib.isNaN(zeroOverZero)).to.equal(true);
  });
});
