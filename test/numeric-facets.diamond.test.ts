// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { Interface } from "ethers";
import {
  deployFullSmartSolve,
  type FullSmartSolveDeployment,
} from "../scripts/deploy";
import {
  DIAMOND_ESTIMATE_CALL,
  formatCallbackBenchmark,
  formatBenchmarkExecution,
} from "./test-utils";

type FacetWithInterface = { interface: Interface };

/**
 * Small deterministic production-path samples for every non-root-finding
 * numerical category. Calls use facet ABIs only to encode calls to the Diamond
 * address; SmartSolve fallback then performs the selector lookup/delegatecall.
 */
describe("SmartSolve Diamond numerical-facet production-path gas estimates", function () {
  let deployment: FullSmartSolveDeployment;
  let loupe: any;
  let integration: any;
  let differentiation: any;
  let ode: any;
  let polynomial: any;
  let matrix: any;
  let linear: any;
  let optimization: any;
  let scalarCallback: any;
  let odeCallback: any;
  let objective: any;

  async function assertRouted(
    facet: FacetWithInterface,
    method: string,
    facetName: string
  ): Promise<void> {
    const selector = facet.interface.getFunction(method)!.selector;
    expect(await loupe.facetAddress(selector)).to.equal(deployment.facetAddresses[facetName]);
  }

  function report(category: string, method: string, gas: bigint, callbackStaticCalls?: bigint): void {
    console.log(
      `Diamond-routed production-path gas estimate | ${category} | ${method}: ${gas} | ` +
      `execution=${formatBenchmarkExecution(DIAMOND_ESTIMATE_CALL)}` +
      (callbackStaticCalls === undefined
        ? ""
        : ` | callback=${formatCallbackBenchmark({ staticCalls: callbackStaticCalls })}`)
    );
  }

  before(async function () {
    deployment = await deployFullSmartSolve();
    loupe = await ethers.getContractAt("IDiamondLoupe", deployment.diamondAddress);
    integration = await ethers.getContractAt("IntegrationFacet", deployment.diamondAddress);
    differentiation = await ethers.getContractAt("DifferentiationFacet", deployment.diamondAddress);
    ode = await ethers.getContractAt("ODESolverFacet", deployment.diamondAddress);
    polynomial = await ethers.getContractAt("PolynomialFacet", deployment.diamondAddress);
    matrix = await ethers.getContractAt("MatrixMasterFacet", deployment.diamondAddress);
    linear = await ethers.getContractAt("LinearSolversFacet", deployment.diamondAddress);
    optimization = await ethers.getContractAt("SteepestDescentFacet", deployment.diamondAddress);

    const RootFindingHarness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    scalarCallback = await RootFindingHarness.deploy();
    await scalarCallback.waitForDeployment();

    const ODESolverHarness = await ethers.getContractFactory("ODESolverHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    odeCallback = await ODESolverHarness.deploy();
    await odeCallback.waitForDeployment();

    const SphericalObjectiveHarness = await ethers.getContractFactory("SphericalObjectiveHarness", {
      libraries: { MathLib: deployment.mathLibAddress },
    });
    objective = await SphericalObjectiveHarness.deploy();
    await objective.waitForDeployment();
  });

  it("benchmarks integration through the Diamond", async function () {
    await assertRouted(integration, "integrateSimpson13WithN", "IntegrationFacet");
    const selector = scalarCallback.interface.getFunction("f_x2_minus_4")!.selector;
    const a = await scalarCallback.qFromInt(0n);
    const b = await scalarCallback.qFromInt(2n);
    const args = [await scalarCallback.getAddress(), selector, a, b, 2n] as const;

    const gas = await integration.getFunction("integrateSimpson13WithN").estimateGas(...args);
    const value = await integration.integrateSimpson13WithN(...args);

    report("integration", "Simpson 1/3, n=2", gas, 3n);
    expect(BigInt(await scalarCallback.toFloat(value))).to.be.lessThan(0n);
  });

  it("benchmarks differentiation through the Diamond", async function () {
    await assertRouted(differentiation, "centeredDiff", "DifferentiationFacet");
    const selector = scalarCallback.interface.getFunction("f_x2_minus_4")!.selector;
    const x = await scalarCallback.qFromInt(2n);
    const h = await scalarCallback.qFromFrac(1n, 100n);
    const args = [await scalarCallback.getAddress(), selector, x, h] as const;

    const gas = await differentiation.getFunction("centeredDiff").estimateGas(...args);
    const derivative = await differentiation.centeredDiff(...args);
    const scaled = BigInt(await scalarCallback.toFloat(derivative));

    report("differentiation", "centered difference", gas, 2n);
    expect(scaled >= 3_999_999_000_000_000_000n).to.equal(true);
    expect(scaled <= 4_000_001_000_000_000_000n).to.equal(true);
  });

  it("benchmarks ODE solving through the Diamond", async function () {
    await assertRouted(ode, "eulerIter", "ODESolverFacet");
    const selector = odeCallback.interface.getFunction("f_const5")!.selector;
    const zero = await scalarCallback.qFromInt(0n);
    const one = await scalarCallback.qFromInt(1n);
    const args = [await odeCallback.getAddress(), selector, zero, zero, one, 3n] as const;

    const gas = await ode.getFunction("eulerIter").estimateGas(...args);
    const yFinal = await ode.eulerIter(...args);

    report("ODE solving", "Euler, 3 steps", gas, 3n);
    expect(BigInt(await scalarCallback.toFloat(yFinal))).to.equal(15_000_000_000_000_000_000n);
  });

  it("benchmarks polynomial evaluation through the Diamond", async function () {
    await assertRouted(polynomial, "polyEvaluate", "PolynomialFacet");
    const coeffs = await Promise.all([5n, 2n, 3n].map((value) => scalarCallback.qFromInt(value)));
    const x = await scalarCallback.qFromInt(2n);

    const gas = await polynomial.getFunction("polyEvaluate").estimateGas(coeffs, x);
    const value = await polynomial.polyEvaluate(coeffs, x);

    report("polynomial", "Horner evaluation of 3x^2 + 2x + 5", gas);
    expect(BigInt(await scalarCallback.toFloat(value))).to.equal(21_000_000_000_000_000_000n);
  });

  it("benchmarks matrix multiplication through the Diamond", async function () {
    await assertRouted(matrix, "multiplyMatrices", "MatrixMasterFacet");
    const left = await Promise.all([1n, 2n, 3n, 4n].map((value) => scalarCallback.qFromInt(value)));
    const right = await Promise.all([5n, 6n, 7n, 8n].map((value) => scalarCallback.qFromInt(value)));
    const args = [2n, 2n, left, 2n, 2n, right] as const;

    const gas = await matrix.getFunction("multiplyMatrices").estimateGas(...args);
    const [rows, cols, data] = await matrix.multiplyMatrices(...args);

    report("matrix operations", "2x2 matrix multiplication", gas);
    expect(rows).to.equal(2n);
    expect(cols).to.equal(2n);
    expect(BigInt(await scalarCallback.toFloat(data[3]))).to.equal(50_000_000_000_000_000_000n);
  });

  it("benchmarks linear solving through the Diamond", async function () {
    await assertRouted(linear, "gaussianElimination", "LinearSolversFacet");
    const coefficients = await Promise.all([2n, 1n, 1n, 3n].map((value) => scalarCallback.qFromInt(value)));
    const rhs = await Promise.all([1n, 2n].map((value) => scalarCallback.qFromInt(value)));

    const gas = await linear.getFunction("gaussianElimination").estimateGas(2n, coefficients, rhs);
    const solution = await linear.gaussianElimination(2n, coefficients, rhs);

    report("linear solvers", "2x2 Gaussian elimination", gas);
    expect(BigInt(await scalarCallback.toFloat(solution[0]))).to.equal(200_000_000_000_000_000n);
    const secondComponent = BigInt(await scalarCallback.toFloat(solution[1]));
    expect(secondComponent >= 599_999_999_000_000_000n).to.equal(true);
    expect(secondComponent <= 600_000_001_000_000_000n).to.equal(true);

    await assertRouted(linear, "jacobiWithStatus", "LinearSolversFacet");
    const zero = await scalarCallback.qFromInt(0n);
    const tolerance = await scalarCallback.qFromFrac(1n, 1_000_000_000_000_000n);
    const [iterativeSolution, iterations, converged] = await linear.jacobiWithStatus(
      2n,
      [coefficients[0], zero, zero, coefficients[0]],
      rhs,
      [zero, zero],
      3n,
      tolerance
    );
    expect(iterativeSolution).to.have.lengthOf(2);
    expect(iterations).to.equal(2n);
    expect(converged).to.equal(true);
  });

  it("benchmarks steepest descent through the Diamond", async function () {
    await assertRouted(optimization, "steepestDescent", "SteepestDescentFacet");
    const x0 = await Promise.all([1n, -1n].map((value) => scalarCallback.qFromInt(value)));
    const tolerance = await scalarCallback.qFromFrac(1n, 1_000_000n);
    const args = [await objective.getAddress(), x0, 3n, tolerance] as const;

    const gas = await optimization.getFunction("steepestDescent").estimateGas(...args);
    const result = await optimization.steepestDescent(...args);

    report("steepest descent / optimization", "spherical objective, dimension 2", gas);
    expect(result.x).to.have.lengthOf(2);
    expect(result.status).to.equal(1n);
  });
});
