import { expect } from "chai";
import { ethers } from "hardhat";

describe("Decision-point accumulation hardening", function () {
  this.timeout(120_000);

  // Captured with the same compiler/network settings immediately before the
  // production decision-verification change.
  const JACOBI_BASELINE_GAS = 998_671n;
  const GAUSS_SEIDEL_BASELINE_GAS = 569_217n;

  let math: any;
  let linear: any;
  let matrix: any;
  let zero: string;
  let one: string;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    math = await MathLib.deploy();
    await math.waitForDeployment();
    const libraries = { MathLib: await math.getAddress() };
    linear = await (await ethers.getContractFactory("LinearSolversHarness", { libraries })).deploy();
    matrix = await (await ethers.getContractFactory("PowerIterationDecisionHarness", { libraries })).deploy();
    await Promise.all([linear.waitForDeployment(), matrix.waitForDeployment()]);
    zero = await math.fromInt(0n);
    one = await math.fromInt(1n);
  });

  async function q(value: bigint): Promise<string> {
    return await math.fromInt(value);
  }

  async function fraction(numerator: bigint, denominator: bigint): Promise<string> {
    return await math.div(await q(numerator), await q(denominator));
  }

  it("rejects cancellation-corrupted stationary-solver false success", async function () {
    const large = await q(1n << 116n);
    const negativeLarge = await q(-(1n << 116n));
    const half = await fraction(1n, 2n);
    const A = [
      one, large, negativeLarge,
      zero, one, zero,
      zero, zero, one,
    ];
    const b = [one, one, one];
    const exactSolution = [one, one, one];

    for (const method of ["jacobiWithStatus", "gaussSeidelWithStatus"] as const) {
      const [x, iterations, converged] = await linear[method](
        3n, A, b, exactSolution, 3n, half
      );
      expect(iterations).to.equal(3n);
      expect(converged).to.equal(false);
      expect([...x]).to.not.deep.equal(exactSolution);
    }
  });

  it("retries a cancellation-corrupted zero image before power-iteration termination", async function () {
    const seed = ethers.id("power-zero-image-verification");
    const x0 = await matrix.initialVector(3n, seed);
    const large = await q(1n << 116n);
    const negativeLarge = await q(-(1n << 116n));

    // Tailor a rank-one matrix so the first row products are exactly
    // [1, 2^116, -2^116] for production's deterministic initial vector.
    const A = [
      await math.div(one, x0[0]),
      await math.div(large, x0[1]),
      await math.div(negativeLarge, x0[2]),
      zero, zero, zero,
      zero, zero, zero,
    ];
    const tolerance = await fraction(1n, 10n ** 20n);
    const legacy = await matrix.legacyPowerIteration(3n, A, seed, tolerance, 4n);
    const [lambda, eigenvector, iterations, converged] =
      await matrix.powerIterationWithStatusAndMaxIter(
        3n, A, seed, tolerance, 4n
      );

    expect(legacy[2]).to.equal(0n);
    expect(legacy[3]).to.equal(false);
    expect(legacy[0]).to.equal(zero);
    expect(iterations).to.equal(2n);
    expect(converged).to.equal(true);
    expect(lambda).to.not.equal(zero);
    expect(eigenvector).to.deep.equal([one, zero, zero]);
  });

  it("reports ordinary decision-point gas", async function () {
    const two = await q(2n);
    const three = await q(3n);
    const four = await q(4n);
    const six = await q(6n);
    const eight = await q(8n);
    const tolerance = await fraction(1n, 10n ** 20n);
    const A = [four, one, two, three];
    const b = [six, eight];
    const x0 = [zero, zero];

    const jacobiGas = await linear.jacobiWithStatus.estimateGas(2n, A, b, x0, 48n, tolerance);
    const gsGas = await linear.gaussSeidelWithStatus.estimateGas(2n, A, b, x0, 48n, tolerance);
    const powerArgs = [
      2n,
      [four, zero, zero, two],
      ethers.id("ordinary-power-decision-gas"),
      await fraction(1n, 10n ** 12n),
      64n,
    ] as const;
    const legacyPowerGas = await matrix.legacyPowerIteration.estimateGas(...powerArgs);
    const powerGas = await matrix.powerIterationWithStatusAndMaxIter.estimateGas(...powerArgs);

    expect(jacobiGas).to.be.at.most(JACOBI_BASELINE_GAS * 110n / 100n);
    expect(gsGas).to.be.at.most(GAUSS_SEIDEL_BASELINE_GAS * 110n / 100n);
    expect(powerGas).to.be.at.most(legacyPowerGas * 101n / 100n);

    console.log(
      `DECISION_POINT_GAS | jacobiBefore=${JACOBI_BASELINE_GAS} | jacobiAfter=${jacobiGas} | ` +
      `gaussSeidelBefore=${GAUSS_SEIDEL_BASELINE_GAS} | gaussSeidelAfter=${gsGas} | ` +
      `powerLegacy=${legacyPowerGas} | powerVerified=${powerGas}`
    );
  });
});
