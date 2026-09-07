// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { BaseContract } from "ethers";
import { binary128ToRational, type ExactRational } from "./precision-utils";

type Strategy = "naive" | "compensated";
type SolverOutput = {
  vector: string[];
  iterations: bigint;
  converged: boolean;
  finalMetric: string;
  lambda?: string;
  gas: bigint;
};

function abs(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function gcd(a: bigint, b: bigint): bigint {
  a = abs(a);
  b = abs(b);
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
}

function rational(num: bigint, den = 1n): ExactRational {
  if (den < 0n) return rational(-num, -den);
  if (num === 0n) return { num: 0n, den: 1n };
  const divisor = gcd(num, den);
  return { num: num / divisor, den: den / divisor };
}

function add(a: ExactRational, b: ExactRational): ExactRational {
  return rational(a.num * b.den + b.num * a.den, a.den * b.den);
}

function sub(a: ExactRational, b: ExactRational): ExactRational {
  return rational(a.num * b.den - b.num * a.den, a.den * b.den);
}

function mul(a: ExactRational, b: ExactRational): ExactRational {
  return rational(a.num * b.num, a.den * b.den);
}

function square(a: ExactRational): ExactRational {
  return mul(a, a);
}

function compare(a: ExactRational, b: ExactRational): number {
  const left = a.num * b.den;
  const right = b.num * a.den;
  return left < right ? -1 : left > right ? 1 : 0;
}

function absolute(a: ExactRational): ExactRational {
  return { num: abs(a.num), den: a.den };
}

function decimal(value: ExactRational, digits = 54): string {
  if (value.num === 0n) return "0";
  const negative = value.num < 0n;
  const numerator = abs(value.num);
  const scale = 10n ** BigInt(digits);
  const scaled = numerator * scale / value.den;
  const whole = scaled / scale;
  const fraction = (scaled % scale).toString().padStart(digits, "0").replace(/0+$/, "");
  const rendered = fraction ? `${whole}.${fraction}` : whole.toString();
  return negative ? `-${rendered}` : rendered;
}

function maxVectorError(actual: string[], expected: ExactRational[]): ExactRational {
  let maximum = rational(0n);
  for (let i = 0; i < actual.length; ++i) {
    const error = absolute(sub(binary128ToRational(actual[i]), expected[i]));
    if (compare(error, maximum) > 0) maximum = error;
  }
  return maximum;
}

function exactResidualSquared(a: string[], x: string[], b: string[], m: number, n: number): ExactRational {
  let norm2 = rational(0n);
  for (let i = 0; i < m; ++i) {
    let row = rational(0n);
    for (let j = 0; j < n; ++j) {
      row = add(row, mul(binary128ToRational(a[i * n + j]), binary128ToRational(x[j])));
    }
    norm2 = add(norm2, square(sub(row, binary128ToRational(b[i]))));
  }
  return norm2;
}

function exactGradientNormSquared(a: string[], x: string[], b: string[], m: number, n: number): ExactRational {
  const residual: ExactRational[] = [];
  for (let i = 0; i < m; ++i) {
    let row = rational(0n);
    for (let j = 0; j < n; ++j) {
      row = add(row, mul(binary128ToRational(a[i * n + j]), binary128ToRational(x[j])));
    }
    residual.push(sub(row, binary128ToRational(b[i])));
  }

  let norm2 = rational(0n);
  for (let j = 0; j < n; ++j) {
    let component = rational(0n);
    for (let i = 0; i < m; ++i) {
      component = add(component, mul(binary128ToRational(a[i * n + j]), residual[i]));
    }
    norm2 = add(norm2, square(component));
  }
  return norm2;
}

function exactEigenResidualSquared(a: string[], x: string[], lambda: string, n: number): ExactRational {
  const eigenvalue = binary128ToRational(lambda);
  let norm2 = rational(0n);
  for (let i = 0; i < n; ++i) {
    let row = rational(0n);
    for (let j = 0; j < n; ++j) {
      row = add(row, mul(binary128ToRational(a[i * n + j]), binary128ToRational(x[j])));
    }
    norm2 = add(norm2, square(sub(row, mul(eigenvalue, binary128ToRational(x[i])))));
  }
  return norm2;
}

describe("Solver-level accumulation strategy A/B", function () {
  this.timeout(120_000);

  let math: BaseContract;
  let harness: BaseContract;
  let productionLinear: BaseContract;
  let productionMatrix: BaseContract;
  let zero: string;
  let one: string;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    math = await MathLib.deploy();
    await math.waitForDeployment();
    const Harness = await ethers.getContractFactory("SolverAccumulationBenchmarkHarness", {
      libraries: { MathLib: await math.getAddress() },
    });
    harness = await Harness.deploy();
    await harness.waitForDeployment();
    const ProductionLinear = await ethers.getContractFactory("LinearSolversHarness", {
      libraries: { MathLib: await math.getAddress() },
    });
    productionLinear = await ProductionLinear.deploy();
    await productionLinear.waitForDeployment();
    const ProductionMatrix = await ethers.getContractFactory("MatrixMasterHarness", {
      libraries: { MathLib: await math.getAddress() },
    });
    productionMatrix = await ProductionMatrix.deploy();
    await productionMatrix.waitForDeployment();
    zero = await (math as any).fromInt(0n);
    one = await (math as any).fromInt(1n);

    console.log("============================================================");
    console.log("SOLVER-LEVEL ACCUMULATION A/B — TEST ONLY");
    console.log("Strategies: production-order naive vs Neumaier compensated accumulation");
    console.log("Accuracy oracle: exact rational arithmetic over binary128 inputs/outputs");
    console.log("Power iteration: deterministic supplied x0; no random initialization gas");
    console.log("============================================================");
  });

  async function q(value: bigint): Promise<string> {
    return await (math as any).fromInt(value);
  }

  async function fraction(numerator: bigint, denominator: bigint): Promise<string> {
    return await (math as any).div(await q(numerator), await q(denominator));
  }

  function unpack(raw: any): { vector: string[]; iterations: bigint; converged: boolean; finalMetric: string } {
    const tuple = raw.vector !== undefined ? raw : raw[0];
    return {
      vector: [...tuple.vector],
      iterations: tuple.iterations,
      converged: tuple.converged,
      finalMetric: tuple.finalMetric,
    };
  }

  async function execute(method: string, args: readonly unknown[], power = false): Promise<SolverOutput> {
    const callable = harness.getFunction(method);
    const raw: any = await callable.staticCall(...args);
    const result = unpack(power ? raw[0] : raw);
    return {
      ...result,
      lambda: power ? raw[1] : undefined,
      gas: await callable.estimateGas(...args),
    };
  }

  function report(
    workload: string,
    solver: string,
    strategy: Strategy,
    output: SolverOutput,
    resultError: ExactRational,
    oracleMetric?: ExactRational
  ): void {
    console.log(
      `SOLVER_ACCUM_AB | workload=${workload} | solver=${solver} | strategy=${strategy} | ` +
      `converged=${output.converged} | iterations=${output.iterations} | gas=${output.gas} | ` +
      `resultError=${decimal(resultError)} | reportedMetric=${decimal(binary128ToRational(output.finalMetric))}` +
      (oracleMetric ? ` | oracleMetric=${decimal(oracleMetric)}` : "")
    );
  }

  async function compareLinear(
    workload: string,
    solver: "jacobi" | "gaussSeidel" | "gradientDescent",
    args: readonly unknown[],
    expected: ExactRational[],
    a: string[],
    b: string[],
    m: number,
    n: number
  ): Promise<Record<Strategy, SolverOutput>> {
    const outputs = {} as Record<Strategy, SolverOutput>;
    for (const strategy of ["naive", "compensated"] as const) {
      const suffix = strategy === "naive" ? "Naive" : "Compensated";
      const output = await execute(`${solver}${suffix}`, args);
      const error = maxVectorError(output.vector, expected);
      const oracleMetric = solver === "gradientDescent"
        ? exactGradientNormSquared(a, output.vector, b, m, n)
        : exactResidualSquared(a, output.vector, b, m, n);
      report(workload, solver, strategy, output, error, oracleMetric);
      outputs[strategy] = output;
    }
    return outputs;
  }

  it("matches the production naive paths before comparing accumulation", async function () {
    const two = await q(2n);
    const three = await q(3n);
    const four = await q(4n);
    const six = await q(6n);
    const eight = await q(8n);
    const a = [four, one, two, three];
    const b = [six, eight];
    const x0 = [zero, zero];
    const tolerance = await fraction(1n, 10n ** 12n);

    for (const solver of ["jacobi", "gaussSeidel"] as const) {
      const benchmark = await execute(`${solver}Naive`, [2n, a, b, x0, 32n, tolerance]);
      const production: any = await (productionLinear as any)[`${solver}WithStatus`](2n, a, b, x0, 32n, tolerance);
      expect(benchmark.vector).to.deep.equal([...production[0]]);
      expect(benchmark.iterations).to.equal(production[1]);
      expect(benchmark.converged).to.equal(production[2]);
    }

    const half = await fraction(1n, 2n);
    const identity = [one, zero, zero, one];
    const benchmarkGd = await execute(
      "gradientDescentNaive",
      [2n, 2n, identity, [one, two], x0, half, 32n, tolerance]
    );
    const productionGd: any = await (productionLinear as any).gradientDescentLeastSquares(
      2n, 2n, identity, [one, two], x0, half, 32n, tolerance
    );
    expect(benchmarkGd.vector).to.deep.equal([...productionGd[0]]);
    expect(benchmarkGd.iterations).to.equal(productionGd[1]);

    const seed = ethers.id("solver-accumulation-production-parity");
    const random: any = await (productionMatrix as any).randomVectorHarness(2n, seed);
    const initial = [...random[2]];
    const benchmarkPower = await execute(
      "powerIterationNaive",
      [2n, [four, zero, zero, two], initial, 48n, tolerance],
      true
    );
    const productionPower: any = await (productionMatrix as any).powerIterationWithStatusAndMaxIterHarness(
      2n, 2n, [four, zero, zero, two], seed, tolerance, 48n
    );
    expect(benchmarkPower.lambda).to.equal(productionPower[0]);
    expect(benchmarkPower.vector).to.deep.equal([...productionPower[3]]);
    expect(benchmarkPower.iterations).to.equal(productionPower[4]);
    expect(benchmarkPower.converged).to.equal(productionPower[5]);
  });

  async function comparePower(
    workload: string,
    args: readonly unknown[],
    expectedVector: ExactRational[],
    expectedLambda: ExactRational,
    a: string[],
    n: number
  ): Promise<Record<Strategy, SolverOutput>> {
    const outputs = {} as Record<Strategy, SolverOutput>;
    for (const strategy of ["naive", "compensated"] as const) {
      const suffix = strategy === "naive" ? "Naive" : "Compensated";
      const output = await execute(`powerIteration${suffix}`, args, true);
      const vectorError = maxVectorError(output.vector, expectedVector);
      const lambdaError = absolute(sub(binary128ToRational(output.lambda!), expectedLambda));
      const resultError = compare(vectorError, lambdaError) >= 0 ? vectorError : lambdaError;
      const oracleResidual = exactEigenResidualSquared(a, output.vector, output.lambda!, n);
      report(workload, "powerIteration", strategy, output, resultError, oracleResidual);
      outputs[strategy] = output;
    }
    return outputs;
  }

  it("preserves decisions and accuracy on ordinary well-conditioned controls", async function () {
    const two = await q(2n);
    const three = await q(3n);
    const four = await q(4n);
    const six = await q(6n);
    const eight = await q(8n);
    const half = await fraction(1n, 2n);
    const tolerance = await fraction(1n, 10n ** 20n);
    const expected = [rational(1n), rational(2n)];
    const a = [four, one, two, three];
    const b = [six, eight];
    const x0 = [zero, zero];

    for (const solver of ["jacobi", "gaussSeidel"] as const) {
      const outputs = await compareLinear(
        "ordinary-2x2",
        solver,
        [2n, a, b, x0, 48n, tolerance],
        expected,
        a,
        b,
        2,
        2
      );
      expect(outputs.naive.converged).to.equal(true);
      expect(outputs.compensated.converged).to.equal(true);
      expect(outputs.compensated.iterations).to.equal(outputs.naive.iterations);
      expect(outputs.compensated.vector).to.deep.equal(outputs.naive.vector);
    }

    const identity = [one, zero, zero, one];
    const gd = await compareLinear(
      "ordinary-identity-2x2",
      "gradientDescent",
      [2n, 2n, identity, [one, two], x0, half, 64n, tolerance],
      expected,
      identity,
      [one, two],
      2,
      2
    );
    expect(gd.naive.converged).to.equal(true);
    expect(gd.compensated.converged).to.equal(true);
    expect(gd.compensated.iterations).to.equal(gd.naive.iterations);
    expect(gd.compensated.vector).to.deep.equal(gd.naive.vector);

    const power = await comparePower(
      "ordinary-diagonal-2x2",
      [2n, [four, zero, zero, two], [one, one], 64n, await fraction(1n, 10n ** 12n)],
      [rational(1n), rational(0n)],
      rational(4n),
      [four, zero, zero, two],
      2
    );
    expect(power.naive.converged).to.equal(true);
    expect(power.compensated.converged).to.equal(true);
    expect(power.compensated.iterations).to.equal(power.naive.iterations);
    expect(power.compensated.vector).to.deep.equal(power.naive.vector);
    expect(power.compensated.lambda).to.equal(power.naive.lambda);
  });

  it("preserves exhaustion decisions on scaled and slow-convergence controls", async function () {
    const scale = await fraction(1n, 1n << 40n);
    const half = await fraction(1n, 2n);
    const tinyTolerance = await fraction(1n, 10n ** 50n);
    const expected = [rational(1n), rational(1n)];
    const diagonal = [one, zero, zero, scale];
    const rhs = [one, scale];
    const x0 = [zero, zero];

    for (const solver of ["jacobi", "gaussSeidel"] as const) {
      const outputs = await compareLinear(
        "scaled-diagonal-condition-2^40",
        solver,
        [2n, diagonal, rhs, x0, 8n, tinyTolerance],
        expected,
        diagonal,
        rhs,
        2,
        2
      );
      expect(outputs.naive.converged).to.equal(true);
      expect(outputs.compensated.converged).to.equal(true);
      expect(outputs.compensated.iterations).to.equal(outputs.naive.iterations);
      expect(outputs.compensated.vector).to.deep.equal(outputs.naive.vector);
    }

    const gd = await compareLinear(
      "scaled-gradient-condition-2^40",
      "gradientDescent",
      [2n, 2n, diagonal, rhs, x0, half, 16n, tinyTolerance],
      expected,
      diagonal,
      rhs,
      2,
      2
    );
    expect(gd.naive.converged).to.equal(false);
    expect(gd.compensated.converged).to.equal(false);
    expect(gd.compensated.iterations).to.equal(gd.naive.iterations);
    expect(gd.compensated.vector).to.deep.equal(gd.naive.vector);

    const clustered = [one, zero, zero, await (math as any).sub(one, scale)];
    const power = await comparePower(
      "clustered-eigenvalues-gap-2^-40",
      [2n, clustered, [one, one], 16n, await fraction(1n, 10n ** 20n)],
      [rational(1n), rational(0n)],
      rational(1n),
      clustered,
      2
    );
    expect(power.naive.converged).to.equal(false);
    expect(power.compensated.converged).to.equal(false);
    expect(power.compensated.iterations).to.equal(power.naive.iterations);
    expect(power.compensated.vector).to.deep.equal(power.naive.vector);
    expect(power.compensated.lambda).to.equal(power.naive.lambda);
  });

  it("exposes convergence-decision changes under adversarial cancellation", async function () {
    const large = await q(1n << 116n);
    const negativeLarge = await q(-(1n << 116n));
    const quarter = await fraction(1n, 4n);
    const cancellationSystem = [
      one, large, negativeLarge,
      zero, one, zero,
      zero, zero, one,
    ];
    const rhs = [one, one, one];
    const solution = [one, one, one];
    const expected = solution.map(() => rational(1n));

    for (const solver of ["jacobi", "gaussSeidel"] as const) {
      const outputs = await compareLinear(
        "cancellation-exact-initial-solution",
        solver,
        [3n, cancellationSystem, rhs, solution, 3n, quarter],
        expected,
        cancellationSystem,
        rhs,
        3,
        3
      );
      expect(outputs.naive.converged).to.equal(false);
      expect(outputs.naive.iterations).to.equal(3n);
      expect(outputs.compensated.converged).to.equal(true);
      expect(outputs.compensated.iterations).to.equal(1n);
      expect(maxVectorError(outputs.naive.vector, expected).num).to.not.equal(0n);
      expect(maxVectorError(outputs.compensated.vector, expected).num).to.equal(0n);
    }

    // With a looser tolerance, the same corrupted naive residual is small
    // enough to produce false success even though the exact residual is 49.
    const half = await fraction(1n, 2n);
    for (const solver of ["jacobi", "gaussSeidel"] as const) {
      const outputs = await compareLinear(
        "cancellation-loose-tolerance-false-success",
        solver,
        [3n, cancellationSystem, rhs, solution, 3n, half],
        expected,
        cancellationSystem,
        rhs,
        3,
        3
      );
      expect(outputs.naive.converged).to.equal(true);
      expect(outputs.naive.iterations).to.equal(2n);
      expect(binary128ToRational(outputs.naive.finalMetric)).to.deep.equal(rational(1n));
      expect(exactResidualSquared(cancellationSystem, outputs.naive.vector, rhs, 3, 3)).to.deep.equal(rational(49n));
      expect(outputs.compensated.converged).to.equal(true);
      expect(outputs.compensated.iterations).to.equal(1n);
      expect(maxVectorError(outputs.compensated.vector, expected).num).to.equal(0n);
    }

    const alpha = await fraction(1n, 1n << 240n);
    const gd = await compareLinear(
      "cancellation-exact-least-squares-solution",
      "gradientDescent",
      [3n, 3n, cancellationSystem, rhs, solution, alpha, 1n, quarter],
      expected,
      cancellationSystem,
      rhs,
      3,
      3
    );
    expect(gd.naive.converged).to.equal(false);
    expect(gd.naive.iterations).to.equal(1n);
    expect(gd.compensated.converged).to.equal(true);
    expect(gd.compensated.iterations).to.equal(0n);

    const cancellationEigen = [
      one, large, negativeLarge,
      zero, zero, zero,
      zero, zero, zero,
    ];
    const power = await comparePower(
      "cancellation-zero-image-false-stop",
      [3n, cancellationEigen, solution, 4n, await fraction(1n, 10n ** 20n)],
      [rational(1n), rational(0n), rational(0n)],
      rational(1n),
      cancellationEigen,
      3
    );
    expect(power.naive.converged).to.equal(false);
    expect(power.naive.iterations).to.equal(0n);
    expect(power.compensated.converged).to.equal(true);
    expect(power.compensated.iterations).to.equal(2n);
    expect(binary128ToRational(power.naive.lambda!).num).to.equal(0n);
    expect(binary128ToRational(power.compensated.lambda!)).to.deep.equal(rational(1n));
  });
});
