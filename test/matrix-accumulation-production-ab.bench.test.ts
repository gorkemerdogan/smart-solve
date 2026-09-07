// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { BaseContract, ContractFactory, ContractTransactionResponse } from "ethers";
import { binary128ToRational } from "./precision-utils";

type Strategy = "naive" | "pairwise" | "compensated";
type ErrorValue = { numerator: bigint; denominator: bigint };
type GasRow = { workload: string; gas: Record<Strategy, bigint> };

const STRATEGIES: Strategy[] = ["naive", "pairwise", "compensated"];

function exactIntegerError(value: string, expected: bigint): ErrorValue {
  const actual = binary128ToRational(value);
  const difference = actual.num - expected * actual.den;
  return { numerator: difference < 0n ? -difference : difference, denominator: actual.den };
}

function compareErrors(a: ErrorValue, b: ErrorValue): number {
  const left = a.numerator * b.denominator;
  const right = b.numerator * a.denominator;
  return left < right ? -1 : left > right ? 1 : 0;
}

function maxError(values: string[], expected: bigint[]): ErrorValue {
  let maximum: ErrorValue = { numerator: 0n, denominator: 1n };
  for (let i = 0; i < values.length; ++i) {
    const error = exactIntegerError(values[i], expected[i]);
    if (compareErrors(error, maximum) > 0) maximum = error;
  }
  return maximum;
}

function relativeError(absolute: ErrorValue, expectedMagnitude: bigint): ErrorValue {
  if (expectedMagnitude === 0n) return absolute;
  return { numerator: absolute.numerator, denominator: absolute.denominator * expectedMagnitude };
}

function decimal(value: ErrorValue, digits = 36): string {
  if (value.numerator === 0n) return "0";
  const scale = 10n ** BigInt(digits);
  const scaled = value.numerator * scale / value.denominator;
  const whole = scaled / scale;
  const fraction = (scaled % scale).toString().padStart(digits, "0").replace(/0+$/, "");
  return fraction.length === 0 ? whole.toString() : `${whole}.${fraction}`;
}

function byteLength(bytecode: string): number {
  return (bytecode.length - 2) / 2;
}

async function deploymentGas(contract: BaseContract): Promise<bigint> {
  const transaction = contract.deploymentTransaction() as ContractTransactionResponse | null;
  if (!transaction) throw new Error("Missing deployment transaction");
  const receipt = await transaction.wait();
  if (!receipt) throw new Error("Missing deployment receipt");
  return receipt.gasUsed;
}

describe("Production-shaped dot and matrix accumulation A/B", function () {
  this.timeout(120_000);

  const contracts = {} as Record<Strategy, BaseContract>;
  const factories = {} as Record<Strategy, ContractFactory>;
  const rows: GasRow[] = [];
  let mathLib: BaseContract;
  let one: string;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();
    const libraries = { MathLib: await mathLib.getAddress() };
    const definitions: Array<[Strategy, string]> = [
      ["naive", "MatrixAccumulationNaiveHarness"],
      ["pairwise", "MatrixAccumulationPairwiseHarness"],
      ["compensated", "MatrixAccumulationCompensatedHarness"],
    ];
    for (const [strategy, name] of definitions) {
      factories[strategy] = await ethers.getContractFactory(name, { libraries });
      contracts[strategy] = await factories[strategy].deploy();
      await contracts[strategy].waitForDeployment();
    }
    one = await (mathLib as any).fromInt(1n);

    console.log("============================================================");
    console.log("PRODUCTION-SHAPED DOT/MATRIX ACCUMULATION A/B — TEST ONLY");
    console.log("Naive path: unchanged MatrixMaster dot/matvec/subtract implementation");
    console.log("Oracle: exact BigInt rational decoding of binary128 outputs");
    console.log("Execution model: direct harness estimateGas with comparable calldata copies");
    console.log("============================================================");
  });

  async function q(value: bigint): Promise<string> {
    return await (mathLib as any).fromInt(value);
  }

  async function compareScalar(
    workload: string,
    method: "dot" | "residualNormSquared",
    args: readonly unknown[],
    expected: bigint
  ): Promise<Record<Strategy, ErrorValue>> {
    const gas = {} as Record<Strategy, bigint>;
    const errors = {} as Record<Strategy, ErrorValue>;
    for (const strategy of STRATEGIES) {
      const callable = contracts[strategy].getFunction(method);
      const output = await callable.staticCall(...args) as string;
      gas[strategy] = await callable.estimateGas(...args);
      errors[strategy] = exactIntegerError(output, expected);
      const relative = relativeError(errors[strategy], expected < 0n ? -expected : expected);
      console.log(
        `MATRIX_ACCUM_AB | workload=${workload} | strategy=${strategy} | gas=${gas[strategy]} | ` +
        `absError=${decimal(errors[strategy])} | relError=${decimal(relative)} | output=${output}`
      );
    }
    rows.push({ workload, gas });
    return errors;
  }

  async function compareVector(
    workload: string,
    args: readonly unknown[],
    expected: bigint[]
  ): Promise<Record<Strategy, ErrorValue>> {
    const gas = {} as Record<Strategy, bigint>;
    const errors = {} as Record<Strategy, ErrorValue>;
    const expectedScale = expected.reduce((max, value) => {
      const magnitude = value < 0n ? -value : value;
      return magnitude > max ? magnitude : max;
    }, 0n);
    for (const strategy of STRATEGIES) {
      const callable = contracts[strategy].getFunction("matrixVector");
      const output = await callable.staticCall(...args) as string[];
      gas[strategy] = await callable.estimateGas(...args);
      errors[strategy] = maxError([...output], expected);
      console.log(
        `MATRIX_ACCUM_AB | workload=${workload} | strategy=${strategy} | gas=${gas[strategy]} | ` +
        `maxAbsError=${decimal(errors[strategy])} | maxScaleRelativeError=${decimal(relativeError(errors[strategy], expectedScale))} | ` +
        `output=${output.join(",")}`
      );
    }
    rows.push({ workload, gas });
    return errors;
  }

  it("compares an ordinary well-conditioned dot product and matvec", async function () {
    const a = await Promise.all(Array.from({ length: 16 }, (_, i) => q(BigInt(i + 1))));
    const x = await Promise.all(Array.from({ length: 16 }, (_, i) => q(BigInt(16 - i))));
    const dotErrors = await compareScalar("dot-ordinary-16", "dot", [a, x], 816n);
    for (const strategy of STRATEGIES) expect(dotErrors[strategy].numerator).to.equal(0n);

    const secondRow = Array<string>(16).fill(one);
    const matvecErrors = await compareVector("matvec-ordinary-2x16", [2n, 16n, [...a, ...secondRow], x], [816n, 136n]);
    for (const strategy of STRATEGIES) expect(matvecErrors[strategy].numerator).to.equal(0n);
  });

  it("compares cancellation-heavy dot and matvec accumulation", async function () {
    const large = await q(1n << 116n);
    const negativeLarge = await q(-(1n << 116n));
    const cancellationRow = [large, ...Array<string>(32).fill(one), negativeLarge];
    const x = Array<string>(34).fill(one);

    const dotErrors = await compareScalar("dot-cancellation-34", "dot", [cancellationRow, x], 32n);
    expect(decimal(dotErrors.naive)).to.equal("32");
    expect(decimal(dotErrors.pairwise)).to.equal("8");
    expect(dotErrors.compensated.numerator).to.equal(0n);

    const ordinaryRow = Array<string>(34).fill(one);
    const matvecErrors = await compareVector(
      "matvec-cancellation-2x34",
      [2n, 34n, [...cancellationRow, ...ordinaryRow], x],
      [32n, 34n]
    );
    expect(decimal(matvecErrors.naive)).to.equal("32");
    expect(decimal(matvecErrors.pairwise)).to.equal("8");
    expect(matvecErrors.compensated.numerator).to.equal(0n);
  });

  it("compares a wide-dynamic-range nonnegative squared norm", async function () {
    const large = await q(1n << 58n);
    const values = [large, ...Array<string>(32).fill(one)];
    const expected = (1n << 116n) + 32n;
    const errors = await compareScalar("squared-norm-wide-range-33", "dot", [values, values], expected);
    expect(decimal(errors.naive)).to.equal("32");
    expect(decimal(errors.pairwise)).to.equal("16");
    expect(errors.compensated.numerator).to.equal(0n);
  });

  it("propagates accumulation error through a solver-shaped residual norm", async function () {
    const large = await q(1n << 116n);
    const negativeLarge = await q(-(1n << 116n));
    const firstRow = [large, ...Array<string>(32).fill(one), negativeLarge];
    const secondRow = Array<string>(34).fill(one);
    const x = Array<string>(34).fill(one);
    const b = [await q(31n), await q(34n)];
    const errors = await compareScalar(
      "residual-norm-cancellation-2x34",
      "residualNormSquared",
      [2n, 34n, [...firstRow, ...secondRow], x, b],
      1n
    );
    expect(decimal(errors.naive)).to.equal("960");
    expect(decimal(errors.pairwise)).to.equal("80");
    expect(errors.compensated.numerator).to.equal(0n);
  });

  after(async function () {
    console.log("------------------------------------------------------------");
    console.log("MATRIX_ACCUMULATION_DEPLOYMENT_AND_SIZE");
    for (const strategy of STRATEGIES) {
      const runtime = await ethers.provider.getCode(await contracts[strategy].getAddress());
      console.log(
        `strategy=${strategy} | creationBytes=${byteLength(factories[strategy].bytecode)} | ` +
        `runtimeBytes=${byteLength(runtime)} | deploymentGas=${await deploymentGas(contracts[strategy])}`
      );
    }
    for (const row of rows) {
      console.log(
        `MATRIX_ACCUM_GAS | workload=${row.workload} | naive=${row.gas.naive} | ` +
        `pairwise=${row.gas.pairwise} | compensated=${row.gas.compensated}`
      );
    }
    console.log("============================================================");
  });
});
