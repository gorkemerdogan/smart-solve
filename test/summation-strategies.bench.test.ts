// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { BaseContract, ContractFactory, ContractTransactionResponse } from "ethers";
import { binary128ToRational } from "./precision-utils";

type Strategy = "naive" | "compensated" | "pairwise";
type GasRow = { workload: string; gas: Record<Strategy, bigint> };

const STRATEGIES: Strategy[] = ["naive", "compensated", "pairwise"];

function byteLength(bytecode: string): number {
  return (bytecode.length - 2) / 2;
}

function exactIntegerError(value: string, expected: bigint): { numerator: bigint; denominator: bigint } {
  const actual = binary128ToRational(value);
  const signedError = actual.num - expected * actual.den;
  return {
    numerator: signedError < 0n ? -signedError : signedError,
    denominator: actual.den,
  };
}

function formatError(error: { numerator: bigint; denominator: bigint }): string {
  if (error.numerator === 0n) return "0";
  if (error.denominator === 1n) return error.numerator.toString();
  return `${error.numerator}/${error.denominator}`;
}

async function deploymentGas(contract: BaseContract): Promise<bigint> {
  const transaction = contract.deploymentTransaction() as ContractTransactionResponse | null;
  if (!transaction) throw new Error("Missing deployment transaction");
  const receipt = await transaction.wait();
  if (!receipt) throw new Error("Missing deployment receipt");
  return receipt.gasUsed;
}

describe("Naive vs compensated vs pairwise binary128 accumulation", function () {
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

    const definitions: Array<[Strategy, string]> = [
      ["naive", "NaiveSummationBenchmarkHarness"],
      ["compensated", "CompensatedSummationBenchmarkHarness"],
      ["pairwise", "PairwiseSummationBenchmarkHarness"],
    ];
    for (const [strategy, contractName] of definitions) {
      factories[strategy] = await ethers.getContractFactory(contractName, {
        libraries: { MathLib: await mathLib.getAddress() },
      });
      contracts[strategy] = await factories[strategy].deploy();
      await contracts[strategy].waitForDeployment();
    }
    one = await (mathLib as any).fromInt(1n);

    console.log("============================================================");
    console.log("BINARY128 SUMMATION STRATEGY BENCHMARK — TEST ONLY");
    console.log("Execution model: direct harness estimateGas + eth_call result");
    console.log("Oracle: exact BigInt rational decoding of returned binary128 words");
    console.log("Math path: production-style linked MathLib calls");
    console.log("============================================================");
  });

  async function compareScalar(
    workload: string,
    method: "sum" | "dot" | "squaredNorm",
    args: readonly unknown[],
    expected: bigint
  ): Promise<Record<Strategy, string>> {
    const outputs = {} as Record<Strategy, string>;
    const gas = {} as Record<Strategy, bigint>;
    for (const strategy of STRATEGIES) {
      const callable = contracts[strategy].getFunction(method);
      outputs[strategy] = await callable.staticCall(...args) as string;
      gas[strategy] = await callable.estimateGas(...args);
      const error = exactIntegerError(outputs[strategy], expected);
      console.log(
        `SUMMATION_AB | workload=${workload} | strategy=${strategy} | gas=${gas[strategy]} | ` +
        `result=${outputs[strategy]} | exactError=${formatError(error)}`
      );
    }
    rows.push({ workload, gas });
    return outputs;
  }

  it("compares ordinary and cancellation-heavy integration-style sample sums", async function () {
    const ordinary = await Promise.all(
      Array.from({ length: 32 }, (_, index) => (mathLib as any).fromInt(BigInt(index + 1)))
    );
    const ordinaryOutputs = await compareScalar("integration-samples-ordinary-32", "sum", [ordinary], 528n);
    for (const strategy of STRATEGIES) {
      expect(exactIntegerError(ordinaryOutputs[strategy], 528n).numerator).to.equal(0n);
    }

    const large = await (mathLib as any).fromInt(1n << 116n);
    const negativeLarge = await (mathLib as any).fromInt(-(1n << 116n));
    const cancellation = [large, ...Array<string>(32).fill(one), negativeLarge];
    const cancellationOutputs = await compareScalar(
      "integration-samples-cancellation-34",
      "sum",
      [cancellation],
      32n
    );
    const naiveError = exactIntegerError(cancellationOutputs.naive, 32n).numerator;
    const pairwiseError = exactIntegerError(cancellationOutputs.pairwise, 32n).numerator;
    expect(naiveError).to.be.greaterThan(0n);
    expect(exactIntegerError(cancellationOutputs.compensated, 32n).numerator).to.equal(0n);
    expect(pairwiseError).to.be.lessThan(naiveError);
  });

  it("compares dot-product cancellation and nonnegative squared norms", async function () {
    const ordinaryA = await Promise.all(
      Array.from({ length: 16 }, (_, index) => (mathLib as any).fromInt(BigInt(index + 1)))
    );
    const ordinaryB = await Promise.all(
      Array.from({ length: 16 }, (_, index) => (mathLib as any).fromInt(BigInt(16 - index)))
    );
    const ordinaryDotOutputs = await compareScalar("dot-ordinary-16", "dot", [ordinaryA, ordinaryB], 816n);
    for (const strategy of STRATEGIES) {
      expect(exactIntegerError(ordinaryDotOutputs[strategy], 816n).numerator).to.equal(0n);
    }

    const large = await (mathLib as any).fromInt(1n << 116n);
    const negativeLarge = await (mathLib as any).fromInt(-(1n << 116n));
    const cancellation = [large, ...Array<string>(32).fill(one), negativeLarge];
    const weights = Array<string>(34).fill(one);
    const dotOutputs = await compareScalar("dot-cancellation-34", "dot", [cancellation, weights], 32n);
    const naiveDotError = exactIntegerError(dotOutputs.naive, 32n).numerator;
    const pairwiseDotError = exactIntegerError(dotOutputs.pairwise, 32n).numerator;
    expect(naiveDotError).to.be.greaterThan(0n);
    expect(exactIntegerError(dotOutputs.compensated, 32n).numerator).to.equal(0n);
    expect(pairwiseDotError).to.be.lessThan(naiveDotError);

    const largeNormEntry = await (mathLib as any).fromInt(1n << 58n);
    const normValues = [largeNormEntry, ...Array<string>(32).fill(one)];
    const exactNormSquared = (1n << 116n) + 32n;
    const normOutputs = await compareScalar("squared-norm-wide-range-33", "squaredNorm", [normValues], exactNormSquared);
    const naiveNormError = exactIntegerError(normOutputs.naive, exactNormSquared).numerator;
    const pairwiseNormError = exactIntegerError(normOutputs.pairwise, exactNormSquared).numerator;
    expect(naiveNormError).to.be.greaterThan(0n);
    expect(exactIntegerError(normOutputs.compensated, exactNormSquared).numerator).to.equal(0n);
    expect(pairwiseNormError).to.be.lessThan(naiveNormError);
  });

  it("compares a cancellation-heavy matrix-product cell", async function () {
    const ordinaryRow = await Promise.all(
      Array.from({ length: 16 }, (_, index) => (mathLib as any).fromInt(BigInt(index + 1)))
    );
    const ordinaryColumn = await Promise.all(
      Array.from({ length: 16 }, (_, index) => (mathLib as any).fromInt(BigInt(16 - index)))
    );
    for (const strategy of STRATEGIES) {
      const callable = contracts[strategy].getFunction("multiplyMatrices");
      const args = [1n, 16n, 1n, ordinaryRow, ordinaryColumn];
      const output = await callable.staticCall(...args) as string[];
      const ordinaryGas = await callable.estimateGas(...args);
      const error = exactIntegerError(output[0], 816n);
      expect(error.numerator).to.equal(0n);
      console.log(
        `SUMMATION_AB | workload=matrix-cell-ordinary-16 | strategy=${strategy} | ` +
        `gas=${ordinaryGas} | result=${output[0]} | exactError=${formatError(error)}`
      );
    }

    const large = await (mathLib as any).fromInt(1n << 116n);
    const negativeLarge = await (mathLib as any).fromInt(-(1n << 116n));
    const row = [large, ...Array<string>(32).fill(one), negativeLarge];
    const column = Array<string>(34).fill(one);
    const gas = {} as Record<Strategy, bigint>;

    for (const strategy of STRATEGIES) {
      const callable = contracts[strategy].getFunction("multiplyMatrices");
      const args = [1n, 34n, 1n, row, column];
      const output = await callable.staticCall(...args) as string[];
      gas[strategy] = await callable.estimateGas(...args);
      const error = exactIntegerError(output[0], 32n);
      console.log(
        `SUMMATION_AB | workload=matrix-cell-cancellation-34 | strategy=${strategy} | ` +
        `gas=${gas[strategy]} | result=${output[0]} | exactError=${formatError(error)}`
      );
      if (strategy === "naive") expect(error.numerator).to.equal(32n);
      if (strategy === "compensated") expect(error.numerator).to.equal(0n);
      if (strategy === "pairwise") expect(error.numerator).to.equal(8n);
    }
    rows.push({ workload: "matrix-cell-cancellation-34", gas });
  });

  after(async function () {
    console.log("------------------------------------------------------------");
    console.log("SUMMATION_AB_DEPLOYMENT_AND_SIZE");
    for (const strategy of STRATEGIES) {
      const runtime = await ethers.provider.getCode(await contracts[strategy].getAddress());
      console.log(
        `strategy=${strategy} | creationBytes=${byteLength(factories[strategy].bytecode)} | ` +
        `runtimeBytes=${byteLength(runtime)} | deploymentGas=${await deploymentGas(contracts[strategy])}`
      );
    }
    for (const row of rows) {
      console.log(
        `SUMMATION_AB_GAS_SUMMARY | workload=${row.workload} | naive=${row.gas.naive} | ` +
        `compensated=${row.gas.compensated} | pairwise=${row.gas.pairwise}`
      );
    }
    console.log("============================================================");
  });
});
