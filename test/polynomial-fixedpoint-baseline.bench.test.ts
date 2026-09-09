// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { BaseContract, ContractTransactionReceipt, ContractTransactionResponse } from "ethers";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";
import { binary128ToRational, formatScaledInt, type ExactRational } from "./precision-utils";
import { FACET_ESTIMATE_CALL, HARNESS_ESTIMATE_CALL } from "./test-utils";

type Fraction = ExactRational;

type BaselineCase = {
  name: string;
  coefficients: Fraction[]; // ascending powers
  x: Fraction;
  fixedPointExact: boolean;
};

const WAD = 10n ** 18n;
const REPORT_SCALE = 10n ** 36n;

const cases: BaselineCase[] = [
  {
    name: "small-integer-coefficients",
    coefficients: [{ num: 5n, den: 1n }, { num: 2n, den: 1n }, { num: 3n, den: 1n }],
    x: { num: 5n, den: 4n },
    fixedPointExact: true,
  },
  {
    name: "terminating-fraction-coefficients",
    coefficients: [{ num: 1n, den: 8n }, { num: -3n, den: 4n }, { num: 5n, den: 2n }, { num: -1n, den: 4n }],
    x: { num: 7n, den: 8n },
    fixedPointExact: true,
  },
  {
    // 1/3 is deliberately truncated at the 1e18 input scale. The binary128
    // path is converted from the same exact rational source, then evaluated
    // by the production PolynomialFacet implementation.
    name: "one-third-input-rounding",
    coefficients: [{ num: 1n, den: 3n }, { num: -2n, den: 3n }, { num: 1n, den: 3n }],
    x: { num: 1n, den: 3n },
    fixedPointExact: false,
  },
];

function abs(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function normalize(value: Fraction): Fraction {
  if (value.den === 0n) throw new Error("Rational denominator must be non-zero");
  const sign = value.den < 0n ? -1n : 1n;
  let numerator = value.num * sign;
  let denominator = value.den * sign;
  let a = abs(numerator);
  let b = denominator;
  while (b !== 0n) [a, b] = [b, a % b];
  numerator /= a;
  denominator /= a;
  return { num: numerator, den: denominator };
}

function add(left: Fraction, right: Fraction): Fraction {
  return normalize({
    num: left.num * right.den + right.num * left.den,
    den: left.den * right.den,
  });
}

function multiply(left: Fraction, right: Fraction): Fraction {
  return normalize({ num: left.num * right.num, den: left.den * right.den });
}

function evaluateExactly(coefficients: Fraction[], x: Fraction): Fraction {
  let value: Fraction = { num: 0n, den: 1n };
  for (let index = coefficients.length - 1; index >= 0; --index) {
    value = add(multiply(value, x), coefficients[index]);
  }
  return value;
}

function toFixed(value: Fraction): bigint {
  // BigInt division, like Solidity signed division, rounds toward zero.
  return (value.num * WAD) / value.den;
}

function scaledAbsoluteError(actual: Fraction, expected: Fraction): bigint {
  return (abs(actual.num * expected.den - expected.num * actual.den) * REPORT_SCALE)
    / (actual.den * expected.den);
}

function scaledRelativeError(actual: Fraction, expected: Fraction): bigint | undefined {
  if (expected.num === 0n) return undefined;
  return (abs(actual.num * expected.den - expected.num * actual.den) * REPORT_SCALE)
    / (actual.den * abs(expected.num));
}

function rationalText(value: Fraction): string {
  return `${value.num}/${value.den}`;
}

function byteLength(bytecode: string): number {
  return (bytecode.length - 2) / 2;
}

async function deploymentReceipt(contract: BaseContract): Promise<ContractTransactionReceipt> {
  const transaction = contract.deploymentTransaction() as ContractTransactionResponse | null;
  if (!transaction) throw new Error("Missing deployment transaction");
  const receipt = await transaction.wait();
  if (!receipt) throw new Error("Missing deployment receipt");
  return receipt;
}

describe("Polynomial Horner fixed-point 1e18 representation baseline", function () {
  this.timeout(120_000);

  let converter: BaseContract;
  let binary128Facet: BaseContract;
  let fixedPointHarness: BaseContract;
  let mathLibDeploymentGas: bigint;
  let binary128DeploymentGas: bigint;
  let fixedPointDeploymentGas: bigint;
  let mathLibRuntimeBytes: number;
  let binary128RuntimeBytes: number;
  let fixedPointRuntimeBytes: number;
  let writer: BenchmarkResultWriter;

  before(async function () {
    writer = await BenchmarkResultWriter.create({
      suite: "polynomial-fixedpoint-horner-baseline",
    });

    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    const mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();

    const libraries = { MathLib: await mathLib.getAddress() };
    const PolynomialHarness = await ethers.getContractFactory("PolynomialHarness", { libraries });
    const PolynomialFacet = await ethers.getContractFactory("PolynomialFacet", { libraries });
    const FixedPointHarness = await ethers.getContractFactory("FixedPointHornerBaselineHarness");
    [converter, binary128Facet, fixedPointHarness] = await Promise.all([
      PolynomialHarness.deploy(),
      PolynomialFacet.deploy(),
      FixedPointHarness.deploy(),
    ]);
    await Promise.all([
      converter.waitForDeployment(),
      binary128Facet.waitForDeployment(),
      fixedPointHarness.waitForDeployment(),
    ]);

    const [mathLibReceipt, binary128Receipt, fixedPointReceipt, mathLibCode, binary128Code, fixedPointCode] = await Promise.all([
      deploymentReceipt(mathLib),
      deploymentReceipt(binary128Facet),
      deploymentReceipt(fixedPointHarness),
      ethers.provider.getCode(await mathLib.getAddress()),
      ethers.provider.getCode(await binary128Facet.getAddress()),
      ethers.provider.getCode(await fixedPointHarness.getAddress()),
    ]);
    mathLibDeploymentGas = mathLibReceipt.gasUsed;
    binary128DeploymentGas = binary128Receipt.gasUsed;
    fixedPointDeploymentGas = fixedPointReceipt.gasUsed;
    mathLibRuntimeBytes = byteLength(mathLibCode);
    binary128RuntimeBytes = byteLength(binary128Code);
    fixedPointRuntimeBytes = byteLength(fixedPointCode);

    console.log("============================================================");
    console.log("POLYNOMIAL HORNER — REPRESENTATION BASELINE");
    console.log("Binary128: production PolynomialFacet.polyEvaluate (direct facet call)");
    console.log("Fixed point: isolated test-only signed 1e18 Horner harness");
    console.log("Gas model: estimateGas; result model: eth_call_result");
    console.log("Oracle: exact rational arithmetic from the unrounded source inputs");
    console.log("============================================================");
    console.log(`Linked MathLib runtime/deployment: ${mathLibRuntimeBytes} bytes / ${mathLibDeploymentGas} gas`);
    console.log(`Production PolynomialFacet runtime/deployment: ${binary128RuntimeBytes} bytes / ${binary128DeploymentGas} gas (facet only)`);
    console.log(`Production facet + required MathLib deployment: ${binary128DeploymentGas + mathLibDeploymentGas} gas`);
    console.log(`Test-only fixed-point harness runtime/deployment: ${fixedPointRuntimeBytes} bytes / ${fixedPointDeploymentGas} gas`);

    writer.record({
      benchmark: "linked-deployment-footprint",
      category: "polynomial",
      operation: "Horner evaluation deployment footprint",
      execution: benchmarkExecutionRecord(FACET_ESTIMATE_CALL),
      input: {
        runtimeBytes: binary128RuntimeBytes,
        linkedMathLibRuntimeBytes: mathLibRuntimeBytes,
        facetDeploymentGas: binary128DeploymentGas.toString(),
        linkedMathLibDeploymentGas: mathLibDeploymentGas.toString(),
        scope: "full production PolynomialFacet plus required linked MathLib deployment",
      },
      gas: (binary128DeploymentGas + mathLibDeploymentGas).toString(),
      status: "success",
      variant: "binary128_production_facet_direct_not_horner_only_linked_deployment",
    });
    writer.record({
      benchmark: "deployment-footprint",
      category: "polynomial",
      operation: "Horner evaluation deployment footprint",
      execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
      input: { runtimeBytes: fixedPointRuntimeBytes, scope: "isolated test-only fixed-point Horner harness" },
      gas: fixedPointDeploymentGas.toString(),
      status: "success",
      variant: "fixedpoint_1e18_test_only_horner_harness",
    });
  });

  for (const benchmarkCase of cases) {
    it(`compares binary128 and 1e18 fixed point for ${benchmarkCase.name}`, async function () {
      const expected = evaluateExactly(benchmarkCase.coefficients, benchmarkCase.x);
      const qFromFrac = converter.getFunction("qFromFrac");
      const [binaryCoefficients, binaryX] = await Promise.all([
        Promise.all(benchmarkCase.coefficients.map(({ num, den }) => qFromFrac.staticCall(num, den))),
        qFromFrac.staticCall(benchmarkCase.x.num, benchmarkCase.x.den),
      ]);
      const fixedCoefficients = benchmarkCase.coefficients.map(toFixed);
      const fixedX = toFixed(benchmarkCase.x);

      const binaryMethod = binary128Facet.getFunction("polyEvaluate");
      const fixedMethod = fixedPointHarness.getFunction("evaluateHorners");
      const [binaryValue, fixedValue, binaryGas, fixedGas] = await Promise.all([
        binaryMethod.staticCall(binaryCoefficients, binaryX),
        fixedMethod.staticCall(fixedCoefficients, fixedX),
        binaryMethod.estimateGas(binaryCoefficients, binaryX),
        fixedMethod.estimateGas(fixedCoefficients, fixedX),
      ]);

      const binaryActual = binary128ToRational(binaryValue as string);
      const fixedActual = normalize({ num: fixedValue as bigint, den: WAD });
      const binaryAbsoluteError = scaledAbsoluteError(binaryActual, expected);
      const fixedAbsoluteError = scaledAbsoluteError(fixedActual, expected);
      const binaryRelativeError = scaledRelativeError(binaryActual, expected);
      const fixedRelativeError = scaledRelativeError(fixedActual, expected);

      if (benchmarkCase.fixedPointExact) {
        expect(fixedAbsoluteError, `${benchmarkCase.name} fixed-point error`).to.equal(0n);
      } else {
        expect(fixedAbsoluteError, `${benchmarkCase.name} must exercise 1e18 rounding`).to.be.greaterThan(0n);
      }

      const commonInput = {
        case: benchmarkCase.name,
        degree: benchmarkCase.coefficients.length - 1,
        coefficientSourceRationals: benchmarkCase.coefficients.map(rationalText),
        xSourceRational: rationalText(benchmarkCase.x),
        exactOracle: rationalText(expected),
      };
      const errorMetrics = (absoluteError: bigint, relativeError: bigint | undefined) => ({
        absoluteErrorScaled1e36: absoluteError.toString(),
        ...(relativeError === undefined ? {} : { relativeErrorScaled1e36: relativeError.toString() }),
        exactOracle: rationalText(expected),
      });

      writer.record({
        benchmark: benchmarkCase.name,
        category: "polynomial",
        operation: "Horner evaluation",
        execution: benchmarkExecutionRecord(FACET_ESTIMATE_CALL),
        input: commonInput,
        gas: binaryGas.toString(),
        status: "success",
        errorMetrics: errorMetrics(binaryAbsoluteError, binaryRelativeError),
        variant: "binary128_production_polynomial_facet_direct",
      });
      writer.record({
        benchmark: benchmarkCase.name,
        category: "polynomial",
        operation: "Horner evaluation",
        execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
        input: { ...commonInput, fixedPointScale: WAD.toString() },
        gas: fixedGas.toString(),
        status: "success",
        errorMetrics: errorMetrics(fixedAbsoluteError, fixedRelativeError),
        variant: "fixedpoint_1e18_test_only_harness",
      });

      console.log(
        `REPRESENTATION BASELINE | ${benchmarkCase.name} | degree=${benchmarkCase.coefficients.length - 1}` +
        ` | binary128_facet_gas=${binaryGas} | fixedpoint_1e18_gas=${fixedGas}` +
        ` | binary_abs_error_1e36=${formatScaledInt(binaryAbsoluteError, 36)}` +
        ` | fixed_abs_error_1e36=${formatScaledInt(fixedAbsoluteError, 36)}`,
      );
    });
  }

  after(async function () {
    await writer.flush();
  });
});
