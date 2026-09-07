// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { binary128ToRational } from "./precision-utils";

type Method = "trapezoidal" | "simpson13" | "simpson38";
type Variant = "naive" | "compensated";

type ErrorValue = {
  numerator: bigint;
  denominator: bigint;
};

type ResultRow = {
  callback: string;
  method: Method;
  variant: Variant;
  output: string;
  absoluteError: ErrorValue;
  relativeError: ErrorValue;
  gas: bigint;
  evaluations: bigint;
};

function errorAgainstInteger(value: string, expected: bigint): ErrorValue {
  const actual = binary128ToRational(value);
  const difference = actual.num - expected * actual.den;
  return {
    numerator: difference < 0n ? -difference : difference,
    denominator: actual.den,
  };
}

function relativeError(absolute: ErrorValue, expected: bigint): ErrorValue {
  const magnitude = expected < 0n ? -expected : expected;
  if (magnitude === 0n) return absolute;
  return { numerator: absolute.numerator, denominator: absolute.denominator * magnitude };
}

function decimal(value: ErrorValue, digits = 36): string {
  if (value.numerator === 0n) return "0";
  const scale = 10n ** BigInt(digits);
  const scaled = value.numerator * scale / value.denominator;
  const integer = scaled / scale;
  const fraction = (scaled % scale).toString().padStart(digits, "0").replace(/0+$/, "");
  return fraction.length === 0 ? integer.toString() : `${integer}.${fraction}`;
}

describe("Production-shaped compensated integration summation benchmark", function () {
  this.timeout(120_000);

  const methods: Method[] = ["trapezoidal", "simpson13", "simpson38"];
  const rows: ResultRow[] = [];
  let naive: any;
  let compensated: any;
  let target: any;
  let lower: string;
  let upper: string;

  before(async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    const mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();
    const libraries = { MathLib: await mathLib.getAddress() };

    const Naive = await ethers.getContractFactory("IntegrationNaiveBenchmarkHarness", { libraries });
    const Compensated = await ethers.getContractFactory("IntegrationCompensatedBenchmarkHarness", { libraries });
    const Target = await ethers.getContractFactory("IntegrationCancellationBenchmarkTarget", { libraries });
    [naive, compensated, target] = await Promise.all([
      Naive.deploy(),
      Compensated.deploy(),
      Target.deploy(),
    ]);
    await Promise.all([naive.waitForDeployment(), compensated.waitForDeployment(), target.waitForDeployment()]);

    lower = await (mathLib as any).fromInt(0n);
    upper = await (mathLib as any).fromInt(48n);

    console.log("============================================================");
    console.log("INTEGRATION COMPENSATED-SUMMATION A/B — TEST ONLY");
    console.log("Grid: [0,48], n=48, 49 deterministic callback evaluations per call");
    console.log("Execution gas: full harness estimateGas, including callback staticcalls and callback arithmetic");
    console.log("Oracle: analytic exact integer integral; binary128 decoded as exact BigInt rational");
    console.log("============================================================");
  });

  async function runCase(
    callback: "constantFive" | "centeredLinearModerate" | "centeredLinearStress" | "centeredCubic",
    expected: bigint
  ): Promise<void> {
    const selector = target.interface.getFunction(callback).selector;
    const address = await target.getAddress();
    let strictImprovements = 0;

    for (const method of methods) {
      const perVariant = {} as Record<Variant, ResultRow>;
      for (const [variant, harness] of [["naive", naive], ["compensated", compensated]] as const) {
        const callable = harness.getFunction(method);
        const args = [address, selector, lower, upper, 48n];
        const output = await callable.staticCall(...args) as string;
        const gas = await callable.estimateGas(...args);
        const absolute = errorAgainstInteger(output, expected);
        const relative = relativeError(absolute, expected);
        const row: ResultRow = {
          callback,
          method,
          variant,
          output,
          absoluteError: absolute,
          relativeError: relative,
          gas,
          evaluations: 49n,
        };
        rows.push(row);
        perVariant[variant] = row;
        console.log(
          `INTEGRATION_SUM_AB | callback=${callback} | method=${method} | variant=${variant} | ` +
          `absError=${decimal(absolute)} | relError=${decimal(relative)} | gas=${gas} | ` +
          `functionEvaluations=49 | callbackGasIncluded=true | output=${output}`
        );
      }

      const naiveError = perVariant.naive.absoluteError;
      const compensatedError = perVariant.compensated.absoluteError;
      expect(
        compensatedError.numerator * naiveError.denominator,
        `${callback}/${method}: compensation must not increase absolute error`
      ).to.be.at.most(naiveError.numerator * compensatedError.denominator);
      if (
        compensatedError.numerator * naiveError.denominator <
        naiveError.numerator * compensatedError.denominator
      ) strictImprovements++;
    }

    if (callback === "centeredLinearStress") {
      expect(strictImprovements, `${callback}: expected at least one strict accuracy improvement`).to.be.greaterThan(0);
    } else {
      // The positive control and symmetrically sampled cubic establish that a
      // sign-changing integrand does not automatically benefit from compensation.
      expect(strictImprovements, `${callback}: compensation should not alter an already accurate sum`).to.equal(0);
    }
  }

  it("compares an ordinary positive constant control", async function () {
    await runCase("constantFive", 240n);
  });

  it("compares a sign-changing centered linear callback", async function () {
    await runCase("centeredLinearStress", 48n);
  });

  it("compares the same centered linear callback at moderate scale", async function () {
    await runCase("centeredLinearModerate", 48n);
  });

  it("compares a sign-changing centered cubic callback", async function () {
    await runCase("centeredCubic", 48n);
  });

  after(function () {
    console.log("------------------------------------------------------------");
    for (const method of methods) {
      for (const callback of ["constantFive", "centeredLinearModerate", "centeredLinearStress", "centeredCubic"]) {
        const naiveRow = rows.find(row => row.method === method && row.callback === callback && row.variant === "naive");
        const compensatedRow = rows.find(
          row => row.method === method && row.callback === callback && row.variant === "compensated"
        );
        if (!naiveRow || !compensatedRow) continue;
        console.log(
          `INTEGRATION_SUM_GAS | callback=${callback} | method=${method} | naive=${naiveRow.gas} | ` +
          `compensated=${compensatedRow.gas} | delta=${compensatedRow.gas - naiveRow.gas} | ` +
          `functionEvaluations=${naiveRow.evaluations} | callbackGasIncluded=true`
        );
      }
    }
    console.log("============================================================");
  });
});
