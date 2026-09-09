// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { BaseContract } from "ethers";
import { deployFullSmartSolve } from "../scripts/deploy";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";
import { binary128ToRational, type ExactRational } from "./precision-utils";
import {
  classifyExecutionFailure,
  DIAMOND_ESTIMATE_CALL,
  formatBenchmarkExecution,
  type ExecutionFailureKind,
} from "./test-utils";

type ErrorValue = { num: bigint; den: bigint };
type FailureCounts = Record<ExecutionFailureKind, number>;

type IntegrationAttempt = {
  n: bigint;
  gas?: bigint;
  error?: ErrorValue;
  failure?: ExecutionFailureKind;
};

type RootAttempt = {
  tolerance: string;
  maxIter: bigint;
  gas?: bigint;
  rootError?: ErrorValue;
  residual?: ErrorValue;
  iterations?: bigint;
  converged?: boolean;
  failure?: ExecutionFailureKind;
};

const INTEGRATION_SUBDIVISIONS = [2n, 4n, 8n, 16n];
const INTEGRATION_EPSILONS = ["1e-2", "1e-3"] as const;
const ROOT_CONFIGURATIONS = [
  { label: "1e-4", denominator: 10n ** 4n, maxIter: 16n },
  { label: "1e-8", denominator: 10n ** 8n, maxIter: 32n },
  { label: "1e-12", denominator: 10n ** 12n, maxIter: 64n },
] as const;
const ROOT_EPSILONS = ["1e-3", "1e-7", "1e-11"] as const;

function abs(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function error(actual: ExactRational, expected: ExactRational): ErrorValue {
  return {
    num: abs(actual.num * expected.den - expected.num * actual.den),
    den: actual.den * expected.den,
  };
}

function relativeError(value: ErrorValue | undefined, expected: ExactRational): ErrorValue | undefined {
  if (!value || expected.num === 0n) return undefined;
  return {
    num: value.num * expected.den,
    den: value.den * abs(expected.num),
  };
}

function atMost(value: ErrorValue, threshold: ErrorValue): boolean {
  return value.num * threshold.den <= threshold.num * value.den;
}

function decimalEpsilon(label: string): ErrorValue {
  const match = /^1e-(\d+)$/.exec(label);
  if (!match) throw new Error(`Unsupported epsilon ${label}`);
  return { num: 1n, den: 10n ** BigInt(match[1]) };
}

function formatRational(value: ErrorValue | undefined, digits = 20): string {
  if (!value) return "n/a";
  if (value.num === 0n) return "0";
  const integer = value.num / value.den;
  const fractional = ((value.num % value.den) * (10n ** BigInt(digits))) / value.den;
  const suffix = fractional.toString().padStart(digits, "0").replace(/0+$/, "");
  return `${integer}${suffix ? `.${suffix}` : ""}`;
}

function failureCounts(attempts: Array<{ failure?: ExecutionFailureKind }>): FailureCounts {
  const counts: FailureCounts = { revert: 0, "out-of-gas": 0, failure: 0 };
  for (const attempt of attempts) {
    if (attempt.failure) counts[attempt.failure]++;
  }
  return counts;
}

function lowestGas<T extends { gas?: bigint }>(attempts: T[]): T | undefined {
  return attempts.reduce<T | undefined>((best, candidate) => {
    if (candidate.gas === undefined) return best;
    if (!best || best.gas === undefined || candidate.gas < best.gas) return candidate;
    return best;
  }, undefined);
}

function bestError<T extends { rootError?: ErrorValue; error?: ErrorValue }>(attempts: T[]): T | undefined {
  return attempts.reduce<T | undefined>((best, candidate) => {
    const candidateError = candidate.rootError ?? candidate.error;
    const bestValue = best?.rootError ?? best?.error;
    if (!candidateError) return best;
    if (!bestValue || candidateError.num * bestValue.den < bestValue.num * candidateError.den) return candidate;
    return best;
  }, undefined);
}

/**
 * Bounded production-path equal-accuracy benchmark. The numerical calls enter
 * the fully deployed Diamond; callback harnesses only provide deterministic
 * functions and binary128 input conversion.
 */
describe("Diamond-routed equal-accuracy benchmark: integration and root finding", function () {
  this.timeout(120_000);

  let integration: BaseContract;
  let rootFinding: BaseContract;
  let numericConfig: BaseContract;
  let integrationCallback: BaseContract;
  let rootCallback: BaseContract;
  let writer: BenchmarkResultWriter;

  before(async function () {
    writer = await BenchmarkResultWriter.create({
      suite: "equal-accuracy.diamond.integration-root-finding",
    });

    const deployment = await deployFullSmartSolve();
    integration = await ethers.getContractAt("IntegrationFacet", deployment.diamondAddress);
    rootFinding = await ethers.getContractAt("RootFindingFacet", deployment.diamondAddress);
    numericConfig = await ethers.getContractAt("NumericConfigFacet", deployment.diamondAddress);
    const loupe = await ethers.getContractAt("IDiamondLoupe", deployment.diamondAddress);

    expect(await loupe.facetAddress(integration.interface.getFunction("integrateTrapezoidalWithN")!.selector))
      .to.equal(deployment.facetAddresses.IntegrationFacet);
    expect(await loupe.facetAddress(rootFinding.interface.getFunction("rootFindingBisection")!.selector))
      .to.equal(deployment.facetAddresses.RootFindingFacet);
    expect(await loupe.facetAddress(numericConfig.interface.getFunction("setTol")!.selector))
      .to.equal(deployment.facetAddresses.NumericConfigFacet);

    const localMath = deployment.mathLibAddress;
    const trigMathFactory = await ethers.getContractFactory(
      "@gorkemerdogan/trigonometry-master/contracts/libraries/MathLib.sol:MathLib",
    );
    const trigMath = await trigMathFactory.deploy();
    await trigMath.waitForDeployment();
    const IntegrationHarness = await ethers.getContractFactory("IntegrationHarness", {
      libraries: {
        "contracts/libraries/MathLib.sol:MathLib": localMath,
        "@gorkemerdogan/trigonometry-master/contracts/libraries/MathLib.sol:MathLib": await trigMath.getAddress(),
      },
    });
    const RootFindingHarness = await ethers.getContractFactory("RootFindingHarness", {
      libraries: { MathLib: localMath },
    });
    [integrationCallback, rootCallback] = await Promise.all([
      IntegrationHarness.deploy(),
      RootFindingHarness.deploy(),
    ]);
    await Promise.all([integrationCallback.waitForDeployment(), rootCallback.waitForDeployment()]);

    console.log("Diamond-routed equal-accuracy benchmark");
    console.log(`  execution model: ${formatBenchmarkExecution(DIAMOND_ESTIMATE_CALL)}`);
    console.log("  route: full SmartSolve Diamond fallback; callback gas is included and not subtracted");
  });

  after(async function () {
    await writer.flush();
  });

  it("selects minimum Diamond-routed gas for target integration error", async function () {
    const qFromInt = integrationCallback.getFunction("qFromInt");
    const [zero, one] = await Promise.all([qFromInt.staticCall(0n), qFromInt.staticCall(1n)]);
    const callbackAddress = await integrationCallback.getAddress();
    const selector = integrationCallback.interface.getFunction("f_square")!.selector;
    const exactIntegral = { num: 1n, den: 3n };
    const attempts: IntegrationAttempt[] = [];
    const method = integration.getFunction("integrateTrapezoidalWithN");

    for (const n of INTEGRATION_SUBDIVISIONS) {
      try {
        const args = [callbackAddress, selector, zero, one, n] as const;
        const [gas, result] = await Promise.all([
          method.estimateGas(...args),
          method.staticCall(...args),
        ]);
        attempts.push({ n, gas, error: error(binary128ToRational(result as string), exactIntegral) });
      } catch (cause) {
        attempts.push({ n, failure: classifyExecutionFailure(cause) });
      }
    }

    const counts = failureCounts(attempts);
    let selectedCount = 0;
    console.log("\nIntegration: trapezoidal x^2 on [0,1]");
    for (const epsilonLabel of INTEGRATION_EPSILONS) {
      const epsilon = decimalEpsilon(epsilonLabel);
      const selected = lowestGas(attempts.filter(item => item.error && atMost(item.error, epsilon)));
      const reported = selected ?? bestError(attempts);
      if (selected) selectedCount++;
      console.log(
        `  eps=${epsilonLabel} status=${selected ? "success" : "no-match"}` +
        ` n=${selected?.n ?? "-"} gas=${selected?.gas ?? "-"}` +
        ` error=${formatRational(reported?.error)} evals=${selected ? selected.n + 1n : "-"}` +
        ` failures=revert:${counts.revert}/oog:${counts["out-of-gas"]}/other:${counts.failure}`,
      );

      writer.record({
        benchmark: "minimum-gas-at-target-error",
        category: "integration-equal-accuracy",
        operation: "trapezoidal",
        execution: benchmarkExecutionRecord(DIAMOND_ESTIMATE_CALL),
        input: {
          problem: "x^2 on [0,1]",
          epsilon: epsilonLabel,
          testedSubdivisions: INTEGRATION_SUBDIVISIONS.map(value => value.toString()),
          selectedSubdivisions: selected?.n.toString() ?? null,
          selection: "minimum Diamond-routed estimateGas among tested configurations satisfying absolute error <= epsilon",
          oracle: "analytic integral 1/3 (exact rational)",
        },
        callback: {
          model: "target_staticcall",
          functionEvaluations: selected ? (selected.n + 1n).toString() : undefined,
          detail: "integrand f(x)=x^2; end-to-end callback gas included",
          gasIncludesCallback: true,
        },
        gas: selected?.gas?.toString(),
        status: selected ? "success" : "failure",
        errorMetrics: {
          absoluteError: formatRational(reported?.error),
          absoluteErrorNumerator: reported?.error?.num.toString() ?? null,
          absoluteErrorDenominator: reported?.error?.den.toString() ?? null,
          relativeError: formatRational(relativeError(reported?.error, exactIntegral)),
          threshold: epsilonLabel,
          comparison: "exact rational comparison after binary128 decoding",
        },
        convergence: {
          classification: selected ? "accuracy-threshold-reached" : "accuracy-threshold-not-reached",
          attempts: attempts.length,
          successfulAttempts: attempts.filter(item => item.gas !== undefined).length,
          reverted: counts.revert,
          outOfGas: counts["out-of-gas"],
          otherFailures: counts.failure,
        },
      });
    }

    expect(selectedCount, "each bounded integration target should be reached").to.equal(INTEGRATION_EPSILONS.length);
  });

  it("selects minimum Diamond-routed gas for target root error", async function () {
    const qFromInt = rootCallback.getFunction("qFromInt");
    const qFromFrac = rootCallback.getFunction("qFromFrac");
    const [one, three, four, tolerances] = await Promise.all([
      qFromInt.staticCall(1n),
      qFromInt.staticCall(3n),
      qFromInt.staticCall(4n),
      Promise.all(ROOT_CONFIGURATIONS.map(item => qFromFrac.staticCall(1n, item.denominator))),
    ]);
    const callbackAddress = await rootCallback.getAddress();
    const functionSelector = rootCallback.interface.getFunction("f_x2_minus_4")!.selector;
    const derivativeSelector = rootCallback.interface.getFunction("df_2x")!.selector;
    const exactRoot = { num: 2n, den: 1n };
    const setTol = numericConfig.getFunction("setTol");
    const setMaxIter = numericConfig.getFunction("setMaxIter");
    let selectedCount = 0;

    console.log("\nRoot finding: x^2 - 4 = 0");
    for (const methodName of ["bisection", "newton", "secant"] as const) {
      const method = rootFinding.getFunction(`rootFinding${methodName[0].toUpperCase()}${methodName.slice(1)}`);
      const attempts: RootAttempt[] = [];
      for (const [index, configuration] of ROOT_CONFIGURATIONS.entries()) {
        try {
          // RootFindingFacet deliberately consumes these proxy-held effective
          // values rather than accepting per-call tolerances.
          await (await setTol.send(tolerances[index])).wait();
          await (await setMaxIter.send(configuration.maxIter)).wait();
          const args = methodName === "bisection"
            ? [callbackAddress, functionSelector, one, four]
            : methodName === "newton"
              ? [callbackAddress, functionSelector, callbackAddress, derivativeSelector, three]
              : [callbackAddress, functionSelector, one, four];
          const [gas, result] = await Promise.all([
            method.estimateGas(...args),
            method.staticCall(...args),
          ]);
          const decoded = result as { root: string; iterations: bigint; converged: boolean; fAtRoot: string };
          attempts.push({
            tolerance: configuration.label,
            maxIter: configuration.maxIter,
            gas,
            iterations: decoded.iterations,
            converged: decoded.converged,
            rootError: error(binary128ToRational(decoded.root), exactRoot),
            residual: error(binary128ToRational(decoded.fAtRoot), { num: 0n, den: 1n }),
          });
        } catch (cause) {
          attempts.push({
            tolerance: configuration.label,
            maxIter: configuration.maxIter,
            failure: classifyExecutionFailure(cause),
          });
        }
      }

      const counts = failureCounts(attempts);
      for (const epsilonLabel of ROOT_EPSILONS) {
        const epsilon = decimalEpsilon(epsilonLabel);
        const selected = lowestGas(attempts.filter(
          item => item.converged === true && item.rootError && atMost(item.rootError, epsilon),
        ));
        const reported = selected ?? bestError(attempts.filter(item => item.converged));
        const functionEvaluations = selected?.iterations === undefined
          ? undefined
          : methodName === "newton"
            ? 1n + selected.iterations
            : 2n + selected.iterations;
        const derivativeEvaluations = methodName === "newton" && selected?.iterations !== undefined
          ? selected.iterations
          : 0n;
        const totalStaticCalls = functionEvaluations === undefined
          ? undefined
          : functionEvaluations + derivativeEvaluations;
        if (selected) selectedCount++;

        console.log(
          `  ${methodName.padEnd(9)} eps=${epsilonLabel} status=${selected ? "success" : "no-match"}` +
          ` tol=${selected?.tolerance ?? "-"} maxIter=${selected?.maxIter ?? "-"}` +
          ` gas=${selected?.gas ?? "-"} iter=${selected?.iterations ?? "-"}` +
          ` error=${formatRational(reported?.rootError)} fEvals=${functionEvaluations ?? "-"}` +
          ` dfEvals=${derivativeEvaluations ?? "-"}` +
          ` failures=revert:${counts.revert}/oog:${counts["out-of-gas"]}/other:${counts.failure}`,
        );

        writer.record({
          benchmark: "minimum-gas-at-target-root-error",
          category: "root-finding-equal-accuracy",
          operation: methodName,
          execution: benchmarkExecutionRecord(DIAMOND_ESTIMATE_CALL),
          input: {
            problem: "x^2 - 4 = 0",
            epsilon: epsilonLabel,
            testedConfigurations: ROOT_CONFIGURATIONS.map(item => `${item.label}/maxIter=${item.maxIter}`),
            selectedTolerance: selected?.tolerance ?? null,
            selectedMaxIter: selected?.maxIter.toString() ?? null,
            configurationApplication: "setTol and setMaxIter through NumericConfigFacet on the Diamond; setter transaction gas excluded",
            selection: "minimum Diamond-routed estimateGas among converged tested configurations satisfying root error <= epsilon",
            oracle: "analytic root 2 (exact integer)",
          },
          callback: {
            model: "target_staticcall",
            functionEvaluations: totalStaticCalls?.toString(),
            detail: methodName === "newton"
              ? `f evaluations=${functionEvaluations ?? "-"}; derivative evaluations=${derivativeEvaluations ?? "-"}; end-to-end callback gas included`
              : `f evaluations=${functionEvaluations ?? "-"}; end-to-end callback gas included`,
            gasIncludesCallback: true,
          },
          gas: selected?.gas?.toString(),
          status: selected ? "success" : "failure",
          errorMetrics: {
            absoluteError: formatRational(reported?.rootError),
            absoluteErrorNumerator: reported?.rootError?.num.toString() ?? null,
            absoluteErrorDenominator: reported?.rootError?.den.toString() ?? null,
            relativeError: formatRational(relativeError(reported?.rootError, exactRoot)),
            residualMagnitude: formatRational(reported?.residual),
            threshold: epsilonLabel,
            comparison: "exact rational comparison after binary128 decoding",
          },
          convergence: {
            classification: selected ? "converged-and-accurate" : "no-converged-configuration-met-error",
            converged: selected?.converged ?? false,
            iterations: selected?.iterations?.toString() ?? null,
            functionEvaluations: functionEvaluations?.toString() ?? null,
            derivativeEvaluations: derivativeEvaluations?.toString() ?? null,
            attempts: attempts.length,
            nonConvergedAttempts: attempts.filter(item => item.converged === false).length,
            reverted: counts.revert,
            outOfGas: counts["out-of-gas"],
            otherFailures: counts.failure,
          },
        });
      }
    }

    expect(selectedCount, "each root method/epsilon pair should be reached").to.equal(3 * ROOT_EPSILONS.length);
  });
});
