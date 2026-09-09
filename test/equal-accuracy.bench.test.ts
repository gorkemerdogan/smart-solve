// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";
import { binary128ToRational, type ExactRational } from "./precision-utils";
import {
    classifyExecutionFailure,
    formatBenchmarkExecution,
    HARNESS_ESTIMATE_CALL,
    type ExecutionFailureKind,
} from "./test-utils";

type IntegrationMethod = "trapezoidal" | "simpson13" | "simpson38";
type RootMethod = "bisection" | "newton" | "secant";

type AttemptFailure = {
    classification: ExecutionFailureKind;
};

type ErrorValue = {
    num: bigint;
    den: bigint;
};

type IntegrationAttempt = {
    n: bigint;
    gas?: bigint;
    error?: ErrorValue;
    failure?: AttemptFailure;
};

type RootAttempt = {
    toleranceLabel: string;
    maxIter: bigint;
    gas?: bigint;
    error?: ErrorValue;
    residual?: ErrorValue;
    iterations?: bigint;
    converged?: boolean;
    failure?: AttemptFailure;
};

type IntegrationHarness = Contract & {
    qFromInt(value: bigint): Promise<string>;
};

type RootFindingHarness = Contract & {
    qFromInt(value: bigint): Promise<string>;
    qFromFrac(num: bigint, den: bigint): Promise<string>;
};

const INTEGRATION_METHODS: ReadonlyArray<{
    name: IntegrationMethod;
    validN: (n: bigint) => boolean;
}> = [
    { name: "trapezoidal", validN: n => n > 0n },
    { name: "simpson13", validN: n => n > 0n && n % 2n === 0n },
    { name: "simpson38", validN: n => n > 0n && n % 3n === 0n },
];

const N_SWEEP = [2n, 3n, 4n, 6n, 8n, 12n, 16n, 24n, 32n, 48n, 64n, 96n, 128n, 192n];
const INTEGRATION_EPSILONS = ["1e-3", "1e-5", "1e-7"];
const ROOT_EPSILONS = ["1e-6", "1e-12", "1e-20", "1e-28"];

function pow10(exponent: number): bigint {
    if (!Number.isInteger(exponent) || exponent < 0) throw new Error(`Invalid power of ten: ${exponent}`);
    return 10n ** BigInt(exponent);
}

/** Parse a finite decimal string as an exact rational, without using Number for its value. */
function decimalToRational(value: string): ExactRational {
    const match = value.trim().match(/^([+-]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/);
    if (!match) throw new Error(`Invalid decimal value: ${value}`);
    const negative = match[1] === "-";
    const fraction = match[3] ?? "";
    const exponent = Number(match[4] ?? "0") - fraction.length;
    let num = BigInt(`${match[2]}${fraction}`);
    let den = 1n;
    if (exponent >= 0) num *= pow10(exponent);
    else den = pow10(-exponent);
    return { num: negative ? -num : num, den };
}

function abs(value: bigint): bigint {
    return value < 0n ? -value : value;
}

function absoluteError(actual: ExactRational, expected: ExactRational): ErrorValue {
    return {
        num: abs(actual.num * expected.den - expected.num * actual.den),
        den: actual.den * expected.den,
    };
}

function relativeError(error: ErrorValue, expected: ExactRational): ErrorValue | undefined {
    if (expected.num === 0n) return undefined;
    return { num: error.num * expected.den, den: error.den * abs(expected.num) };
}

function atMost(value: ErrorValue, threshold: ExactRational): boolean {
    return value.num * threshold.den <= threshold.num * value.den;
}

function formatRational(value: ErrorValue | ExactRational | undefined, digits = 34): string {
    if (!value) return "n/a";
    if (value.num === 0n) return "0";
    const negative = value.num < 0n;
    const numerator = abs(value.num);
    const integer = numerator / value.den;
    const fraction = ((numerator % value.den) * pow10(digits)) / value.den;
    const fractionText = fraction.toString().padStart(digits, "0").replace(/0+$/, "");
    if (integer === 0n && fraction === 0n) return `${negative ? "-" : ""}<1e-${digits}`;
    return `${negative ? "-" : ""}${integer}${fractionText ? `.${fractionText}` : ""}`;
}

function errorCounts(attempts: Array<{ failure?: AttemptFailure }>): Record<ExecutionFailureKind, number> {
    const counts: Record<ExecutionFailureKind, number> = { revert: 0, "out-of-gas": 0, failure: 0 };
    for (const attempt of attempts) {
        if (attempt.failure) counts[attempt.failure.classification]++;
    }
    return counts;
}

function lowestGas<T extends { gas?: bigint }>(attempts: T[]): T | undefined {
    return attempts.reduce<T | undefined>((best, current) => {
        if (current.gas === undefined) return best;
        if (!best || best.gas === undefined || current.gas < best.gas) return current;
        return best;
    }, undefined);
}

function lowestError<T extends { error?: ErrorValue }>(attempts: T[]): T | undefined {
    return attempts.reduce<T | undefined>((best, current) => {
        if (!current.error) return best;
        if (!best?.error || current.error.num * best.error.den < best.error.num * current.error.den) return current;
        return best;
    }, undefined);
}

describe("Equal-accuracy gas benchmark: integration and root finding", function () {
    this.timeout(0);

    let integration: IntegrationHarness;
    let rootFinding: RootFindingHarness;
    let integrationTarget: string;
    let rootTarget: string;
    let writer: BenchmarkResultWriter;

    before(async function () {
        writer = await BenchmarkResultWriter.create({ suite: "equal-accuracy.integration-root-finding" });

        const localMathFactory = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
        const localMath = await localMathFactory.deploy();
        await localMath.waitForDeployment();

        const trigMathFactory = await ethers.getContractFactory(
            "@gorkemerdogan/trigonometry-master/contracts/libraries/MathLib.sol:MathLib",
        );
        const trigMath = await trigMathFactory.deploy();
        await trigMath.waitForDeployment();

        const integrationFactory = await ethers.getContractFactory("IntegrationHarness", {
            libraries: {
                "contracts/libraries/MathLib.sol:MathLib": await localMath.getAddress(),
                "@gorkemerdogan/trigonometry-master/contracts/libraries/MathLib.sol:MathLib":
                    await trigMath.getAddress(),
            },
        });
        integration = (await integrationFactory.deploy()) as unknown as IntegrationHarness;
        await integration.waitForDeployment();
        integrationTarget = await integration.getAddress();

        const rootFactory = await ethers.getContractFactory("RootFindingHarness", {
            libraries: { MathLib: await localMath.getAddress() },
        });
        rootFinding = (await rootFactory.deploy()) as unknown as RootFindingHarness;
        await rootFinding.waitForDeployment();
        rootTarget = await rootFinding.getAddress();

        console.log(`Execution model: ${formatBenchmarkExecution(HARNESS_ESTIMATE_CALL)}`);
        console.log("Accuracy model: exact binary128 decoding versus exact/high-precision rational host references");
        console.log("Gas includes callback STATICCALL execution; callback gas is not subtracted.");
    });

    after(async function () {
        await writer.flush();
    });

    it("reports minimum gas at target integration error", async function () {
        const zero = await integration.qFromInt(0n);
        const one = await integration.qFromInt(1n);
        const two = await integration.qFromInt(2n);
        const problems = [
            {
                name: "square_[0,1]",
                interval: "[0,1]",
                a: zero,
                b: one,
                selector: integration.interface.getFunction("f_square")!.selector,
                reference: { num: 1n, den: 3n },
                oracle: "analytic integral 1/3 (exact rational)",
            },
            {
                name: "inverse_[1,2]",
                interval: "[1,2]",
                selector: integration.interface.getFunction("f_inv")!.selector,
                a: one,
                b: two,
                reference: decimalToRational(
                    "0.693147180559945309417232121458176568075500134360255254120680009",
                ),
                oracle: "analytic integral ln(2), 63 significant decimal digits",
            },
        ];

        let records = 0;
        let successes = 0;
        const successfulPairs = new Set<string>();
        console.log("\nEqual-accuracy integration summary");
        for (const problem of problems) {
            for (const method of INTEGRATION_METHODS) {
                const attempts: IntegrationAttempt[] = [];
                for (const n of N_SWEEP.filter(method.validN)) {
                    try {
                        const callable = integration.getFunction(method.name);
                        const args = [integrationTarget, problem.selector, problem.a, problem.b, n] as const;
                        const gas = await callable.estimateGas(...args);
                        const result = await callable.staticCall(...args) as string;
                        attempts.push({ n, gas, error: absoluteError(binary128ToRational(result), problem.reference) });
                    } catch (error) {
                        attempts.push({
                            n,
                            failure: {
                                classification: classifyExecutionFailure(error),
                            },
                        });
                    }
                }

                for (const epsilonLabel of INTEGRATION_EPSILONS) {
                    const epsilon = decimalToRational(epsilonLabel);
                    const selected = lowestGas(attempts.filter(
                        attempt => attempt.error !== undefined && atMost(attempt.error, epsilon),
                    ));
                    const bestAccuracy = lowestError(attempts);
                    const reported = selected ?? bestAccuracy;
                    const reportedEvaluations = reported ? reported.n + 1n : undefined;
                    const counts = errorCounts(attempts);
                    const status = selected ? "success" : "failure";
                    records++;
                    if (selected) {
                        successes++;
                        successfulPairs.add(`${problem.name}/${method.name}`);
                    }

                    console.log(
                        `${problem.name.padEnd(18)} ${method.name.padEnd(11)} eps=${epsilonLabel.padEnd(5)} ` +
                        `status=${selected ? "success" : "no-match"} n=${reported?.n ?? "-"} ` +
                        `gas=${selected?.gas ?? "unreached"} bestAccuracyGas=${bestAccuracy?.gas ?? "-"} ` +
                        `error=${formatRational(reported?.error)} evals=${reportedEvaluations ?? "-"} ` +
                        `failures=revert:${counts.revert}/oog:${counts["out-of-gas"]}/other:${counts.failure}`,
                    );

                    writer.record({
                        benchmark: "minimum-gas-at-target-error",
                        category: "integration-equal-accuracy",
                        operation: method.name,
                        execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
                        input: {
                            problem: problem.name,
                            interval: problem.interval,
                            epsilon: epsilonLabel,
                            testedSubdivisions: attempts.map(attempt => attempt.n.toString()),
                            selectedSubdivisions: selected?.n.toString() ?? null,
                            bestTestedSubdivisions: bestAccuracy?.n.toString() ?? null,
                            bestTestedGas: bestAccuracy?.gas?.toString() ?? null,
                            bestTestedFunctionEvaluations: bestAccuracy ? (bestAccuracy.n + 1n).toString() : null,
                            selection: "minimum estimated gas among tested configurations satisfying absolute error <= epsilon",
                            oracle: problem.oracle,
                        },
                        callback: {
                            model: "target_staticcall",
                            functionEvaluations: selected ? (selected.n + 1n).toString() : undefined,
                            detail: "end-to-end IntegrationHarness estimate; callback gas included",
                            gasIncludesCallback: true,
                        },
                        gas: selected?.gas?.toString(),
                        status,
                        errorMetrics: {
                            absoluteError: reported ? formatRational(reported.error) : null,
                            absoluteErrorNumerator: reported?.error?.num.toString() ?? null,
                            absoluteErrorDenominator: reported?.error?.den.toString() ?? null,
                            relativeError: reported
                                ? formatRational(relativeError(reported.error!, problem.reference))
                                : null,
                            threshold: epsilonLabel,
                            comparison: "exact rational comparison after direct binary128 decoding",
                        },
                        convergence: {
                            classification: selected ? "accuracy-threshold-reached" : "accuracy-threshold-not-reached",
                            attempts: attempts.length,
                            executionFailures: counts.failure + counts.revert + counts["out-of-gas"],
                            reverted: counts.revert,
                            outOfGas: counts["out-of-gas"],
                            otherFailures: counts.failure,
                        },
                    });
                }
            }
        }

        expect(records).to.equal(problems.length * INTEGRATION_METHODS.length * INTEGRATION_EPSILONS.length);
        expect(successes, "bounded integration sweep should produce threshold-reaching configurations").to.be.greaterThan(0);
        expect(successfulPairs.size, "every integration problem/method pair should reach at least one bounded target")
            .to.equal(problems.length * INTEGRATION_METHODS.length);
    });

    it("reports minimum gas at target root error", async function () {
        const zero = await rootFinding.qFromInt(0n);
        const one = await rootFinding.qFromInt(1n);
        const two = await rootFinding.qFromInt(2n);
        const three = await rootFinding.qFromInt(3n);
        const problems = [
            {
                name: "x2_minus_4",
                fSelector: rootFinding.interface.getFunction("f_x2_minus_4")!.selector,
                dfSelector: rootFinding.interface.getFunction("df_2x")!.selector,
                reference: { num: 2n, den: 1n },
                oracle: "analytic root 2 (exact integer)",
                bisection: [zero, three] as const,
                newton: three,
                secant: [zero, three] as const,
            },
            {
                name: "x3_minus_x_minus_2",
                fSelector: rootFinding.interface.getFunction("f_cubic")!.selector,
                dfSelector: rootFinding.interface.getFunction("df_cubic")!.selector,
                reference: decimalToRational(
                    "1.521379706804567569604080832254438514428389828427903908901",
                ),
                oracle: "high-precision real root, 58 significant decimal digits",
                bisection: [one, two] as const,
                newton: await rootFinding.qFromFrac(3n, 2n),
                secant: [one, two] as const,
            },
        ];
        const sweep = [
            { label: "1e-4", tol: await rootFinding.qFromFrac(1n, 10n ** 4n), maxIter: 16n },
            { label: "1e-8", tol: await rootFinding.qFromFrac(1n, 10n ** 8n), maxIter: 32n },
            { label: "1e-12", tol: await rootFinding.qFromFrac(1n, 10n ** 12n), maxIter: 48n },
            { label: "1e-18", tol: await rootFinding.qFromFrac(1n, 10n ** 18n), maxIter: 80n },
            { label: "1e-24", tol: await rootFinding.qFromFrac(1n, 10n ** 24n), maxIter: 112n },
            { label: "1e-30", tol: await rootFinding.qFromFrac(1n, 10n ** 30n), maxIter: 160n },
        ];

        let records = 0;
        let successes = 0;
        console.log("\nEqual-accuracy root-finding summary");
        for (const problem of problems) {
            for (const method of ["bisection", "newton", "secant"] as RootMethod[]) {
                const attempts: RootAttempt[] = [];
                for (const configuration of sweep) {
                    try {
                        const callable = rootFinding.getFunction(`rootFinding${method[0].toUpperCase()}${method.slice(1)}`);
                        const common = [rootTarget, problem.fSelector];
                        const args = method === "bisection"
                            ? [...common, ...problem.bisection, configuration.tol, configuration.maxIter]
                            : method === "newton"
                                ? [...common, rootTarget, problem.dfSelector, problem.newton, configuration.tol, configuration.maxIter]
                                : [...common, ...problem.secant, configuration.tol, configuration.maxIter];
                        const gas = await callable.estimateGas(...args);
                        const result = await callable.staticCall(...args) as [string, bigint, boolean, string];
                        attempts.push({
                            toleranceLabel: configuration.label,
                            maxIter: configuration.maxIter,
                            gas,
                            iterations: result[1],
                            converged: result[2],
                            error: absoluteError(binary128ToRational(result[0]), problem.reference),
                            residual: absoluteError(binary128ToRational(result[3]), { num: 0n, den: 1n }),
                        });
                    } catch (error) {
                        attempts.push({
                            toleranceLabel: configuration.label,
                            maxIter: configuration.maxIter,
                            failure: {
                                classification: classifyExecutionFailure(error),
                            },
                        });
                    }
                }

                for (const epsilonLabel of ROOT_EPSILONS) {
                    const epsilon = decimalToRational(epsilonLabel);
                    const selected = lowestGas(attempts.filter(
                        attempt => attempt.converged === true && attempt.error !== undefined && atMost(attempt.error, epsilon),
                    ));
                    const bestAccuracy = lowestError(attempts.filter(attempt => attempt.converged === true));
                    const reported = selected ?? bestAccuracy;
                    const counts = errorCounts(attempts);
                    const initialization = method === "bisection"
                        ? "sign-changing bracket"
                        : method === "newton"
                            ? "one initial point"
                            : "two initial points";
                    const functionEvaluations = selected?.iterations === undefined
                        ? undefined
                        : method === "newton"
                            ? 1n + selected.iterations
                            : 2n + selected.iterations;
                    const derivativeEvaluations = method === "newton" ? selected?.iterations : 0n;
                    records++;
                    if (selected) successes++;

                    console.log(
                        `${problem.name.padEnd(20)} ${method.padEnd(9)} eps=${epsilonLabel.padEnd(5)} ` +
                        `status=${selected ? "success" : "no-match"} tol=${selected?.toleranceLabel ?? "-"} ` +
                        `iter=${selected?.iterations ?? "-"} gas=${selected?.gas ?? "-"} ` +
                        `error=${formatRational(reported?.error)} ` +
                        `fEvals=${functionEvaluations ?? "-"} dfEvals=${derivativeEvaluations ?? "-"} ` +
                        `failures=revert:${counts.revert}/oog:${counts["out-of-gas"]}/other:${counts.failure}`,
                    );

                    writer.record({
                        benchmark: "minimum-gas-at-target-root-error",
                        category: "root-finding-equal-accuracy",
                        operation: method,
                        execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
                        input: {
                            problem: problem.name,
                            epsilon: epsilonLabel,
                            testedConfigurations: sweep.map(item => `${item.label}/maxIter=${item.maxIter}`),
                            selectedTolerance: selected?.toleranceLabel ?? null,
                            selectedMaxIter: selected?.maxIter.toString() ?? null,
                            bestTestedTolerance: bestAccuracy?.toleranceLabel ?? null,
                            bestTestedMaxIter: bestAccuracy?.maxIter.toString() ?? null,
                            bestTestedGas: bestAccuracy?.gas?.toString() ?? null,
                            initializationPolicy: initialization,
                            fairnessNote: "same equation and root; method-specific initialization data are necessarily different",
                            selection: "minimum estimated gas among converged tested configurations satisfying root error <= epsilon",
                            oracle: problem.oracle,
                        },
                        callback: {
                            model: "target_staticcall",
                            functionEvaluations: functionEvaluations?.toString(),
                            detail: method === "newton"
                                ? `${derivativeEvaluations ?? 0n} derivative evaluations; end-to-end callback gas included`
                                : "function evaluations; end-to-end callback gas included",
                            gasIncludesCallback: true,
                        },
                        gas: selected?.gas?.toString(),
                        status: selected ? "success" : "failure",
                        errorMetrics: {
                            absoluteError: reported ? formatRational(reported.error) : null,
                            absoluteErrorNumerator: reported?.error?.num.toString() ?? null,
                            absoluteErrorDenominator: reported?.error?.den.toString() ?? null,
                            relativeError: reported
                                ? formatRational(relativeError(reported.error!, problem.reference))
                                : null,
                            residualMagnitude: reported ? formatRational(reported.residual) : null,
                            threshold: epsilonLabel,
                            comparison: "exact rational comparison after direct binary128 decoding",
                        },
                        convergence: {
                            classification: selected ? "converged-and-accurate" : "no-converged-configuration-met-error",
                            converged: selected?.converged ?? false,
                            iterations: selected?.iterations?.toString() ?? null,
                            functionEvaluations: functionEvaluations?.toString() ?? null,
                            derivativeEvaluations: derivativeEvaluations?.toString() ?? null,
                            attempts: attempts.length,
                            nonConvergedAttempts: attempts.filter(item => item.converged === false).length,
                            executionFailures: counts.failure + counts.revert + counts["out-of-gas"],
                            reverted: counts.revert,
                            outOfGas: counts["out-of-gas"],
                            otherFailures: counts.failure,
                        },
                    });
                }
            }
        }

        expect(records).to.equal(problems.length * 3 * ROOT_EPSILONS.length);
        expect(successes, "bounded root sweep should produce threshold-reaching configurations").to.be.greaterThan(0);
    });
});
