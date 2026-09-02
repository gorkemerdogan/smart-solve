import { expect } from "chai";
import { ethers } from "hardhat";
import {
    classifyExecutionFailure,
    formatBenchmarkExecution,
    HARNESS_ESTIMATE_CALL,
    type ExecutionFailureKind,
} from "./test-utils";
import { binary128ToScaledInt, printPrecisionMetadata } from "./precision-utils";

describe("DifferentiationHarness - Binary128-Aware Method-Error Tests (324 tests)", function () {
    let harness: any;
    let mathLib: any;

    /**
     * Decimal reporting scale applied after directly decoding bytes16.
     */
    const SCALE_DECIMALS = 30;
    const SCALE = 10n ** BigInt(SCALE_DECIMALS);

    // Conservative binary128 unit-roundoff proxy (1e-34) and operation safety factor.
    const BINARY128_ROUNDOFF_DEN = 10n ** 34n;
    const ROUNDING_SAFETY_FACTOR = 64n;

    /**
     * Six input points.
     */
    const xValues = [-10, -5, -1, 1, 5, 10];

    /**
     * Six step sizes represented as rational numbers.
     */
    const hValues = [
        { num: 1, den: 10 },        // 1e-1
        { num: 1, den: 100 },       // 1e-2
        { num: 1, den: 1000 },      // 1e-3
        { num: 1, den: 10000 },     // 1e-4
        { num: 1, den: 100000 },    // 1e-5
        { num: 1, den: 1000000 },   // 1e-6
    ];

    /**
     * Three differentiation methods.
     */
    const methods = [
        { name: "forwardDiff", fn: "forwardDiffHarness" },
        { name: "backwardDiff", fn: "backwardDiffHarness" },
        { name: "centeredDiff", fn: "centeredDiffHarness" },
    ];

    /**
     * Three benchmark functions.
     */
    const functions = [
        {
            name: "f_cube",
            displayName: "f(x)=x^3",
            exactDerivativeScaled: (x: number) => 3n * BigInt(x) * BigInt(x) * SCALE,
        },
        {
            name: "f_linear",
            displayName: "f(x)=3x-2",
            exactDerivativeScaled: (_x: number) => 3n * SCALE,
        },
        {
            name: "f_square",
            displayName: "f(x)=x^2",
            exactDerivativeScaled: (x: number) => 2n * BigInt(x) * SCALE,
        },
    ];

    type TestResult = {
        method: string;
        functionName: string;
        x: number;
        hNum: number;
        hDen: number;
        numericalScaled: bigint;
        exactScaled: bigint;
        absoluteErrorScaled: bigint;
        relativeErrorScaled: bigint;
        toleranceScaled: bigint;
        toleranceUsageScaled: bigint;
        estimatedGas: bigint;
        passed: boolean;
    };

    /**
     * Shared in-memory result store.
     */
    const allResults: TestResult[] = [];
    const executionFailures: {
        method: string;
        functionName: string;
        kind: ExecutionFailureKind;
        reason: string;
    }[] = [];

    before(async function () {
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        mathLib = await MathLibFactory.deploy();
        await mathLib.waitForDeployment();

        const HarnessFactory = await ethers.getContractFactory("DifferentiationHarness", {
            libraries: {
                MathLib: await mathLib.getAddress(),
            },
        });

        harness = await HarnessFactory.deploy();
        await harness.waitForDeployment();
    });

    async function qFromInt(x: number) {
        return await harness.qFromInt(x);
    }

    async function qFromFrac(num: number, den: number) {
        return await harness.qFromFrac(num, den);
    }

    async function toScaledInt(q: string): Promise<bigint> {
        return binary128ToScaledInt(q, SCALE_DECIMALS);
    }

    function absBigInt(x: bigint): bigint {
        return x < 0n ? -x : x;
    }

    function scaledRatio(numerator: bigint, denominator: bigint): bigint {
        if (denominator === 0n) {
            return 0n;
        }
        return (numerator * SCALE) / denominator;
    }

    function expectedErrorBoundScaled(args: {
        methodName: string;
        functionName: string;
        x: number;
        hNum: number;
        hDen: number;
    }): bigint {
        const hNum = BigInt(args.hNum);
        const hDen = BigInt(args.hDen);
        let errorNum = 0n;
        let errorDen = 1n;

        if (args.functionName === "f(x)=x^2" && args.methodName !== "centeredDiff") {
            errorNum = hNum;
            errorDen = hDen;
        } else if (args.functionName === "f(x)=x^3") {
            if (args.methodName === "centeredDiff") {
                errorNum = hNum * hNum;
                errorDen = hDen * hDen;
            } else if (args.methodName === "forwardDiff") {
                errorNum = 3n * BigInt(args.x) * hNum * hDen + hNum * hNum;
                errorDen = hDen * hDen;
            } else {
                errorNum = -3n * BigInt(args.x) * hNum * hDen + hNum * hNum;
                errorDen = hDen * hDen;
            }
        }

        const magnitude = absBigInt(errorNum);
        const scaledCeiling = (magnitude * SCALE + errorDen - 1n) / errorDen;

        // Finite differences subtract nearby function values and divide by h.
        // Bound the resulting cancellation amplification by C*u*max(1,|f|)/h,
        // with u approximated conservatively as 1e-34 for binary128.
        const xMagnitude = BigInt(Math.abs(args.x) + 1);
        const functionMagnitudeBound = xMagnitude ** 3n;
        const roundoffNum = ROUNDING_SAFETY_FACTOR * functionMagnitudeBound * hDen * SCALE;
        const roundoffDen = hNum * BINARY128_ROUNDOFF_DEN;
        const roundoffAllowance = (roundoffNum + roundoffDen - 1n) / roundoffDen;
        return scaledCeiling + roundoffAllowance;
    }

    function formatScaled(value: bigint, decimals: number = SCALE_DECIMALS): string {
        const negative = value < 0n;
        const absValue = negative ? -value : value;

        const integerPart = absValue / SCALE;
        const fractionalPart = absValue % SCALE;

        const fracStr = fractionalPart.toString().padStart(SCALE_DECIMALS, "0").slice(0, decimals);
        return `${negative ? "-" : ""}${integerPart.toString()}.${fracStr}`;
    }

    function summarize(results: TestResult[]) {
        const count = results.length;

        let withinToleranceCount = 0;
        let exceededToleranceCount = 0;
        let sumAbsError = 0n;
        let sumRelError = 0n;
        let sumTolUsage = 0n;
        let sumGas = 0n;
        let maxAbsError = 0n;
        let minGas = 0n;
        let maxGas = 0n;
        let first = true;

        for (const r of results) {
            if (r.passed) {
                withinToleranceCount++;
            } else {
                exceededToleranceCount++;
            }

            sumAbsError += r.absoluteErrorScaled;
            sumRelError += r.relativeErrorScaled;
            sumTolUsage += r.toleranceUsageScaled;
            sumGas += r.estimatedGas;

            if (r.absoluteErrorScaled > maxAbsError) {
                maxAbsError = r.absoluteErrorScaled;
            }

            if (first) {
                minGas = r.estimatedGas;
                maxGas = r.estimatedGas;
                first = false;
            } else {
                if (r.estimatedGas < minGas) minGas = r.estimatedGas;
                if (r.estimatedGas > maxGas) maxGas = r.estimatedGas;
            }
        }

        const avgAbsError = count > 0 ? sumAbsError / BigInt(count) : 0n;
        const avgRelError = count > 0 ? sumRelError / BigInt(count) : 0n;
        const avgTolUsage = count > 0 ? sumTolUsage / BigInt(count) : 0n;
        const avgGas = count > 0 ? sumGas / BigInt(count) : 0n;

        return {
            count,
            successfulCount: count,
            failedCount: exceededToleranceCount,
            withinToleranceCount,
            exceededToleranceCount,
            passRate: count > 0 ? withinToleranceCount / count : 0,
            avgAbsError,
            avgRelError,
            avgTolUsage,
            avgGas,
            minGas,
            maxGas,
            maxAbsError,
        };
    }

    function printSummary(title: string, results: TestResult[], failures = executionFailures) {
        const s = summarize(results);
        const total = s.count + failures.length;
        const reverted = failures.filter(f => f.kind === "revert").length;
        const outOfGas = failures.filter(f => f.kind === "out-of-gas").length;
        const otherFailures = failures.filter(f => f.kind === "failure").length;

        console.log("\n============================================================");
        console.log(title);
        console.log("============================================================");
        console.log(`Execution Model       : ${formatBenchmarkExecution(HARNESS_ESTIMATE_CALL)}`);
        console.log(`Total Tests           : ${total}`);
        console.log(`Successful Executions : ${s.successfulCount}`);
        console.log(`Failed Cases          : ${s.failedCount + failures.length}`);
        console.log(`Numerical Failures    : ${s.failedCount}`);
        console.log(`Reverted Cases        : ${reverted}`);
        console.log(`Out-of-Gas Cases      : ${outOfGas}`);
        console.log(`Other Failures        : ${otherFailures}`);
        console.log(`Pass Rate             : ${(total > 0 ? s.withinToleranceCount / total * 100 : 0).toFixed(2)}%`);
        console.log(`Within Tolerance      : ${s.withinToleranceCount}`);
        console.log(`Exceeded Tolerance    : ${s.exceededToleranceCount}`);
        console.log(`Average Abs. Error    : ${formatScaled(s.avgAbsError)} (successful executions only)`);
        console.log(`Average Rel. Error    : ${formatScaled(s.avgRelError)} (successful executions only)`);
        console.log(`Average Tol. Usage    : ${formatScaled(s.avgTolUsage)}`);
        console.log(`Max Abs. Error        : ${formatScaled(s.maxAbsError)}`);
        console.log("Tolerance             : exact polynomial method-error bound + C*u*max(1,|f|)/h cancellation allowance");
        printPrecisionMetadata({
            classification: "binary128-aware comparison",
            comparisonScale: "1e30 (direct bytes16 decoding; truncation toward zero)",
            oraclePrecision: "exact integer/rational polynomial derivative and truncation formulas",
            conversion: "binary128 bytes16 -> exact BigInt rational -> 1e30 integer",
            claim: "30-decimal reporting of method error; not zero-error differentiation",
        });
        console.log(`Average Gas           : ${s.avgGas.toString()} (successful executions only)`);
        console.log(`Min Gas               : ${s.minGas.toString()}`);
        console.log(`Max Gas               : ${s.maxGas.toString()}`);
    }

    function printCaseResult(result: TestResult) {
        console.log("\n------------------------------------------------------------");
        console.log(`Method             : ${result.method}`);
        console.log(`Function           : ${result.functionName}`);
        console.log(`Input              : x=${result.x}, h=${result.hNum}/${result.hDen}`);
        console.log(`Expected Output    : ${formatScaled(result.exactScaled)}`);
        console.log(`Actual Output      : ${formatScaled(result.numericalScaled)}`);
        console.log(`Absolute Error     : ${formatScaled(result.absoluteErrorScaled)}`);
        console.log(`Relative Error     : ${formatScaled(result.relativeErrorScaled)}`);
        console.log(`Tolerance          : ${formatScaled(result.toleranceScaled)}`);
        console.log(`Tolerance Usage    : ${formatScaled(result.toleranceUsageScaled)}`);
        console.log(`Estimated Gas      : ${result.estimatedGas.toString()}`);
        console.log(`Status             : ${result.passed ? "PASS" : "FAIL"}`);
        console.log("------------------------------------------------------------");
    }

    async function runCase(params: {
        methodName: string;
        methodFn: string;
        targetSelector: string;
        functionDisplayName: string;
        x: number;
        hNum: number;
        hDen: number;
        exactScaled: bigint;
    }) {
        const xQuad = await qFromInt(params.x);
        const hQuad = await qFromFrac(params.hNum, params.hDen);

        let estimatedGas: bigint;
        let numericalQuad: string;
        try {
            estimatedGas = BigInt(
                (
                    await harness[params.methodFn].estimateGas(
                        await harness.getAddress(),
                        params.targetSelector,
                        xQuad,
                        hQuad
                    )
                ).toString()
            );

            numericalQuad = await harness[params.methodFn](
                await harness.getAddress(),
                params.targetSelector,
                xQuad,
                hQuad
            );
        } catch (error) {
            const kind = classifyExecutionFailure(error);
            const reason = error instanceof Error ? error.message : String(error);
            executionFailures.push({
                method: params.methodName,
                functionName: params.functionDisplayName,
                kind,
                reason,
            });
            expect.fail(
                `${params.methodName} ${kind} for ${params.functionDisplayName} at ` +
                `x=${params.x}, h=${params.hNum}/${params.hDen}: ${reason}`
            );
        }

        const numericalScaled = await toScaledInt(numericalQuad);
        const absoluteErrorScaled = absBigInt(numericalScaled - params.exactScaled);

        const exactAbsScaled = absBigInt(params.exactScaled);
        const relativeErrorScaled = scaledRatio(absoluteErrorScaled, exactAbsScaled);
        const toleranceScaled = expectedErrorBoundScaled({
            methodName: params.methodName,
            functionName: params.functionDisplayName,
            x: params.x,
            hNum: params.hNum,
            hDen: params.hDen,
        });
        const toleranceUsageScaled = scaledRatio(absoluteErrorScaled, toleranceScaled);
        const passed = absoluteErrorScaled <= toleranceScaled;

        const result: TestResult = {
            method: params.methodName,
            functionName: params.functionDisplayName,
            x: params.x,
            hNum: params.hNum,
            hDen: params.hDen,
            numericalScaled,
            exactScaled: params.exactScaled,
            absoluteErrorScaled,
            relativeErrorScaled,
            toleranceScaled,
            toleranceUsageScaled,
            estimatedGas,
            passed,
        };

        allResults.push(result);
        printCaseResult(result);

        expect(
            passed,
            `${params.methodName} exceeded absolute-error tolerance for ${params.functionDisplayName} ` +
            `at x=${params.x}, h=${params.hNum}/${params.hDen}: ` +
            `error=${formatScaled(absoluteErrorScaled)}, tolerance=${formatScaled(toleranceScaled)}`
        ).to.equal(true);
    }

    describe("Sanity checks", function () {
        it("should define 108 test cases for f(x)=x^3", async function () {
            const total = xValues.length * hValues.length * methods.length;
            expect(total).to.equal(108);
        });

        it("should define 108 test cases for f(x)=x^2", async function () {
            const total = xValues.length * hValues.length * methods.length;
            expect(total).to.equal(108);
        });

        it("should define 108 test cases for f(x)=3x-2", async function () {
            const total = xValues.length * hValues.length * methods.length;
            expect(total).to.equal(108);
        });

        it("should define 324 total accuracy tests", async function () {
            const total = functions.length * xValues.length * hValues.length * methods.length;
            expect(total).to.equal(324);
        });
    });

    /**
     * 3 functions × 3 methods × 6 input points × 6 step sizes
     */
    for (const fnObj of functions) {
        describe(`Accuracy tests for ${fnObj.displayName}`, function () {
            for (const method of methods) {
                for (const x of xValues) {
                    for (const h of hValues) {
                        it(`${method.name} | ${fnObj.displayName} | x=${x} | h=${h.num}/${h.den}`, async function () {
                            const selector = harness.interface.getFunction(fnObj.name)!.selector;
                            await runCase({
                                methodName: method.name,
                                methodFn: method.fn,
                                targetSelector: selector,
                                functionDisplayName: fnObj.displayName,
                                x,
                                hNum: h.num,
                                hDen: h.den,
                                exactScaled: fnObj.exactDerivativeScaled(x),
                            });
                        });
                    }
                }
            }
        });
    }

    describe("Summary reports", function () {
        it("should print overall summary", async function () {
            expect(allResults.length + executionFailures.length).to.equal(324);
            printSummary("OVERALL SUMMARY", allResults);
        });

        for (const fnObj of functions) {
            it(`should print summary for ${fnObj.displayName}`, async function () {
                const filtered = allResults.filter((r) => r.functionName === fnObj.displayName);
                const failures = executionFailures.filter(f => f.functionName === fnObj.displayName);
                expect(filtered.length + failures.length).to.equal(108);

                printSummary(`SUMMARY - ${fnObj.displayName}`, filtered, failures);
            });
        }

        for (const method of methods) {
            it(`should print summary for ${method.name}`, async function () {
                const filtered = allResults.filter((r) => r.method === method.name);
                const failures = executionFailures.filter(f => f.method === method.name);
                expect(filtered.length + failures.length).to.equal(108);

                printSummary(`SUMMARY - ${method.name}`, filtered, failures);
            });
        }

        for (const fnObj of functions) {
            for (const method of methods) {
                it(`should print summary for ${method.name} on ${fnObj.displayName}`, async function () {
                    const filtered = allResults.filter(
                        (r) => r.method === method.name && r.functionName === fnObj.displayName
                    );
                    const failures = executionFailures.filter(
                        f => f.method === method.name && f.functionName === fnObj.displayName
                    );
                    expect(filtered.length + failures.length).to.equal(36);

                    printSummary(`SUMMARY - ${method.name} | ${fnObj.displayName}`, filtered, failures);
                });
            }
        }
    });
});
