// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { classifyExecutionFailure, printBlockRegular } from "./test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type ODESolverHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;
    fromFloat(n: bigint): Promise<string>;

    eulerIter(
        target: string,
        selector: string,
        x: string,
        y: string,
        h: string,
        steps: bigint
    ): Promise<string>;

    rk2MidpointIter(
        target: string,
        selector: string,
        x: string,
        y: string,
        h: string,
        steps: bigint
    ): Promise<string>;

    rk2HeunIter(
        target: string,
        selector: string,
        x: string,
        y: string,
        h: string,
        steps: bigint
    ): Promise<string>;

    rk4Iter(
        target: string,
        selector: string,
        x: string,
        y: string,
        h: string,
        steps: bigint
    ): Promise<string>;
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

let SCALE_DECIMALS = 12n;
let SCALE = 10n ** SCALE_DECIMALS;

type ODEMethod = "eulerIter" | "rk2MidpointIter" | "rk2HeunIter" | "rk4Iter";
type MethodLabel = "Euler" | "RK2 Midpoint" | "RK2 Heun" | "RK4";
type BenchmarkKey = "const5" | "linear" | "square" | "cubicPoly";

interface BenchmarkDef {
    key: BenchmarkKey;
    label: string;
    selector: string;
    exactSolution: (y0: number, xFinal: number) => number;
    rhs: (x: number, y: number) => number;
}

interface TestCase {
    caseNo: number;
    y0: number;
    hNum: number;
    hDen: number;
    hLabel: string;
    steps: number;
    xFinal: number;
}

interface StepCase {
    hNum: number;
    hDen: number;
    hLabel: string;
    steps: number;
}

interface MethodStats {
    count: number;
    failed: number;
    reverted: number;
    outOfGas: number;
    otherFailures: number;
    skipped: number;
    withinTol: number;
    exceededTol: number;
    absSum: bigint;
    relSumScaled: bigint;
    maxAbs: bigint;
    minAbs: bigint;
    maxRelScaled: bigint;
    minRelScaled: bigint;
    gasSum: bigint;
    minGas: bigint;
    maxGas: bigint;
}

interface RunResult {
    output: string;
    estimatedGas: bigint;
}

function asBigInt(v: unknown): bigint {
    if (typeof v === "bigint") return v;
    if (typeof v === "number") return BigInt(v);
    if (typeof v === "string") return BigInt(v);

    if (v && typeof v === "object") {
        const maybeToString = (v as { toString?: () => string }).toString;
        if (typeof maybeToString === "function") {
            return BigInt(maybeToString.call(v));
        }
    }

    throw new Error(`Cannot convert value to bigint: ${String(v)}`);
}

function inferScaleDecimals(scale: bigint): bigint {
    const s = scale.toString();
    if (!/^10*$/.test(s) || s[0] !== "1") {
        return SCALE_DECIMALS;
    }

    return BigInt(s.length - 1);
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");

    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function formatPercentScaled(v: bigint): string {
    return formatScaledInt(v * 100n);
}

async function outScaled(harness: ODESolverHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

function scaledAbsError(actual: bigint, expected: bigint): bigint {
    return absBigInt(actual - expected);
}

function scaledRelErrorScaled(actual: bigint, expected: bigint): bigint {
    const num = absBigInt(actual - expected);
    const den = absBigInt(expected);

    if (den === 0n) {
        return 0n;
    }

    return (num * SCALE) / den;
}

function decimalStringToScaledBigInt(value: string, scaleDecimals: number): bigint {
    const trimmed = value.trim();
    const match = trimmed.match(/^([+-])?(\d+(?:\.\d+)?)(?:[eE]([+-]?\d+))?$/);

    if (!match) {
        throw new Error(`Invalid decimal/scientific string: ${value}`);
    }

    const sign = match[1] === "-" ? -1n : 1n;
    const mantissa = match[2];
    const exponent = match[3] ? parseInt(match[3], 10) : 0;

    const [intPartRaw, fracPartRaw = ""] = mantissa.split(".");
    const digits = (intPartRaw + fracPartRaw).replace(/^0+/, "") || "0";
    const decimalPlaces = fracPartRaw.length;

    const netExponent = exponent - decimalPlaces + scaleDecimals;

    let scaledAbs: bigint;

    if (netExponent >= 0) {
        scaledAbs = BigInt(digits) * 10n ** BigInt(netExponent);
    } else {
        scaledAbs = BigInt(digits) / 10n ** BigInt(-netExponent);
    }

    return sign * scaledAbs;
}

function numberToScaledBigInt(value: number): bigint {
    if (!Number.isFinite(value)) {
        throw new Error(`Non-finite numeric value encountered: ${value}`);
    }

    return decimalStringToScaledBigInt(value.toExponential(24), Number(SCALE_DECIMALS));
}

function referenceStep(method: MethodLabel, rhs: BenchmarkDef["rhs"], x: number, y: number, h: number): number {
    const k1 = rhs(x, y);
    if (method === "Euler") return y + h * k1;

    if (method === "RK2 Midpoint") {
        return y + h * rhs(x + h / 2, y + h * k1 / 2);
    }

    if (method === "RK2 Heun") {
        const k2 = rhs(x + h, y + h * k1);
        return y + h * (k1 + k2) / 2;
    }

    const k2 = rhs(x + h / 2, y + h * k1 / 2);
    const k3 = rhs(x + h / 2, y + h * k2 / 2);
    const k4 = rhs(x + h, y + h * k3);
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

function methodAwareToleranceScaled(
    method: MethodLabel,
    benchmark: BenchmarkDef,
    tc: TestCase,
    exactValue: number
): bigint {
    const h = tc.hNum / tc.hDen;
    let x = 0;
    let y = tc.y0;
    for (let i = 0; i < tc.steps; i++) {
        y = referenceStep(method, benchmark.rhs, x, y, h);
        x += h;
    }

    // The reference method captures the method's expected truncation error.
    // The additional 1e-9 relative/absolute allowance covers conversion from
    // IEEE-754 reference values into the suite's 1e12 reporting scale.
    const discretizationError = Math.abs(y - exactValue);
    const conversionAllowance = 1e-9 * Math.max(1, Math.abs(exactValue));
    return numberToScaledBigInt(discretizationError + conversionAllowance);
}

function initStats(): MethodStats {
    return {
        count: 0,
        failed: 0,
        reverted: 0,
        outOfGas: 0,
        otherFailures: 0,
        skipped: 0,
        withinTol: 0,
        exceededTol: 0,
        absSum: 0n,
        relSumScaled: 0n,
        maxAbs: 0n,
        minAbs: -1n,
        maxRelScaled: 0n,
        minRelScaled: -1n,
        gasSum: 0n,
        minGas: -1n,
        maxGas: 0n,
    };
}

function updateStats(
    stats: MethodStats,
    absErr: bigint,
    relErrScaled: bigint,
    estimatedGas: bigint,
    passed: boolean
): void {
    stats.count += 1;
    stats.absSum += absErr;
    stats.relSumScaled += relErrScaled;
    stats.gasSum += estimatedGas;

    if (passed) {
        stats.withinTol += 1;
    } else {
        stats.exceededTol += 1;
    }

    if (stats.minAbs === -1n || absErr < stats.minAbs) stats.minAbs = absErr;
    if (absErr > stats.maxAbs) stats.maxAbs = absErr;

    if (stats.minRelScaled === -1n || relErrScaled < stats.minRelScaled) {
        stats.minRelScaled = relErrScaled;
    }

    if (relErrScaled > stats.maxRelScaled) {
        stats.maxRelScaled = relErrScaled;
    }

    if (stats.minGas === -1n || estimatedGas < stats.minGas) stats.minGas = estimatedGas;
    if (estimatedGas > stats.maxGas) stats.maxGas = estimatedGas;
}

function averageScaled(sum: bigint, count: number): bigint {
    return count === 0 ? 0n : sum / BigInt(count);
}

function errorToString(err: unknown): string {
    if (err instanceof Error) return err.message;
    return String(err);
}

function safeErrorToString(err: unknown): string {
    if (err instanceof Error) return err.message;
    return String(err);
}

function shouldSkipCase(benchmark: BenchmarkDef, tc: TestCase): boolean {
    /*
     * The exponential benchmark y' = y grows very rapidly.
     * Large x_final values produce huge expected values, so absolute-error
     * statistics become dominated by exponential growth rather than by the
     * numerical behavior of the ODE method.
     */
    return benchmark.key === "linear" && tc.xFinal > 15;
}

async function runMethod(
    harness: ODESolverHarness,
    method: ODEMethod,
    target: string,
    selector: string,
    x0: string,
    y0: string,
    h: string,
    steps: number
): Promise<RunResult> {
    const fn = (harness as any).getFunction(method);

    const estimatedGas = asBigInt(
        await fn.estimateGas(
            target,
            selector,
            x0,
            y0,
            h,
            BigInt(steps)
        )
    );

    const output = await fn.staticCall(
        target,
        selector,
        x0,
        y0,
        h,
        BigInt(steps),
        { gasLimit: 30_000_000n }
    );

    return { output, estimatedGas };
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("ODESolver Library - Multi-Case Accuracy Tests", function () {
    this.timeout(0);

    let harness: ODESolverHarness;
    let target: string;

    let selConst5: string;
    let selLinear: string;
    let selSquare: string;
    let selCubicPoly: string;

    const METHODS: Array<{ key: ODEMethod; label: MethodLabel }> = [
        { key: "eulerIter", label: "Euler" },
        { key: "rk2MidpointIter", label: "RK2 Midpoint" },
        { key: "rk2HeunIter", label: "RK2 Heun" },
        { key: "rk4Iter", label: "RK4" },
    ];

    const Y0_VALUES = [0, 1, 2];

    const ACCURACY_STEP_CASES: StepCase[] = [
        { hNum: 1, hDen: 10, hLabel: "0.1", steps: 10 },
        { hNum: 1, hDen: 10, hLabel: "0.1", steps: 20 },
        { hNum: 1, hDen: 20, hLabel: "0.05", steps: 20 },
        { hNum: 1, hDen: 20, hLabel: "0.05", steps: 40 },
        { hNum: 1, hDen: 100, hLabel: "0.01", steps: 100 },
        { hNum: 1, hDen: 100, hLabel: "0.01", steps: 200 },
    ];

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );

        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HarnessFactory = await ethers.getContractFactory("ODESolverHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HarnessFactory.deploy()) as unknown as ODESolverHarness;
        await harness.waitForDeployment();

        target = await harness.getAddress();

        selConst5 = harness.interface.getFunction("f_const5")!.selector;
        selLinear = harness.interface.getFunction("f_linear")!.selector;
        selSquare = harness.interface.getFunction("f_square")!.selector;
        selCubicPoly = harness.interface.getFunction("f_cubic_poly")!.selector;

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1n)));

        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    it("evaluates bounded ODE accuracy at paired step refinements", async function () {
        const x0 = await qInt(0n);

        const benchmarks: BenchmarkDef[] = [
            {
                key: "const5",
                label: "y' = 5",
                selector: selConst5,
                exactSolution: (y0: number, xFinal: number) => y0 + 5 * xFinal,
                rhs: () => 5,
            },
            {
                key: "linear",
                label: "y' = y",
                selector: selLinear,
                exactSolution: (y0: number, xFinal: number) => y0 * Math.exp(xFinal),
                rhs: (_x, y) => y,
            },
            {
                key: "square",
                label: "y' = x^2",
                selector: selSquare,
                exactSolution: (y0: number, xFinal: number) => y0 + (xFinal ** 3) / 3,
                rhs: (x) => x ** 2,
            },
            {
                key: "cubicPoly",
                label: "y' = x^3 + x^2 + 2x + 3",
                selector: selCubicPoly,
                exactSolution: (y0: number, xFinal: number) =>
                    y0 + (xFinal ** 4) / 4 + (xFinal ** 3) / 3 + xFinal ** 2 + 3 * xFinal,
                rhs: (x) => x ** 3 + x ** 2 + 2 * x + 3,
            },
        ];

        const testCases: TestCase[] = [];
        let caseNo = 0;

        for (const y0 of Y0_VALUES) {
            for (const sc of ACCURACY_STEP_CASES) {
                caseNo += 1;

                testCases.push({
                    caseNo,
                    y0,
                    hNum: sc.hNum,
                    hDen: sc.hDen,
                    hLabel: sc.hLabel,
                    steps: sc.steps,
                    xFinal: (sc.hNum / sc.hDen) * sc.steps,
                });
            }
        }

        const overallStatsByMethod = new Map<MethodLabel, MethodStats>();

        for (const method of METHODS) {
            overallStatsByMethod.set(method.label, initStats());
        }

        let overallAttempts = 0;
        let overallSkipped = 0;

        for (const benchmark of benchmarks) {
            const statsByMethod = new Map<MethodLabel, MethodStats>();

            for (const method of METHODS) {
                statsByMethod.set(method.label, initStats());
            }

            console.log("");
            console.log("############################################################");
            console.log(`Benchmark: ${benchmark.label}`);
            console.log(`Systematic cases: ${testCases.length}`);
            console.log("############################################################");

            for (const tc of testCases) {
                if (shouldSkipCase(benchmark, tc)) {
                    overallSkipped += METHODS.length;

                    for (const method of METHODS) {
                        statsByMethod.get(method.label)!.skipped += 1;
                        overallStatsByMethod.get(method.label)!.skipped += 1;
                    }

                    console.log("------------------------------------------------------------");
                    console.log(`Test: ${benchmark.key}-${tc.caseNo}`);
                    console.log("Method: ALL");
                    console.log(
                        `Explanation: SKIPPED | Benchmark=${benchmark.label}, y0=${tc.y0}, h=${tc.hLabel}, steps=${tc.steps}, x_final=${tc.xFinal}`
                    );
                    console.log("Reason: exponential benchmark skipped for x_final > 15");
                    console.log("------------------------------------------------------------");

                    continue;
                }

                const y0Q = await qInt(tc.y0);
                const hQ = await qFrac(tc.hNum, tc.hDen);

                const expectedValue = benchmark.exactSolution(tc.y0, tc.xFinal);
                const expectedScaled = numberToScaledBigInt(expectedValue);

                let expectedHex = "N/A";
                let expectedHexStatus = "OK";

                try {
                    expectedHex = await harness.fromFloat(expectedScaled);
                } catch (err) {
                    expectedHexStatus = `OUT_OF_RANGE: ${safeErrorToString(err)}`;
                    expectedHex = "OUT_OF_RANGE";
                }

                for (const method of METHODS) {
                    overallAttempts += 1;

                    try {
                        const run = await runMethod(
                            harness,
                            method.key,
                            target,
                            benchmark.selector,
                            x0,
                            y0Q,
                            hQ,
                            tc.steps
                        );

                        const actualScaled = await outScaled(harness, run.output);
                        const absErr = scaledAbsError(actualScaled, expectedScaled);
                        const relErrScaled = scaledRelErrorScaled(actualScaled, expectedScaled);
                        const tolerance = methodAwareToleranceScaled(method.label, benchmark, tc, expectedValue);
                        const passed = absErr <= tolerance;

                        updateStats(
                            statsByMethod.get(method.label)!,
                            absErr,
                            relErrScaled,
                            run.estimatedGas,
                            passed
                        );

                        updateStats(
                            overallStatsByMethod.get(method.label)!,
                            absErr,
                            relErrScaled,
                            run.estimatedGas,
                            passed
                        );

                        printBlockRegular({
                            t: `${benchmark.key}-${tc.caseNo}`,
                            method: method.label,
                            explanation:
                                `SUCCESS | Benchmark=${benchmark.label}, x0=0, y0=${tc.y0}, ` +
                                `h=${tc.hLabel}, steps=${tc.steps}, x_final=${tc.xFinal}, ` +
                                `expectedHexStatus=${expectedHexStatus}`,
                            gas: run.estimatedGas.toString(),
                            inHex: `x0=${x0} | y0=${y0Q} | h=${hQ} | steps=${tc.steps}`,
                            expectedHex,
                            outHex: run.output,
                            expectedDec: formatScaledInt(expectedScaled),
                            outDec: formatScaledInt(actualScaled),
                            maxAbsError: formatScaledInt(absErr),
                            avgAbsError: formatScaledInt(absErr),
                            residual: formatScaledInt(absErr),
                            normalizedResidual: formatScaledInt(relErrScaled),
                            tolerance: `abs<=${formatScaledInt(tolerance)} (reference ${method.label} discretization error + 1e-9 conversion allowance)`,
                            withinTol: passed ? "Yes" : "No",
                            exceededTol: passed ? "No" : "Yes",
                            worstCase:
                                `Case ${tc.caseNo} | AbsErr=${formatScaledInt(absErr)} | ` +
                                `RelErr=${formatPercentScaled(relErrScaled)}%`,
                        });
                    } catch (err) {
                        const message = errorToString(err);
                        const failureKind = classifyExecutionFailure(err);

                        statsByMethod.get(method.label)!.failed += 1;
                        overallStatsByMethod.get(method.label)!.failed += 1;
                        if (failureKind === "revert") {
                            statsByMethod.get(method.label)!.reverted += 1;
                            overallStatsByMethod.get(method.label)!.reverted += 1;
                        } else if (failureKind === "out-of-gas") {
                            statsByMethod.get(method.label)!.outOfGas += 1;
                            overallStatsByMethod.get(method.label)!.outOfGas += 1;
                        } else {
                            statsByMethod.get(method.label)!.otherFailures += 1;
                            overallStatsByMethod.get(method.label)!.otherFailures += 1;
                        }

                        printBlockRegular({
                            t: `${benchmark.key}-${tc.caseNo}`,
                            method: method.label,
                            explanation:
                                `FAILED (${failureKind}) | Benchmark=${benchmark.label}, x0=0, y0=${tc.y0}, ` +
                                `h=${tc.hLabel}, steps=${tc.steps}, x_final=${tc.xFinal}, ` +
                                `expectedHexStatus=${expectedHexStatus}`,
                            gas: "FAILED",
                            inHex: `x0=${x0} | y0=${y0Q} | h=${hQ} | steps=${tc.steps}`,
                            expectedHex,
                            outHex: "REVERTED / OUT_OF_GAS",
                            expectedDec: formatScaledInt(expectedScaled),
                            outDec: "FAILED",
                            maxAbsError: "FAILED",
                            avgAbsError: "FAILED",
                            residual: "FAILED",
                            normalizedResidual: "FAILED",
                            tolerance: "method-aware reference tolerance",
                            withinTol: "FAILED",
                            exceededTol: "FAILED",
                            worstCase: message,
                        });
                    }
                }
            }

            console.log("");
            console.log("******************** BENCHMARK SUMMARY *********************");

            for (const method of METHODS) {
                const stats = statsByMethod.get(method.label)!;
                const avgAbs = averageScaled(stats.absSum, stats.count);
                const avgRel = averageScaled(stats.relSumScaled, stats.count);
                const avgGas = averageScaled(stats.gasSum, stats.count);
                const totalCases = stats.count + stats.failed + stats.skipped;
                const passRate = totalCases === 0 ? 0 : (stats.withinTol * 100) / totalCases;

                printBlockRegular({
                    t: `${benchmark.key}-summary`,
                    method: method.label,
                    explanation:
                        `Summary for benchmark ${benchmark.label} | total=${totalCases} | successful executions=${stats.count} | ` +
                        `failed=${stats.failed} (revert=${stats.reverted}, outOfGas=${stats.outOfGas}, other=${stats.otherFailures}) | ` +
                        `skipped=${stats.skipped} | passRate=${passRate.toFixed(2)}% | averages=successful executions only`,
                    gas: stats.count > 0 ? avgGas.toString() : "N/A",
                    inHex: "-",
                    expectedHex: "-",
                    outHex: "-",
                    expectedDec: "-",
                    outDec: "-",
                    maxAbsError: stats.count > 0 ? formatScaledInt(stats.maxAbs) : "N/A",
                    avgAbsError: stats.count > 0 ? formatScaledInt(avgAbs) : "N/A",
                    residual: stats.count > 0 ? formatScaledInt(stats.maxAbs) : "N/A",
                    normalizedResidual: stats.count > 0 ? formatScaledInt(avgRel) : "N/A",
                    tolerance: "reference method discretization error + 1e-9 conversion allowance",
                    withinTol: stats.withinTol.toString(),
                    exceededTol: stats.exceededTol.toString(),
                    worstCase:
                        stats.count > 0
                            ? `success=${stats.count}, failed=${stats.failed}, skipped=${stats.skipped}, ` +
                            `minAbs=${formatScaledInt(stats.minAbs)}, ` +
                            `maxRel=${formatPercentScaled(stats.maxRelScaled)}%, ` +
                            `minGas=${stats.minGas.toString()}, maxGas=${stats.maxGas.toString()}`
                            : `success=0, failed=${stats.failed}, skipped=${stats.skipped}`,
                });
            }

            const eulerStats = statsByMethod.get("Euler")!;
            const rk4Stats = statsByMethod.get("RK4")!;

            if (eulerStats.count > 0 && rk4Stats.count > 0) {
                const eulerAvgAbs = averageScaled(eulerStats.absSum, eulerStats.count);
                const rk4AvgAbs = averageScaled(rk4Stats.absSum, rk4Stats.count);

                expect(rk4AvgAbs <= eulerAvgAbs).to.equal(true);
            }
        }

        console.log("");
        console.log("********************** OVERALL SUMMARY *********************");

        for (const method of METHODS) {
            const stats = overallStatsByMethod.get(method.label)!;
            const avgAbs = averageScaled(stats.absSum, stats.count);
            const avgRel = averageScaled(stats.relSumScaled, stats.count);
            const avgGas = averageScaled(stats.gasSum, stats.count);
            const totalCases = stats.count + stats.failed + stats.skipped;
            const passRate = totalCases === 0 ? 0 : (stats.withinTol * 100) / totalCases;

            printBlockRegular({
                t: "overall-summary",
                method: method.label,
                explanation:
                    `Overall summary across all benchmarks | total=${totalCases} | successful executions=${stats.count} | ` +
                    `failed=${stats.failed} (revert=${stats.reverted}, outOfGas=${stats.outOfGas}, other=${stats.otherFailures}) | ` +
                    `skipped=${stats.skipped} | passRate=${passRate.toFixed(2)}% | averages=successful executions only`,
                gas: stats.count > 0 ? avgGas.toString() : "N/A",
                inHex: "-",
                expectedHex: "-",
                outHex: "-",
                expectedDec: "-",
                outDec: "-",
                maxAbsError: stats.count > 0 ? formatScaledInt(stats.maxAbs) : "N/A",
                avgAbsError: stats.count > 0 ? formatScaledInt(avgAbs) : "N/A",
                residual: stats.count > 0 ? formatScaledInt(stats.maxAbs) : "N/A",
                normalizedResidual: stats.count > 0 ? formatScaledInt(avgRel) : "N/A",
                tolerance: "reference method discretization error + 1e-9 conversion allowance",
                withinTol: stats.withinTol.toString(),
                exceededTol: stats.exceededTol.toString(),
                worstCase:
                    stats.count > 0
                        ? `success=${stats.count}, failed=${stats.failed}, skipped=${stats.skipped}, ` +
                        `minAbs=${formatScaledInt(stats.minAbs)}, ` +
                        `maxRel=${formatPercentScaled(stats.maxRelScaled)}%, ` +
                        `minGas=${stats.minGas.toString()}, maxGas=${stats.maxGas.toString()}`
                        : `success=0, failed=${stats.failed}, skipped=${stats.skipped}`,
            });
        }

        expect(overallAttempts + overallSkipped).to.equal(
            testCases.length * benchmarks.length * METHODS.length
        );

        for (const method of METHODS) {
            const stats = overallStatsByMethod.get(method.label)!;
            expect(
                stats.failed,
                `${method.label} had unexpected execution failures: revert=${stats.reverted}, outOfGas=${stats.outOfGas}, other=${stats.otherFailures}`
            ).to.equal(0);
            expect(
                stats.exceededTol,
                `${method.label} exceeded its documented hybrid accuracy threshold in ${stats.exceededTol} executed cases`
            ).to.equal(0);
        }
    });

    it("reports ODE iteration-count feasibility limits separately from accuracy", async function () {
        const x0 = await qInt(0n);
        const y0 = await qInt(0n);
        const h = await qFrac(1, 10);
        const stepBudgets = [600, 800, 1000, 1500];
        let successful = 0;
        let reverted = 0;
        let outOfGas = 0;
        let otherFailures = 0;

        for (const steps of stepBudgets) {
            for (const method of METHODS) {
                try {
                    const run = await runMethod(
                        harness,
                        method.key,
                        target,
                        selConst5,
                        x0,
                        y0,
                        h,
                        steps
                    );
                    successful++;
                    console.log(`ODE FEASIBILITY | method=${method.label} | steps=${steps} | status=success | estimatedGas=${run.estimatedGas}`);
                } catch (error) {
                    const kind = classifyExecutionFailure(error);
                    if (kind === "revert") reverted++;
                    else if (kind === "out-of-gas") outOfGas++;
                    else otherFailures++;
                    console.log(`ODE FEASIBILITY | method=${method.label} | steps=${steps} | status=${kind}`);
                }
            }
        }

        const total = stepBudgets.length * METHODS.length;
        const failed = reverted + outOfGas + otherFailures;
        console.log(`ODE FEASIBILITY SUMMARY | total=${total} | successful=${successful} | failed=${failed} | reverted=${reverted} | outOfGas=${outOfGas} | other=${otherFailures} | successRate=${((successful / total) * 100).toFixed(2)}%`);

        expect(successful + failed).to.equal(total);
        expect(successful, "at least one high-step ODE case should remain feasible").to.be.greaterThan(0);
    });
});
