// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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

function initStats(): MethodStats {
    return {
        count: 0,
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
    tol: bigint
): void {
    stats.count += 1;
    stats.absSum += absErr;
    stats.relSumScaled += relErrScaled;
    stats.gasSum += estimatedGas;

    if (absErr <= tol) stats.withinTol += 1;
    else stats.exceededTol += 1;

    if (stats.minAbs === -1n || absErr < stats.minAbs) stats.minAbs = absErr;
    if (absErr > stats.maxAbs) stats.maxAbs = absErr;

    if (stats.minRelScaled === -1n || relErrScaled < stats.minRelScaled) stats.minRelScaled = relErrScaled;
    if (relErrScaled > stats.maxRelScaled) stats.maxRelScaled = relErrScaled;

    if (stats.minGas === -1n || estimatedGas < stats.minGas) stats.minGas = estimatedGas;
    if (estimatedGas > stats.maxGas) stats.maxGas = estimatedGas;
}

function averageScaled(sum: bigint, count: number): bigint {
    return count === 0 ? 0n : sum / BigInt(count);
}

function printMethodSummary(args: {
    benchmark: string;
    method: MethodLabel;
    stats: MethodStats;
    tol: bigint;
}) {
    const avgAbs = averageScaled(args.stats.absSum, args.stats.count);
    const avgRel = averageScaled(args.stats.relSumScaled, args.stats.count);
    const avgGas = averageScaled(args.stats.gasSum, args.stats.count);

    console.log("============================================================");
    console.log(`Benchmark           : ${args.benchmark}`);
    console.log(`Method              : ${args.method}`);
    console.log(`Total Tests         : ${args.stats.count}`);
    console.log(`Within Tolerance    : ${args.stats.withinTol}`);
    console.log(`Exceeded Tolerance  : ${args.stats.exceededTol}`);
    console.log(`Tolerance           : ${formatScaledInt(args.tol)}`);
    console.log(`Average Abs. Error  : ${formatScaledInt(avgAbs)}`);
    console.log(`Average Rel. Error  : ${formatScaledInt(avgRel)}`);
    console.log(`Average Rel. Error% : ${formatPercentScaled(avgRel)}%`);
    console.log(`Min Abs. Error      : ${formatScaledInt(args.stats.minAbs)}`);
    console.log(`Max Abs. Error      : ${formatScaledInt(args.stats.maxAbs)}`);
    console.log(`Min Rel. Error      : ${formatScaledInt(args.stats.minRelScaled)}`);
    console.log(`Max Rel. Error      : ${formatScaledInt(args.stats.maxRelScaled)}`);
    console.log(`Average Est. Gas    : ${avgGas.toString()}`);
    console.log(`Min Est. Gas        : ${args.stats.minGas.toString()}`);
    console.log(`Max Est. Gas        : ${args.stats.maxGas.toString()}`);
    console.log("============================================================");
}

function printCaseBlock(args: {
    benchmark: string;
    caseNo: number;
    method: MethodLabel;
    y0: number;
    hLabel: string;
    steps: number;
    xFinal: number;
    expectedScaled: bigint;
    actualScaled: bigint;
    absErr: bigint;
    relErrScaled: bigint;
    estimatedGas: bigint;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Benchmark          : ${args.benchmark}`);
    console.log(`Case               : ${args.caseNo}`);
    console.log(`Method             : ${args.method}`);
    console.log(`Input              : x0=0, y0=${args.y0}, h=${args.hLabel}, steps=${args.steps}, x_final=${args.xFinal}`);
    console.log(`Expected Output    : ${formatScaledInt(args.expectedScaled)}`);
    console.log(`Actual Output      : ${formatScaledInt(args.actualScaled)}`);
    console.log(`Absolute Error     : ${formatScaledInt(args.absErr)}`);
    console.log(`Relative Error     : ${formatScaledInt(args.relErrScaled)}`);
    console.log(`Relative Error (%) : ${formatPercentScaled(args.relErrScaled)}%`);
    console.log(`Estimated Gas      : ${args.estimatedGas.toString()}`);
    console.log("------------------------------------------------------------");
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

    const STEP_CASES: StepCase[] = [
        { hNum: 1, hDen: 10, hLabel: "0.1", steps: 10 },
        { hNum: 1, hDen: 10, hLabel: "0.1", steps: 50 },
        { hNum: 1, hDen: 20, hLabel: "0.05", steps: 20 },
        { hNum: 1, hDen: 20, hLabel: "0.05", steps: 100 },
        { hNum: 1, hDen: 100, hLabel: "0.01", steps: 100 },
        { hNum: 1, hDen: 100, hLabel: "0.01", steps: 200 }
    ];

    const ABS_TOL = 1_000_000n; // 1e-6 at SCALE = 1e12

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
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

    it("should evaluate average absolute and relative error over 18 systematic test instances per benchmark", async function () {
        const x0 = await qInt(0n);

        const benchmarks: BenchmarkDef[] = [
            {
                key: "const5",
                label: "y' = 5",
                selector: selConst5,
                exactSolution: (y0: number, xFinal: number) => y0 + 5 * xFinal,
            },
            {
                key: "linear",
                label: "y' = y",
                selector: selLinear,
                exactSolution: (y0: number, xFinal: number) => y0 * Math.exp(xFinal),
            },
            {
                key: "square",
                label: "y' = x^2",
                selector: selSquare,
                exactSolution: (y0: number, xFinal: number) => y0 + (xFinal ** 3) / 3,
            },
            {
                key: "cubicPoly",
                label: "y' = x^3 + x^2 + 2x + 3",
                selector: selCubicPoly,
                exactSolution: (y0: number, xFinal: number) =>
                    y0 + (xFinal ** 4) / 4 + (xFinal ** 3) / 3 + xFinal ** 2 + 3 * xFinal,
            },
        ];

        const testCases: TestCase[] = [];
        let caseNo = 0;

        for (const y0 of Y0_VALUES) {
            for (const sc of STEP_CASES) {
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

        let overallExecutions = 0;

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
                const y0Q = await qInt(tc.y0);
                const hQ = await qFrac(tc.hNum, tc.hDen);

                const expectedValue = benchmark.exactSolution(tc.y0, tc.xFinal);
                const expectedScaled = numberToScaledBigInt(expectedValue);

                for (const method of METHODS) {
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

                    updateStats(
                        statsByMethod.get(method.label)!,
                        absErr,
                        relErrScaled,
                        run.estimatedGas,
                        ABS_TOL
                    );
                    overallExecutions += 1;

                    printCaseBlock({
                        benchmark: benchmark.label,
                        caseNo: tc.caseNo,
                        method: method.label,
                        y0: tc.y0,
                        hLabel: tc.hLabel,
                        steps: tc.steps,
                        xFinal: tc.xFinal,
                        expectedScaled,
                        actualScaled,
                        absErr,
                        relErrScaled,
                        estimatedGas: run.estimatedGas,
                    });
                }
            }

            console.log("");
            console.log("******************** BENCHMARK SUMMARY *********************");
            for (const method of METHODS) {
                printMethodSummary({
                    benchmark: benchmark.label,
                    method: method.label,
                    stats: statsByMethod.get(method.label)!,
                    tol: ABS_TOL,
                });
            }

            const eulerAvgAbs = averageScaled(statsByMethod.get("Euler")!.absSum, 18);
            const rk4AvgAbs = averageScaled(statsByMethod.get("RK4")!.absSum, 18);
            expect(rk4AvgAbs <= eulerAvgAbs).to.equal(true);
        }

        expect(overallExecutions).to.equal(18 * 4 * 4);
    });
});