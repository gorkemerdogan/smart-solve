// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { classifyExecutionFailure } from "./test-utils";
import { printPrecisionMetadata } from "./precision-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type SteepestDescentHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;

    solve(
        objective: string,
        x0: string[],
        maxIter: bigint,
        tol: string
    ): Promise<[string[], string, bigint, bigint]>;
};

type ObjectiveHarness = Contract & {
    g(x: string[]): Promise<string>;
};

type SolveWithGasResult = {
    xRaw: string[];
    gRaw: string;
    iterations: bigint;
    status: bigint;
    gasUsed: bigint;
};

type BenchmarkSummary = {
    objectiveName: string;
    numberOfTests: number;
    averageDistance: bigint;
    minDistance: bigint;
    maxDistance: bigint;
    averageFinalObjectiveValue: bigint;
    averageIterationsScaled: bigint;
    averageGasScaled: bigint;
    minGas: bigint;
    maxGas: bigint;
    executionSuccessCount: number;
    successCount: number;
    numericalPassCount: number;
    failedCount: number;
    revertedCount: number;
    outOfGasCount: number;
    otherFailureCount: number;
};

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

const STATUS_SUCCESS = 0n;
const STATUS_ZERO_GRADIENT = 1n;
const STATUS_NO_LIKELY_IMPROVEMENT = 2n;
const STATUS_MAX_ITER_EXCEEDED = 3n;

const FIXED_SEED = 37n;
const NUM_TESTS = 30;
const RANDOM_MIN = -10n;
const RANDOM_MAX = 10n;
const DISTANCE_TOL_SCALED = 100_000_000n; // 1e-4 at the harness scale
const OBJECTIVE_TOL_SCALED = 10_000n;     // 1e-8 at the harness scale

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

let SCALE_DECIMALS = 12n;
let SCALE = 10n ** SCALE_DECIMALS;

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

function formatIntegerAverageScaled(vScaledBy1000: bigint): string {
    const integerPart = vScaledBy1000 / 1000n;
    const fracPart = absBigInt(vScaledBy1000 % 1000n).toString().padStart(3, "0");
    return `${integerPart.toString()}.${fracPart}`.replace(/\.?0+$/, "");
}

function vecInfNorm(v: bigint[]): bigint {
    let m = 0n;
    for (const x of v) {
        const ax = absBigInt(x);
        if (ax > m) m = ax;
    }
    return m;
}

function subVec(a: bigint[], b: bigint[]): bigint[] {
    return a.map((x, i) => x - b[i]);
}

async function scaledVec(
    harness: { toFloat: (q: string) => Promise<unknown> },
    values: string[]
): Promise<bigint[]> {
    return Promise.all(values.map(async v => asBigInt(await harness.toFloat(v))));
}

function avgBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    const total = values.reduce((acc, x) => acc + x, 0n);
    return total / BigInt(values.length);
}

function minBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    let m = values[0];
    for (const v of values) {
        if (v < m) m = v;
    }
    return m;
}

function maxBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    let m = values[0];
    for (const v of values) {
        if (v > m) m = v;
    }
    return m;
}

function isSuccessfulStatus(status: bigint): boolean {
    return (
        status === STATUS_SUCCESS ||
        status === STATUS_ZERO_GRADIENT
    );
}

/**
 * @notice Deterministically maps a Keccak-256 hash into the inclusive integer interval [min, max].
 * @dev    This helper is used to generate reproducible pseudo-random integer coordinates for the
 *         initial vector. The generation is deterministic with respect to the seed, objective id,
 *         test id, and coordinate index.
 */
function deterministicIntInRange(
    seed: bigint,
    objectiveId: bigint,
    testId: bigint,
    dimId: bigint,
    min: bigint,
    max: bigint
): bigint {
    const encoded = ethers.solidityPacked(
        ["uint256", "uint256", "uint256", "uint256"],
        [seed, objectiveId, testId, dimId]
    );

    const hash = ethers.keccak256(encoded);
    const raw = BigInt(hash);
    const span = max - min + 1n;

    return min + (raw % span);
}

/**
 * @notice Builds a deterministic pseudo-random initial point in the integer interval [min, max].
 * @dev    The produced vector is reproducible across runs because all coordinates are generated
 *         from the fixed seed and explicit benchmark/test indices.
 */
async function buildDeterministicInitialVector(
    solver: SteepestDescentHarness,
    objectiveId: bigint,
    testId: bigint,
    dimension: number,
    min: bigint,
    max: bigint
): Promise<string[]> {
    const values: string[] = [];

    for (let i = 0; i < dimension; i++) {
        const intVal = deterministicIntInRange(
            FIXED_SEED,
            objectiveId,
            testId,
            BigInt(i),
            min,
            max
        );
        values.push(await solver.qFromInt(intVal));
    }

    return values;
}

/**
 * @notice Executes the solver twice: once for the numerical result and once for gas measurement.
 * @dev    The eth_call result is used to obtain the returned values directly. The transaction run
 *         is used only to measure gas consumption under the same input configuration.
 */
async function solveWithGas(
    solver: SteepestDescentHarness,
    objectiveAddress: string,
    x0: string[],
    maxIter: bigint,
    tol: string
): Promise<SolveWithGasResult> {
    const [xRaw, gRaw, iterations, status] = await solver.solve(
        objectiveAddress,
        x0,
        maxIter,
        tol
    );

    const gasUsed = await solver.getFunction("solve").estimateGas(
        objectiveAddress,
        x0,
        maxIter,
        tol
    );

    return {
        xRaw,
        gRaw,
        iterations,
        status,
        gasUsed: BigInt(gasUsed.toString()),
    };
}

function printBenchmarkSummary(summary: BenchmarkSummary): void {
    const successRateScaled = (BigInt(summary.successCount) * 100000n) / BigInt(summary.numberOfTests);

    console.log("============================================================");
    console.log(`Benchmark                : ${summary.objectiveName}`);
    console.log(`Number of Tests          : ${summary.numberOfTests}`);
    console.log(`Successful Executions    : ${summary.executionSuccessCount}`);
    console.log(`Execution Failures       : ${summary.failedCount}`);
    console.log(`Reverted Cases           : ${summary.revertedCount}`);
    console.log(`Out-of-Gas Cases         : ${summary.outOfGasCount}`);
    console.log(`Other Failures           : ${summary.otherFailureCount}`);
    console.log(`Numerical Passes         : ${summary.numericalPassCount}`);
    console.log(`Numerical Failures       : ${summary.executionSuccessCount - summary.numericalPassCount}`);
    console.log(`Average Distance to Optimum : ${formatScaledInt(summary.averageDistance)}`);
    console.log(`Min Distance             : ${formatScaledInt(summary.minDistance)}`);
    console.log(`Max Distance             : ${formatScaledInt(summary.maxDistance)}`);
    console.log(`Average Final Objective Value : ${formatScaledInt(summary.averageFinalObjectiveValue)}`);
    console.log(`Average Iterations       : ${formatIntegerAverageScaled(summary.averageIterationsScaled)}`);
    console.log(`Average Gas Consumption (successful executions only): ${formatIntegerAverageScaled(summary.averageGasScaled)}`);
    console.log(`Min Gas                  : ${summary.minGas.toString()}`);
    console.log(`Max Gas                  : ${summary.maxGas.toString()}`);
    console.log(`Success Rate             : ${formatIntegerAverageScaled(successRateScaled)}% (${summary.successCount}/${summary.numberOfTests})`);
    console.log(`Numerical Pass Rate      : ${((summary.numericalPassCount * 100) / summary.numberOfTests).toFixed(2)}%`);
    console.log("============================================================");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("SteepestDescent Library - 1e12 Fixed-Point Accuracy & Gas Benchmarks", function () {
    let solver: SteepestDescentHarness & { toFloat(q: string): Promise<unknown> };
    let spherical: ObjectiveHarness;
    let weighted: ObjectiveHarness;
    let semiWeighted: ObjectiveHarness;

    let TOL_1E_9: string;

    before(async () => {
        printPrecisionMetadata({
            classification: "fixed-point/truncated comparison",
            comparisonScale: "1e12 Solidity toFloat output with BigInt error arithmetic",
            oraclePrecision: "exact known minimizers/objective values for selected objectives",
            conversion: "binary128 -> truncating 1e12 fixed-point integer -> BigInt",
            claim: "1e12-scale optimizer output agreement at method-specific tolerances; not binary128 accuracy",
        });
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const SolverFactory = await ethers.getContractFactory("SteepestDescentHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });
        const deployedSolver = (await SolverFactory.deploy()) as unknown as SteepestDescentHarness & {
            toFloat(q: string): Promise<unknown>;
        };

        const helperAbi = [
            "function qFromInt(int256 n) external pure returns (bytes16)",
            "function qFromFrac(int256 num, int256 den) external pure returns (bytes16)",
            "function toFloat(bytes16 x) external pure returns (int256)"
        ];

        const combinedAbi = [...helperAbi, ...SolverFactory.interface.fragments];

        solver = new ethers.Contract(await deployedSolver.getAddress(), combinedAbi, ethers.provider) as any;

        const SphericalFactory = await ethers.getContractFactory("SphericalObjectiveHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });
        spherical = (await SphericalFactory.deploy()) as unknown as ObjectiveHarness;
        await spherical.waitForDeployment();

        const WeightedFactory = await ethers.getContractFactory("WeightedQuadraticObjectiveHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });
        weighted = (await WeightedFactory.deploy()) as unknown as ObjectiveHarness;
        await weighted.waitForDeployment();

        const SemiFactory = await ethers.getContractFactory("SemiWeightedQuadraticObjectiveHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });
        semiWeighted = (await SemiFactory.deploy()) as unknown as ObjectiveHarness;
        await semiWeighted.waitForDeployment();

        TOL_1E_9 = await solver.qFromFrac(1n, 1_000_000_000n);

        const oneScaled = asBigInt(await solver.toFloat(await solver.qFromInt(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    /**
     * @notice Runs a 30-test deterministic benchmark for one objective function and prints summary metrics.
     * @dev    The known minimizer for all objectives is the zero vector. For each run, a deterministic
     *         pseudo-random initial point is generated in [-10, 10].
     */
    async function runBenchmark(args: {
        objectiveName: string;
        objectiveId: bigint;
        objective: ObjectiveHarness;
        dimension: number;
        maxIter: bigint;
    }): Promise<BenchmarkSummary> {
        const zeroVector = new Array<bigint>(args.dimension).fill(0n);

        const distances: bigint[] = [];
        const finalObjectiveValues: bigint[] = [];
        const iterationCounts: bigint[] = [];
        const gasValues: bigint[] = [];
        let executionSuccessCount = 0;
        let successCount = 0;
        let numericalPassCount = 0;
        let failedCount = 0;
        let revertedCount = 0;
        let outOfGasCount = 0;
        let otherFailureCount = 0;

        for (let testNo = 1; testNo <= NUM_TESTS; testNo++) {
            const x0 = await buildDeterministicInitialVector(
                solver,
                args.objectiveId,
                BigInt(testNo),
                args.dimension,
                RANDOM_MIN,
                RANDOM_MAX
            );

            try {
                const result = await solveWithGas(
                    solver,
                    await args.objective.getAddress(),
                    x0,
                    args.maxIter,
                    TOL_1E_9
                );

                const xComputed = await scaledVec(solver, result.xRaw);
                const distance = vecInfNorm(subVec(xComputed, zeroVector));
                const finalObjective = absBigInt(asBigInt(await solver.toFloat(result.gRaw)));

                executionSuccessCount++;
                distances.push(distance);
                finalObjectiveValues.push(finalObjective);
                iterationCounts.push(result.iterations);
                gasValues.push(result.gasUsed);

                const successfulStatus = isSuccessfulStatus(result.status);
                if (successfulStatus) successCount++;
                if (
                    successfulStatus &&
                    distance <= DISTANCE_TOL_SCALED &&
                    finalObjective <= OBJECTIVE_TOL_SCALED
                ) {
                    numericalPassCount++;
                }
            } catch (error) {
                failedCount++;
                const kind = classifyExecutionFailure(error);
                if (kind === "revert") revertedCount++;
                else if (kind === "out-of-gas") outOfGasCount++;
                else otherFailureCount++;

                console.log(`Unexpected ${kind} in ${args.objectiveName}, case ${testNo}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }

        const iterationSum = iterationCounts.reduce((acc, x) => acc + x, 0n);
        const gasSum = gasValues.reduce((acc, x) => acc + x, 0n);

        const summary: BenchmarkSummary = {
            objectiveName: args.objectiveName,
            numberOfTests: NUM_TESTS,
            averageDistance: avgBigInt(distances),
            minDistance: minBigInt(distances),
            maxDistance: maxBigInt(distances),
            averageFinalObjectiveValue: avgBigInt(finalObjectiveValues),
            averageIterationsScaled: executionSuccessCount === 0 ? 0n : (iterationSum * 1000n) / BigInt(executionSuccessCount),
            averageGasScaled: executionSuccessCount === 0 ? 0n : (gasSum * 1000n) / BigInt(executionSuccessCount),
            minGas: minBigInt(gasValues),
            maxGas: maxBigInt(gasValues),
            executionSuccessCount,
            successCount,
            numericalPassCount,
            failedCount,
            revertedCount,
            outOfGasCount,
            otherFailureCount,
        };

        printBenchmarkSummary(summary);
        expect(summary.failedCount, `${args.objectiveName} had unexpected execution failures`).to.equal(0);
        expect(summary.successCount, `${args.objectiveName} returned a non-success solver status`).to.equal(NUM_TESTS);
        expect(
            summary.numericalPassCount,
            `${args.objectiveName} exceeded distance<=1e-4 or objective<=1e-8 in one or more cases`
        ).to.equal(NUM_TESTS);
        return summary;
    }

    it("prints summary results for spherical objective", async function () {
        const summary = await runBenchmark({
            objectiveName: "Spherical objective",
            objectiveId: 1n,
            objective: spherical,
            dimension: 4,
            maxIter: 1000n,
        });

        expect(summary.numberOfTests).to.equal(NUM_TESTS);
    });

    it("prints summary results for semi-weighted objective", async function () {
        const summary = await runBenchmark({
            objectiveName: "Semi-weighted objective",
            objectiveId: 2n,
            objective: semiWeighted,
            dimension: 8,
            maxIter: 1000n,
        });

        expect(summary.numberOfTests).to.equal(NUM_TESTS);
    });

    it("prints summary results for weighted objective", async function () {
        const summary = await runBenchmark({
            objectiveName: "Weighted objective",
            objectiveId: 3n,
            objective: weighted,
            dimension: 4,
            maxIter: 1000n,
        });

        expect(summary.numberOfTests).to.equal(NUM_TESTS);
    });
});
