// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockOptimization } from "./test-utils";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type SteepestDescentHarness = Contract & {
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(num: bigint, den: bigint): Promise<string>;
    solve(
        objective: string,
        x0: string[],
        maxIter: bigint,
        tol: string
    ): Promise<[string[], string, bigint, bigint]>;
};

type ObjectiveHarness = Contract & {
    g(x: string[]): Promise<string>;
    grad(x: string[]): Promise<string[]>;
};

const STATUS_ZERO_GRADIENT = 1n;
const STATUS_NO_LIKELY_IMPROVEMENT = 2n;
const STATUS_MAX_ITER_EXCEEDED = 3n;

const RUNS_PER_CASE = 10;
const DEFAULT_SEED = "STEepestDescent-Keccak-Seed-v1";
const REPORT_TOL = 1e-12;

// ------------------------------------------------------------
//  Deployment
// ------------------------------------------------------------

async function newHarnesses(): Promise<{
    solver: SteepestDescentHarness;
    weighted: ObjectiveHarness;
    spherical: ObjectiveHarness;
    semiWeighted: ObjectiveHarness;
}> {
    const MathLibFactory = await ethers.getContractFactory(
        "contracts/libraries/MathLib.sol:MathLib"
    );
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

    const SolverFactory = await ethers.getContractFactory("SteepestDescentHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const WeightedFactory = await ethers.getContractFactory("WeightedQuadraticObjectiveHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const SphericalFactory = await ethers.getContractFactory("SphericalObjectiveHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const SemiWeightedFactory = await ethers.getContractFactory("SemiWeightedQuadraticObjectiveHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const solver = await SolverFactory.deploy();
    await solver.waitForDeployment();

    const weighted = await WeightedFactory.deploy();
    await weighted.waitForDeployment();

    const spherical = await SphericalFactory.deploy();
    await spherical.waitForDeployment();

    const semiWeighted = await SemiWeightedFactory.deploy();
    await semiWeighted.waitForDeployment();

    return {
        solver: solver as unknown as SteepestDescentHarness,
        weighted: weighted as unknown as ObjectiveHarness,
        spherical: spherical as unknown as ObjectiveHarness,
        semiWeighted: semiWeighted as unknown as ObjectiveHarness,
    };
}

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

function fmtHexArr(arr: string[]) {
    if (arr.length > 6) {
        return `[${arr.slice(0, 3).join(", ")}, ..., ${arr.slice(-3).join(", ")}] (len=${arr.length})`;
    }
    return `[${arr.join(", ")}]`;
}

function asResult(tuple: [string[], string, bigint, bigint]) {
    const [x, gx, iters, status] = tuple;
    return { x: [...x], gx, iters, status };
}

function quadHexToApproxNumber(hex: string): number {
    const bits = BigInt(hex);
    const sign = ((bits >> 127n) & 1n) === 1n ? -1 : 1;
    const exp = Number((bits >> 112n) & 0x7fffn);
    const fracMask = (1n << 112n) - 1n;
    const frac = bits & fracMask;
    const bias = 16383;

    if (exp === 0 && frac === 0n) return sign * 0;
    if (exp === 0x7fff) return frac === 0n ? sign * Infinity : NaN;

    const fracApprox = Number(frac) / Math.pow(2, 112);

    if (exp === 0) {
        return sign * fracApprox * Math.pow(2, 1 - bias);
    }

    const mantissa = 1 + fracApprox;
    return sign * mantissa * Math.pow(2, exp - bias);
}

function makeTestId(section: number, index: number): string {
    return `${section}.${index}`;
}

async function qVec(
    solver: SteepestDescentHarness,
    vals: bigint[]
): Promise<string[]> {
    return Promise.all(vals.map(v => solver.qFromInt(v)));
}

// ------------------------------------------------------------
//  Deterministic Keccak Random Helpers
// ------------------------------------------------------------

function keccakBigInt(...parts: (string | number | bigint)[]): bigint {
    const encoded = parts.map(String).join("|");
    const hex = ethers.keccak256(ethers.toUtf8Bytes(encoded));
    return BigInt(hex);
}

function keccakSignedInt(
    minInclusive: bigint,
    maxInclusive: bigint,
    ...parts: (string | number | bigint)[]
): bigint {
    const span = maxInclusive - minInclusive + 1n;
    const r = keccakBigInt(DEFAULT_SEED, ...parts) % span;
    return minInclusive + r;
}

function makeRandomVectorByKeccak(
    dim: number,
    minInclusive: bigint,
    maxInclusive: bigint,
    ...parts: (string | number | bigint)[]
): bigint[] {
    return Array.from({ length: dim }, (_, i) =>
        keccakSignedInt(minInclusive, maxInclusive, ...parts, "idx", i)
    );
}

function addVectors(a: bigint[], b: bigint[]): bigint[] {
    return a.map((v, i) => v + b[i]);
}

function summarizeBigint(values: bigint[]) {
    const min = values.reduce((a, b) => (a < b ? a : b));
    const max = values.reduce((a, b) => (a > b ? a : b));
    const avg = values.reduce((a, b) => a + b, 0n) / BigInt(values.length);
    return { min, max, avg };
}

function summarizeFiniteNumber(values: number[]) {
    const finite = values.filter(v => Number.isFinite(v));
    if (finite.length === 0) {
        return { min: NaN, max: NaN, avg: NaN };
    }
    return {
        min: Math.min(...finite),
        max: Math.max(...finite),
        avg: finite.reduce((a, b) => a + b, 0) / finite.length
    };
}

function formatNum(v: number, digits = 12): string {
    if (!Number.isFinite(v)) return String(v);
    if (v === 0) return "0";
    return v.toExponential(digits);
}

async function computeXErrorInfNorm(x: string[]): Promise<number> {
    let maxAbs = 0;
    for (const xi of x) {
        const v = Math.abs(quadHexToApproxNumber(xi));
        if (v > maxAbs) maxAbs = v;
    }
    return maxAbs;
}

async function computeFErrorFromGx(gx: string): Promise<number> {
    return Math.abs(quadHexToApproxNumber(gx));
}

async function computeGradNormInf(
    objective: ObjectiveHarness,
    x: string[]
): Promise<number> {
    const grad = await objective.grad(x);
    let maxAbs = 0;
    for (const gi of grad) {
        const v = Math.abs(quadHexToApproxNumber(gi));
        if (v > maxAbs) maxAbs = v;
    }
    return maxAbs;
}

function countWithinTol(values: number[], tol: number): { within: number; exceeded: number } {
    let within = 0;
    let exceeded = 0;
    for (const v of values) {
        if (Number.isFinite(v) && v <= tol) within++;
        else exceeded++;
    }
    return { within, exceeded };
}

function makeWorstCaseLabel(
    gradNormStats: { max: number },
    xErrorStats: { max: number },
    fErrorStats: { max: number }
): string {
    return `gradNorm=${formatNum(gradNormStats.max, 6)}, xError=${formatNum(xErrorStats.max, 6)}, fError=${formatNum(fErrorStats.max, 6)}`;
}

function statusHistogram(statuses: bigint[]): string {
    const map = new Map<string, number>();
    for (const s of statuses) {
        const k = s.toString();
        map.set(k, (map.get(k) ?? 0) + 1);
    }
    return Array.from(map.entries())
        .sort((a, b) => Number(a[0]) - Number(b[0]))
        .map(([k, v]) => `${k}:${v}`)
        .join(", ");
}

type BatchRunResult = {
    gasValues: bigint[];
    iterValues: bigint[];
    statuses: bigint[];
    finalXs: string[][];
    finalGxs: string[];
    inputs: string[][];
    gradNormValues: number[];
    xErrorValues: number[];
    fErrorValues: number[];
};

async function runBatch(
    solver: SteepestDescentHarness,
    objective: ObjectiveHarness,
    objectiveAddr: string,
    x0Batch: string[][],
    maxIter: bigint,
    tol: string
): Promise<BatchRunResult> {
    const gasValues: bigint[] = [];
    const iterValues: bigint[] = [];
    const statuses: bigint[] = [];
    const finalXs: string[][] = [];
    const finalGxs: string[] = [];
    const inputs: string[][] = [];
    const gradNormValues: number[] = [];
    const xErrorValues: number[] = [];
    const fErrorValues: number[] = [];

    for (const x0 of x0Batch) {
        await touchGas(solver, "solve", [objectiveAddr, x0, maxIter, tol]);
        const gasRaw = await estimateGas(solver, "solve", [objectiveAddr, x0, maxIter, tol]);
        const gas = BigInt(gasRaw);
        const out = asResult(await solver.solve(objectiveAddr, x0, maxIter, tol));

        const gradNorm = await computeGradNormInf(objective, out.x);
        const xError = await computeXErrorInfNorm(out.x);
        const fError = await computeFErrorFromGx(out.gx);

        gasValues.push(gas);
        iterValues.push(out.iters);
        statuses.push(out.status);
        finalXs.push(out.x);
        finalGxs.push(out.gx);
        inputs.push(x0);
        gradNormValues.push(gradNorm);
        xErrorValues.push(xError);
        fErrorValues.push(fError);
    }

    return {
        gasValues,
        iterValues,
        statuses,
        finalXs,
        finalGxs,
        inputs,
        gradNormValues,
        xErrorValues,
        fErrorValues
    };
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("SteepestDescent - Gas Growth Tests", function () {
    let solver: SteepestDescentHarness;
    let weighted: ObjectiveHarness;
    let spherical: ObjectiveHarness;
    let semiWeighted: ObjectiveHarness;

    let tolTight: string;
    let tolZero: string;

    before(async () => {
        const deployed = await newHarnesses();
        solver = deployed.solver;
        weighted = deployed.weighted;
        spherical = deployed.spherical;
        semiWeighted = deployed.semiWeighted;

        tolTight = await solver.qFromFrac(1n, 1_000_000_000_000_000_000_000_000n); // 1e-24
        tolZero = await solver.qFromInt(0n);
    });

    // --------------------------------------------------------
    //  Section 1: Gas Growth with Respect to Dimension
    // --------------------------------------------------------

    describe.skip("Section 1: Gas Growth with Respect to Dimension", function () {
        const DIMS = [2, 3, 4, 6, 8, 12, 14, 16, 18];
        let testIndex = 0;

        for (const dim of DIMS) {
            const testId = makeTestId(1, ++testIndex);

            it(`Test ${testId}: Spherical objective gas growth with dimension n=${dim}`, async function () {
                const objectiveAddr = await spherical.getAddress();

                const x0Batch = await Promise.all(
                    Array.from({ length: RUNS_PER_CASE }, async (_, runIdx) => {
                        const vals = makeRandomVectorByKeccak(
                            dim,
                            -10n,
                            10n,
                            "sec1",
                            "spherical",
                            dim,
                            runIdx
                        );
                        return qVec(solver, vals);
                    })
                );

                const batch = await runBatch(solver, spherical, objectiveAddr, x0Batch, 10n, tolTight);

                batch.statuses.forEach((s) => {
                    expect([STATUS_ZERO_GRADIENT, STATUS_MAX_ITER_EXCEEDED]).to.include(s);
                });

                const gasStats = summarizeBigint(batch.gasValues);
                const iterStats = summarizeBigint(batch.iterValues);
                const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
                const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
                const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
                const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

                printBlockOptimization({
                    t: testId,
                    method: "solve",
                    explanation: `Gas growth with respect to dimension for spherical objective at n=${dim}.`,
                    gas: gasStats.avg,
                    x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                    xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                    gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                    status: statusHistogram(batch.statuses),
                    iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                    gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                    xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                    fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                    withinTol: `${xTolCount.within}/${RUNS_PER_CASE}`,
                    exceededTol: `${xTolCount.exceeded}/${RUNS_PER_CASE}`,
                    tolerance: "1e-12 (xError)",
                    worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                    extra: `runs=${RUNS_PER_CASE}, objective=spherical, dim=${dim}, gas[min=${gasStats.min}, max=${gasStats.max}]`
                });
            });
        }

        testIndex = 0;

        for (const dim of DIMS) {
            const testId = makeTestId(2, ++testIndex);

            it(`Test ${testId}: Semi-weighted quadratic objective gas growth with dimension n=${dim}`, async function () {
                const objectiveAddr = await semiWeighted.getAddress();

                const x0Batch = await Promise.all(
                    Array.from({ length: RUNS_PER_CASE }, async (_, runIdx) => {
                        const vals = makeRandomVectorByKeccak(
                            dim,
                            -10n,
                            10n,
                            "sec1",
                            "semiWeighted",
                            dim,
                            runIdx
                        );
                        return qVec(solver, vals);
                    })
                );

                const batch = await runBatch(solver, semiWeighted, objectiveAddr, x0Batch, 20n, tolTight);

                batch.statuses.forEach((s) => {
                    expect([
                        STATUS_ZERO_GRADIENT,
                        STATUS_MAX_ITER_EXCEEDED,
                        STATUS_NO_LIKELY_IMPROVEMENT,
                    ]).to.include(s);
                });

                const gasStats = summarizeBigint(batch.gasValues);
                const iterStats = summarizeBigint(batch.iterValues);
                const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
                const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
                const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
                const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

                printBlockOptimization({
                    t: testId,
                    method: "solve",
                    explanation: `Gas growth with respect to dimension for semi-weighted quadratic objective at n=${dim}.`,
                    gas: gasStats.avg,
                    x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                    xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                    gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                    status: statusHistogram(batch.statuses),
                    iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                    gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                    xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                    fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                    withinTol: `${xTolCount.within}/${RUNS_PER_CASE}`,
                    exceededTol: `${xTolCount.exceeded}/${RUNS_PER_CASE}`,
                    tolerance: "1e-12 (xError)",
                    worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                    extra: `runs=${RUNS_PER_CASE}, objective=semiWeighted, dim=${dim}, gas[min=${gasStats.min}, max=${gasStats.max}]`
                });
            });
        }
    });

    // --------------------------------------------------------
    //  Section 2: Gas Sensitivity to Iteration Budget
    // --------------------------------------------------------

    describe.skip("Section 2: Gas Sensitivity to Iteration Budget", function () {
        const ITERS = [1n, 2n, 5n, 10n, 20n];
        let testIndex = 0;

        for (const maxIter of ITERS) {
            const testId = makeTestId(2, ++testIndex);

            it(`Test ${testId}: Spherical gas sensitivity with maxIter=${maxIter}`, async function () {
                const objectiveAddr = await spherical.getAddress();

                const x0Batch = await Promise.all(
                    Array.from({ length: RUNS_PER_CASE }, async (_, runIdx) => {
                        const vals = makeRandomVectorByKeccak(
                            8,
                            -10n,
                            10n,
                            "sec2",
                            "spherical",
                            maxIter,
                            runIdx
                        );
                        return qVec(solver, vals);
                    })
                );

                const batch = await runBatch(solver, spherical, objectiveAddr, x0Batch, maxIter, tolTight);

                const gasStats = summarizeBigint(batch.gasValues);
                const iterStats = summarizeBigint(batch.iterValues);
                const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
                const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
                const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
                const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

                printBlockOptimization({
                    t: testId,
                    method: "solve",
                    explanation: `Gas sensitivity to iteration budget for spherical objective at fixed dimension n=8 and maxIter=${maxIter}.`,
                    gas: gasStats.avg,
                    x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                    xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                    gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                    status: statusHistogram(batch.statuses),
                    iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                    gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                    xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                    fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                    withinTol: `${xTolCount.within}/${RUNS_PER_CASE}`,
                    exceededTol: `${xTolCount.exceeded}/${RUNS_PER_CASE}`,
                    tolerance: "1e-12 (xError)",
                    worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                    extra: `runs=${RUNS_PER_CASE}, objective=spherical, dim=8, maxIter=${maxIter}, gas[min=${gasStats.min}, max=${gasStats.max}]`
                });
            });
        }
    });

    // --------------------------------------------------------
    //  Section 3: Gas Sensitivity to Initial Magnitude
    // --------------------------------------------------------

    describe.skip("Section 3: Gas Sensitivity to Initial Point Magnitude", function () {
        type MagnitudeCase = {
            label: string;
            base: bigint;
        };

        const cases: MagnitudeCase[] = [
            { label: "small", base: 1n },
            { label: "medium", base: 10n },
            { label: "large", base: 100n },
            { label: "veryLarge", base: 1000n },
        ];

        const DIM = 8;
        const MAX_ITER = 30n;

        let testIndex = 0;

        for (const p of cases) {
            const testId = makeTestId(3, ++testIndex);

            it(`Test ${testId}: Spherical quadratic gas sensitivity for initial magnitude ${p.label}`, async function () {
                const objectiveAddr = await spherical.getAddress();

                const x0Batch = await Promise.all(
                    Array.from({ length: RUNS_PER_CASE }, async (_, runIdx) => {
                        const signs = makeRandomVectorByKeccak(
                            DIM,
                            -1n,
                            1n,
                            "sec3",
                            p.label,
                            "signs",
                            runIdx
                        ).map((v) => (v === 0n ? 1n : v));

                        const jitter = makeRandomVectorByKeccak(
                            DIM,
                            -2n,
                            2n,
                            "sec3",
                            p.label,
                            "jitter",
                            runIdx
                        );

                        const vals = signs.map((s, i) => s * (p.base + jitter[i]));
                        return qVec(solver, vals);
                    })
                );

                const batch = await runBatch(
                    solver,
                    spherical,
                    objectiveAddr,
                    x0Batch,
                    MAX_ITER,
                    tolTight
                );

                batch.statuses.forEach((s) => {
                    expect(s).to.eq(STATUS_ZERO_GRADIENT);
                });

                const gasStats = summarizeBigint(batch.gasValues);
                const iterStats = summarizeBigint(batch.iterValues);
                const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
                const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
                const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
                const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

                printBlockOptimization({
                    t: testId,
                    method: "solve",
                    explanation: `Gas sensitivity to initial-point magnitude for spherical objective using ${p.label} scale.`,
                    gas: gasStats.avg,
                    x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                    xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                    gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                    status: statusHistogram(batch.statuses),
                    iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                    gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                    xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                    fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                    withinTol: `${xTolCount.within}/${RUNS_PER_CASE}`,
                    exceededTol: `${xTolCount.exceeded}/${RUNS_PER_CASE}`,
                    tolerance: "1e-12 (xError)",
                    worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                    extra: `runs=${RUNS_PER_CASE}, objective=spherical, dim=${DIM}, maxIter=${MAX_ITER}, magnitude=${p.label}, gas[min=${gasStats.min}, max=${gasStats.max}]`
                });
            });
        }

        const mixedLargeTestId = makeTestId(3, ++testIndex);

        it(`Test ${mixedLargeTestId}: Spherical quadratic gas sensitivity for initial magnitude mixedLarge`, async function () {
            const objectiveAddr = await spherical.getAddress();

            const x0Batch = await Promise.all(
                Array.from({ length: RUNS_PER_CASE }, async (_, runIdx) => {
                    const base = Array.from(
                        { length: DIM },
                        (_, i) => (i % 2 === 0 ? -1000n : 1000n)
                    );

                    const jitter = makeRandomVectorByKeccak(
                        DIM,
                        -10n,
                        10n,
                        "sec3",
                        "mixedLarge",
                        runIdx
                    );

                    const vals = addVectors(base, jitter);
                    return qVec(solver, vals);
                })
            );

            const batch = await runBatch(
                solver,
                spherical,
                objectiveAddr,
                x0Batch,
                MAX_ITER,
                tolTight
            );

            batch.statuses.forEach((s) => {
                expect(s).to.eq(STATUS_ZERO_GRADIENT);
            });

            const gasStats = summarizeBigint(batch.gasValues);
            const iterStats = summarizeBigint(batch.iterValues);
            const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
            const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
            const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
            const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

            printBlockOptimization({
                t: mixedLargeTestId,
                method: "solve",
                explanation: `Gas sensitivity to initial-point magnitude for spherical objective using mixedLarge scale.`,
                gas: gasStats.avg,
                x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                status: statusHistogram(batch.statuses),
                iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                withinTol: `${xTolCount.within}/${RUNS_PER_CASE}`,
                exceededTol: `${xTolCount.exceeded}/${RUNS_PER_CASE}`,
                tolerance: "1e-12 (xError)",
                worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                extra: `runs=${RUNS_PER_CASE}, objective=spherical, dim=${DIM}, maxIter=${MAX_ITER}, magnitude=mixedLarge, gas[min=${gasStats.min}, max=${gasStats.max}]`
            });
        });
    });

    // --------------------------------------------------------
    //  Section 4: Gas Sensitivity to Initial Pattern
    // --------------------------------------------------------

    describe("Section 4: Gas Sensitivity to Initial Point Range", function () {
        type PatternCase = {
            label: string;
            min: bigint;
            max: bigint;
        };

        type ObjectiveCase = {
            label: string;
            objective: () => ObjectiveHarness;
            getAddress: () => Promise<string>;
            expectedStatuses: bigint[];
            dim: number;
            maxIter: bigint;
        };

        const RANDOM_INPUTS_PER_PATTERN = 5;

        const patterns: PatternCase[] = [
            { label: "smallRandom", min: -2n, max: 2n },
            { label: "mediumRandom", min: -10n, max: 10n },
            { label: "largeRandom", min: -100n, max: 100n },
            { label: "positiveRandom", min: 1n, max: 10n },
            { label: "negativeRandom", min: -10n, max: -1n }
        ];

        const objectives: ObjectiveCase[] = [
            {
                label: "spherical",
                objective: () => spherical,
                getAddress: async () => spherical.getAddress(),
                expectedStatuses: [STATUS_ZERO_GRADIENT],
                dim: 8,
                maxIter: 30n
            },
            {
                label: "semiWeighted",
                objective: () => semiWeighted,
                getAddress: async () => semiWeighted.getAddress(),
                expectedStatuses: [
                    STATUS_ZERO_GRADIENT,
                    STATUS_MAX_ITER_EXCEEDED,
                    STATUS_NO_LIKELY_IMPROVEMENT
                ],
                dim: 4,
                maxIter: 20n
            }
        ];

        let testIndex = 0;

        for (const objCase of objectives) {
            for (const p of patterns) {
                const testId = makeTestId(4, ++testIndex);

                it(`Test ${testId}: ${objCase.label} gas sensitivity for ${p.label} initial inputs`, async function () {
                    const objective = objCase.objective();
                    const objectiveAddr = await objCase.getAddress();

                    const x0Batch = await Promise.all(
                        Array.from({ length: RANDOM_INPUTS_PER_PATTERN }, async (_, runIdx) => {
                            const vals = makeRandomVectorByKeccak(
                                objCase.dim,
                                p.min,
                                p.max,
                                "sec4",
                                objCase.label,
                                p.label,
                                runIdx
                            );

                            return qVec(solver, vals);
                        })
                    );

                    const batch = await runBatch(
                        solver,
                        objective,
                        objectiveAddr,
                        x0Batch,
                        objCase.maxIter,
                        tolTight
                    );

                    batch.statuses.forEach((s) => {
                        expect(objCase.expectedStatuses).to.include(s);
                    });

                    const gasStats = summarizeBigint(batch.gasValues);
                    const iterStats = summarizeBigint(batch.iterValues);
                    const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
                    const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
                    const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
                    const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

                    printBlockOptimization({
                        t: testId,
                        method: "solve",
                        explanation: `Gas sensitivity to random initial inputs for ${objCase.label} objective using ${p.label} range.`,
                        gas: gasStats.avg,
                        x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                        xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                        gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                        status: statusHistogram(batch.statuses),
                        iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                        gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                        xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                        fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                        withinTol: `${xTolCount.within}/${RANDOM_INPUTS_PER_PATTERN}`,
                        exceededTol: `${xTolCount.exceeded}/${RANDOM_INPUTS_PER_PATTERN}`,
                        tolerance: "1e-12 (xError)",
                        worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                        extra: `runs=${RANDOM_INPUTS_PER_PATTERN}, objective=${objCase.label}, dim=${objCase.dim}, maxIter=${objCase.maxIter}, pattern=${p.label}, gas[min=${gasStats.min}, max=${gasStats.max}]`
                    });
                });
            }
        }
    });

    // --------------------------------------------------------
    //  Section 5: Gas Growth with Respect to Iteration Budget
    // --------------------------------------------------------

    describe.skip("Section 5: Gas Growth with Respect to Iteration Budget", function () {
        const ITERS = [1n, 2n, 5n, 10n, 15n, 17n, 19n, 20n, 21n, 22n, 23n, 24n, 25n];
        const DIMS = [12];
        let testIndex = 0;

        const fixedX0Batches = new Map<number, string[][]>();

        before(async function () {
            for (const dim of DIMS) {
                const fixedX0Batch = await Promise.all(
                    Array.from({length: RUNS_PER_CASE}, async (_, runIdx) => {
                        const vals = makeRandomVectorByKeccak(
                            dim,
                            -10n,
                            10n,
                            "sec5",
                            "semiWeighted",
                            "fixed-x0",
                            dim,
                            runIdx
                        );

                        return qVec(solver, vals);
                    })
                );

                fixedX0Batches.set(dim, fixedX0Batch);
            }
        });

        for (const dim of DIMS) {
            for (const maxIter of ITERS) {
                const testId = makeTestId(5, ++testIndex);

                it(`Test ${testId}: Semi-weighted gas growth at fixed dimension n=${dim} with maxIter=${maxIter}`, async function () {
                    const objectiveAddr = await semiWeighted.getAddress();
                    const fixedX0Batch = fixedX0Batches.get(dim);

                    if (!fixedX0Batch) {
                        throw new Error(`Missing fixed initial-point batch for dim=${dim}`);
                    }

                    const batch = await runBatch(
                        solver,
                        semiWeighted,
                        objectiveAddr,
                        fixedX0Batch,
                        maxIter,
                        tolTight
                    );

                    batch.statuses.forEach((s) => {
                        expect([
                            STATUS_ZERO_GRADIENT,
                            STATUS_MAX_ITER_EXCEEDED,
                            STATUS_NO_LIKELY_IMPROVEMENT,
                        ]).to.include(s);
                    });

                    const gasStats = summarizeBigint(batch.gasValues);
                    const iterStats = summarizeBigint(batch.iterValues);
                    const gradNormStats = summarizeFiniteNumber(batch.gradNormValues);
                    const xErrorStats = summarizeFiniteNumber(batch.xErrorValues);
                    const fErrorStats = summarizeFiniteNumber(batch.fErrorValues);
                    const xTolCount = countWithinTol(batch.xErrorValues, REPORT_TOL);

                    printBlockOptimization({
                        t: testId,
                        method: "solve",
                        explanation: `Gas growth with respect to iteration budget for semi-weighted objective at fixed dimension n=${dim}.`,
                        gas: gasStats.avg,
                        x0: `sample=${fmtHexArr(batch.inputs[0])}`,
                        xFinal: `sample=${fmtHexArr(batch.finalXs[0])}`,
                        gx: `sample=${batch.finalGxs[0]} (~ ${quadHexToApproxNumber(batch.finalGxs[0])})`,
                        status: statusHistogram(batch.statuses),
                        iters: `avg=${iterStats.avg}, min=${iterStats.min}, max=${iterStats.max}`,
                        gradNorm: `avg=${formatNum(gradNormStats.avg)}, max=${formatNum(gradNormStats.max)}`,
                        xError: `avg=${formatNum(xErrorStats.avg)}, max=${formatNum(xErrorStats.max)}`,
                        fError: `avg=${formatNum(fErrorStats.avg)}, max=${formatNum(fErrorStats.max)}`,
                        withinTol: `${xTolCount.within}/${RUNS_PER_CASE}`,
                        exceededTol: `${xTolCount.exceeded}/${RUNS_PER_CASE}`,
                        tolerance: "1e-12 (xError)",
                        worstCase: makeWorstCaseLabel(gradNormStats, xErrorStats, fErrorStats),
                        extra: `runs=${RUNS_PER_CASE}, objective=semiWeighted, dim=${dim}, maxIter=${maxIter}, gas[min=${gasStats.min}, max=${gasStats.max}]`
                    });
                });
            }
        }
    });
});
