// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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
    grad(x: string[]): Promise<string[]>;
};

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

const STATUS_SUCCESS = 0n;
const STATUS_ZERO_GRADIENT = 1n;
const STATUS_NO_LIKELY_IMPROVEMENT = 2n;
const STATUS_MAX_ITER_EXCEEDED = 3n;

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

function fmtVec(v: bigint[]): string {
    return `[${v.map(formatScaledInt).join(", ")}]`;
}

async function scaledVec(harness: { toFloat: (q: string) => Promise<unknown> }, values: string[]): Promise<bigint[]> {
    return Promise.all(values.map(async v => asBigInt(await harness.toFloat(v))));
}

function printSDResult(args: {
    t: string;
    method: string;
    explanation: string;
    objective: string;
    x0: string;
    expectedX: string;
    outputX: string;
    distanceToOptimum: string;
    finalObjective: string;
    iterations: string;
    status: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Objective: ${args.objective}`);
    console.log(`Initial x0: ${args.x0}`);
    console.log(`Expected Minimizer: ${args.expectedX}`);
    console.log(`Output x: ${args.outputX}`);
    console.log(`Distance to Optimum (inf-norm): ${args.distanceToOptimum}`);
    console.log(`Final Objective Value: ${args.finalObjective}`);
    console.log(`Iterations: ${args.iterations}`);
    console.log(`Status: ${args.status}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("SteepestDescent Library - Numerical Accuracy Tests", function () {
    let solver: SteepestDescentHarness & { toFloat(q: string): Promise<unknown> };
    let spherical: ObjectiveHarness;
    let weighted: ObjectiveHarness;
    let semiWeighted: ObjectiveHarness;

    let TOL_1E_9: string;

    const qInt = async (x: number | bigint) => await solver.qFromInt(x);

    async function qVecFromInts(vals: Array<number | bigint>): Promise<string[]> {
        return Promise.all(vals.map(v => qInt(v)));
    }

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
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

        const WeightedFactory = await ethers.getContractFactory("WeightedQuadraticObjectiveHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });
        weighted = (await WeightedFactory.deploy()) as unknown as ObjectiveHarness;

        const SemiFactory = await ethers.getContractFactory("SemiWeightedQuadraticObjectiveHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });
        semiWeighted = (await SemiFactory.deploy()) as unknown as ObjectiveHarness;

        TOL_1E_9 = await solver.qFromFrac(1n, 1_000_000_000n);

        const oneScaled = asBigInt(await solver.toFloat(await solver.qFromInt(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    // ------------------------------------------------------------
    // Section 1: Spherical quadratic objective
    // ------------------------------------------------------------

    describe("Section 1: Spherical objective benchmark", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: solver converges close to zero on spherical objective`, async function () {
            const x0 = await qVecFromInts([4n, -3n, 2n, -1n]);
            const xTrue = [0n, 0n, 0n, 0n];

            const [xRaw, gxRaw, iters, status] = await solver.solve(
                await spherical.getAddress(),
                x0,
                300n,
                TOL_1E_9
            );

            const xComp = await scaledVec(solver, xRaw);
            const gx = asBigInt(await solver.toFloat(gxRaw));

            const dist = vecInfNorm(subVec(xComp, xTrue));

            printSDResult({
                t: `1.${testNo}`,
                method: "Steepest Descent",
                explanation: "For the spherical quadratic objective, the minimizer is the zero vector and the solver should converge close to it.",
                objective: "g(x)=sum x_i^2",
                x0: "[4,-3,2,-1]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                distanceToOptimum: formatScaledInt(dist),
                finalObjective: formatScaledInt(gx),
                iterations: iters.toString(),
                status: status.toString(),
            });

            expect(dist < 100_000n).to.equal(true);
            expect(absBigInt(gx) < 100_000n).to.equal(true);
            expect(
                status === STATUS_SUCCESS ||
                status === STATUS_ZERO_GRADIENT ||
                status === STATUS_NO_LIKELY_IMPROVEMENT
            ).to.equal(true);
        });

        it(`Test 1.${++testNo}: zero initial point is recognized as optimum on spherical objective`, async function () {
            const x0 = await qVecFromInts([0n, 0n, 0n, 0n]);
            const xTrue = [0n, 0n, 0n, 0n];

            const [xRaw, gxRaw, iters, status] = await solver.solve(
                await spherical.getAddress(),
                x0,
                50n,
                TOL_1E_9
            );

            const xComp = await scaledVec(solver, xRaw);
            const gx = asBigInt(await solver.toFloat(gxRaw));
            const dist = vecInfNorm(subVec(xComp, xTrue));

            printSDResult({
                t: `1.${testNo}`,
                method: "Steepest Descent",
                explanation: "If the initial point is already the minimizer, the solver should stop immediately or nearly immediately.",
                objective: "g(x)=sum x_i^2",
                x0: "[0,0,0,0]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                distanceToOptimum: formatScaledInt(dist),
                finalObjective: formatScaledInt(gx),
                iterations: iters.toString(),
                status: status.toString(),
            });

            expect(dist).to.equal(0n);
            expect(absBigInt(gx)).to.equal(0n);
            expect(
                status === STATUS_ZERO_GRADIENT ||
                status === STATUS_SUCCESS
            ).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Weighted objectives
    // ------------------------------------------------------------

    describe("Section 2: Weighted objective benchmarks", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: solver converges on softly weighted quadratic objective`, async function () {
            const x0 = await qVecFromInts([5n, -4n, 3n, -2n, 1n, -1n, 2n, -3n]);
            const xTrue = new Array<bigint>(8).fill(0n);

            const [xRaw, gxRaw, iters, status] = await solver.solve(
                await semiWeighted.getAddress(),
                x0,
                500n,
                TOL_1E_9
            );

            const xComp = await scaledVec(solver, xRaw);
            const gx = asBigInt(await solver.toFloat(gxRaw));
            const dist = vecInfNorm(subVec(xComp, xTrue));

            printSDResult({
                t: `2.${testNo}`,
                method: "Steepest Descent",
                explanation: "The semi-weighted quadratic objective remains anisotropic, but should still converge close to the zero minimizer.",
                objective: "semi-weighted quadratic",
                x0: "[5,-4,3,-2,1,-1,2,-3]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xComp),
                distanceToOptimum: formatScaledInt(dist),
                finalObjective: formatScaledInt(gx),
                iterations: iters.toString(),
                status: status.toString(),
            });

            expect(dist < 100_000_000n).to.equal(true);
            expect(absBigInt(gx) < 100_000_000n).to.equal(true);
            expect(
                status === STATUS_SUCCESS ||
                status === STATUS_ZERO_GRADIENT ||
                status === STATUS_NO_LIKELY_IMPROVEMENT
            ).to.equal(true);
        });

        it(`Test 2.${++testNo}: strongly weighted quadratic is harder than spherical`, async function () {
            const x0 = await qVecFromInts([5n, -4n, 3n, -2n]);

            const [xSRaw, gSRaw, itS] = await solver.solve(
                await spherical.getAddress(),
                x0,
                300n,
                TOL_1E_9
            );

            const [xWRaw, gWRaw, itW] = await solver.solve(
                await weighted.getAddress(),
                x0,
                300n,
                TOL_1E_9
            );

            const xS = await scaledVec(solver, xSRaw);
            const xW = await scaledVec(solver, xWRaw);

            const distS = vecInfNorm(xS);
            const distW = vecInfNorm(xW);

            const gS = absBigInt(asBigInt(await solver.toFloat(gSRaw)));
            const gW = absBigInt(asBigInt(await solver.toFloat(gWRaw)));

            printSDResult({
                t: `2.${testNo}.1`,
                method: "Steepest Descent",
                explanation: "Spherical objective baseline.",
                objective: "g(x)=sum x_i^2",
                x0: "[5,-4,3,-2]",
                expectedX: fmtVec(new Array<bigint>(4).fill(0n)),
                outputX: fmtVec(xS),
                distanceToOptimum: formatScaledInt(distS),
                finalObjective: formatScaledInt(gS),
                iterations: itS.toString(),
                status: "baseline",
            });

            printSDResult({
                t: `2.${testNo}.2`,
                method: "Steepest Descent",
                explanation: "Strongly weighted quadratic benchmark.",
                objective: "weighted quadratic",
                x0: "[5,-4,3,-2]",
                expectedX: fmtVec(new Array<bigint>(4).fill(0n)),
                outputX: fmtVec(xW),
                distanceToOptimum: formatScaledInt(distW),
                finalObjective: formatScaledInt(gW),
                iterations: itW.toString(),
                status: "comparison",
            });

            expect(distW >= distS || gW >= gS).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Initial-point robustness
    // ------------------------------------------------------------

    describe("Section 3: Initial-point robustness", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: different initial points still converge close to the same minimizer`, async function () {
            const x0A = await qVecFromInts([2n, -2n, 1n, -1n]);
            const x0B = await qVecFromInts([10n, -10n, 5n, -5n]);
            const xTrue = [0n, 0n, 0n, 0n];

            const [xARaw, gARaw] = await solver.solve(
                await spherical.getAddress(),
                x0A,
                300n,
                TOL_1E_9
            );

            const [xBRaw, gBRaw] = await solver.solve(
                await spherical.getAddress(),
                x0B,
                300n,
                TOL_1E_9
            );

            const xA = await scaledVec(solver, xARaw);
            const xB = await scaledVec(solver, xBRaw);

            const distA = vecInfNorm(subVec(xA, xTrue));
            const distB = vecInfNorm(subVec(xB, xTrue));
            const between = vecInfNorm(subVec(xA, xB));

            const gA = absBigInt(asBigInt(await solver.toFloat(gARaw)));
            const gB = absBigInt(asBigInt(await solver.toFloat(gBRaw)));

            printSDResult({
                t: `3.${testNo}.1`,
                method: "Steepest Descent",
                explanation: "Result from moderate initial point.",
                objective: "g(x)=sum x_i^2",
                x0: "[2,-2,1,-1]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xA),
                distanceToOptimum: formatScaledInt(distA),
                finalObjective: formatScaledInt(gA),
                iterations: "N/A",
                status: "A",
            });

            printSDResult({
                t: `3.${testNo}.2`,
                method: "Steepest Descent",
                explanation: "Result from farther initial point.",
                objective: "g(x)=sum x_i^2",
                x0: "[10,-10,5,-5]",
                expectedX: fmtVec(xTrue),
                outputX: fmtVec(xB),
                distanceToOptimum: formatScaledInt(distB),
                finalObjective: formatScaledInt(gB),
                iterations: "N/A",
                status: "B",
            });

            expect(distA < 100_000n).to.equal(true);
            expect(distB < 100_000n).to.equal(true);
            expect(between < 100_000n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Objective decrease from initial point
    // ------------------------------------------------------------

    describe("Section 4: Objective decrease", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: final objective is lower than initial objective on spherical benchmark`, async function () {
            const x0 = await qVecFromInts([6n, -5n, 4n, -3n]);

            const g0Raw = await spherical.g(x0);
            const g0 = asBigInt(await solver.toFloat(g0Raw));

            const [xRaw, gFinalRaw, iters, status] = await solver.solve(
                await spherical.getAddress(),
                x0,
                300n,
                TOL_1E_9
            );

            const xComp = await scaledVec(solver, xRaw);
            const gFinal = asBigInt(await solver.toFloat(gFinalRaw));

            printSDResult({
                t: `4.${testNo}`,
                method: "Steepest Descent",
                explanation: "The optimization process should reduce the objective value relative to the starting point.",
                objective: "g(x)=sum x_i^2",
                x0: "[6,-5,4,-3]",
                expectedX: fmtVec([0n, 0n, 0n, 0n]),
                outputX: fmtVec(xComp),
                distanceToOptimum: formatScaledInt(vecInfNorm(xComp)),
                finalObjective: `${formatScaledInt(gFinal)} (initial=${formatScaledInt(g0)})`,
                iterations: iters.toString(),
                status: status.toString(),
            });

            expect(absBigInt(gFinal) <= absBigInt(g0)).to.equal(true);
        });
    });
});