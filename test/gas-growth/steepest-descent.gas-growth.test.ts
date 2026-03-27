// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockOptimization } from "../test-utils";

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

const STATUS_SUCCESS = 0n;
const STATUS_ZERO_GRADIENT = 1n;
const STATUS_NO_LIKELY_IMPROVEMENT = 2n;
const STATUS_MAX_ITER_EXCEEDED = 3n;

// ------------------------------------------------------------
//  Deployment
// ------------------------------------------------------------

async function newHarnesses(): Promise<{
    solver: SteepestDescentHarness;
    quadratic: ObjectiveHarness;
    weighted: ObjectiveHarness;
    spherical: ObjectiveHarness
}> {
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

    const SolverFactory = await ethers.getContractFactory("SteepestDescentHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const QuadraticFactory = await ethers.getContractFactory("QuadraticObjectiveHarness", {
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

    const solver = await SolverFactory.deploy();
    await solver.waitForDeployment();

    const quadratic = await QuadraticFactory.deploy();
    await quadratic.waitForDeployment();

    const weighted = await WeightedFactory.deploy();
    await weighted.waitForDeployment();

    const spherical = await SphericalFactory.deploy();
    await spherical.waitForDeployment();

    return {
        solver: solver as unknown as SteepestDescentHarness,
        quadratic: quadratic as unknown as ObjectiveHarness,
        weighted: weighted as unknown as ObjectiveHarness,
        spherical: spherical as unknown as ObjectiveHarness,
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

function makeVector(dim: number, kind: "ones" | "increasing" | "large" | "mixed" | "zero"): bigint[] {
    if (kind === "zero") return new Array(dim).fill(0n);
    if (kind === "ones") return new Array(dim).fill(1n);
    if (kind === "large") return Array.from({ length: dim }, (_, i) => BigInt((i + 1) * 100));
    if (kind === "mixed") return Array.from({ length: dim }, (_, i) => BigInt(i % 2 === 0 ? i + 1 : -(i + 1)));
    return Array.from({ length: dim }, (_, i) => BigInt(i + 1));
}

async function qVec(
    solver: SteepestDescentHarness,
    vals: bigint[]
): Promise<string[]> {
    return Promise.all(vals.map(v => solver.qFromInt(v)));
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("SteepestDescent - Gas Growth Tests", function () {
    let solver: SteepestDescentHarness;
    let quadratic: ObjectiveHarness;
    let weighted: ObjectiveHarness;
    let spherical: ObjectiveHarness;
    let t = 0;

    let tolTight: string;
    let tolZero: string;

    before(async () => {
        const deployed = await newHarnesses();
        solver = deployed.solver;
        quadratic = deployed.quadratic;
        weighted = deployed.weighted;
        spherical = deployed.spherical;

        tolTight = await solver.qFromFrac(1n, 1_000_000_000_000_000_000n); // 1e-18
        tolZero = await solver.qFromInt(0n);
    });

    // --------------------------------------------------------
    //  Section 1: Gas Growth with Respect to Dimension
    // --------------------------------------------------------

    describe("Section 1: Gas Growth with Respect to Dimension", function () {
        const DIMS = [2, 4, 8, 16];

        for (const dim of DIMS) {
            it(`Test ${++t}: Quadratic objective gas growth with dimension n=${dim}`, async function () {
                const x0 = await qVec(solver, makeVector(dim, "increasing"));
                const objectiveAddr = await quadratic.getAddress();

                await touchGas(solver, "solve", [objectiveAddr, x0, 10n, tolTight]);
                const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, 10n, tolTight]);
                const out = asResult(await solver.solve(objectiveAddr, x0, 10n, tolTight));

                expect([STATUS_SUCCESS, STATUS_MAX_ITER_EXCEEDED]).to.include(out.status);

                printBlockOptimization({
                    t,
                    method: "solve",
                    explanation: `Gas growth with respect to dimension for spherical quadratic objective in n=${dim}.`,
                    gas,
                    x0: fmtHexArr(x0),
                    xFinal: fmtHexArr(out.x),
                    gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                    status: `${out.status}`,
                    iters: `${out.iters}`,
                    extra: `objective=quadratic, dim=${dim}`
                });
            });
        }

        for (const dim of DIMS) {
            it(`Test ${++t}: Weighted quadratic objective gas growth with dimension n=${dim}`, async function () {
                const x0 = await qVec(solver, makeVector(dim, "increasing"));
                const objectiveAddr = await weighted.getAddress();

                await touchGas(solver, "solve", [objectiveAddr, x0, 20n, tolTight]);
                const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, 20n, tolTight]);
                const out = asResult(await solver.solve(objectiveAddr, x0, 20n, tolTight));

                expect([STATUS_SUCCESS, STATUS_MAX_ITER_EXCEEDED, STATUS_NO_LIKELY_IMPROVEMENT]).to.include(out.status);

                printBlockOptimization({
                    t,
                    method: "solve",
                    explanation: `Gas growth with respect to dimension for weighted quadratic objective in n=${dim}.`,
                    gas,
                    x0: fmtHexArr(x0),
                    xFinal: fmtHexArr(out.x),
                    gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                    status: `${out.status}`,
                    iters: `${out.iters}`,
                    extra: `objective=weighted, dim=${dim}`
                });
            });
        }
    });

    // --------------------------------------------------------
    //  Section 2: Gas Sensitivity to Iteration Budget
    // --------------------------------------------------------

    describe("Section 2: Gas Sensitivity to Iteration Budget", function () {
        const ITERS = [1n, 2n, 5n, 10n, 20n];

        for (const maxIter of ITERS) {
            it(`Test ${++t}: Weighted quadratic gas sensitivity with maxIter=${maxIter}`, async function () {
                const x0 = await qVec(solver, [1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n]);
                const objectiveAddr = await weighted.getAddress();

                await touchGas(solver, "solve", [objectiveAddr, x0, maxIter, tolTight]);
                const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, maxIter, tolTight]);
                const out = asResult(await solver.solve(objectiveAddr, x0, maxIter, tolTight));

                printBlockOptimization({
                    t,
                    method: "solve",
                    explanation: `Gas sensitivity to iteration budget for weighted quadratic objective with fixed dimension n=8 and maxIter=${maxIter}.`,
                    gas,
                    x0: fmtHexArr(x0),
                    xFinal: fmtHexArr(out.x),
                    gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                    status: `${out.status}`,
                    iters: `${out.iters}`,
                    extra: `objective=weighted`
                });
            });
        }
    });

    // --------------------------------------------------------
    //  Section 3: Gas Sensitivity to Initial Magnitude
    // --------------------------------------------------------

    describe("Section 3: Gas Sensitivity to Initial Point Magnitude", function () {
        type MagnitudeCase = {
            label: string;
            values: number[];
        };

        const cases: MagnitudeCase[] = [
            { label: "small", values: [1, 1, 1, 1, 1, 1, 1, 1] },
            { label: "medium", values: [10, 10, 10, 10, 10, 10, 10, 10] },
            { label: "large", values: [100, 100, 100, 100, 100, 100, 100, 100] },
            { label: "veryLarge", values: [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] },
            { label: "mixedLarge", values: [-1000, 1000, -1000, 1000, -1000, 1000, -1000, 1000] }
        ];

        for (const p of cases) {
            it(`Test ${++t}: Spherical quadratic gas sensitivity for initial magnitude ${p.label}`, async function () {
                const x0 = await Promise.all(p.values.map((v) => solver.qFromInt(BigInt(v))));
                const tol = await solver.qFromFrac(1n, 1000000000000n); // 1e-12
                const maxIter = 20n;

                const objectiveAddr = await spherical.getAddress();

                await touchGas(solver, "solve", [objectiveAddr, x0, maxIter, tol]);
                const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, maxIter, tol]);

                const out = asResult(await solver.solve(objectiveAddr, x0, maxIter, tol));

                expect(out.status).to.eq(STATUS_SUCCESS);

                printBlockOptimization({
                    t,
                    method: "solve",
                    explanation: `Gas sensitivity to initial point magnitude for spherical quadratic objective using ${p.label} starting vector.`,
                    gas,
                    x0: fmtHexArr(x0),
                    xFinal: fmtHexArr(out.x),
                    gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                    status: `${out.status}`,
                    iters: `${out.iters}`,
                    extra: `objective=spherical, magnitude=${p.label}`
                });
            });
        }
    });

    // --------------------------------------------------------
    //  Section 4: Gas Sensitivity to Initial Pattern
    // --------------------------------------------------------

    describe("Section 4: Gas Sensitivity to Initial Point Pattern", function () {
        type PatternCase = {
            label: string;
            values: number[];
        };

        const cases: PatternCase[] = [
            { label: "ones", values: [5, 5, 5, 5, 5, 5, 5, 5] },
            { label: "increasing", values: [1, 2, 3, 4, 5, 6, 7, 8] },
            { label: "mixedSign", values: [1, -2, 3, -4, 5, -6, 7, -8] },
            { label: "singleSpike", values: [20, 0, 0, 0, 0, 0, 0, 0] },
            { label: "twoBlock", values: [8, 8, 8, 8, -8, -8, -8, -8] }
        ];

        for (const p of cases) {
            it(`Test ${++t}: Spherical quadratic gas sensitivity for initial pattern ${p.label}`, async function () {
                const x0 = await Promise.all(p.values.map((v) => solver.qFromInt(BigInt(v))));
                const tol = await solver.qFromFrac(1n, 1000000000000n); // 1e-12
                const maxIter = 20n;

                const objectiveAddr = await spherical.getAddress();

                await touchGas(solver, "solve", [objectiveAddr, x0, maxIter, tol]);
                const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, maxIter, tol]);

                const out = asResult(await solver.solve(objectiveAddr, x0, maxIter, tol));

                expect(out.status).to.eq(STATUS_SUCCESS);

                printBlockOptimization({
                    t,
                    method: "solve",
                    explanation: `Gas sensitivity to initial point pattern for spherical quadratic objective using ${p.label} vector.`,
                    gas,
                    x0: fmtHexArr(x0),
                    xFinal: fmtHexArr(out.x),
                    gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                    status: `${out.status}`,
                    iters: `${out.iters}`,
                    extra: `objective=spherical, pattern=${p.label}`
                });
            });
        }
    });
    // --------------------------------------------------------
    //  Section 5: Gas Sensitivity to Objective Structure
    // --------------------------------------------------------

    describe("Section 5: Gas Sensitivity to Objective Structure", function () {
        it(`Test ${++t}: Quadratic vs weighted quadratic comparison on same start`, async function () {
            const x0 = await qVec(solver, [3n, 4n, 5n, 6n, 7n, 8n]);
            const quadraticAddr = await quadratic.getAddress();
            const weightedAddr = await weighted.getAddress();

            await touchGas(solver, "solve", [quadraticAddr, x0, 20n, tolTight]);
            const gasQuadratic = await estimateGas(solver, "solve", [quadraticAddr, x0, 20n, tolTight]);
            const outQuadratic = asResult(await solver.solve(quadraticAddr, x0, 20n, tolTight));

            await touchGas(solver, "solve", [weightedAddr, x0, 20n, tolTight]);
            const gasWeighted = await estimateGas(solver, "solve", [weightedAddr, x0, 20n, tolTight]);
            const outWeighted = asResult(await solver.solve(weightedAddr, x0, 20n, tolTight));

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Gas sensitivity to objective structure by comparing spherical and weighted quadratic objectives on the same initial point.",
                gas: `${gasQuadratic} / ${gasWeighted}`,
                x0: fmtHexArr(x0),
                xFinal: `quadratic=${fmtHexArr(outQuadratic.x)} | weighted=${fmtHexArr(outWeighted.x)}`,
                gx: `quadratic=${outQuadratic.gx} (~ ${quadHexToApproxNumber(outQuadratic.gx)}) | weighted=${outWeighted.gx} (~ ${quadHexToApproxNumber(outWeighted.gx)})`,
                status: `quadratic=${outQuadratic.status} | weighted=${outWeighted.status}`,
                iters: `quadratic=${outQuadratic.iters} | weighted=${outWeighted.iters}`,
                extra: `gasQuadratic=${gasQuadratic}, gasWeighted=${gasWeighted}`
            });
        });
    });

    // --------------------------------------------------------
    //  Section 6: Gas Sensitivity to Termination Path
    // --------------------------------------------------------

    describe("Section 6: Gas Sensitivity to Termination Path", function () {
        it(`Test ${++t}: Success path`, async function () {
            const x0 = await qVec(solver, [3n, 4n]);
            const objectiveAddr = await quadratic.getAddress();

            await touchGas(solver, "solve", [objectiveAddr, x0, 10n, tolTight]);
            const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, 10n, tolTight]);
            const out = asResult(await solver.solve(objectiveAddr, x0, 10n, tolTight));

            expect(out.status).to.equal(STATUS_SUCCESS);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Gas for normal success path.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it(`Test ${++t}: Zero-gradient path`, async function () {
            const x0 = await qVec(solver, [0n, 0n, 0n, 0n]);
            const objectiveAddr = await quadratic.getAddress();

            await touchGas(solver, "solve", [objectiveAddr, x0, 10n, tolTight]);
            const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, 10n, tolTight]);
            const out = asResult(await solver.solve(objectiveAddr, x0, 10n, tolTight));

            expect([STATUS_SUCCESS, STATUS_ZERO_GRADIENT]).to.include(out.status);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Gas for zero-gradient / already-optimal path.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it(`Test ${++t}: Max-iteration-exceeded path`, async function () {
            const x0 = await qVec(solver, [3n, 4n, 5n, 6n]);
            const objectiveAddr = await weighted.getAddress();

            await touchGas(solver, "solve", [objectiveAddr, x0, 1n, tolTight]);
            const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, 1n, tolTight]);
            const out = asResult(await solver.solve(objectiveAddr, x0, 1n, tolTight));

            expect(out.status).to.equal(STATUS_MAX_ITER_EXCEEDED);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Gas for forced max-iteration-exceeded path using maxIter=1.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it(`Test ${++t}: Tolerance fallback path (tol=0)`, async function () {
            const x0 = await qVec(solver, [6n, 8n, 10n]);
            const objectiveAddr = await weighted.getAddress();

            await touchGas(solver, "solve", [objectiveAddr, x0, 20n, tolZero]);
            const gas = await estimateGas(solver, "solve", [objectiveAddr, x0, 20n, tolZero]);
            const out = asResult(await solver.solve(objectiveAddr, x0, 20n, tolZero));

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Gas when explicit tolerance is zero and fallback tolerance path is used.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`,
                extra: "tol=0"
            });
        });
    });
});