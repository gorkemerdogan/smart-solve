// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockOptimization } from "../test-utils";

/**
 * @title  Steepest Descent: General Optimization using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive tests for the textbook-style steepest descent solver
 *         using a simple quadratic objective g(x) = sum_i x_i^2 and grad g(x) = 2x.
 *         Includes convergence checks, deterministic behavior, tolerance fallback,
 *         and edge cases such as zero-gradient start, max-iteration cutoff, and invalid input.
 */

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

type QuadraticObjectiveHarness = Contract & {
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
    objective: QuadraticObjectiveHarness;
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

    const solver = await SolverFactory.deploy();
    await solver.waitForDeployment();

    const ObjectiveFactory = await ethers.getContractFactory("QuadraticObjectiveHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const objective = await ObjectiveFactory.deploy();
    await objective.waitForDeployment();

    return {
        solver: solver as unknown as SteepestDescentHarness,
        objective: objective as unknown as QuadraticObjectiveHarness,
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

/**
 * Approximate IEEE-754 binary128 (quad) hex -> JS number.
 * Good enough for readable debug / sanity checks.
 */
function quadHexToApproxNumber(hex: string): number {
    const bits = BigInt(hex);
    const sign = ((bits >> 127n) & 1n) === 1n ? -1 : 1;
    const exp = Number((bits >> 112n) & 0x7fffn);
    const fracMask = (1n << 112n) - 1n;
    const frac = bits & fracMask;
    const bias = 16383;

    if (exp === 0 && frac === 0n) return sign * 0;

    if (exp === 0x7fff) {
        if (frac === 0n) return sign * Infinity;
        return NaN;
    }

    const fracApprox = Number(frac) / Math.pow(2, 112);

    if (exp === 0) {
        return sign * fracApprox * Math.pow(2, 1 - bias);
    }

    const mantissa = 1 + fracApprox;
    return sign * mantissa * Math.pow(2, exp - bias);
}

function expectNearMinimum(out: { x: string[]; gx: string }) {
    const gx = quadHexToApproxNumber(out.gx);
    expect(gx).to.be.lessThan(1e-20);

    for (let i = 0; i < out.x.length; ++i) {
        const xi = quadHexToApproxNumber(out.x[i]);
        expect(Math.abs(xi)).to.be.lessThan(1e-10);
    }
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("SteepestDescent — General Solver", function () {
    let solver: SteepestDescentHarness;
    let objective: QuadraticObjectiveHarness;
    let t = 0;
    let tol: string;

    beforeEach(async () => {
        const deployed = await newHarnesses();
        solver = deployed.solver;
        objective = deployed.objective;
        tol = await solver.qFromFrac(1n, 1_000_000_000_000_000_000n); // 1e-18
    });

    const qInt = async (n: number | string | bigint) =>
        await solver.qFromInt(BigInt(n));

    describe("Section 1: Steepest Descent", function () {

        it("Test 1: converges from [3,4] to the minimizer of x1^2 + x2^2", async function () {
            t++;

            const x0 = [await qInt(3), await qInt(4)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();
            const g0 = await objective.g(x0);

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Steepest descent minimizes g(x)=x1^2+x2^2 starting from [3,4].",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`,
                extra: `g0=${g0} (~ ${quadHexToApproxNumber(g0)})`
            });
        });

        it("Test 2: converges from [-5,12] with mixed signs", async function () {
            t++;

            const x0 = [await qInt(-5), await qInt(12)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Handles a 2D starting point with negative and positive components.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it("Test 3: converges from 3D point [8,15,0]", async function () {
            t++;

            const x0 = [await qInt(8), await qInt(15), await qInt(0)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Converges in 3D for the spherical quadratic objective.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it("Test 4: converges from large-magnitude point [7,24]", async function () {
            t++;

            const x0 = [await qInt(7), await qInt(24)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Works correctly for a larger-norm starting point.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it("Test 5: objective value decreases from initial point [9,12]", async function () {
            t++;

            const x0 = [await qInt(9), await qInt(12)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            const g0 = await objective.g(x0);

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            const g0Approx = quadHexToApproxNumber(g0);
            const gFinalApprox = quadHexToApproxNumber(out.gx);

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(gFinalApprox).to.be.lessThan(g0Approx);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Verifies that the solver reduces the quadratic objective value.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${gFinalApprox})`,
                status: `${out.status}`,
                iters: `${out.iters}`,
                extra: `g0=${g0} (~ ${g0Approx})`
            });
        });

        it("Test 6: tol = 0 triggers fallback tolerance path", async function () {
            t++;

            const x0 = [await qInt(6), await qInt(8)];
            const zeroTol = await qInt(0);
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, zeroTol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, zeroTol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, zeroTol));

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Uses the library/config default tolerance when explicit tol = 0.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`,
                extra: "tol=0 => fallback path"
            });
        });

        it("Test 7: deterministic output for repeated solve on [9,40]", async function () {
            t++;

            const x0 = [await qInt(9), await qInt(40)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out1 = asResult(await solver.solve(objAddr, x0, maxIter, tol));
            const out2 = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out1.status).to.equal(out2.status);
            expect(out1.iters).to.equal(out2.iters);
            expect(out1.gx.toLowerCase()).to.equal(out2.gx.toLowerCase());
            expect(out1.x.length).to.equal(out2.x.length);

            for (let i = 0; i < out1.x.length; ++i) {
                expect(out1.x[i].toLowerCase()).to.equal(out2.x[i].toLowerCase());
            }

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Repeated calls with the same input must produce identical output.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out1.x),
                gx: `${out1.gx} (~ ${quadHexToApproxNumber(out1.gx)})`,
                status: `${out1.status}`,
                iters: `${out1.iters}`,
                extra: `run2.x=${fmtHexArr(out2.x)}, run2.gx=${out2.gx}, run2.status=${out2.status}, run2.iters=${out2.iters}`
            });
        });

        it("Test 8: converges from 4D point [1,2,2,1]", async function () {
            t++;

            const x0 = [await qInt(1), await qInt(2), await qInt(2), await qInt(1)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "Extends the same minimization logic to a 4D vector.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        // --------------------------------------------------------
        //  Edge Cases
        // --------------------------------------------------------

        it("Test 9 (Edge): zero-gradient start returns success immediately", async function () {
            t++;

            const x0 = [await qInt(0), await qInt(0)];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            const qZero = await qInt(0);
            expect(out.status).to.equal(STATUS_SUCCESS);
            expect(out.iters).to.equal(0n);
            expect(out.gx.toLowerCase()).to.equal(qZero.toLowerCase());
            expect(out.x[0].toLowerCase()).to.equal(qZero.toLowerCase());
            expect(out.x[1].toLowerCase()).to.equal(qZero.toLowerCase());

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "If the initial point is already optimal, the solver exits immediately with success.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it("Test 10 (Edge): maxIter = 1 returns STATUS_MAX_ITER_EXCEEDED after one update", async function () {
            t++;

            const x0 = [await qInt(3), await qInt(4)];
            const maxIter = 1n;
            const objAddr = await objective.getAddress();

            await touchGas(solver, "solve", [objAddr, x0, maxIter, tol]);
            const gas = await estimateGas(solver, "solve", [objAddr, x0, maxIter, tol]);

            const out = asResult(await solver.solve(objAddr, x0, maxIter, tol));

            expect(out.status).to.equal(STATUS_MAX_ITER_EXCEEDED);
            expect(out.iters).to.equal(1n);
            expectNearMinimum(out);

            printBlockOptimization({
                t,
                method: "solve",
                explanation: "With maxIter = 1, the solver performs one update then exits due to iteration limit.",
                gas,
                x0: fmtHexArr(x0),
                xFinal: fmtHexArr(out.x),
                gx: `${out.gx} (~ ${quadHexToApproxNumber(out.gx)})`,
                status: `${out.status}`,
                iters: `${out.iters}`
            });
        });

        it("Test 11 (Edge): empty initial vector reverts", async function () {
            t++;

            const x0: string[] = [];
            const maxIter = 10n;
            const objAddr = await objective.getAddress();

            await expect(
                solver.solve(objAddr, x0, maxIter, tol)
            ).to.be.revertedWith("SteepestDescent: empty initial point");
        });
    });
});