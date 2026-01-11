// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests covers essential vector utilities and numerical iterative methods: random vector, normalization, convergence, power iteration
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

    // normalization
    euclideanNormHarness(vRows: bigint, vCols: bigint, vData: string[]): Promise<string>;
    normalizeHarness(vRows: bigint, vCols: bigint, vData: string[]): Promise<[bigint, bigint, string[]]>;

    // create vector
    createVectorHarness(data: string[]): Promise<[bigint, bigint, string[]]>;

    // random vector
    randomVectorHarness(n: bigint, seed: string): Promise<[bigint, bigint, string[]]>;

    // convergence
    hasConvergedHarness(xRows: bigint, xCols: bigint, xData: string[], yRows: bigint, yCols: bigint, yData: string[], tol: string): Promise<boolean>;

    // power iteration
    powerIterationHarness(aRows: bigint, aCols: bigint, aData: string[], seed: string, tol: string): Promise<[string, bigint, bigint, string[]]>;
};

async function newHarness(): Promise<MatrixMasterHarness> {
    // Deploy MathLib (library)
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();
    const mathAddr = await math.getAddress();

    // Deploy MatrixMasterHarness with linked MathLib
    const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
        libraries: {
            "contracts/libraries/MathLib.sol:MathLib": mathAddr,
        },
    });

    const harness = await Factory.deploy();
    await harness.waitForDeployment();
    return harness as unknown as MatrixMasterHarness;
}

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

function asMatrix(tuple: [bigint, bigint, string[]]) {
    const [rows, cols, data] = tuple;
    return { rows, cols, data: [...data] };
}

const fmtHexArr = (arr: string[]) => `[${arr.join(", ")}]`;

// ABDK quad zero can appear as +0 or -0 at the bit level.
// These two encodings are numerically equivalent.
const QUAD_POS_ZERO = "0x00000000000000000000000000000000";
const QUAD_NEG_ZERO = "0x80000000000000000000000000000000";

function isQuadZero(hex: string): boolean {
    const h = hex.toLowerCase();
    return h === QUAD_POS_ZERO || h === QUAD_NEG_ZERO;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("MatrixMaster (library) : norm, random, convergence, power iteration over ABDK quad", function () {
    let harness: MatrixMasterHarness;
    let t = 0; // global test counter for reporting

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    const qInt = async (n: number | string | bigint): Promise<string> =>
        harness.qFromInt(BigInt(n));

    // ------------------------------------------------------------
    //  Section 11: Norm & Normalize
    // ------------------------------------------------------------

    describe("Section 11: Norm & Normalize", function () {
        it("Test 1: euclideanNorm of [3,4] is 5", async function () {
            t++;
            const v = [await qInt(3), await qInt(4)];

            await touchGas(harness, "euclideanNormHarness", [2n, 1n, v]);
            const gas = await estimateGas(harness, "euclideanNormHarness", [2n, 1n, v]);
            const out = await harness.euclideanNormHarness(2n, 1n, v);

            expect(out.toLowerCase()).to.equal((await qInt(5)).toLowerCase());

            printBlockMatrix({
                t,
                method: "euclideanNormHarness",
                explanation: "Computes L2 Norm ||[3,4]|| = 5.",
                gas,
                shapeIn: "2x1",
                shapeOut: "scalar",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: out
            });
        });

        it("Test 2: euclideanNorm reverts on row vector", async function () {
            t++;
            const v = [await qInt(1), await qInt(1)];

            await expect(harness.euclideanNormHarness(1n, 2n, v)).to.be.revertedWith("MatrixMaster: euclideanNorm requires column vector");

            printBlockMatrix({
                t,
                method: "euclideanNormHarness",
                explanation: "Rejects row vectors (1xN).",
                gas: "Revert",
                shapeIn: "1x2",
                shapeOut: "revert",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: "-"
            });
        });

        it("Test 3: normalize scales [3,0,4] to unit length", async function () {
            t++;
            const v = [await qInt(3), await qInt(0), await qInt(4)];

            await touchGas(harness, "normalizeHarness", [3n, 1n, v]);
            const gas = await estimateGas(harness, "normalizeHarness", [3n, 1n, v]);
            const out = asMatrix(await harness.normalizeHarness(3n, 1n, v));

            const ex0 = await harness.qFromFrac(3n, 5n);
            const ex2 = await harness.qFromFrac(4n, 5n);

            expect(out.data[0].toLowerCase()).to.equal(ex0.toLowerCase());
            expect(out.data[2].toLowerCase()).to.equal(ex2.toLowerCase());

            printBlockMatrix({
                t,
                method: "normalizeHarness",
                explanation: "Normalizes vector to unit length.",
                gas,
                shapeIn: "3x1",
                shapeOut: "3x1",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: fmtHexArr(out.data)
            });
        });

        it("Test 4: normalize reverts on zero vector", async function () {
            t++;
            const v = [await qInt(0), await qInt(0)];

            await expect(harness.normalizeHarness(2n, 1n, v)).to.be.revertedWith("MatrixMaster: cannot normalize zero vector");

            printBlockMatrix({
                t,
                method: "normalizeHarness",
                explanation: "Rejects zero vector normalization (div by zero).",
                gas: "Revert",
                shapeIn: "2x1",
                shapeOut: "revert",
                inHex: "Zeros",
                outHex: "-"
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 12: Random Vector & Convergence
    // ------------------------------------------------------------

    describe("Section 12: Creation & Random & Convergence", function () {
        it("Test 5: createVector creates n×1 column vector", async function () {
            t++;
            const v = [await qInt(1), await qInt(2), await qInt(3)];

            await touchGas(harness, "createVectorHarness", [v]);
            const gas = await estimateGas(harness, "createVectorHarness", [v]);
            const [rows, cols, data] = await harness.createVectorHarness(v);

            expect(rows).to.equal(3n);
            expect(cols).to.equal(1n);

            printBlockMatrix({
                t,
                method: "createVectorHarness",
                explanation: "Wraps array into Column Vector (Nx1).",
                gas,
                shapeIn: "3",
                shapeOut: "3x1",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: fmtHexArr(data)
            });
        });

        it("Test 6: createVector reverts on empty array", async function () {
            t++;
            const v: string[] = [];

            await expect(harness.createVectorHarness(v)).to.be.revertedWith("MatrixMaster: empty vector");

            printBlockMatrix({
                t,
                method: "createVectorHarness",
                explanation: "Rejects empty input array.",
                gas: "Revert",
                shapeIn: "0",
                shapeOut: "revert",
                inHex: "[]",
                outHex: "-"
            });
        });

        it("Test 7: randomVector deterministic generation", async function () {
            t++;
            const n = 5n;
            const seed = ethers.ZeroHash;

            await touchGas(harness, "randomVectorHarness", [n, seed]);
            const gas = await estimateGas(harness, "randomVectorHarness", [n, seed]);

            const res1 = await harness.randomVectorHarness(n, seed);
            const res2 = await harness.randomVectorHarness(n, seed);

            expect(res1[2]).to.deep.equal(res2[2]); // Data match

            printBlockMatrix({
                t,
                method: "randomVectorHarness",
                explanation: "Same seed produces identical vectors.",
                gas,
                shapeIn: `n=${n}`,
                shapeOut: "5x1",
                inHex: `seed=${seed}`,
                outHex: "Deterministic"
            });
        });

        it("Test 8: randomVector seed entropy", async function () {
            t++;
            const n = 5n;
            const seedA = ethers.ZeroHash;
            const seedB = ethers.keccak256(ethers.toUtf8Bytes("B"));

            const resA = await harness.randomVectorHarness(n, seedA);
            const resB = await harness.randomVectorHarness(n, seedB);

            expect(resA[2]).to.not.deep.equal(resB[2]);

            printBlockMatrix({
                t,
                method: "randomVectorHarness",
                explanation: "Different seeds produce different vectors.",
                gas: "N/A",
                shapeIn: `n=${n}`,
                shapeOut: "5x1",
                inHex: "Mixed seeds",
                outHex: "Differ"
            });
        });

        it("Test 9: hasConverged check", async function () {
            t++;
            const vOld = [await qInt(0), await qInt(0)];
            const vNew = [await qInt(1), await qInt(0)]; // Distance = 1.0

            const tolPass = await qInt(2); // 1.0 < 2.0 -> True
            const tolFail = await qInt(1); // 1.0 < 1.0 -> False

            await touchGas(harness, "hasConvergedHarness", [2n, 1n, vNew, 2n, 1n, vOld, tolPass]);
            const gas = await estimateGas(harness, "hasConvergedHarness", [2n, 1n, vNew, 2n, 1n, vOld, tolPass]);

            const yes = await harness.hasConvergedHarness(2n, 1n, vNew, 2n, 1n, vOld, tolPass);
            const no = await harness.hasConvergedHarness(2n, 1n, vNew, 2n, 1n, vOld, tolFail);

            expect(yes).to.be.true;
            expect(no).to.be.false;

            printBlockMatrix({
                t,
                method: "hasConvergedHarness",
                explanation: "Checks convergence ||vNew - vOld|| < tol.",
                gas,
                shapeIn: "2x1, 2x1",
                shapeOut: "bool",
                inHex: "Dist=1.0",
                outHex: `Pass=${yes}, Fail=${no}`
            });
        });

        it("Test 10: hasConverged reverts on row vectors", async function () {
            t++;
            const vRow = [await qInt(1), await qInt(2)];
            const tol = await qInt(1);

            await expect(harness.hasConvergedHarness(1n, 2n, vRow, 1n, 2n, vRow, tol)).to.be.revertedWith("MatrixMaster: converged requires vectors");

            printBlockMatrix({
                t,
                method: "hasConvergedHarness",
                explanation: "Rejects row vectors for convergence check.",
                gas: "Revert",
                shapeIn: "1x2, 1x2",
                shapeOut: "revert",
                inHex: "Row Vecs",
                outHex: "-"
            });
        });
    });

    // ---------------------------------------------------------
    //  Section 13: Power Iteration
    // ---------------------------------------------------------

    describe("Section 13: PowerIteration", function () {
        const DIVISOR = 1_000_000_000_000; // PowerIteration returns eigenvalue scaled by 1e12

        it("Test 11: diagonal 2x2 matrix – dominant eigenvalue 5", async function () {
            t++;
            const A = [await qInt(5), await qInt(0), await qInt(0), await qInt(2)];
            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1000n);

            // Calculate gas first
            await touchGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(2n, 2n, A, seed, tol);

            const rawLambda = Number(await harness.toFloat(lambda));
            const actualValue = rawLambda / DIVISOR;

            const diff = Math.abs(actualValue - 5.0);
            expect(diff <= 0.01).to.be.true;

            printBlockMatrix({
                t,
                method: "powerIterationHarness",
                explanation: "Dominant eigenvalue ~5 for Diag(5,2).",
                gas,
                shapeIn: "2x2",
                shapeOut: "Eigenpair",
                inHex: `A=${fmtHexArr(A)}`,
                outHex: `λ=${actualValue.toFixed(4)}`
            });
        });

        it("Test 12: symmetric 2x2 matrix – dominant eigenvalue 4", async function () {
            t++;
            const A = [await qInt(3), await qInt(1), await qInt(1), await qInt(3)];
            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1000n);

            await touchGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);

            const [lambda] = await harness.powerIterationHarness(2n, 2n, A, seed, tol);

            const rawLambda = Number(await harness.toFloat(lambda));
            const actualValue = rawLambda / DIVISOR;

            expect(Math.abs(actualValue - 4.0) <= 0.01).to.be.true;

            printBlockMatrix({
                t,
                method: "powerIterationHarness",
                explanation: "Symmetric [[3,1],[1,3]] -> Eigenvalue ~4.",
                gas,
                shapeIn: "2x2",
                shapeOut: "Eigenpair",
                inHex: `A=${fmtHexArr(A)}`,
                outHex: `λ=${actualValue.toFixed(4)}`
            });
        });

        it("Test 13: non-symmetric 3x3 – dominant eigenvalue 3", async function () {
            t++;
            const A = [
                await qInt(2), await qInt(1), await qInt(0),
                await qInt(0), await qInt(3), await qInt(1),
                await qInt(0), await qInt(0), await qInt(1)
            ];
            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1000n);

            await touchGas(harness, "powerIterationHarness", [3n, 3n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [3n, 3n, A, seed, tol]);

            const [lambda] = await harness.powerIterationHarness(3n, 3n, A, seed, tol);

            const rawLambda = Number(await harness.toFloat(lambda));
            const actualValue = rawLambda / DIVISOR;

            expect(Math.abs(actualValue - 3.0) <= 0.01).to.be.true;

            printBlockMatrix({
                t,
                method: "powerIterationHarness",
                explanation: "Upper triangular -> Eigenvalue ~3.",
                gas,
                shapeIn: "3x3",
                shapeOut: "Eigenpair",
                inHex: `A=${fmtHexArr(A)}`,
                outHex: `λ=${actualValue.toFixed(4)}`
            });
        });
    });
});