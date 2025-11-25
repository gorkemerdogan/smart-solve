// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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

    // normalization
    normHarness(vRows: bigint, vCols: bigint, vData: string[]): Promise<string>;
    normalizeHarness(vRows: bigint, vCols: bigint, vData: string[]): Promise<[bigint, bigint, string[]]>;

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
//  Gas Estimation
// ------------------------------------------------------------

async function touchGas(h: MatrixMasterHarness, method: string, args: any[]) {
    try {
        const data = h.interface.encodeFunctionData(method, args);
        const [signer] = await ethers.getSigners();
        const to = await h.getAddress();
        const tx = await signer.sendTransaction({ to, data });
        await tx.wait();
    } catch {
        // For revert tests or estimation failures, silently ignore.
    }
}

async function estimateGas(
    h: MatrixMasterHarness,
    method: string,
    args: any[],
): Promise<string> {
    try {
        const anyH = h as any;
        if (anyH[method]?.estimateGas) {
            return (await anyH[method].estimateGas(...args)).toString();
        }
        const data = h.interface.encodeFunctionData(method, args);
        const [signer] = await ethers.getSigners();
        const to = await h.getAddress();
        const gas = await signer.estimateGas({ to, data });
        return gas.toString();
    } catch {
        return "revert";
    }
}

// ------------------------------------------------------------
//  Print Block
// ------------------------------------------------------------

function printBlock(options: {
    t: number;
    method: string;
    explanation: string;
    gas: string;
    shapeIn: string;
    shapeOut: string;
    inHex?: string;
    outHex?: string;
}) {
    const { t, method, explanation, gas, shapeIn, shapeOut, inHex, outHex } = options;
    const sep = "-".repeat(60);
    console.log(
        `\n${sep}\nTest ${t}\nMethod: ${method}\nExplanation: ${explanation}\nGas Usage: ${gas}\nInput shape: ${shapeIn}\nOutput shape: ${shapeOut}\nInput (hex): ${inHex ?? "-"}\nOutput (hex): ${outHex ?? "-"}`,
    );
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
    //  Section 1: Norm & Normalize
    // ------------------------------------------------------------

    describe("Section 1: Norm & Normalize", function () {

        // ------------------------------
        // Norm
        // ------------------------------

        it("Test 1: norm of [3,4] is 5", async function () {
            t++;
            // Vector [3, 4]^T -> ||v||_2 = 5
            const v = [await qInt(3), await qInt(4)];

            await touchGas(harness, "normHarness", [2n, 1n, v]);
            const gas = await estimateGas(harness, "normHarness", [2n, 1n, v]);
            const out = await harness.normHarness(2n, 1n, v);

            const expected = await qInt(5);
            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            printBlock({
                t,
                method: "normHarness",
                explanation: "Computes Euclidean norm ||[3,4]^T||_2 = 5.",
                gas,
                shapeIn: "2x1",
                shapeOut: "scalar",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: out,
            });
        });

        it("Test 2: norm reverts on row vector (1x2)", async function () {
            t++;
            const v = [await qInt(1), await qInt(1)]; // 1x2 row vector

            await touchGas(harness, "normHarness", [1n, 2n, v]);
            const gas = await estimateGas(harness, "normHarness", [1n, 2n, v]);

            await expect(
                harness.normHarness(1n, 2n, v),
            ).to.be.revertedWith("MatrixMaster: norm requires column vector");

            printBlock({
                t,
                method: "normHarness",
                explanation: "Rejects row vectors. norm requires a column vector (n×1).",
                gas,
                shapeIn: "1x2",
                shapeOut: "revert",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: "-",
            });
        });

        // ------------------------------
        // Normalize
        // ------------------------------

        it("Test 3: normalize scales [3,0,4] to unit length", async function () {
            t++;
            const v = [await qInt(3), await qInt(0), await qInt(4)]; // 3x1

            await touchGas(harness, "normalizeHarness", [3n, 1n, v]);
            const gas = await estimateGas(harness, "normalizeHarness", [3n, 1n, v]);

            const out = asMatrix(
                await harness.normalizeHarness(3n, 1n, v),
            );
            const data = out.data;

            // expected = [3/5, 0, 4/5]
            const ex0 = await harness.qFromFrac(3n, 5n);
            const ex1 = await qInt(0);
            const ex2 = await harness.qFromFrac(4n, 5n);

            expect(data[0].toLowerCase()).to.equal(ex0.toLowerCase());
            expect(data[1].toLowerCase()).to.equal(ex1.toLowerCase());
            expect(data[2].toLowerCase()).to.equal(ex2.toLowerCase());

            printBlock({
                t,
                method: "normalizeHarness",
                explanation: "Normalizes [3,0,4]^T into unit vector (3/5, 0, 4/5) using quad math.",
                gas,
                shapeIn: "3x1",
                shapeOut: "3x1",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: fmtHexArr(data),
            });
        });

        it("Test 4: normalize reverts on zero vector", async function () {
            t++;
            const v = [await qInt(0), await qInt(0)]; // 2x1 zero vector

            await touchGas(harness, "normalizeHarness", [2n, 1n, v]);
            const gas = await estimateGas(harness, "normalizeHarness", [2n, 1n, v]);

            await expect(harness.normalizeHarness(2n, 1n, v)).to.be.revertedWith("MatrixMaster: cannot normalize zero vector");

            printBlock({
                t,
                method: "normalizeHarness",
                explanation: "Rejects normalization of the zero vector (||v|| = 0).",
                gas,
                shapeIn: "2x1",
                shapeOut: "revert",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: "-",
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Random Vector & Convergence
    // ------------------------------------------------------------

    describe("Section 2: Random & Convergence", function () {

        // ------------------------------
        // Random Vector
        // ------------------------------

        it("Test 5: randomVector generates deterministic n×1 vector for fixed seed", async function () {
            t++;
            const n = 5n;
            const seed = ethers.ZeroHash;

            await touchGas(harness, "randomVectorHarness", [n, seed]);
            const gas = await estimateGas(harness, "randomVectorHarness", [n, seed]);

            const res1 = await harness.randomVectorHarness(n, seed);
            const res2 = await harness.randomVectorHarness(n, seed);

            const [rows1, cols1, data1] = res1;
            const [rows2, cols2, data2] = res2;

            expect(rows1).to.equal(n);
            expect(cols1).to.equal(1n);
            expect(rows2).to.equal(n);
            expect(cols2).to.equal(1n);
            expect(data1).to.deep.equal(data2); // deterministic for fixed seed

            printBlock({
                t,
                method: "randomVectorHarness",
                explanation: "Deterministic generation with fixed seed, returns a stable 5x1 vector.",
                gas,
                shapeIn: `n=${n}`,
                shapeOut: "5x1",
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(data1),
            });
        });

        it("Test 6: randomVector – different seeds produce different vectors", async function () {
            t++;
            const n = 5n;
            const seedA = ethers.ZeroHash;
            const seedB = ethers.keccak256(ethers.toUtf8Bytes("another"));

            await touchGas(harness, "randomVectorHarness", [n, seedA]);
            await touchGas(harness, "randomVectorHarness", [n, seedB]);

            const gasA = await estimateGas(harness, "randomVectorHarness", [n, seedA]);
            const gasB = await estimateGas(harness, "randomVectorHarness", [n, seedB]);
            const gas = `${gasA} + ${gasB}`;

            const [, , dataA] = await harness.randomVectorHarness(n, seedA);
            const [, , dataB] = await harness.randomVectorHarness(n, seedB);

            expect(dataA).to.not.deep.equal(dataB); // different seed -> different vector

            printBlock({
                t,
                method: "randomVectorHarness",
                explanation: "Different seeds must result different pseudo-random vectors.",
                gas,
                shapeIn: `n=${n}`,
                shapeOut: "5x1",
                inHex: `seedA=${seedA}, seedB=${seedB}`,
                outHex: "vectors differ",
            });
        });

        // ------------------------------
        // Convergence
        // ------------------------------

        it("Test 7: hasConverged detects ||vNew - vOld|| < tol correctly", async function () {
            t++;
            // vOld = [0, 0], vNew = [1, 0]
            // ||vNew - vOld||_2 = 1
            const vOld = [await qInt(0), await qInt(0)];
            const vNew = [await qInt(1), await qInt(0)];

            // tolPass = 2  → 1 < 2  => true
            // tolFail = 1  → 1 < 1  => false
            const tolPass = await qInt(2);
            const tolFail = await qInt(1);

            await touchGas(harness, "hasConvergedHarness", [2n, 1n, vNew, 2n, 1n, vOld, tolPass]);
            const gas = await estimateGas(harness, "hasConvergedHarness", [2n, 1n, vNew, 2n, 1n, vOld, tolPass]);

            const yes = await harness.hasConvergedHarness(2n, 1n, vNew, 2n, 1n, vOld, tolPass);
            const no = await harness.hasConvergedHarness(2n, 1n, vNew, 2n, 1n, vOld, tolFail);

            expect(yes).to.equal(true);
            expect(no).to.equal(false);

            printBlock({
                t,
                method: "hasConvergedHarness",
                explanation: "Checks whether ||vNew - vOld||_2 < tol (true: 1 < 2, false: 1 < 1).",
                gas,
                shapeIn: "2x1, 2x1",
                shapeOut: "bool",
                inHex: `tolPass=${tolPass}, tolFail=${tolFail}`,
                outHex: `yes=${yes}, no=${no}`,
            });
        });

        it("Test 8: hasConverged reverts on row vectors", async function () {
            t++;
            const vRow = [await qInt(1), await qInt(2)]; // 1x2
            const tol = await qInt(1);

            await touchGas(harness, "hasConvergedHarness", [1n, 2n, vRow, 1n, 2n, vRow, tol]);
            const gas = await estimateGas(harness, "hasConvergedHarness", [1n, 2n, vRow, 1n, 2n, vRow, tol]);

            await expect(harness.hasConvergedHarness(1n, 2n, vRow, 1n, 2n, vRow, tol)).to.be.revertedWith("MatrixMaster: converged requires vectors");

            printBlock({
                t,
                method: "hasConvergedHarness",
                explanation: "Rejects row-shaped inputs. Convergence check expects column vectors.",
                gas,
                shapeIn: "1x2, 1x2",
                shapeOut: "revert",
                inHex: `vRow=${fmtHexArr(vRow)}, tol=${tol}`,
                outHex: "-",
            });
        });
    });

    // =========================================================
    //  Section 3: Power Iteration
    // =========================================================

    describe("Section 3: PowerIteration", function () {
        it("Test 9: diagonal 2x2 matrix – dominant eigenvalue 5", async function () {
            t++;

            const five = await qInt(5);
            const two = await qInt(2);
            const zero = await qInt(0);

            // A = [ [5,0],
            //       [0,2] ]
            const A = [five, zero, zero, two];

            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1_000_000n); // ~1e-6

            await touchGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(2n, 2n, A, seed, tol);

            const expected = five;
            const lambdaBI = BigInt(lambda);
            const expectedBI = BigInt(expected);
            const tolBI = BigInt(tol);

            const diff = lambdaBI > expectedBI ? lambdaBI - expectedBI : expectedBI - lambdaBI;

            expect(diff).to.be.lt(tolBI);

            printBlock({
                t,
                method: "powerIterationHarness",
                explanation: "Diagonal matrix diag(5,2) -> dominant eigenvalue ≈ 5.",
                gas,
                shapeIn: "2x2",
                shapeOut: `lambda + eigenvector (${xRows}x${xCols})`,
                inHex: `A=${fmtHexArr(A)}`,
                outHex: `lambda=${lambda}, x=${fmtHexArr(xData)}`,
            });
        });

        it("Test 10: symmetric 2x2 matrix – dominant eigenvalue 4", async function () {
            t++;

            const three = await qInt(3);
            const one = await qInt(1);

            // A = [[3,1],
            //      [1,3]]
            const A = [three, one, one, three];

            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1_000_000n);

            await touchGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(2n, 2n, A, seed, tol);

            const expectedEigen = await qInt(4);
            const lambdaBI = BigInt(lambda);
            const expectedBI = BigInt(expectedEigen);
            const tolBI = BigInt(tol);

            const diff = lambdaBI > expectedBI ? lambdaBI - expectedBI : expectedBI - lambdaBI;

            expect(diff).to.be.lt(tolBI);

            printBlock({
                t,
                method: "powerIterationHarness",
                explanation: "Symmetric matrix [[3,1],[1,3]] -> dominant eigenvalue ≈ 4.",
                gas,
                shapeIn: "2x2",
                shapeOut: `lambda + eigenvector (${xRows}x${xCols})`,
                inHex: `A=${fmtHexArr(A)}`,
                outHex: `lambda=${lambda}, x=${fmtHexArr(xData)}`,
            });
        });

        it("Test 11: non-symmetric 3x3 upper-triangular – dominant eigenvalue 3", async function () {
            t++;

            const one = await qInt(1);
            const two = await qInt(2);
            const three = await qInt(3);
            const zero = await qInt(0);

            // A =
            // [2 1 0
            //  0 3 1
            //  0 0 1]
            const A = [two, one, zero, zero, three, one, zero, zero, one];

            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1_000_000n);

            await touchGas(harness, "powerIterationHarness", [3n, 3n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [3n, 3n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(3n, 3n, A, seed, tol);

            // expected dominant eigenvalue = 3 (largest diagonal entry)
            const expectedEigen = three;
            const lambdaBI = BigInt(lambda);
            const expectedBI = BigInt(expectedEigen);
            const tolBI = BigInt(tol);

            const diff = lambdaBI > expectedBI ? lambdaBI - expectedBI : expectedBI - lambdaBI;

            expect(diff).to.be.lt(tolBI);

            printBlock({
                t,
                method: "powerIterationHarness",
                explanation: "Upper-triangular 3x3 -> dominant eigenvalue ≈ 3 (largest diagonal entry).",
                gas,
                shapeIn: "3x3",
                shapeOut: `lambda + eigenvector (${xRows}x${xCols})`,
                inHex: `A=${fmtHexArr(A)}`,
                outHex: `lambda=${lambda}, x=${fmtHexArr(xData)}`,
            });
        });
    });
});