// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests includes functions for matrix creation (zeros, ones, identity, diagonal, random),
 *         element access/modification, matrix algebra (addition, multiplication), and advanced operations (determinant, inverse).
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;

    // creation
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    onesHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    createIdentityMatrixHarness(n: bigint): Promise<[bigint, bigint, string[]]>;
    fromDiagonalHarness(diag: string[]): Promise<[bigint, bigint, string[]]>;
    randomMatrixHarness(rows: bigint, cols: bigint, seed: string): Promise<[bigint, bigint, string[]]>;

    // element access
    getHarness(rows: bigint, cols: bigint, dataFlat: string[], row: bigint, col: bigint): Promise<string>;
    setHarness(rows: bigint, cols: bigint, dataFlat: string[], row: bigint, col: bigint, val: string): Promise<[bigint, bigint, string[]]>;
};

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

function asMatrix(tuple: [bigint, bigint, string[]]) {
    const [rows, cols, data] = tuple;
    return { rows, cols, data: [...data] };
}

function fmtHexArr(arr: string[]) {
    if (arr.length > 6) {
        return `[${arr.slice(0, 3).join(", ")}, ..., ${arr
            .slice(-3)
            .join(", ")}] (len=${arr.length})`;
    }
    return `[${arr.join(", ")}]`;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("MatrixMaster — Creation & Element Access", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    // --------------------------------------------------------
    //  Quad helper wrappers
    // --------------------------------------------------------

    const qInt = async (n: number | bigint) => await harness.qFromInt(BigInt(n));

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const math = await MathLibFactory.deploy();
        await math.waitForDeployment();

        const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": await math.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();
    });

    // ------------------------------------------------------------
    //  Section 1: Creation
    // ------------------------------------------------------------

    describe("Section 1: Creation", function () {

        it("Test 1: Zeros (2x3)", async function () {
            t++;
            const rows = 2n;
            const cols = 3n;

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(6);

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "zerosHarness",
                explanation: "Creates a 2×3 dense matrix fully initialized with quad-precision zeros.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 2: Zeros Stress (20x20)", async function () {
            t++;
            const rows = 20n;
            const cols = 20n;
            const total = Number(rows * cols);

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(total);

            const zero = await qInt(0);

            // Check first, last, and random middle
            expect(m.data[0].toLowerCase()).to.equal(zero.toLowerCase());
            expect(m.data[total - 1].toLowerCase()).to.equal(zero.toLowerCase());
            expect(m.data[Math.floor(total / 2)].toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "zerosHarness",
                explanation: `Stress-allocates a ${rows}x${cols} dense matrix and verifies initialization.`,
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: "See Dec (Too Large)",
            });
        });

        it("Test 3: Ones (2x2)", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "onesHarness", [rows, cols]);
            const gas = await estimateGas(harness, "onesHarness", [rows, cols]);

            const m = asMatrix(await harness.onesHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(4);

            const one = await qInt(1);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(one.toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "onesHarness",
                explanation: "Allocates a 2×2 matrix filled with quad-precision ones.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 4: Identity (3x3)", async function () {
            t++;
            const n = 3n;

            await touchGas(harness, "createIdentityMatrixHarness", [n]);
            const gas = await estimateGas(harness, "createIdentityMatrixHarness", [n]);

            const m = asMatrix(await harness.createIdentityMatrixHarness(n));
            const zero = await qInt(0);
            const one = await qInt(1);

            const rows = Number(m.rows);
            const cols = Number(m.cols);

            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected = i === j ? one : zero;
                    expect(m.data[idx].toLowerCase()).to.equal(expected.toLowerCase());
                }
            }

            printBlockMatrix({
                t,
                method: "createIdentityMatrixHarness",
                explanation: "Builds a 3×3 identity matrix with ones on the diagonal.",
                gas,
                shapeIn: `n=${n.toString()}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 5: Identity (n=0 revert)", async function () {
            t++;
            const n = 0n;

            await expect(harness.createIdentityMatrixHarness(n)).to.be.revertedWith(
                "MatrixMaster: n must be > 0",
            );

            printBlockMatrix({
                t,
                method: "createIdentityMatrixHarness",
                explanation: "Ensures identity construction rejects zero-sized matrices.",
                gas: "Revert",
                shapeIn: `n=${n.toString()}`,
                shapeOut: "revert",
                inHex: "-",
                outHex: "-",
            });
        });

        it("Test 6: From Diagonal (3 elements)", async function () {
            t++;
            const d0 = await qInt(1);
            const d1 = await qInt(2);
            const d2 = await qInt(3);
            const diag = [d0, d1, d2];

            await touchGas(harness, "fromDiagonalHarness", [diag]);
            const gas = await estimateGas(harness, "fromDiagonalHarness", [diag]);
            const m = asMatrix(await harness.fromDiagonalHarness(diag));
            const zero = await qInt(0);

            expect(m.rows).to.equal(3n);
            expect(m.cols).to.equal(3n);

            const rows = Number(m.rows);
            const cols = Number(m.cols);

            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected =
                        i === j ? [d0, d1, d2][i].toLowerCase() : zero.toLowerCase();
                    expect(m.data[idx].toLowerCase()).to.equal(expected);
                }
            }

            printBlockMatrix({
                t,
                method: "fromDiagonalHarness",
                explanation: "Lifts a 3-element quad vector into a 3×3 diagonal matrix.",
                gas,
                shapeIn: `diag length=${diag.length}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: fmtHexArr(diag),
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 7: From Diagonal (empty revert)", async function () {
            t++;
            const diag: string[] = [];

            await expect(harness.fromDiagonalHarness(diag)).to.be.revertedWith(
                "MatrixMaster: empty diagonal",
            );

            printBlockMatrix({
                t,
                method: "fromDiagonalHarness",
                explanation: "Checks that constructing a diagonal matrix from an empty vector is rejected.",
                gas: "Revert",
                shapeIn: "diag length=0",
                shapeOut: "revert",
                inHex: "[]",
                outHex: "-",
            });
        });

        it("Test 8: From Diagonal (mixed entries)", async function () {
            t++;
            const d0 = await qInt(-5);
            const d1 = await qInt(0);
            const d2 = await qInt(42);
            const d3 = await qInt(1_000_000);
            const diag = [d0, d1, d2, d3];

            await touchGas(harness, "fromDiagonalHarness", [diag]);
            const gas = await estimateGas(harness, "fromDiagonalHarness", [diag]);
            const m = asMatrix(await harness.fromDiagonalHarness(diag));
            const zero = await qInt(0);

            expect(m.rows).to.equal(4n);
            expect(m.cols).to.equal(4n);

            const rows = Number(m.rows);
            const cols = Number(m.cols);

            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected =
                        i === j ? diag[i].toLowerCase() : zero.toLowerCase();
                    expect(m.data[idx].toLowerCase()).to.equal(expected);
                }
            }

            printBlockMatrix({
                t,
                method: "fromDiagonalHarness",
                explanation: "Builds a 4×4 diagonal matrix with mixed negative and zero entries.",
                gas,
                shapeIn: `diag length=${diag.length}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: fmtHexArr(diag),
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 9: Random Matrix (Determinism)", async function () {
            t++;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("seed-123"));
            const rows = 3n;
            const cols = 4n;

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m1 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));
            const m2 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            expect(m1.rows).to.equal(rows);
            expect(m1.cols).to.equal(cols);
            expect(m1.data.length).to.equal(12);

            // Verify Determinism
            for (let i = 0; i < m1.data.length; ++i) {
                expect(m1.data[i]).to.equal(m2.data[i]);
            }

            printBlockMatrix({
                t,
                method: "randomMatrixHarness",
                explanation: "Generates pseudo-random quad entries and verifies deterministic seeding.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m1.data),
            });
        });

        it("Test 10: Random Matrix (Different seeds)", async function () {
            t++;
            const seed1 = ethers.keccak256(ethers.toUtf8Bytes("seed-A"));
            const seed2 = ethers.keccak256(ethers.toUtf8Bytes("seed-B"));
            const rows = 2n;
            const cols = 3n;

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed1]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed1]);

            const m1 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed1));
            const m2 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed2));

            const same = m1.data.every((v, i) => v === m2.data[i]);

            // Verify Divergence
            expect(same).to.equal(false);

            printBlockMatrix({
                t,
                method: "randomMatrixHarness",
                explanation: "Uses two different seeds and checks that generated patterns diverge.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed1=${seed1}, seed2=${seed2}`,
                outHex: "Matrices differ (Pass)",
            });
        });

        it("Test 11: Random Matrix (Range [0,1) Precision Check)", async function () {
            t++;
            const rows = 3n;
            const cols = 3n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("range-check"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            const oneScaled = await harness.toFloat(await harness.qFromInt(1n));

            for (const v of m.data) {
                const val = await harness.toFloat(v);

                // Perform the range check against the scaled limits
                expect(val).to.be.at.least(0, `Value ${val} should not be negative`);
                expect(val).to.be.below(oneScaled, `Value ${val} must be < 1.0 (Scaled: ${oneScaled})`);
            }

            const sampleHex = m.data.slice(0, 3).map(h => h.substring(0, 10) + "...").join(", ");

            printBlockMatrix({
                t,
                method: "randomMatrixHarness",
                explanation: `Verified entries are in [0, 1.0) relative to scale ${oneScaled}.`,
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: sampleHex,
            });
        });

        it("Test 12: Random Matrix (1xN row vector)", async function () {
            t++;
            const rows = 1n;
            const cols = 5n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-1xN"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            printBlockMatrix({
                t,
                method: "randomMatrixHarness",
                explanation: "Generates a 1×N random row vector for use as simple quad noise or weights.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 13: Random Matrix (Nx1 column vector)", async function () {
            t++;
            const rows = 5n;
            const cols = 1n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-Nx1"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            printBlockMatrix({
                t,
                method: "randomMatrixHarness",
                explanation: "Generates an N×1 random column vector suitable for stochastic gradient toy examples.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("Test 14: Random Matrix (Diversity)", async function () {
            t++;
            const rows = 4n;
            const cols = 4n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-diversity"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));
            const zeroQ = await qInt(0);

            const distinct = new Set(m.data);
            const hasNonZero = m.data.some(
                (v) => v.toLowerCase() !== zeroQ.toLowerCase(),
            );

            expect(distinct.size).to.be.greaterThan(1);
            expect(hasNonZero).to.equal(true);

            printBlockMatrix({
                t,
                method: "randomMatrixHarness",
                explanation: "Samples a 4×4 random matrix and verifies it exhibits non-trivial value diversity.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Element Access (get / set)
    // ------------------------------------------------------------

    describe("Section 2: Element Access (get / set)", function () {
        it("Test 15: get reads correct element", async function () {
            t++;
            // 2x3: [[1,2,3],[4,5,6]]
            const vals = [
                await qInt(1), await qInt(2), await qInt(3),
                await qInt(4), await qInt(5), await qInt(6),
            ];

            const rows = 2n;
            const cols = 3n;
            const row = 1n;
            const col = 2n;

            await touchGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const gas = await estimateGas(harness, "getHarness", [rows, cols, vals, row, col]);

            const v = await harness.getHarness(rows, cols, vals, row, col);
            expect(v.toLowerCase()).to.equal(vals[5].toLowerCase());

            printBlockMatrix({
                t,
                method: "getHarness",
                explanation: "Fetches a single quad entry (row 1, col 2) from a 2x3 flat row-major matrix.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: v,
            });
        });

        it("Test 16: get out-of-bounds access reverts", async function () {
            t++;
            const vals = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const rows = 2n;
            const cols = 2n;

            await expect(harness.getHarness(rows, cols, vals, 2n, 0n)).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlockMatrix({
                t,
                method: "getHarness",
                explanation: "Checks that reading with an out-of-range row index triggers bounds protection.",
                gas: "Revert",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });

            await expect(harness.getHarness(rows, cols, vals, 0n, 2n)).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlockMatrix({
                t,
                method: "getHarness",
                explanation: "Checks that reading with an out-of-range column index reverts via bounds check.",
                gas: "Revert",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("Test 17: get retrieves corner elements", async function () {
            t++;
            const rows = 3n;
            const cols = 3n;
            const vals: string[] = [];

            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }

            await touchGas(harness, "getHarness", [rows, cols, vals, 0n, 0n]);
            const gas1 = await estimateGas(harness, "getHarness", [rows, cols, vals, 0n, 0n]);
            const topLeft = await harness.getHarness(rows, cols, vals, 0n, 0n);
            expect(topLeft.toLowerCase()).to.equal(vals[0].toLowerCase());

            printBlockMatrix({
                t,
                method: "getHarness",
                explanation: "Accesses the top-left corner element (0,0).",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: topLeft,
            });

            const lastRow = rows - 1n;
            const lastCol = cols - 1n;
            const expectedIdx = Number(lastRow * cols + lastCol);

            const gas2 = await estimateGas(harness, "getHarness", [rows, cols, vals, lastRow, lastCol]);
            const bottomRight = await harness.getHarness(rows, cols, vals, lastRow, lastCol);

            expect(bottomRight.toLowerCase()).to.equal(vals[expectedIdx].toLowerCase());

            printBlockMatrix({
                t,
                method: "getHarness",
                explanation: "Accesses the bottom-right corner element (2,2).",
                gas: gas2,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: bottomRight,
            });
        });

        it("Test 18: get random interior element inside 10x10", async function () {
            t++;
            const rows = 10n;
            const cols = 10n;
            const vals: string[] = [];

            for (let i = 1; i <= 100; ++i) {
                vals.push(await qInt(i));
            }

            const row = 7n;
            const col = 3n;
            const expectedIdx = Number(row * cols + col);

            await touchGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const gas = await estimateGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const v = await harness.getHarness(rows, cols, vals, row, col);

            expect(v.toLowerCase()).to.equal(vals[expectedIdx].toLowerCase());

            printBlockMatrix({
                t,
                method: "getHarness",
                explanation: "Indexes an interior element (7,3) of a 10×10 matrix.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: "Too large to show",
                outHex: v,
            });
        });

        it("Test 19: set writes correct element", async function () {
            t++;

            const zero = await qInt(0);
            const one = await qInt(1);
            const init = [zero, zero, zero, zero];

            const rows = 2n;
            const cols = 2n;
            const row = 1n;
            const col = 0n;

            await touchGas(harness, "setHarness", [rows, cols, init, row, col, one]);
            const gas = await estimateGas(harness, "setHarness", [rows, cols, init, row, col, one]);

            const out = asMatrix(await harness.setHarness(rows, cols, init, row, col, one));

            expect(out.rows).to.equal(rows);
            expect(out.cols).to.equal(cols);
            expect(out.data[2].toLowerCase()).to.equal(one.toLowerCase());
            expect(out.data[3].toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "setHarness",
                explanation: "Mutates cell (1,0) to 1.0 and verifies surrounding cells remain 0.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: fmtHexArr(init),
                outHex: fmtHexArr(out.data),
            });
        });

        it("Test 20: set out-of-bounds write reverts", async function () {
            t++;
            const zero = await qInt(0);
            const init = [zero, zero, zero, zero];
            const rows = 2n;
            const cols = 2n;


            // Case 1
            await expect(harness.setHarness(rows, cols, init, 2n, 0n, zero)).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlockMatrix({
                t,
                method: "setHarness",
                explanation: "Verifies that assigning outside valid row indices reverts.",
                gas: "Revert",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(init),
                outHex: "-",
            });

            // Case 2
            await expect(harness.setHarness(rows, cols, init, 0n, 2n, zero)).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlockMatrix({
                t,
                method: "setHarness",
                explanation: "Verifies that assigning outside valid column indices reverts.",
                gas: "Revert",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(init),
                outHex: "-",
            });
        });

        it("Test 21: set + get reflect mutation", async function () {
            t++;
            const zero = await qInt(0);
            const one = await qInt(1);

            const rows = 3n;
            const cols = 3n;
            const init = Array.from({ length: 9 }, () => zero);

            const row = 2n;
            const col = 1n;

            await touchGas(harness, "setHarness", [rows, cols, init, row, col, one]);
            const gas = await estimateGas(harness, "setHarness", [rows, cols, init, row, col, one]);

            const out = asMatrix(await harness.setHarness(rows, cols, init, row, col, one));
            const readBack = await harness.getHarness(out.rows, out.cols, out.data, row, col);

            expect(readBack.toLowerCase()).to.equal(one.toLowerCase());

            printBlockMatrix({
                t,
                method: "setHarness",
                explanation: "set() a value and confirm get() retrieves the new value.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: "Zeros",
                outHex: fmtHexArr(out.data),
            });
        });
    });
});