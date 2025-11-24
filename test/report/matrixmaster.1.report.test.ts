// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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
//  Gas Estimation
// ------------------------------------------------------------

async function touchGas(harness: MatrixMasterHarness, method: string, args: any[]) {
    try {
        const data = harness.interface.encodeFunctionData(method, args);
        const [signer] = await ethers.getSigners();
        const to = await harness.getAddress();
        const tx = await signer.sendTransaction({ to, data });
        await tx.wait();
    } catch {
        // Ignored for reverts (we only care about side-effectful gas touching)
    }
}

async function estimateGas(harness: MatrixMasterHarness, method: string, args: any[]) {
    try {
        const anyH = harness as any;
        if (anyH[method]?.estimateGas) {
            return (await anyH[method].estimateGas(...args)).toString();
        }
        const data = harness.interface.encodeFunctionData(method, args);
        const [signer] = await ethers.getSigners();
        const to = await harness.getAddress();
        const gas = await signer.estimateGas({ to, data });
        return gas.toString();
    } catch {
        return "revert";
    }
}

// ------------------------------------------------------------
//  Print Block
// ------------------------------------------------------------

function printBlock({
    t,
    method,
    explanation,
    gas,
    shapeIn,
    shapeOut,
    inHex,
    outHex,
}: any) {
    const sep = "-".repeat(60);
    console.log(
        `
        ${sep}
        Test ${t}
        Method: ${method}
        Explanation: ${explanation}
        Gas Usage: ${gas}
        Shape In: ${shapeIn}
        Shape Out: ${shapeOut}
        Input: ${inHex || "-"}
        Output: ${outHex || "-"}
`.trim(),
    );
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

            printBlock({
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

        it("Test 2: Zeros (100x100)", async function () {
            t++;
            const rows = 100n;
            const cols = 100n;

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "zerosHarness",
                explanation: "Stress-allocates a 100×100 dense matrix and verifies every entry is an exact quad zero.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
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

            printBlock({
                t,
                method: "onesHarness",
                explanation: "Allocates a 2×2 matrix filled with quad-precision ones for baseline sanity checks.",
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

            expect(m.rows).to.equal(n);
            expect(m.cols).to.equal(n);
            expect(m.data.length).to.equal(9);

            const rows = Number(m.rows);
            const cols = Number(m.cols);

            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected = i === j ? one : zero;
                    expect(m.data[idx].toLowerCase()).to.equal(expected.toLowerCase());
                }
            }

            printBlock({
                t,
                method: "createIdentityMatrixHarness",
                explanation: "Builds a 3×3 identity matrix with ones on the diagonal and strict zeros elsewhere.",
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

            await touchGas(harness, "createIdentityMatrixHarness", [n]);
            const gas = await estimateGas(harness, "createIdentityMatrixHarness", [n]);

            await expect(harness.createIdentityMatrixHarness(n)).to.be.revertedWith(
                "MatrixMaster: n must be > 0",
            );

            printBlock({
                t,
                method: "createIdentityMatrixHarness",
                explanation: "Ensures identity construction rejects zero-sized matrices via dimension guard.",
                gas,
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

            printBlock({
                t,
                method: "fromDiagonalHarness",
                explanation: "Lifts a 3-element quad vector into a 3×3 diagonal matrix with zeros off the diagonal.",
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

            await touchGas(harness, "fromDiagonalHarness", [diag]);
            const gas = await estimateGas(harness, "fromDiagonalHarness", [diag]);

            await expect(harness.fromDiagonalHarness(diag)).to.be.revertedWith(
                "MatrixMaster: empty diagonal",
            );

            printBlock({
                t,
                method: "fromDiagonalHarness",
                explanation: "Checks that constructing a diagonal matrix from an empty vector is rejected.",
                gas,
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

            printBlock({
                t,
                method: "fromDiagonalHarness",
                explanation: "Builds a 4×4 diagonal matrix with mixed negative, zero, and large-magnitude quad entries.",
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
            expect(m2.rows).to.equal(rows);
            expect(m2.cols).to.equal(cols);
            expect(m1.data.length).to.equal(m2.data.length);

            for (let i = 0; i < m1.data.length; ++i) {
                expect(m1.data[i]).to.equal(m2.data[i]);
            }

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation: "Generates pseudo-random quad entries in [0,1) and verifies deterministic seeding.",
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

            expect(m1.data.length).to.equal(6);
            expect(m2.data.length).to.equal(6);

            const same = m1.data.every((v, i) => v === m2.data[i]);
            expect(same).to.equal(false);

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation: "Uses two different seeds and checks that generated patterns diverge as expected.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed1=${seed1}, seed2=${seed2}`,
                outHex: `m1=${fmtHexArr(m1.data)}, m2=${fmtHexArr(m2.data)}`,
            });
        });

        it("Test 11: Random Matrix (Range [0,1))", async function () {
            t++;
            const rows = 3n;
            const cols = 3n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("range-check"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));
            const zeroQ = await qInt(0);
            const oneQ = await qInt(1);
            const zeroBI = BigInt(zeroQ);
            const oneBI = BigInt(oneQ);

            for (const v of m.data) {
                const bi = BigInt(v);
                expect(bi).to.be.gte(zeroBI);
                expect(bi).to.be.lt(oneBI);
            }

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation: "Samples a 3×3 random matrix and enforces every encoded quad lies in [0,1).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
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

            printBlock({
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

            printBlock({
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

            printBlock({
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
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ];
            const rows = 2n;
            const cols = 3n;
            const row = 1n;
            const col = 2n;

            await touchGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const gas = await estimateGas(harness, "getHarness", [rows, cols, vals, row, col]);

            const v = await harness.getHarness(rows, cols, vals, row, col); // row=1,col=2 -> 6
            expect(v.toLowerCase()).to.equal(vals[5].toLowerCase());

            printBlock({
                t,
                method: "getHarness",
                explanation: "Fetches a single quad entry by row/col indices from a flat row-major matrix.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: v,
            });
        });

        it("Test 16: get out-of-bounds access reverts", async function () {
            t++;
            const vals = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];
            const rows = 2n;
            const cols = 2n;

            // row=2 out of range
            await touchGas(harness, "getHarness", [rows, cols, vals, 2n, 0n]);
            const gas1 = await estimateGas(harness, "getHarness", [rows, cols, vals, 2n, 0n]);

            await expect(
                harness.getHarness(rows, cols, vals, 2n, 0n),
            ).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlock({
                t,
                method: "getHarness",
                explanation: "Checks that reading with an out-of-range row index triggers bounds protection.",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });

            // col=2 out of range
            await touchGas(harness, "getHarness", [rows, cols, vals, 0n, 2n]);
            const gas2 = await estimateGas(harness, "getHarness", [rows, cols, vals, 0n, 2n]);

            await expect(
                harness.getHarness(rows, cols, vals, 0n, 2n),
            ).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlock({
                t,
                method: "getHarness",
                explanation: "Checks that reading with an out-of-range column index reverts via bounds check.",
                gas: gas2,
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

            // (0,0)
            await touchGas(harness, "getHarness", [rows, cols, vals, 0n, 0n]);
            const gas1 = await estimateGas(harness, "getHarness", [rows, cols, vals, 0n, 0n]);
            const topLeft = await harness.getHarness(rows, cols, vals, 0n, 0n);
            expect(topLeft.toLowerCase()).to.equal(vals[0].toLowerCase());

            printBlock({
                t,
                method: "getHarness",
                explanation: "Accesses the top-left corner element (0,0) in a 3×3 matrix and checks correct decoding.",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: topLeft,
            });

            // (rows-1, cols-1)
            const lastRow = rows - 1n;
            const lastCol = cols - 1n;

            await touchGas(harness, "getHarness", [rows, cols, vals, lastRow, lastCol]);
            const gas2 = await estimateGas(harness, "getHarness", [rows, cols, vals, lastRow, lastCol]);
            const bottomRight = await harness.getHarness(rows, cols, vals, lastRow, lastCol);
            const expectedIdx = Number(lastRow * cols + lastCol);

            expect(bottomRight.toLowerCase()).to.equal(
                vals[expectedIdx].toLowerCase(),
            );

            printBlock({
                t,
                method: "getHarness",
                explanation: "Accesses the bottom-right corner element (rows-1, cols-1) and verifies flat-index mapping.",
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
            const col = 3n; // deterministic but random-looking interior cell
            const expectedIdx = Number(row * cols + col);
            const expectedVal = vals[expectedIdx];

            await touchGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const gas = await estimateGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const v = await harness.getHarness(rows, cols, vals, row, col);

            expect(v.toLowerCase()).to.equal(expectedVal.toLowerCase());

            printBlock({
                t,
                method: "getHarness",
                explanation: "Indexes an interior element of a 10×10 matrix and validates large-grid index arithmetic.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: v,
            });
        });

        it("Test 19: set writes correct element", async function () {
            t++;
            // 2x2 all zeros
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

            // Expect only (1,0) to be 1
            expect(out.data[0].toLowerCase()).to.equal(zero.toLowerCase());
            expect(out.data[1].toLowerCase()).to.equal(zero.toLowerCase());
            expect(out.data[2].toLowerCase()).to.equal(one.toLowerCase()); // row1,col0
            expect(out.data[3].toLowerCase()).to.equal(zero.toLowerCase());

            printBlock({
                t,
                method: "setHarness",
                explanation: "Mutates a single matrix cell and verifies only that entry is modified.",
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

            // row=2 out of range
            await touchGas(harness, "setHarness", [rows, cols, init, 2n, 0n, zero]);
            const gas1 = await estimateGas(harness, "setHarness", [rows, cols, init, 2n, 0n, zero]);

            await expect(
                harness.setHarness(rows, cols, init, 2n, 0n, zero),
            ).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlock({
                t,
                method: "setHarness",
                explanation: "Verifies that assigning outside valid row indices reverts with an explicit error.",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(init),
                outHex: "-",
            });

            // col=2 out of range
            await touchGas(harness, "setHarness", [rows, cols, init, 0n, 2n, zero]);
            const gas2 = await estimateGas(harness, "setHarness", [rows, cols, init, 0n, 2n, zero]);

            await expect(
                harness.setHarness(rows, cols, init, 0n, 2n, zero),
            ).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlock({
                t,
                method: "setHarness",
                explanation: "Verifies that assigning outside valid column indices also reverts safely.",
                gas: gas2,
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

            printBlock({
                t,
                method: "setHarness",
                explanation: "Writes a single cell using set() and immediately reads it back via get() to confirm persistence.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: fmtHexArr(init),
                outHex: fmtHexArr(out.data),
            });
        });
    });
});