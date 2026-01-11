// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests covers functions for manipulating the structure and dimensions of matrices: slice & reshape
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromInt(n: bigint): Promise<string>;

    sliceHarness(rows: bigint, cols: bigint, dataFlat: string[], rowStart: bigint, rowEnd: bigint, colStart: bigint, colEnd: bigint): Promise<[bigint, bigint, string[]]>;
    reshapeHarness(rows: bigint, cols: bigint, dataFlat: string[], newRows: bigint, newCols: bigint): Promise<[bigint, bigint, string[]]>;
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

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("MatrixMaster — Shape & Reshape", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---
    async function qInt(n: number | string | bigint): Promise<string> {
        return harness.qFromInt(BigInt(n));
    }

    // ------------------------------------------------------------
    //  Section 4: Slice & Reshape
    // ------------------------------------------------------------

    describe("Section 4: Slice & reshape", function () {
        it("Test 1: slice centered (Interior Block)", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 4n;

            const rowStart = 1n; const rowEnd = 3n;
            const colStart = 1n; const colEnd = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(2n);
            expect(s.cols).to.equal(2n);

            const expected = [await qInt(6), await qInt(7), await qInt(10), await qInt(11)];

            expect(s.data.length).to.equal(4);
            for (let i = 0; i < 4; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "sliceHarness",
                explanation: "Extracts interior 2x2 sub-block [[6,7],[10,11]].",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("Test 2: slice full row", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 4n;
            await touchGas(harness, "sliceHarness", [rows, cols, vals, 1n, 2n, 0n, 4n]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 1n, 2n, 0n, 4n]);

            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, 1n, 2n, 0n, 4n));
            expect(s.rows).to.equal(1n);
            expect(s.cols).to.equal(4n);

            const expected = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];
            for (let i = 0; i < 4; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "sliceHarness",
                explanation: "Slices single full row (Row 1).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("Test 3: slice invalid or out-of-range bounds revert", async function () {
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 2n;
            const cols = 3n;

            // Case 1: rowStart >= rowEnd
            {
                t++;
                await expect(harness.sliceHarness(rows, cols, vals, 1n, 1n, 0n, 2n)).to.be.revertedWith("MatrixMaster: invalid slice range");

                printBlockMatrix({
                    t,
                    method: "sliceHarness",
                    explanation: "Rejects empty row range (Start=End).",
                    gas: "Revert",
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // Case 2: colStart >= colEnd
            {
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 1n, 2n, 2n)).to.be.revertedWith("MatrixMaster: invalid slice range");

                printBlockMatrix({
                    t,
                    method: "sliceHarness",
                    explanation: "Rejects empty column range.",
                    gas: "Revert",
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // Case 3: rowEnd > rows
            {
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 3n, 0n, 2n)).to.be.revertedWith("MatrixMaster: slice out of bounds");

                printBlockMatrix({
                    t,
                    method: "sliceHarness",
                    explanation: "Guards against row overflow.",
                    gas: "Revert",
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // Case 4: colEnd > cols
            {
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 2n, 0n, 4n)).to.be.revertedWith("MatrixMaster: slice out of bounds");

                printBlockMatrix({
                    t,
                    method: "sliceHarness",
                    explanation: "Guards against column overflow.",
                    gas: "Revert",
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }
        });

        it("Test 4: slice full column", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 4n;

            const rowStart = 0n; const rowEnd = 3n;
            const colStart = 2n; const colEnd = 3n;

            const expected = [await qInt(3), await qInt(7), await qInt(11)];

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(3n);
            expect(s.cols).to.equal(1n);
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "sliceHarness",
                explanation: "Extracts full column (Col 2).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("Test 5: slice full matrix returns identical layout", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 2n;
            const cols = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, rows, 0n, cols]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, rows, 0n, cols]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, 0n, rows, 0n, cols));

            expect(s.rows).to.equal(rows);
            expect(s.cols).to.equal(cols);
            expect(s.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "sliceHarness",
                explanation: "Identity slice (full range) preserves data exactly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("Test 6: slice last row boundary", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 3n;

            // Last row: 2..3
            await touchGas(harness, "sliceHarness", [rows, cols, vals, 2n, 3n, 0n, 3n]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 2n, 3n, 0n, 3n]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, 2n, 3n, 0n, 3n));

            expect(s.rows).to.equal(1n);
            expect(s.cols).to.equal(3n);
            const expected = [await qInt(7), await qInt(8), await qInt(9)];
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "sliceHarness",
                explanation: "Slices exactly the last row (boundary check).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("Test 7: slice last column boundary", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 3n;

            // Last col: 2..3
            await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 3n, 2n, 3n]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 3n, 2n, 3n]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, 0n, 3n, 2n, 3n));

            expect(s.rows).to.equal(3n);
            expect(s.cols).to.equal(1n);
            const expected = [await qInt(3), await qInt(6), await qInt(9)];

            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "sliceHarness",
                explanation: "Slices exactly the last column (boundary check).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        // ------------------------------
        //  Reshape
        // ------------------------------

        it("Test 8: reshape 2x6 → 3x4 preserves order", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const srcRows = 2n;
            const srcCols = 6n;
            const dstRows = 3n;
            const dstCols = 4n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const r = asMatrix(await harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols));

            expect(r.rows).to.equal(dstRows);
            expect(r.cols).to.equal(dstCols);
            expect(r.data.length).to.equal(12);

            // Verify order unchanged
            for (let i = 0; i < 12; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "reshapeHarness",
                explanation: "Reshapes 2x6 to 3x4 (same element count) preserving row-major order.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("Test 9: reshape mismatched area reverts", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const srcRows = 2n;
            const srcCols = 3n; // Total 6
            const dstRows = 4n;
            const dstCols = 2n; // Total 8

            await expect(harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols)).to.be.revertedWith("MatrixMaster: reshape area mismatch");

            printBlockMatrix({
                t,
                method: "reshapeHarness",
                explanation: "Rejects reshape if total element count differs (6 vs 8).",
                gas: "Revert",
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("Test 10: reshape MxN to 1x(MN) vector (Flatten)", async function () {
            t++;
            const vals: string[] = [];

            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }

            const srcRows = 2n;
            const srcCols = 3n;
            const dstRows = 1n;
            const dstCols = 6n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const r = asMatrix(await harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols));

            expect(r.rows).to.equal(dstRows);
            expect(r.cols).to.equal(dstCols);

            printBlockMatrix({
                t,
                method: "reshapeHarness",
                explanation: "Flattens 2x3 matrix to 1x6 row vector.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("Test 11: reshape 1x(MN) vector back to MxN (Unflatten)", async function () {
            t++;
            const vals: string[] = [];

            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }

            const srcRows = 1n;
            const srcCols = 6n;
            const dstRows = 2n;
            const dstCols = 3n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const r = asMatrix(await harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols));

            expect(r.rows).to.equal(dstRows);
            expect(r.cols).to.equal(dstCols);

            printBlockMatrix({
                t,
                method: "reshapeHarness",
                explanation: "Restores 1x6 vector back to 2x3 structure.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("Test 12: double reshape returns to original shape", async function () {
            t++;
            const vals: string[] = [];

            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }

            const rows0 = 2n; const cols0 = 3n;
            const rows1 = 3n; const cols1 = 2n;

            // Reshape 2x3 -> 3x2
            await touchGas(harness, "reshapeHarness", [rows0, cols0, vals, rows1, cols1]);
            const gas = await estimateGas(harness, "reshapeHarness", [rows0, cols0, vals, rows1, cols1]);
            const r1 = asMatrix(await harness.reshapeHarness(rows0, cols0, vals, rows1, cols1));

            // Reshape 3x2 -> 2x3
            const r2 = asMatrix(await harness.reshapeHarness(r1.rows, r1.cols, r1.data, rows0, cols0));

            expect(r2.rows).to.equal(rows0);
            expect(r2.cols).to.equal(cols0);
            expect(r2.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r2.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "reshapeHarness",
                explanation: "Round-trip reshape 2x3 -> 3x2 -> 2x3 is lossless.",
                gas,
                shapeIn: `${rows0}x${cols0}`,
                shapeOut: `${r2.rows}x${r2.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r2.data),
            });
        });
    });
});