// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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

async function estimateGas(h: MatrixMasterHarness, method: string, args: any[]): Promise<string> {
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
        return "revert / estimation failed";
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

const fmtHexArr = (arr: string[]) => `[${arr.join(", ")}]`;

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("MatrixMaster — Shape & Reshape", function () {
    let harness: MatrixMasterHarness;
    let t = 0; // global test counter for reporting

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    async function qInt(n: number | string | bigint): Promise<string> {
        return harness.qFromInt(BigInt(n));
    }

    // ------------------------------------------------------------
    //  Slice & Reshape
    // ------------------------------------------------------------

    describe("Slice & reshape", function () {
        it("slice : centered slice", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 4n;
            const rowStart = 1n;
            const rowEnd = 3n;
            const colStart = 1n;
            const colEnd = 3n;

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

            printBlock({
                t,
                method: "sliceHarness",
                explanation: "Extracts an interior sub-block and checks row-major layout is preserved correctly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : full row slice", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 4n;
            await touchGas(harness, "sliceHarness", [rows, cols, vals, 1n, 2n, 0n, 4n]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 1n, 2n, 0n, 4n]);

            // row 1 only, all cols -> [5,6,7,8]
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, 1n, 2n, 0n, 4n));
            expect(s.rows).to.equal(1n);
            expect(s.cols).to.equal(4n);

            const expected = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];
            for (let i = 0; i < 4; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation: "Slices a single full row and verifies contiguous row extraction semantics.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : invalid or out-of-range bounds revert", async function () {
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 2n;
            const cols = 3n;

            // rowStart >= rowEnd
            {
                t++;
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 1n, 1n, 0n, 2n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 1n, 1n, 0n, 2n]);

                await expect(harness.sliceHarness(rows, cols, vals, 1n, 1n, 0n, 2n)).to.be.revertedWith("MatrixMaster: invalid slice range");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation: "Ensures start/end row indices must form a strictly positive slice window.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // colStart >= colEnd
            {
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 1n, 2n, 2n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 1n, 2n, 2n]);
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 1n, 2n, 2n)).to.be.revertedWith("MatrixMaster: invalid slice range");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation: "Ensures start/end column indices also must define a non-empty valid range.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // rowEnd > rows
            {
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 3n, 0n, 2n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 3n, 0n, 2n]);
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 3n, 0n, 2n)).to.be.revertedWith("MatrixMaster: slice out of bounds");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation: "Guards against slice windows that extend beyond the matrix row dimension.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // colEnd > cols
            {
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 2n, 0n, 4n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 2n, 0n, 4n]);
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 2n, 0n, 4n)).to.be.revertedWith("MatrixMaster: slice out of bounds");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation: "Guards against slice windows that overflow the matrix column dimension.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }
        });

        it("slice : full column slice", async function () {
            t++;
            // 3x4 matrix [1..12]
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 4n;

            // take full column 2 (zero-based) => [3,7,11]
            const rowStart = 0n;
            const rowEnd = 3n;
            const colStart = 2n;
            const colEnd = 3n;

            const expected = [await qInt(3), await qInt(7), await qInt(11)];

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(3n);
            expect(s.cols).to.equal(1n);
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation: "Selects a single full column and ensures the resulting 3x1 view matches the expected entries.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("reshape : reshape 2x6 → 3x4 preserves order", async function () {
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

            // reshape does not change data order
            for (let i = 0; i < 12; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation: "Changes matrix shape while reusing the same flat data array and order of entries.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("reshape : mismatched area reverts", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const srcRows = 2n;
            const srcCols = 3n;
            const dstRows = 4n;
            const dstCols = 2n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);

            // 2x3 -> area=6; 4x2 -> area=8 => revert
            await expect(harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols)).to.be.revertedWith("MatrixMaster: reshape area mismatch");

            printBlock({
                t,
                method: "reshapeHarness",
                explanation: "Prevents reshapes that would change total element count and corrupt matrix data.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("slice : full matrix slice returns an identical layout", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 2n;
            const cols = 3n;

            const rowStart = 0n;
            const rowEnd = rows;
            const colStart = 0n;
            const colEnd = cols;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(rows);
            expect(s.cols).to.equal(cols);
            expect(s.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation: "Slices the entire matrix domain and checks it is bitwise identical to the original layout.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : last row boundary slices", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 3n;

            // last row only: [7,8,9]
            const rowStart = 2n;
            const rowEnd = 3n;
            const colStart = 0n;
            const colEnd = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(1n);
            expect(s.cols).to.equal(3n);
            const expected = [await qInt(7), await qInt(8), await qInt(9)];
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation: "Slices the last row of a 3x3 matrix to ensure upper-bound row indices behave correctly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : last column boundary slices", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 3n;

            // last column only: [3,6,9]
            const rowStart = 0n;
            const rowEnd = 3n;
            const colStart = 2n;
            const colEnd = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(3n);
            expect(s.cols).to.equal(1n);
            const expected = [await qInt(3), await qInt(6), await qInt(9)];
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation: "Slices the last column of a 3x3 matrix, checking column upper-bounds are handled cleanly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("reshape : MxN to 1x(MN) vector", async function () {
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
            expect(r.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation: "Flattens a 2x3 matrix into a 1x6 row vector, preserving the original row-major order.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("reshape : 1x(MN) vector back to MxN", async function () {
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
            expect(r.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation: "Takes a 1x6 vector and reshapes it back to 2x3 while keeping the flat data untouched.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("reshape : double reshape returns to the original shape", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows0 = 2n;
            const cols0 = 3n;
            const rows1 = 3n;
            const cols1 = 2n;

            await touchGas(harness, "reshapeHarness", [rows0, cols0, vals, rows1, cols1]);
            const gas = await estimateGas(harness, "reshapeHarness", [rows0, cols0, vals, rows1, cols1]);
            const r1 = asMatrix(await harness.reshapeHarness(rows0, cols0, vals, rows1, cols1));
            const r2 = asMatrix(await harness.reshapeHarness(r1.rows, r1.cols, r1.data, rows0, cols0));

            expect(r2.rows).to.equal(rows0);
            expect(r2.cols).to.equal(cols0);
            expect(r2.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r2.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation: "Reshapes 2x3→3x2→2x3 and verifies the final matrix is identical to the original.",
                gas,
                shapeIn: `${rows0}x${cols0}`,
                shapeOut: `${r2.rows}x${r2.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r2.data),
            });
        });
    });
});