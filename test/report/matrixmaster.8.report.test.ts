// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Test suite validating MatrixMaster sparse matrix utilities using ABDK quad precision, covering CSR-based
 *         sparse matrix construction (zero, identity, diagonal, and COO triplet conversion) and sparse matrix–vector multiplication.
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;

    // Sparse matrix creation (CSR)
    createZeroSparseHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, bigint[], bigint[], string[]]>;
    createIdentitySparseHarness(n: bigint): Promise<[bigint, bigint, bigint[], bigint[], string[]]>;
    createDiagonalSparseHarness(diag: string[]): Promise<[bigint, bigint, bigint[], bigint[], string[]]>;
    createSparseFromTripletsHarness(rows: bigint, cols: bigint, rowInd: bigint[], colInd: bigint[], values: string[]):
        Promise<[bigint, bigint, bigint[], bigint[], string[]]>;

    // Sparse matrix x vector multiplication
    mulSparseMatrixVectorHarness(aRows: bigint, aCols: bigint, rowPtr: bigint[], colInd: bigint[], values: string[], xRows: bigint, xData: string[]):
        Promise<[bigint, bigint, string[]]>;
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

// Converts data containing BigInts into a JSON-compatible string.
export const stringify = (data: any): string => { return JSON.stringify(data, (key, value) => typeof value === 'bigint' ? value.toString() : value); };

// Formats CSR matrix components into a readable string for reports.
const fmtCSR = (rowPtr: bigint[], colInd: bigint[], values: string[]): string => { return `rowPtr=${stringify(rowPtr)}, colInd=${stringify(colInd)}, values=${stringify(values)}`; };

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

describe("MatrixMaster (library) : sparse matrix creation and sparse matrix-vector multiplication over ABDK quad", function () {
    let harness: MatrixMasterHarness;
    let t = 0; // global test counter for reporting

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    const qInt = async (n: number | string | bigint): Promise<string> =>
        harness.qFromInt(BigInt(n));

    // ------------------------------------------------------------
    //  Section 14: Sparse Matrix Creation
    // ------------------------------------------------------------

    describe("Section 14: Sparse Matrix Creation", function () {

        it("Test 1: create sparse zero matrix (3x4)", async function () {
            t++;
            const rows = 3n, cols = 4n;

            await touchGas(harness, "createZeroSparseHarness", [rows, cols]);
            const gas = await estimateGas(harness, "createZeroSparseHarness", [rows, cols]);
            const [r, c, rowPtr, colInd, values] = await harness.createZeroSparseHarness(rows, cols);

            expect(r).to.equal(rows);
            expect(c).to.equal(cols);

            printBlockMatrix({
                t,
                method: "createZeroSparseHarness",
                explanation: "Creates an empty 3x4 CSR matrix (all zeros).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `rows=${rows}, cols=${cols}`,
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 2: create sparse zero matrix (1x1)", async function () {
            t++;
            const rows = 1n, cols = 1n;

            await touchGas(harness, "createZeroSparseHarness", [rows, cols]);
            const gas = await estimateGas(harness, "createZeroSparseHarness", [rows, cols]);
            const [r, c, rowPtr, colInd, values] = await harness.createZeroSparseHarness(rows, cols);

            printBlockMatrix({
                t,
                method: "createZeroSparseHarness",
                explanation: "Creates minimal 1x1 sparse zero matrix.",
                gas,
                shapeIn: "1x1",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: "rows=1, cols=1",
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 3: create zero sparse with zero rows reverts", async function () {
            t++;
            const rows = 0n;
            const cols = 3n;

            await expect(harness.createZeroSparseHarness(rows, cols)).to.be.revertedWith("MatrixMaster: invalid sparse shape");

            printBlockMatrix({
                t,
                method: "createZeroSparseHarness",
                explanation: "Reverts on invalid dimension (rows=0).",
                gas: "Revert",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: `rows=${rows}, cols=${cols}`,
                outHex: "-"
            });
        });

        it("Test 4: create sparse identity matrix (3x3)", async function () {
            t++;
            const n = 3n;

            await touchGas(harness, "createIdentitySparseHarness", [n]);
            const gas = await estimateGas(harness, "createIdentitySparseHarness", [n]);
            const [r, c, rowPtr, colInd, values] = await harness.createIdentitySparseHarness(n);

            printBlockMatrix({
                t,
                method: "createIdentitySparseHarness",
                explanation: "Creates 3x3 Identity matrix in CSR format.",
                gas,
                shapeIn: "3x3",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: "n=3",
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 5: create sparse identity matrix (1x1)", async function () {
            t++;
            const n = 1n;

            await touchGas(harness, "createIdentitySparseHarness", [n]);
            const gas = await estimateGas(harness, "createIdentitySparseHarness", [n]);
            const [r, c, rowPtr, colInd, values] = await harness.createIdentitySparseHarness(n);

            printBlockMatrix({
                t,
                method: "createIdentitySparseHarness",
                explanation: "Creates 1x1 Identity (single value 1.0).",
                gas,
                shapeIn: "1x1",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: "n=1",
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 6: create identity sparse with n=0 reverts", async function () {
            t++;
            const n = 0n;

            await expect(harness.createIdentitySparseHarness(n)).to.be.revertedWith("MatrixMaster: n = 0");

            printBlockMatrix({
                t,
                method: "createIdentitySparseHarness",
                explanation: "Reverts on invalid dimension (n=0).",
                gas: "Revert",
                shapeIn: `${n}x${n}`,
                shapeOut: "revert",
                inHex: `n=${n}`,
                outHex: "-"
            });
        });

        it("Test 7: create diagonal sparse matrix ([1,2,3])", async function () {
            t++;
            const diag = [await qInt(1), await qInt(2), await qInt(3)];

            await touchGas(harness, "createDiagonalSparseHarness", [diag]);
            const gas = await estimateGas(harness, "createDiagonalSparseHarness", [diag]);
            const [r, c, rowPtr, colInd, values] = await harness.createDiagonalSparseHarness(diag);

            printBlockMatrix({
                t,
                method: "createDiagonalSparseHarness",
                explanation: "Creates 3x3 Diagonal matrix from array.",
                gas,
                shapeIn: "diag[3]",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `diag=[1,2,3]`,
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 8: create sparse diagonal matrix with 1 element", async function () {
            t++;
            const diag = [await qInt(5)];

            await touchGas(harness, "createDiagonalSparseHarness", [diag]);
            const gas = await estimateGas(harness, "createDiagonalSparseHarness", [diag]);
            const [r, c, rowPtr, colInd, values] = await harness.createDiagonalSparseHarness(diag);

            printBlockMatrix({
                t,
                method: "createDiagonalSparseHarness",
                explanation: "Creates 1x1 Diagonal matrix.",
                gas,
                shapeIn: "diag[1]",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `diag=[5]`,
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 9: empty diag reverts", async function () {
            t++;
            const diag: string[] = [];

            await expect(harness.createDiagonalSparseHarness(diag)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "createDiagonalSparseHarness",
                explanation: "Reverts on empty input array.",
                gas: "Revert",
                shapeIn: "0x0",
                shapeOut: "revert",
                inHex: "diag=[]",
                outHex: "-"
            });
        });

        it("Test 10: create sparse matrix from COO triplets (Sorted)", async function () {
            t++;
            const rowInd = [0n, 1n];
            const colInd = [0n, 1n];
            const values = [await qInt(1), await qInt(2)]; // Identity-like

            await touchGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const gas = await estimateGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const [r, c, rowPtr, outColInd, outValues] = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colInd, values);

            printBlockMatrix({
                t,
                method: "createSparseFromTripletsHarness",
                explanation: "Converts COO triplets to CSR format.",
                gas,
                shapeIn: "2x2 (COO)",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `COO[2]`,
                outHex: fmtCSR(rowPtr, outColInd, outValues)
            });
        });

        it("Test 11: create sparse matrix from unsorted COO input", async function () {
            t++;
            const rowInd = [1n, 0n];
            const colInd = [1n, 0n];
            const values = [await qInt(4), await qInt(3)];

            await touchGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const gas = await estimateGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const [r, c, rowPtr, outColInd, outValues] = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colInd, values);

            printBlockMatrix({
                t,
                method: "createSparseFromTripletsHarness",
                explanation: "Converts unsorted COO input (1,1) then (0,0).",
                gas,
                shapeIn: "2x2 (Unsorted COO)",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `COO[2]`,
                outHex: fmtCSR(rowPtr, outColInd, outValues)
            });
        });

        it("Test 12: length mismatch reverts", async function () {
            t++;
            const rows = 2n, cols = 2n;
            const rowInd = [0n];
            const colInd = [0n, 1n]; // mismatch
            const values = [await qInt(1)];

            await expect(harness.createSparseFromTripletsHarness(rows, cols, rowInd, colInd, values)).to.be.revertedWith("MatrixMaster: triplet length mismatch");

            printBlockMatrix({
                t,
                method: "createSparseFromTripletsHarness",
                explanation: "Reverts on mismatched COO array lengths.",
                gas: "Revert",
                shapeIn: "2x2",
                shapeOut: "revert",
                inHex: "Mismatch",
                outHex: "-"
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 15: Sparse Matrix - Vector Multiplication
    // ------------------------------------------------------------

    /**
     * @notice Formats an array of hex strings into a comma-separated string wrapped in brackets.
     * @param {string[]} arr - An array of hexadecimal strings (e.g., ["0x123...", "0x456..."]).
     * @returns {string} A formatted string: "[0x123..., 0x456...]"
     */
    const fmtHexArr = (arr: string[]): string => `[${arr.join(", ")}]`;

    describe("Section 15: Sparse Matrix - Vector Multiplication", function () {

        // Helper: Strips Ethers Proxy wrappers to return a standard, mutable JavaScript array
        const unproxy = (arr: any[]): any[] => {
            return arr.map(v => {
                if (typeof v === 'bigint') return v; // Already safe
                if (typeof v === 'string') return v; // Already safe
                return v.toString(); // Force string/BigInt conversion
            });
        };

        it("Test 13: multiply identity matrix (3x3) by vector [1, 2, 3]", async function () {
            t++;
            const n = 3n;
            const res = await harness.createIdentitySparseHarness(n);

            // Explicitly "unproxy" the arrays to make them mutable for the next call
            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(1), await qInt(2), await qInt(3)];

            await touchGas(harness, "mulSparseMatrixVectorHarness", [n, n, rowPtr, colInd, values, n, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [n, n, rowPtr, colInd, values, n, xData]);

            const [yRows, yCols, yData] = await harness.mulSparseMatrixVectorHarness(n, n, rowPtr, colInd, values, n, xData);

            expect(yData[0]).to.equal(xData[0]);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Sparse Identity * Dense Vector.",
                gas,
                shapeIn: "3x3 * 3x1",
                shapeOut: "3x1",
                inHex: `Matrix=Identity(3x3), x=${fmtHexArr(xData)}`,
                outHex: `y=${fmtHexArr(yData)}`
            });
        });

        it("Test 14: zero sparse matrix (3x3) times vector [1,2,3]", async function () {
            t++;
            const n = 3n;
            const res = await harness.createZeroSparseHarness(n, n);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(1), await qInt(2), await qInt(3)];

            await touchGas(harness, "mulSparseMatrixVectorHarness", [n, n, rowPtr, colInd, values, n, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [n, n, rowPtr, colInd, values, n, xData]);

            const [yRows, yCols, yData] = await harness.mulSparseMatrixVectorHarness(n, n, rowPtr, colInd, values, n, xData);

            yData.forEach(v => expect(isQuadZero(v)).to.be.true);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Zero sparse matrix produces zero vector.",
                gas,
                shapeIn: "3x3 * 3x1",
                shapeOut: "3x1",
                inHex: "ZeroMatrix",
                outHex: `y=${fmtHexArr(yData)}`
            });
        });

        it("Test 15: diagonal sparse [2,3,4] times vector [1,1,1]", async function () {
            t++;
            const diag = [await qInt(2), await qInt(3), await qInt(4)];
            const res = await harness.createDiagonalSparseHarness(diag);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(1), await qInt(1), await qInt(1)];

            await touchGas(harness, "mulSparseMatrixVectorHarness", [3n, 3n, rowPtr, colInd, values, 3n, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [3n, 3n, rowPtr, colInd, values, 3n, xData]);

            const [_, __, yData] = await harness.mulSparseMatrixVectorHarness(3n, 3n, rowPtr, colInd, values, 3n, xData);

            expect(yData[0]).to.equal(diag[0]);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Diagonal sparse matrix scales vector entries.",
                gas,
                shapeIn: "3x3 * 3x1",
                shapeOut: "3x1",
                inHex: `diag=${fmtHexArr(diag)}`,
                outHex: `y=${fmtHexArr(yData)}`
            });
        });

        it("Test 16: sparse from triplets (unsorted) × vector", async function () {
            t++;
            const rowInd = [1n, 0n], colIndTri = [0n, 1n];
            const valTri = [await qInt(2), await qInt(5)];
            const res = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colIndTri, valTri);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(10), await qInt(1)];

            await touchGas(harness, "mulSparseMatrixVectorHarness", [2n, 2n, rowPtr, colInd, values, 2n, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [2n, 2n, rowPtr, colInd, values, 2n, xData]);

            const [_, __, yData] = await harness.mulSparseMatrixVectorHarness(2n, 2n, rowPtr, colInd, values, 2n, xData);

            expect(yData[0]).to.equal(await qInt(5));
            expect(yData[1]).to.equal(await qInt(20));

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Multiplication with CSR created from triplets.",
                gas,
                shapeIn: "2x2 * 2x1",
                shapeOut: "2x1",
                inHex: "A=[[0,5],[2,0]]",
                outHex: `y=${fmtHexArr(yData)}`
            });
        });

        it("Test 17: dimension mismatch (A.cols != x.rows) reverts", async function () {
            t++;
            const n = 3n, xRows = 2n;
            const res = await harness.createIdentitySparseHarness(n);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(1), await qInt(2)];

            await expect(harness.mulSparseMatrixVectorHarness(n, n, rowPtr, colInd, values, xRows, xData)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Reverts on dimension mismatch.",
                gas: "Revert",
                shapeIn: "3x3 * 2x1",
                shapeOut: "revert",
                inHex: "Mismatch",
                outHex: "-"
            });
        });

        it("Test 18: empty rows in CSR handled correctly", async function () {
            t++;
            const rowInd = [0n], colIndTri = [0n], valTri = [await qInt(1)];
            const res = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colIndTri, valTri);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(7), await qInt(9)];

            await touchGas(harness, "mulSparseMatrixVectorHarness", [2n, 2n, rowPtr, colInd, values, 2n, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [2n, 2n, rowPtr, colInd, values, 2n, xData]);
            const [_, __, yData] = await harness.mulSparseMatrixVectorHarness(2n, 2n, rowPtr, colInd, values, 2n, xData);

            expect(yData[0]).to.equal(await qInt(7));
            expect(isQuadZero(yData[1])).to.be.true;

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Row with no non-zeros results in 0.",
                gas,
                shapeIn: "2x2 * 2x1",
                shapeOut: "2x1",
                inHex: "A=[[1,0],[0,0]]",
                outHex: `y=${fmtHexArr(yData)}`
            });
        });

        it("Test 19: xRows != xData.length reverts", async function () {
            t++;
            const n = 2n, xRows = 3n;
            const res = await harness.createIdentitySparseHarness(n);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = [await qInt(1), await qInt(2)];

            await expect(harness.mulSparseMatrixVectorHarness(n, n, rowPtr, colInd, values, xRows, xData)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Reverts on vector length mismatch.",
                gas: "Revert",
                shapeIn: "Mismatch",
                shapeOut: "revert",
                inHex: "Len != Rows",
                outHex: "-"
            });
        });

        it("Test 20: extreme sparse matrix × vector stress test", async function () {
            t++;
            const rows = 8n, cols = 10n;
            const rowInd = [7n, 0n, 2n, 5n, 2n, 0n, 7n], colIndTri = [0n, 0n, 1n, 5n, 3n, 9n, 9n];
            const valuesRaw = [5, 1, 3, 4, -1, 2, -6];
            const valTri = await Promise.all(valuesRaw.map(v => qInt(v)));

            const res = await harness.createSparseFromTripletsHarness(rows, cols, rowInd, colIndTri, valTri);

            const rowPtr = unproxy([...res[2]]);
            const colInd = unproxy([...res[3]]);
            const values = unproxy([...res[4]]);

            const xData = await Promise.all(Array.from({ length: 10 }, (_, i) => qInt(i + 1)));

            await touchGas(harness, "mulSparseMatrixVectorHarness", [rows, cols, rowPtr, colInd, values, cols, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [rows, cols, rowPtr, colInd, values, cols, xData]);

            const [yRows, _, yData] = await harness.mulSparseMatrixVectorHarness(rows, cols, rowPtr, colInd, values, cols, xData);

            expect(yData[0]).to.equal(await qInt(21)); // 1*1 + 2*10
            expect(yData[7]).to.equal(await qInt(-55)); // 5*1 - 6*10

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "8x10 Sparse Stress Test with mixed signs and empty rows.",
                gas,
                shapeIn: "8x10 * 10x1",
                shapeOut: "8x1",
                inHex: "Sparse CSR",
                outHex: "y verified"
            });
        });
    });
});