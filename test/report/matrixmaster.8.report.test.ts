// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests covers sparse matrix creations and sparse matrix-vector multiplication
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

function asSparseMatrix(tuple: [bigint, bigint, bigint[], bigint[], string[]]) {
    const [rows, cols, rowPtr, colInd, values] = tuple;
    return {
        rows,
        cols,
        rowPtr: rowPtr.map(n => Number(n)),
        colInd: colInd.map(n => Number(n)),
        values
    };
}

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
    //  Section 1: Sparse Matrix Creation
    // ------------------------------------------------------------

    describe("Section 1: Sparse Matrix Creation", function () {

        it("Test 1: create sparse matrix (3x4)", async function () {
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
                explanation: "Sparse zero matrix creation.",
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
                explanation: "Sparse zero matrix creation (1x1).",
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

            await expect(harness.createZeroSparseHarness(rows, cols)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "createZeroSparseHarness",
                explanation: "Reverts when rows = 0 for sparse zero matrix creation.",
                gas: "n/a",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: `rows=${rows}, cols=${cols}`,
                outHex: "revert"
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
                explanation: "Sparse identity matrix creation.",
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
                explanation: "Sparse identity matrix creation (1x1).",
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
            await expect(harness.createIdentitySparseHarness(n)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "createIdentitySparseHarness",
                explanation: "Reverts when n = 0 for sparse identity matrix creation.",
                gas: "n/a",
                shapeIn: `${n}x${n}`,
                shapeOut: "revert",
                inHex: `n=${n}`,
                outHex: "revert"
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
                explanation: "Sparse diagonal matrix creation.",
                gas,
                shapeIn: "diag[3]",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `diag=${JSON.stringify(diag)}`,
                outHex: fmtCSR(rowPtr, colInd, values)
            });
        });

        it("Test 8: create sparse diagonal matrix with 1 element ([5])", async function () {
            t++;
            const diag = [await qInt(5)];

            await touchGas(harness, "createDiagonalSparseHarness", [diag]);
            const gas = await estimateGas(harness, "createDiagonalSparseHarness", [diag]);
            const [r, c, rowPtr, colInd, values] = await harness.createDiagonalSparseHarness(diag);

            printBlockMatrix({
                t,
                method: "createDiagonalSparseHarness",
                explanation: "Sparse diagonal matrix creation (1 element).",
                gas,
                shapeIn: "diag[1]",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `diag=${JSON.stringify(diag)}`,
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
                explanation: "Reverts when diagonal array is empty for sparse diagonal matrix creation.",
                gas: "n/a",
                shapeIn: "0x0",
                shapeOut: "revert",
                inHex: "diag=[]",
                outHex: "revert"
            });
        });

        it("Test 10: create sparse matrix from COO triplets", async function () {
            t++;
            const rowInd = [0n, 1n];
            const colInd = [0n, 1n];
            const values = [await qInt(1), await qInt(2)];

            await touchGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const gas = await estimateGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const [r, c, rowPtr, outColInd, outValues] = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colInd, values);

            printBlockMatrix({
                t,
                method: "createSparseFromTripletsHarness",
                explanation: "Sparse matrix from COO triplets.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `rowInd=${stringify(rowInd)}, colInd=${stringify(colInd)}, values=${stringify(values)}`,
                outHex: fmtCSR(rowPtr, outColInd, outValues)
            });
        });

        it("Test 11: create sparse matrix from unsorted COO input", async function () {
            t++;
            const rowInd = [1n, 0n];
            const colInd = [0n, 1n];
            const values = [await qInt(3), await qInt(4)];

            await touchGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const gas = await estimateGas(harness, "createSparseFromTripletsHarness", [2n, 2n, rowInd, colInd, values]);
            const [r, c, rowPtr, outColInd, outValues] = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colInd, values);

            printBlockMatrix({
                t,
                method: "createSparseFromTripletsHarness",
                explanation: "Sparse matrix from unsorted COO input.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${r}x${c} (CSR)`,
                inHex: `rowInd=${stringify(rowInd)}, colInd=${stringify(colInd)}, values=${stringify(values)}`,
                outHex: fmtCSR(rowPtr, outColInd, outValues)
            });
        });

        it("Test 12: length mismatch reverts", async function () {
            t++;

            const rows = 2n;
            const cols = 2n;
            const rowInd = [0n];
            const colInd = [0n, 1n]; // length mismatch
            const values = [await qInt(1)];

            await expect(harness.createSparseFromTripletsHarness(rows, cols, rowInd, colInd, values)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "createSparseFromTripletsHarness",
                explanation: "Reverts when triplet array lengths do not match.",
                gas: "n/a",
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: `rowInd=${rowInd}, colInd=${colInd}, values=${stringify(values)}`,
                outHex: "revert"
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Sparse Matrix - Vector Multiplication
    // ------------------------------------------------------------

    describe("Section 2: Sparse Matrix - Vector Multiplication", function () {

        it("Test 13: multiply identity matrix (3x3) by vector [1, 2, 3]", async function () {
            t++;

            const n = 3n;

            // Create Identity Matrix components
            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createIdentitySparseHarness(n);

            const rowPtr = [..._rowPtr];
            const colInd = [..._colInd];
            const values = [..._values];

            // Create Vector x = [1.0, 2.0, 3.0]
            const xData = [await qInt(1), await qInt(2), await qInt(3)];
            const xRows = n;

            expect(BigInt(xData.length)).to.equal(xRows);

            await touchGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, colInd, values, xRows, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, colInd, values, xRows, xData]);

            const [yRows, yCols, yData] = await harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, colInd, values, xRows, xData);

            expect(yRows).to.equal(n);
            expect(yCols).to.equal(1n);
            expect(yData[0]).to.equal(xData[0]);
            expect(yData[1]).to.equal(xData[1]);
            expect(yData[2]).to.equal(xData[2]);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Sparse Identity * Dense Vector multiplication.",
                gas,
                shapeIn: "3x3 * 3x1",
                shapeOut: `${yRows}x${yCols}`,
                inHex: `Matrix=${fmtCSR(rowPtr, colInd, values)}, Vector=${stringify(xData)}`,
                outHex: `Result=${stringify(yData)}`
            });
        });

        it("Test 14: zero sparse matrix (3x3) times vector [1,2,3] → zero vector", async function () {
            t++;

            const n = 3n;

            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createZeroSparseHarness(n, n);

            const rowPtr = [..._rowPtr];
            const colInd = [..._colInd];
            const values = [..._values];

            const xData = [await qInt(1), await qInt(2), await qInt(3)];
            const xRows = n;

            await touchGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, colInd, values, xRows, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, colInd, values, xRows, xData]);

            const [yRows, yCols, yData] = await harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, colInd, values, xRows, xData);

            expect(yRows).to.equal(n);
            expect(yCols).to.equal(1n);
            yData.forEach(v => expect(isQuadZero(v)).to.equal(true));

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Zero sparse matrix times vector produces zero vector.",
                gas,
                shapeIn: "3x3 * 3x1",
                shapeOut: `${yRows}x${yCols}`,
                inHex: `A=ZeroCSR, x=${stringify(xData)}`,
                outHex: `y=${stringify(yData)}`
            });
        });

        it("Test 15: diagonal sparse [2,3,4] times vector [1,1,1]", async function () {
            t++;

            const diag = [await qInt(2), await qInt(3), await qInt(4)];

            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createDiagonalSparseHarness(diag);

            const rowPtr = [..._rowPtr];
            const colInd = [..._colInd];
            const values = [..._values];

            const xData = [await qInt(1), await qInt(1), await qInt(1)];
            const xRows = 3n;

            await touchGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, colInd, values, xRows, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, colInd, values, xRows, xData]);

            const [_, __, yData] = await harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, colInd, values, xRows, xData);

            expect(yData[0]).to.equal(diag[0]);
            expect(yData[1]).to.equal(diag[1]);
            expect(yData[2]).to.equal(diag[2]);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Diagonal sparse matrix scales vector entries.",
                gas,
                shapeIn: "3x3 * 3x1",
                shapeOut: "3x1",
                inHex: `diag=${stringify(diag)}, x=${stringify(xData)}`,
                outHex: `y=${stringify(yData)}`
            });
        });

        it("Test 16: sparse from triplets (unsorted) × vector", async function () {
            t++;

            // A =
            // [0 5]
            // [2 0]
            const rowInd = [1n, 0n];
            const colInd = [0n, 1n];
            const values = [await qInt(2), await qInt(5)];

            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colInd, values);

            const rowPtr = [..._rowPtr];
            const cInd = [..._colInd];
            const vals = [..._values];

            const xData = [await qInt(10), await qInt(1)];
            const xRows = 2n;

            await touchGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, cInd, vals, xRows, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, cInd, vals, xRows, xData]);

            const [_, __, yData] = await harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, cInd, vals, xRows, xData);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Sparse (COO→CSR) matrix times vector.",
                gas,
                shapeIn: "2x2 * 2x1",
                shapeOut: "2x1",
                inHex: `A=CSR, x=${stringify(xData)}`,
                outHex: `y=${stringify(yData)}`
            });
        });

        it("Test 17: dimension mismatch (A.cols != x.rows) reverts", async function () {
            t++;

            const n = 3n;
            const xRows = 2n;

            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createIdentitySparseHarness(n);

            const rowPtr = [..._rowPtr];
            const colInd = [..._colInd];
            const values = [..._values];

            const xData = [await qInt(1), await qInt(2)];

            await expect(harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, colInd, values, xRows, xData)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Reverts when A.cols != x.rows in sparse matrix-vector multiplication.",
                gas: "n/a",
                shapeIn: `${aRows}x${aCols} * ${xRows}x1`,
                shapeOut: "revert",
                inHex: `A=CSR(identity ${n}x${n}), xRows=${xRows}, xData=${stringify(xData)}`,
                outHex: "revert"
            });
        });

        it("Test 18: empty rows in CSR handled correctly", async function () {
            t++;
            // A =
            // [1 0]
            // [0 0]
            const rowInd = [0n];
            const colInd = [0n];
            const values = [await qInt(1)];

            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createSparseFromTripletsHarness(2n, 2n, rowInd, colInd, values);

            const xData = [await qInt(7), await qInt(9)];
            const xRows = 2n;

            const [_, __, yData] = await harness.mulSparseMatrixVectorHarness(aRows, aCols, [..._rowPtr], [..._colInd], [..._values], xRows, xData);

            expect(yData[1]).to.satisfy(isQuadZero);

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Rows with no non-zeros produce zero output.",
                gas: "n/a",
                shapeIn: "2x2 * 2x1",
                shapeOut: "2x1",
                inHex: "CSR with empty row",
                outHex: `y=${stringify(yData)}`
            });
        });

        it("Test 19: xRows != xData.length reverts", async function () {
            t++;

            const n = 2n;
            const xRows = 3n; // mismatch

            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createIdentitySparseHarness(n);

            const rowPtr = [..._rowPtr];
            const colInd = [..._colInd];
            const values = [..._values];

            const xData = [await qInt(1), await qInt(2)]; // length = 2, xRows = 3

            await expect(harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, colInd, values, xRows, xData)).to.be.reverted;

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Reverts when xRows does not match xData length.",
                gas: "n/a",
                shapeIn: `${aRows}x${aCols} * ${xRows}x1`,
                shapeOut: "revert",
                inHex: `xRows=${xRows}, xData=${stringify(xData)}`,
                outHex: "revert"
            });
        });

        it("Test 20: extreme sparse matrix × vector stress test", async function () {
            t++;

            /**
             * Matrix A (8 x 10)
             *
             * Rows intentionally irregular:
             *
             * r0: [ 1  0  0  0  0  0  0  0  0  2 ]
             * r1: [ 0  0  0  0  0  0  0  0  0  0 ]   ← EMPTY
             * r2: [ 0  3  0 -1  0  0  0  0  0  0 ]
             * r3: [ 0  0  0  0  0  0  0  0  0  0 ]   ← EMPTY
             * r4: [ 0  0  0  0  0  0  0  0  0  0 ]   ← EMPTY
             * r5: [ 0  0  0  0  0  4  0  0  0  0 ]
             * r6: [ 0  0  0  0  0  0  0  0  0  0 ]   ← EMPTY
             * r7: [ 5  0  0  0  0  0  0  0  0 -6 ]
             */

            const rows = 8n;
            const cols = 10n;

            // COO triplets (INTENTIONALLY unsorted by row)
            const rowInd = [7n, 0n, 2n, 5n, 2n, 0n, 7n];
            const colInd = [0n, 0n, 1n, 5n, 3n, 9n, 9n];
            const values = [
                await qInt(5),
                await qInt(1),
                await qInt(3),
                await qInt(4),
                await qInt(-1),
                await qInt(2),
                await qInt(-6),
            ];

            // Create sparse matrix (COO → CSR)
            const [aRows, aCols, _rowPtr, _colInd, _values] = await harness.createSparseFromTripletsHarness(rows, cols, rowInd, colInd, values);

            const rowPtr = [..._rowPtr];
            const cInd   = [..._colInd];
            const vals   = [..._values];

            /**
             * Dense vector x (10x1)
             * [1,2,3,4,5,6,7,8,9,10]^T
             */
            const xData = [
                await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5),
                await qInt(6), await qInt(7), await qInt(8), await qInt(9), await qInt(10)
            ];
            const xRows = cols;

            await touchGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, cInd, vals, xRows, xData]);
            const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [aRows, aCols, rowPtr, cInd, vals, xRows, xData]);

            const [yRows, yCols, yData] = await harness.mulSparseMatrixVectorHarness(aRows, aCols, rowPtr, cInd, vals, xRows, xData);

            // Shape check
            expect(yRows).to.equal(rows);
            expect(yCols).to.equal(1n);

            /**
             * Expected results:
             *
             * r0 = 1*1 + 2*10 = 21
             * r1 = 0
             * r2 = 3*2 + (-1)*4 = 2
             * r3 = 0
             * r4 = 0
             * r5 = 4*6 = 24
             * r6 = 0
             * r7 = 5*1 + (-6)*10 = -55
             */
            const expected = [
                await qInt(21),
                await qInt(0),
                await qInt(2),
                await qInt(0),
                await qInt(0),
                await qInt(24),
                await qInt(0),
                await qInt(-55),
            ];

            for (let i = 0; i < expected.length; i++) {
                if (await harness.toFloat(expected[i]) === 0n) {
                    expect(isQuadZero(yData[i])).to.equal(true);
                } else {
                    expect(yData[i]).to.equal(expected[i]);
                }
            }

            printBlockMatrix({
                t,
                method: "mulSparseMatrixVectorHarness",
                explanation: "Extreme sparse CSR matrix × vector stress test with empty rows and mixed signs.",
                gas,
                shapeIn: "8x10 * 10x1",
                shapeOut: "8x1",
                inHex: `rowPtr=${JSON.stringify(rowPtr.map(n => n.toString()))}, colInd=${JSON.stringify(cInd.map(n => n.toString()))}`,
                outHex: `y=${stringify(yData)}`
            });
        });
    });
});