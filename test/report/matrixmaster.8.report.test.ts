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
export const stringify = (data: any): string =>
    { return JSON.stringify(data, (key, value) => typeof value === 'bigint' ? value.toString() : value); };

// Formats CSR matrix components into a readable string for reports.
const fmtCSR = (rowPtr: bigint[], colInd: bigint[], values: string[]): string =>
    { return `rowPtr=${stringify(rowPtr)}, colInd=${stringify(colInd)}, values=${stringify(values)}`; };

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

        it("Test 1: createZeroSparse (3x4)", async function () {
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

        it("Test 2: createZeroSparse (1x1)", async function () {
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

        it("Test 3: createZeroSparse with zero rows reverts", async function () {
            await expect(harness.createZeroSparseHarness(0n, 3n)).to.be.reverted;
        });

        it("Test 4: createIdentitySparse (3x3)", async function () {
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

        it("Test 5: createIdentitySparse (1x1)", async function () {
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

        it("Test 6: createIdentitySparse with n=0 reverts", async function () {
            await expect(harness.createIdentitySparseHarness(0n)).to.be.reverted;
        });

        it("Test 7: createDiagonalSparse ([1,2,3])", async function () {
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

        it("Test 8: createDiagonalSparse ([5])", async function () {
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

        it("Test 9: createDiagonalSparse empty diag reverts", async function () {
            await expect(harness.createDiagonalSparseHarness([])).to.be.reverted;
        });

        it("Test 10: createSparseFromTriplets (simple)", async function () {
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

        it("Test 11: createSparseFromTriplets (unsorted COO)", async function () {
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

        it("Test 12: createSparseFromTriplets length mismatch reverts", async function () {
            await expect(harness.createSparseFromTripletsHarness(2n, 2n, [0n], [0n, 1n], [await qInt(1)])).to.be.reverted;
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Sparse Matrix - Vector Multiplication
    // ------------------------------------------------------------

    describe("Section 2: Sparse Matrix - Vector Multiplication", function () {
    });

});
