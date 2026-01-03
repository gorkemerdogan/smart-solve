// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests cover core linear algebra multiplication operations: matrix multiplication,
 *         matrix–vector multiplication, and dot product.
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;

    // matrix comparison helpers
    matricesExactEqual(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<boolean>;

    // creation (helper)
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    createIdentityMatrixHarness(n: bigint): Promise<[bigint, bigint, string[]]>;

    // elementwise arithmetic (helper)
    addHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // matrix multiplication
    mulMatrixHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // matrix-vector multiplication
    mulMatrixVectorHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // dot product
    dotHarness(xRows: bigint, xCols: bigint, xData: string[], yRows: bigint, yCols: bigint, yData: string[]): Promise<string>;
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

describe("MatrixMaster — Matrix multiplication, mat-vec, dot", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    const qInt = async (n: number | string | bigint): Promise<string> =>
        await harness.qFromInt(BigInt(n));

    // ------------------------------------------------------------
    //  Section 1: Matrix multiplication
    // ------------------------------------------------------------

    describe("Section 1: Matrix multiplication", function () {
        it("Test 1: multiplyMatrices valid 2x3 · 3x2 multiplication", async function () {
            t++;
            // A (2x3): [[1,2,3],[4,5,6]]
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ];
            // B (3x2): [[7,8],[9,10],[11,12]]
            const B = [
                await qInt(7),
                await qInt(8),
                await qInt(9),
                await qInt(10),
                await qInt(11),
                await qInt(12),
            ];

            await touchGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 2n, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 2n, B]);

            const C = asMatrix(await harness.mulMatrixHarness(2n, 3n, A, 3n, 2n, B));

            expect(C.rows).to.equal(2n);
            expect(C.cols).to.equal(2n);

            const expected = [
                await qInt(58),
                await qInt(64),
                await qInt(139),
                await qInt(154),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(C.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Computes a standard 2x3·3x2 product and validates dense triple-loop multiplication.",
                gas,
                shapeIn: "A:2x3, B:3x2",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: fmtHexArr(C.data),
            });
        });

        it("Test 2: multiplyMatrices shape mismatch reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)]; // 2x2
            const B = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)]; // 2x2

            await touchGas(harness, "mulMatrixHarness", [1n, 4n, A, 2n, 2n, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [1n, 4n, A, 2n, 2n, B]);

            await expect(
                harness.mulMatrixHarness(1n, 4n, A, 2n, 2n, B),
            ).to.be.revertedWith("MatrixMaster: multiplyMatrices dims a.cols != b.rows");

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Ensures multiplication enforces a.cols == b.rows and fails on incompatible shapes.",
                gas,
                shapeIn: "A:1x4, B:2x2",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: "-",
            });
        });

        it("Test 3: multiplyMatrices 10x10 bilinearity (A+B)·C == A·C + B·C", async function () {
            t++;
            const n = 10n;
            const size = Number(n);

            // Precompute quad ints 0,1,2 for convenience.
            const q0 = await qInt(0);
            const q1 = await qInt(1);
            const q2 = await qInt(2);

            const Adata: string[] = [];
            const Bdata: string[] = [];
            const Cdata: string[] = [];

            // Tiny integer patterns in {0,1,2}. All sums/products stay small,
            // so ABDK quad arithmetic is exact and bilinearity holds bit-for-bit.
            for (let i = 0; i < size; ++i) {
                for (let j = 0; j < size; ++j) {
                    const ai = (i + j) % 3; // 0,1,2
                    const bi = (i * j + 1) % 3; // 0,1,2
                    const ci = (i + 2 * j) % 3; // 0,1,2

                    Adata.push(ai === 0 ? q0 : ai === 1 ? q1 : q2);
                    Bdata.push(bi === 0 ? q0 : bi === 1 ? q1 : q2);
                    Cdata.push(ci === 0 ? q0 : ci === 1 ? q1 : q2);
                }
            }

            // Local JS-side matrices (no asMatrix here – that's only for contract tuples).
            const A = { rows: n, cols: n, data: Adata };
            const B = { rows: n, cols: n, data: Bdata };
            const C = { rows: n, cols: n, data: Cdata };

            // AplusB = A + B
            const AplusB = asMatrix(await harness.addHarness(A.rows, A.cols, A.data, B.rows, B.cols, B.data));

            await touchGas(harness, "mulMatrixHarness", [AplusB.rows, AplusB.cols, AplusB.data, C.rows, C.cols, C.data]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [AplusB.rows, AplusB.cols, AplusB.data, C.rows, C.cols, C.data]);

            const left = asMatrix(await harness.mulMatrixHarness(AplusB.rows, AplusB.cols, AplusB.data, C.rows, C.cols, C.data));
            const AC = asMatrix(await harness.mulMatrixHarness(A.rows, A.cols, A.data, C.rows, C.cols, C.data));
            const BC = asMatrix(await harness.mulMatrixHarness(B.rows, B.cols, B.data, C.rows, C.cols, C.data));
            const right = asMatrix(await harness.addHarness(AC.rows, AC.cols, AC.data, BC.rows, BC.cols, BC.data));

            expect(await harness.matricesExactEqual(left.rows, left.cols, left.data, right.rows, right.cols, right.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Verifies full 10x10 bilinearity with small integer entries where quad arithmetic is exact: (A+B)·C = A·C + B·C bit-for-bit.",
                gas,
                shapeIn: "A:10x10, B:10x10, C:10x10",
                shapeOut: `${left.rows}x${left.cols}`,
                inHex: `A=${fmtHexArr(A.data)}, B=${fmtHexArr(B.data)}, C=${fmtHexArr(C.data)}`,
                outHex: `left=${fmtHexArr(left.data)}, right=${fmtHexArr(right.data)}`,
            });
        });

        it("Test 4: multiplyMatrices identity property A·I == A and I·A == A", async function () {
            t++;
            // A 2x2
            const A = [
                await qInt(2),
                await qInt(3),
                await qInt(5),
                await qInt(7),
            ];
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            // A·I
            await touchGas(harness, "mulMatrixHarness", [2n, 2n, A, I.rows, I.cols, I.data]);
            const gas1 = await estimateGas(harness, "mulMatrixHarness", [2n, 2n, A, I.rows, I.cols, I.data]);
            const AI = asMatrix(await harness.mulMatrixHarness(2n, 2n, A, I.rows, I.cols, I.data));

            // I·A
            await touchGas(harness, "mulMatrixHarness", [I.rows, I.cols, I.data, 2n, 2n, A]);
            const gas2 = await estimateGas(harness, "mulMatrixHarness", [I.rows, I.cols, I.data, 2n, 2n, A]);
            const IA = asMatrix(await harness.mulMatrixHarness(I.rows, I.cols, I.data, 2n, 2n, A));

            expect(await harness.matricesExactEqual(2n, 2n, A, AI.rows, AI.cols, AI.data)).to.equal(true);
            expect(await harness.matricesExactEqual(2n, 2n, A, IA.rows, IA.cols, IA.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies a 2x2 matrix by the identity on both sides and confirms invariance of A.",
                gas: `${gas1} / ${gas2}`,
                shapeIn: "A:2x2, I:2x2",
                shapeOut: "2x2",
                inHex: `A=${fmtHexArr(A)}, I=${fmtHexArr(I.data)}`,
                outHex: `AI=${fmtHexArr(AI.data)}, IA=${fmtHexArr(IA.data)}`,
            });
        });

        it("Test 5: multiplyMatrices A·0 = 0 and 0·A = 0", async function () {
            t++;
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ]; // 2x3
            const rowsA = 2n;
            const colsA = 3n;

            const zeroRight = asMatrix(await harness.zerosHarness(colsA, 2n)); // 3x2
            const zeroLeft = asMatrix(await harness.zerosHarness(rowsA, colsA)); // 2x3

            const zero = await qInt(0);

            // A·0 (2x3 · 3x2 → 2x2)
            await touchGas(harness, "mulMatrixHarness", [rowsA, colsA, A, zeroRight.rows, zeroRight.cols, zeroRight.data]);
            const gas1 = await estimateGas(harness, "mulMatrixHarness", [rowsA, colsA, A, zeroRight.rows, zeroRight.cols, zeroRight.data]);
            const AZ = asMatrix(await harness.mulMatrixHarness(rowsA, colsA, A, zeroRight.rows, zeroRight.cols, zeroRight.data));

            for (const v of AZ.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies A by a zero matrix on the right and confirms the result is all zeros.",
                gas: gas1,
                shapeIn: "A:2x3, 0:3x2",
                shapeOut: `${AZ.rows}x${AZ.cols}`,
                inHex: `A=${fmtHexArr(A)}, 0=${fmtHexArr(zeroRight.data)}`,
                outHex: fmtHexArr(AZ.data),
            });

            // 0·A (2x3 zero · 3x2 A)
            const A3x2 = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ];

            await touchGas(harness, "mulMatrixHarness", [zeroLeft.rows, zeroLeft.cols, zeroLeft.data, 3n, 2n, A3x2]);
            const gas2 = await estimateGas(harness, "mulMatrixHarness", [zeroLeft.rows, zeroLeft.cols, zeroLeft.data, 3n, 2n, A3x2]);
            const ZA = asMatrix(await harness.mulMatrixHarness(zeroLeft.rows, zeroLeft.cols, zeroLeft.data, 3n, 2n, A3x2));

            for (const v of ZA.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies a zero matrix on the left by A and ensures the result is still all zeros.",
                gas: gas2,
                shapeIn: "0:2x3, A:3x2",
                shapeOut: `${ZA.rows}x${ZA.cols}`,
                inHex: `0=${fmtHexArr(zeroLeft.data)}, A=${fmtHexArr(A3x2)}`,
                outHex: fmtHexArr(ZA.data),
            });
        });

        it("Test 6: multiplyMatrices non-square 2x3 · 3x4 multiplication", async function () {
            t++;
            // A (2x3): [[1,2,3],[4,5,6]]
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ];
            // B (3x4): [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
            const B = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
                await qInt(7),
                await qInt(8),
                await qInt(9),
                await qInt(10),
                await qInt(11),
                await qInt(12),
            ];

            await touchGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 4n, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 4n, B]);

            const C = asMatrix(await harness.mulMatrixHarness(2n, 3n, A, 3n, 4n, B));

            expect(C.rows).to.equal(2n);
            expect(C.cols).to.equal(4n);

            const expected = [
                await qInt(38),
                await qInt(44),
                await qInt(50),
                await qInt(56),
                await qInt(83),
                await qInt(98),
                await qInt(113),
                await qInt(128),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(C.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies a 2x3 matrix by a 3x4 matrix and validates each of the 8 output entries.",
                gas,
                shapeIn: "A:2x3, B:3x4",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: fmtHexArr(C.data),
            });
        });

        it("Test 7: multiplyMatrices vector inner product (1xN · Nx1)", async function () {
            t++;

            const row = [await qInt(1), await qInt(2), await qInt(3)]; // row: [1,2,3] as 1x3
            const col = [await qInt(4), await qInt(5), await qInt(6)]; // col: [4,5,6] as 3x1

            await touchGas(harness, "mulMatrixHarness", [1n, 3n, row, 3n, 1n, col]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [1n, 3n, row, 3n, 1n, col]);

            const result = asMatrix(
                await harness.mulMatrixHarness(1n, 3n, row, 3n, 1n, col),
            );

            expect(result.rows).to.equal(1n);
            expect(result.cols).to.equal(1n);

            const expectedDot = await qInt(32); // 1*4 + 2*5 + 3*6 = 32
            expect(result.data[0].toLowerCase()).to.equal(expectedDot.toLowerCase());

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Performs a 1x3 · 3x1 inner product and checks the resulting 1x1 scalar equals the dot product.",
                gas,
                shapeIn: "1x3 · 3x1",
                shapeOut: `${result.rows}x${result.cols}`,
                inHex: `row=${fmtHexArr(row)}, col=${fmtHexArr(col)}`,
                outHex: fmtHexArr(result.data),
            });
        });

        it("Test 8: multiplyMatrices vector outer product (Nx1 · 1xN)", async function () {
            t++;

            const col = [await qInt(1), await qInt(2), await qInt(3)]; // col: [1,2,3] as 3x1
            const row = [await qInt(4), await qInt(5), await qInt(6)]; // row: [4,5,6] as 1x3

            await touchGas(harness, "mulMatrixHarness", [3n, 1n, col, 1n, 3n, row]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [3n, 1n, col, 1n, 3n, row]);

            const result = asMatrix(
                await harness.mulMatrixHarness(3n, 1n, col, 1n, 3n, row),
            );

            expect(result.rows).to.equal(3n);
            expect(result.cols).to.equal(3n);

            const expected = [
                await qInt(4),
                await qInt(5),
                await qInt(6),
                await qInt(8),
                await qInt(10),
                await qInt(12),
                await qInt(12),
                await qInt(15),
                await qInt(18),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(result.data[i].toLowerCase()).to.equal(
                    expected[i].toLowerCase(),
                );
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Computes the outer product of a 3x1 and 1x3 vector and verifies the resulting 3x3 rank-1 matrix.",
                gas,
                shapeIn: "3x1 · 1x3",
                shapeOut: `${result.rows}x${result.cols}`,
                inHex: `col=${fmtHexArr(col)}, row=${fmtHexArr(row)}`,
                outHex: fmtHexArr(result.data),
            });
        });

        it("Test 9: multiplyMatrices stability with tiny entries (no underflow, non-zero result)", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;

            const tiny = await harness.qFromFrac(1n, 1000n); // tiny = 1 / 1000

            // A and B both 2x2 filled with 'tiny'
            const A = [tiny, tiny, tiny, tiny];
            const B = [tiny, tiny, tiny, tiny];

            await touchGas(harness, "mulMatrixHarness", [rows, cols, A, rows, cols, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [rows, cols, A, rows, cols, B]);

            const C = asMatrix(await harness.mulMatrixHarness(rows, cols, A, rows, cols, B));

            const zeroQ = await qInt(0);
            const zeroBI = BigInt(zeroQ);
            const tinyBI = BigInt(tiny);

            // Each entry should be positive and strictly smaller than 'tiny'
            // (since C_ij ≈ 2 * tiny^2 and tiny < 1).
            for (const v of C.data) {
                const cBI = BigInt(v);
                expect(cBI).to.be.gt(zeroBI);
                expect(cBI).to.be.lt(tinyBI);
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies two tiny-valued 2x2 matrices and checks outputs remain small, positive, and non-zero (no underflow).",
                gas,
                shapeIn: `${rows}x${cols} · ${rows}x${cols}`,
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}, tiny=${tiny}`,
                outHex: fmtHexArr(C.data),
            });
        });

        it("Test 10: multiplyMatrices zero row in A produces zero row in A·B", async function () {
            t++;
            const rowsA = 3n;
            const colsA = 4n;
            const rowsB = 4n;
            const colsB = 2n;

            const one = await qInt(1);
            const zero = await qInt(0);

            // A = [[1,1,1,1],
            //      [1,1,1,1],
            //      [0,0,0,0]]
            const Adata = [
                one, one, one, one,
                one, one, one, one,
                zero, zero, zero, zero,
            ];

            // B = [[1,2],
            //      [3,4],
            //      [5,6],
            //      [7,8]]
            const Bdata = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
                await qInt(7),
                await qInt(8),
            ];

            await touchGas(harness, "mulMatrixHarness", [rowsA, colsA, Adata, rowsB, colsB, Bdata]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [rowsA, colsA, Adata, rowsB, colsB, Bdata]);

            const C = asMatrix(await harness.mulMatrixHarness(rowsA, colsA, Adata, rowsB, colsB, Bdata));

            expect(C.rows).to.equal(rowsA);
            expect(C.cols).to.equal(colsB);

            // Last row of C must be all zeros because last row of A is all zeros.
            const colsOut = Number(colsB);
            const lastRowStart = (Number(rowsA) - 1) * colsOut;

            for (let j = 0; j < colsOut; ++j) {
                const v = C.data[lastRowStart + j];
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Uses a 3x4 matrix with a fully zero last row and checks the corresponding row of A·B remains exactly zero.",
                gas,
                shapeIn: "A:3x4, B:4x2",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(Adata)}, B=${fmtHexArr(Bdata)}`,
                outHex: fmtHexArr(C.data),
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Matrix-Vector multiplication
    // ------------------------------------------------------------

    describe("Section 2: Matrix–vector multiplication", function () {
        it("Test 11: multiplyMatrixVector 3x3 · 3x1 produces correct 3x1", async function () {
            t++;

            const A = [
                await qInt(1), await qInt(2), await qInt(3),
                await qInt(4), await qInt(5), await qInt(6),
                await qInt(7), await qInt(8), await qInt(9),
            ]; // 3×3

            const x = [
                await qInt(1),
                await qInt(0),
                await qInt(1),
            ]; // 3×1

            await touchGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 3n, 1n, x]);
            const gas = await estimateGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 3n, 1n, x]);

            const out = asMatrix(await harness.mulMatrixVectorHarness(3n, 3n, A, 3n, 1n, x));

            const e0 = await qInt(4);
            const e1 = await qInt(10);
            const e2 = await qInt(16);

            expect(out.rows).to.equal(3n);
            expect(out.cols).to.equal(1n);
            expect(out.data[0]).to.equal(e0);
            expect(out.data[1]).to.equal(e1);
            expect(out.data[2]).to.equal(e2);

            printBlockMatrix({
                t,
                method: "mulMatrixVectorHarness",
                explanation: "Multiplies a 3x3 dense quad matrix with a 3x1 column vector and validates row-wise dot-products.",
                gas,
                shapeIn: "A:3x3, x:3x1",
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(x)}`,
                outHex: fmtHexArr(out.data),
            });
        });

        it("Test 12: multiplyMatrixVector shape mismatch reverts", async function () {
            t++;

            const A = [
                await qInt(1), await qInt(2), await qInt(3),
                await qInt(4), await qInt(5), await qInt(6),
                await qInt(7), await qInt(8), await qInt(9),
            ]; // 3×3

            const xBad = [await qInt(1), await qInt(2)]; // 2×1

            await touchGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 2n, 1n, xBad]);
            const gas = await estimateGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 2n, 1n, xBad]);

            await expect(
                harness.mulMatrixVectorHarness(3n, 3n, A, 2n, 1n, xBad),
            ).to.be.revertedWith("MatrixMaster: A.cols != x.rows");

            printBlockMatrix({
                t,
                method: "mulMatrixVectorHarness",
                explanation: "Ensures A.cols == x.rows is required for matrix–vector multiplication; mismatched dims revert.",
                gas,
                shapeIn: "A:3x3, x:2x1",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(xBad)}`,
                outHex: "-",
            });
        });

        it("Test 13: multiplyMatrixVector multiplying by zero vector yields zero output", async function () {
            t++;

            const A = [
                await qInt(1), await qInt(2), await qInt(3), await qInt(4),
                await qInt(5), await qInt(6), await qInt(7), await qInt(8),
                await qInt(9), await qInt(10), await qInt(11), await qInt(12),
                await qInt(13), await qInt(14), await qInt(15), await qInt(16),
            ]; // 4×4

            const zero = await qInt(0);
            const xZero = [zero, zero, zero, zero]; // 4×1

            await touchGas(harness, "mulMatrixVectorHarness", [4n, 4n, A, 4n, 1n, xZero]);
            const gas = await estimateGas(harness, "mulMatrixVectorHarness", [4n, 4n, A, 4n, 1n, xZero]);

            const out = asMatrix(await harness.mulMatrixVectorHarness(4n, 4n, A, 4n, 1n, xZero));

            expect(out.rows).to.equal(4n);
            expect(out.cols).to.equal(1n);
            expect(out.data.every((v) => v.toLowerCase() === zero.toLowerCase())).to.equal(true);

            printBlockMatrix({
                t,
                method: "mulMatrixVectorHarness",
                explanation: "Verifies that A·0 = 0 holds for quad-precision matrices: multiplying any 4x4 matrix by a zero vector returns a zero column vector.",
                gas,
                shapeIn: "A:4x4, x:4x1",
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(xZero)}`,
                outHex: fmtHexArr(out.data),
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 3: Dot product
    // ------------------------------------------------------------

    describe("Section 3: Dot product", function () {
        it("Test 14: dot basic 3-element vectors", async function () {
            t++;

            const a = [await qInt(1), await qInt(2), await qInt(3)];
            const b = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "dotHarness", [3n, 1n, a, 3n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [3n, 1n, a, 3n, 1n, b]);

            const out = await harness.dotHarness(3n, 1n, a, 3n, 1n, b);
            const expected = await qInt(32);

            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "dotHarness",
                explanation: "Computes the dot product of two 3x1 vectors and confirms the scalar result matches Sum(aᵢbᵢ).",
                gas,
                shapeIn: "3x1 · 3x1",
                shapeOut: "scalar",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: out,
            });
        });

        it("Test 15: dot handles negative entries", async function () {
            t++;

            const a = [await qInt(-2), await qInt(3)];
            const b = [await qInt(5), await qInt(-4)];

            await touchGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);

            const out = await harness.dotHarness(2n, 1n, a, 2n, 1n, b);
            const expected = await qInt(-22);

            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "dotHarness",
                explanation: "Uses 2x1 vectors with mixed signs and checks the dot product correctly accumulates signed contributions.",
                gas,
                shapeIn: "2x1 · 2x1",
                shapeOut: "scalar",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: out,
            });
        });

        it("Test 16: dot length mismatch reverts", async function () {
            t++;

            const a = [await qInt(1), await qInt(2)];
            const b = [await qInt(3)];

            await touchGas(harness, "dotHarness", [2n, 1n, a, 1n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 1n, 1n, b]);

            await expect(
                harness.dotHarness(2n, 1n, a, 1n, 1n, b),
            ).to.be.revertedWith("MatrixMaster: dot length mismatch");

            printBlockMatrix({
                t,
                method: "dotHarness",
                explanation: "Rejects mismatched vector lengths for the dot product and reverts with an explicit error.",
                gas,
                shapeIn: "2x1 · 1x1",
                shapeOut: "revert",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });
    });
});