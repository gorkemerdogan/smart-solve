// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Test suite validating MatrixMaster linear algebra multiplication operations using ABDK quad precision,
 *         including matrix–matrix multiplication, matrix–vector multiplication, and dot product.
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

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

    const qInt = async (n: number | string | bigint): Promise<string> =>
        await harness.qFromInt(BigInt(n));

    // ------------------------------------------------------------
    //  Section 6: Matrix multiplication
    // ------------------------------------------------------------

    describe("Section 6: Matrix multiplication", function () {
        it("Test 1: multiplyMatrices valid 2x3 · 3x2 multiplication", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];
            const B = [await qInt(7), await qInt(8), await qInt(9), await qInt(10), await qInt(11), await qInt(12)];

            await touchGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 2n, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 2n, B]);

            const C = asMatrix(await harness.mulMatrixHarness(2n, 3n, A, 3n, 2n, B));

            expect(C.rows).to.equal(2n);
            expect(C.cols).to.equal(2n);

            const expected = [await qInt(58), await qInt(64), await qInt(139), await qInt(154)];
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
                outHex: fmtHexArr(C.data)
            });
        });

        it("Test 2: multiplyMatrices shape mismatch reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)]; // 2x2
            const B = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)]; // 2x2

            await expect(harness.mulMatrixHarness(1n, 4n, A, 2n, 2n, B)).to.be.revertedWith("MatrixMaster: multiplyMatrices dims a.cols != b.rows");

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Ensures multiplication enforces a.cols == b.rows and fails on incompatible shapes.",
                gas: "Revert",
                shapeIn: "A:2x2, x:2x2",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: "-"
            });
        });

        it("Test 3: multiplyMatrices 10x10 bilinearity", async function () {
            t++;
            const n = 10n;
            const size = Number(n);
            const q0 = await qInt(0);
            const q1 = await qInt(1);
            const q2 = await qInt(2);

            const Adata: string[] = [];
            const Bdata: string[] = [];
            const Cdata: string[] = [];

            for (let i = 0; i < size; ++i) {
                for (let j = 0; j < size; ++j) {
                    const ai = (i + j) % 3; Adata.push(ai === 0 ? q0 : ai === 1 ? q1 : q2);
                    const bi = (i * j + 1) % 3; Bdata.push(bi === 0 ? q0 : bi === 1 ? q1 : q2);
                    const ci = (i + 2 * j) % 3; Cdata.push(ci === 0 ? q0 : ci === 1 ? q1 : q2);
                }
            }
            const A = { rows: n, cols: n, data: Adata };
            const B = { rows: n, cols: n, data: Bdata };
            const C = { rows: n, cols: n, data: Cdata };

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
                explanation: "Verifies full 10x10 bilinearity: (A+B)C = AC + BC.",
                gas,
                shapeIn: "A:10x10, B:10x10, C:10x10",
                shapeOut: `${left.rows}x${left.cols}`,
                inHex: `A=${fmtHexArr(A.data)}, B=${fmtHexArr(B.data)}, C=${fmtHexArr(C.data)}`,
                outHex: `left=${fmtHexArr(left.data)}, right=${fmtHexArr(right.data)}`
            });
        });

        it("Test 4: multiplyMatrices identity property", async function () {
            t++;
            const A = [await qInt(2), await qInt(3), await qInt(5), await qInt(7)];
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
                explanation: "Confirms invariance: AI = A and IA = A.",
                gas: `${gas1} / ${gas2}`,
                shapeIn: "A:2x2, I:2x2",
                shapeOut: "2x2",
                inHex: `A=${fmtHexArr(A)}, I=${fmtHexArr(I.data)}`,
                outHex: `AI=${fmtHexArr(AI.data)}, IA=${fmtHexArr(IA.data)}`
            });
        });

        it("Test 5: multiplyMatrices A·0 = 0 and 0·A = 0", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];
            const rowsA = 2n; const colsA = 3n;

            const zeroRight = asMatrix(await harness.zerosHarness(colsA, 2n));
            const zeroLeft = asMatrix(await harness.zerosHarness(rowsA, colsA));
            const zero = await qInt(0);

            // A·0
            await touchGas(harness, "mulMatrixHarness", [rowsA, colsA, A, zeroRight.rows, zeroRight.cols, zeroRight.data]);
            const gas1 = await estimateGas(harness, "mulMatrixHarness", [rowsA, colsA, A, zeroRight.rows, zeroRight.cols, zeroRight.data]);
            const AZ = asMatrix(await harness.mulMatrixHarness(rowsA, colsA, A, zeroRight.rows, zeroRight.cols, zeroRight.data));

            for (const v of AZ.data) expect(v.toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "A * ZeroMatrix = ZeroMatrix.",
                gas: gas1,
                shapeIn: "A:2x3, 0:3x2",
                shapeOut: `${AZ.rows}x${AZ.cols}`,
                inHex: `A=${fmtHexArr(A)}, 0=${fmtHexArr(zeroRight.data)}`,
                outHex: fmtHexArr(AZ.data)
            });

            // 0·A
            const A3x2 = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];
            await touchGas(harness, "mulMatrixHarness", [zeroLeft.rows, zeroLeft.cols, zeroLeft.data, 3n, 2n, A3x2]);
            const gas2 = await estimateGas(harness, "mulMatrixHarness", [zeroLeft.rows, zeroLeft.cols, zeroLeft.data, 3n, 2n, A3x2]);
            const ZA = asMatrix(await harness.mulMatrixHarness(zeroLeft.rows, zeroLeft.cols, zeroLeft.data, 3n, 2n, A3x2));

            for (const v of ZA.data) expect(v.toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "ZeroMatrix * A = ZeroMatrix.",
                gas: gas2,
                shapeIn: "0:2x3, A:3x2",
                shapeOut: `${ZA.rows}x${ZA.cols}`,
                inHex: `0=${fmtHexArr(zeroLeft.data)}, A=${fmtHexArr(A3x2)}`,
                outHex: fmtHexArr(ZA.data)
            });
        });

        it("Test 6: multiplyMatrices non-square 2x3 · 3x4", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];
            const B = [
                await qInt(1), await qInt(2), await qInt(3), await qInt(4),
                await qInt(5), await qInt(6), await qInt(7), await qInt(8),
                await qInt(9), await qInt(10), await qInt(11), await qInt(12),
            ];

            await touchGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 4n, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [2n, 3n, A, 3n, 4n, B]);

            const C = asMatrix(await harness.mulMatrixHarness(2n, 3n, A, 3n, 4n, B));
            expect(C.rows).to.equal(2n);
            expect(C.cols).to.equal(4n);

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies 2x3 by 3x4 -> 2x4 matrix.",
                gas,
                shapeIn: "A:2x3, B:3x4",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: fmtHexArr(C.data)
            });
        });

        it("Test 7: multiplyMatrices vector inner product", async function () {
            t++;
            const row = [await qInt(1), await qInt(2), await qInt(3)];
            const col = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "mulMatrixHarness", [1n, 3n, row, 3n, 1n, col]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [1n, 3n, row, 3n, 1n, col]);

            const result = asMatrix(await harness.mulMatrixHarness(1n, 3n, row, 3n, 1n, col));
            const expectedDot = await qInt(32);
            expect(result.data[0].toLowerCase()).to.equal(expectedDot.toLowerCase());

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Row vector * Col vector = Scalar (Inner Product).",
                gas,
                shapeIn: "1x3 · 3x1",
                shapeOut: `${result.rows}x${result.cols}`,
                inHex: `row=${fmtHexArr(row)}, col=${fmtHexArr(col)}`,
                outHex: fmtHexArr(result.data)
            });
        });

        it("Test 8: multiplyMatrices vector outer product", async function () {
            t++;
            const col = [await qInt(1), await qInt(2), await qInt(3)];
            const row = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "mulMatrixHarness", [3n, 1n, col, 1n, 3n, row]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [3n, 1n, col, 1n, 3n, row]);

            const result = asMatrix(await harness.mulMatrixHarness(3n, 1n, col, 1n, 3n, row));
            expect(result.rows).to.equal(3n);
            expect(result.cols).to.equal(3n);

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Col vector * Row vector = Matrix (Outer Product).",
                gas,
                shapeIn: "3x1 · 1x3",
                shapeOut: `${result.rows}x${result.cols}`,
                inHex: `col=${fmtHexArr(col)}, row=${fmtHexArr(row)}`,
                outHex: fmtHexArr(result.data)
            });
        });

        it("Test 9: multiplication stability with tiny entries", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const tiny = await harness.qFromFrac(1n, 1000n);
            const A = [tiny, tiny, tiny, tiny];
            const B = [tiny, tiny, tiny, tiny];

            await touchGas(harness, "mulMatrixHarness", [rows, cols, A, rows, cols, B]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [rows, cols, A, rows, cols, B]);

            const C = asMatrix(await harness.mulMatrixHarness(rows, cols, A, rows, cols, B));

            const tinyVal = await harness.toFloat(tiny);
            for (const v of C.data) {
                const val = await harness.toFloat(v);
                expect(val).to.be.greaterThan(0);
                expect(val).to.be.lessThan(tinyVal);
            }

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Multiplies two tiny-valued matrices; output should be small, positive, non-zero.",
                gas,
                shapeIn: `${rows}x${cols} · ${rows}x${cols}`,
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}, tiny=${tiny}`,
                outHex: fmtHexArr(C.data)
            });
        });

        it("Test 10: zero row propagation", async function () {
            t++;
            const one = await qInt(1);
            const zero = await qInt(0);
            const Adata = [one, one, one, one, one, one, one, one, zero, zero, zero, zero];
            const Bdata = [one, one, one, one, one, one, one, one]; // 4x2

            await touchGas(harness, "mulMatrixHarness", [3n, 4n, Adata, 4n, 2n, Bdata]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [3n, 4n, Adata, 4n, 2n, Bdata]);

            const C = asMatrix(await harness.mulMatrixHarness(3n, 4n, Adata, 4n, 2n, Bdata));
            const lastRowStart = Number(C.cols) * (Number(C.rows) - 1);

            expect(C.data[4].toLowerCase()).to.equal(zero.toLowerCase());
            expect(C.data[5].toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "mulMatrixHarness",
                explanation: "Zero row in A results in zero row in product C.",
                gas,
                shapeIn: "A:3x4, B:4x2",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(Adata)}, B=${fmtHexArr(Bdata)}`,
                outHex: fmtHexArr(C.data)
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 7: Matrix-Vector multiplication
    // ------------------------------------------------------------

    describe("Section 7: Matrix–vector multiplication", function () {
        it("Test 11: multiplyMatrixVector 3x3 · 3x1", async function () {
            t++;

            const A = [
                await qInt(1), await qInt(2), await qInt(3),
                await qInt(4), await qInt(5), await qInt(6),
                await qInt(7), await qInt(8), await qInt(9),
            ];

            const x = [await qInt(1), await qInt(0), await qInt(1)];

            await touchGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 3n, 1n, x]);
            const gas = await estimateGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 3n, 1n, x]);
            const out = asMatrix(await harness.mulMatrixVectorHarness(3n, 3n, A, 3n, 1n, x));

            expect(out.data[0]).to.equal(await qInt(4)); // 1*1 + 0 + 3*1

            printBlockMatrix({
                t,
                method: "mulMatrixVectorHarness",
                explanation: "Multiplies 3x3 matrix with 3x1 column vector.",
                gas,
                shapeIn: "A:3x3, x:3x1",
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(x)}`,
                outHex: fmtHexArr(out.data)
            });
        });

        it("Test 12: multiplyMatrixVector shape mismatch reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const xBad = [await qInt(1)];

            await expect(harness.mulMatrixVectorHarness(2n, 2n, A, 1n, 1n, xBad)).to.be.revertedWith("MatrixMaster: A.cols != x.rows");

            printBlockMatrix({
                t,
                method: "mulMatrixVectorHarness",
                explanation: "Ensures A.cols == x.rows is required.",
                gas: "Revert",
                shapeIn: "A:2x2, x:1x1",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(xBad)}`,
                outHex: "-"
            });
        });

        it("Test 13: multiplyMatrixVector by zero vector", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const xZero = [await qInt(0), await qInt(0)];

            await touchGas(harness, "mulMatrixVectorHarness", [2n, 2n, A, 2n, 1n, xZero]);
            const gas = await estimateGas(harness, "mulMatrixVectorHarness", [2n, 2n, A, 2n, 1n, xZero]);
            const out = asMatrix(await harness.mulMatrixVectorHarness(2n, 2n, A, 2n, 1n, xZero));

            const zero = await qInt(0);
            expect(out.data.every(v => v.toLowerCase() === zero.toLowerCase())).to.be.true;

            printBlockMatrix({
                t,
                method: "mulMatrixVectorHarness",
                explanation: "Mat * ZeroVector = ZeroVector.",
                gas,
                shapeIn: "A:2x2, x:2x1",
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(xZero)}`,
                outHex: fmtHexArr(out.data)
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 8: Dot product
    // ------------------------------------------------------------

    describe("Section 8: Dot product", function () {
        it("Test 14: dot basic", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3)];
            const b = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "dotHarness", [3n, 1n, a, 3n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [3n, 1n, a, 3n, 1n, b]);
            const out = await harness.dotHarness(3n, 1n, a, 3n, 1n, b);

            expect(out.toLowerCase()).to.equal((await qInt(32)).toLowerCase());

            printBlockMatrix({
                t,
                method: "dotHarness",
                explanation: "Computes dot product sum(a*b).",
                gas,
                shapeIn: "3x1 · 3x1",
                shapeOut: "scalar",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: out
            });
        });

        it("Test 15: dot handles negative entries", async function () {
            t++;
            const a = [await qInt(-2), await qInt(3)];
            const b = [await qInt(5), await qInt(-4)];

            await touchGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const out = await harness.dotHarness(2n, 1n, a, 2n, 1n, b);

            expect(out.toLowerCase()).to.equal((await qInt(-22)).toLowerCase());

            printBlockMatrix({
                t,
                method: "dotHarness",
                explanation: "Dot product with negative inputs.",
                gas,
                shapeIn: "2x1 · 2x1",
                shapeOut: "scalar",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: out
            });
        });

        it("Test 16: dot length mismatch reverts", async function () {
            t++;
            const a = [await qInt(1), await qInt(2)];
            const b = [await qInt(3)];

            await expect(harness.dotHarness(2n, 1n, a, 1n, 1n, b)).to.be.revertedWith("MatrixMaster: dot length mismatch");

            printBlockMatrix({
                t,
                method: "dotHarness",
                explanation: "Rejects mismatch vector lengths.",
                gas: "Revert",
                shapeIn: "2x1 · 1x1",
                shapeOut: "revert",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: "-"
            });
        });
    });
});