// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * MatrixMaster (bytes16 / ABDK quad) : Creation, Access, Algebra, det, inverse
 */

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromUInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;

    // matrix comparison helpers
    matricesExactEqual(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<boolean>;

    // creation
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    onesHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    createIdentityMatrixHarness(n: bigint): Promise<[bigint, bigint, string[]]>;
    fromDiagonalHarness(diag: string[]): Promise<[bigint, bigint, string[]]>;
    randomMatrixHarness(rows: bigint, cols: bigint, seed: string): Promise<[bigint, bigint, string[]]>;

    // element access
    getHarness(rows: bigint, cols: bigint, dataFlat: string[], row: bigint, col: bigint): Promise<string>;
    setHarness(rows: bigint, cols: bigint, dataFlat: string[], row: bigint, col: bigint, val: string): Promise<[bigint, bigint, string[]]>;

    // slice & reshape
    sliceHarness(rows: bigint, cols: bigint, dataFlat: string[], rowStart: bigint, rowEnd: bigint, colStart: bigint, colEnd: bigint): Promise<[bigint, bigint, string[]]>;
    reshapeHarness(rows: bigint, cols: bigint, dataFlat: string[], newRows: bigint, newCols: bigint): Promise<[bigint, bigint, string[]]>;

    // transpose
    transposeHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;

    // elementwise arithmetic
    addHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;
    subHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;
    mulScalarHarness(rows: bigint, cols: bigint, dataFlat: string[], k: string): Promise<[bigint, bigint, string[]]>;
    divScalarHarness(rows: bigint, cols: bigint, dataFlat: string[], k: string): Promise<[bigint, bigint, string[]]>;

    // matrix multiplication
    mulMatrixHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // matrix-vector multiplication
    mulMatrixVectorHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // dot product
    dotHarness(xRows: bigint, xCols: bigint, xData: string[], yRows: bigint, yCols: bigint, yData: string[]): Promise<string>;

    // determinant & inverse
    detHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<string>;
    inverseHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;

    // normalization
    normalizeVectorHarness(vRows: bigint, vCols: bigint, vData: string[]): Promise<[bigint, bigint, string[]]>;

    // convergence
    hasConvergedHarness(xRows: bigint, xCols: bigint, xData: string[], yRows: bigint, yCols: bigint, yData: string[], tol: string): Promise<boolean>;
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

// Small helpers for working with the harness
function asMatrix(tuple: [bigint, bigint, string[]]) {
    const [rows, cols, data] = tuple;
    return { rows, cols, data: [...data] };
}

// =========================
// Gas / report utilities
// =========================

/** Fire a real transaction (swallowed errors) so gas reporters can pick it up. */
async function touchGas(
    h: MatrixMasterHarness,
    method: string,
    args: any[],
) {
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

/** Estimate gas for any method; on revert returns a descriptive string. */
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
        return "revert / estimation failed";
    }
}

/** Pretty-print a report block for a single matrix-related test. */
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
    const { t, method, explanation, gas, shapeIn, shapeOut, inHex, outHex } =
        options;
    const sep = "-".repeat(60);
    console.log(
        `\n${sep}\nTest ${t}\nMethod: ${method}\nExplanation: ${explanation}\nGas Usage: ${gas}\nInput shape: ${shapeIn}\nOutput shape: ${shapeOut}\nInput (hex): ${inHex ?? "-"}\nOutput (hex): ${outHex ?? "-"}`,
    );
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

// =========================
/** Main test suite */
// =========================

describe("MatrixMaster (library) : dense matrices over ABDK quad", function () {
    let harness: MatrixMasterHarness;
    let t = 0; // global test counter for reporting

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    async function qInt(n: number | string | bigint): Promise<string> {
        return harness.qFromInt(BigInt(n));
    }

    async function qUInt(n: number | string | bigint): Promise<string> {
        return harness.qFromUInt(BigInt(n));
    }

    // =========================================================
    // Matrix multiplication
    // =========================================================

    describe("Matrix multiplication", function () {
        it("mulMatrix : valid multiplication", async function () {
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

            const expected = [await qInt(58), await qInt(64), await qInt(139), await qInt(154)];

            for (let i = 0; i < expected.length; ++i) {
                expect(C.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Computes a standard 2x3·3x2 product and validates dense triple-loop multiplication.",
                gas,
                shapeIn: "A:2x3, B:3x2",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: fmtHexArr(C.data),
            });
        });

        it("mulMatrix : shape mismatch reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)]; // 2x2
            const B = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)]; // 2x2

            await touchGas(harness, "mulMatrixHarness", [1n, 4n, A, 2n, 2n, B]);

            const gas = await estimateGas(harness, "mulMatrixHarness", [1n, 4n, A, 2n, 2n, B]);
            await expect(harness.mulMatrixHarness(1n, 4n, A, 2n, 2n, B)).to.be.revertedWith("MatrixMaster: mulMatrix dims a.cols != b.rows");

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Ensures multiplication enforces a.cols == b.rows and fails on incompatible shapes.",
                gas,
                shapeIn: "A:1x4, B:2x2",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: "-",
            });
        });

        it("mulMatrix : 10x10 bilinearity (A+B)·C == A·C + B·C (small integer matrices)", async function () {
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
                    const ai = (i + j) % 3;      // 0,1,2
                    const bi = (i * j + 1) % 3;  // 0,1,2
                    const ci = (i + 2 * j) % 3;  // 0,1,2

                    Adata.push(ai === 0 ? q0 : ai === 1 ? q1 : q2);
                    Bdata.push(bi === 0 ? q0 : bi === 1 ? q1 : q2);
                    Cdata.push(ci === 0 ? q0 : ci === 1 ? q1 : q2);
                }
            }

            // Local JS-side matrices (no asMatrix here – that's only for contract tuples)
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

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Verifies full 10x10 bilinearity using small integer matrices where quad arithmetic is exact: (A+B)·C = A·C + B·C bit-for-bit.",
                gas,
                shapeIn: "A:10x10, B:10x10, C:10x10",
                shapeOut: `${left.rows}x${left.cols}`,
                inHex: `A=${fmtHexArr(A.data)}, B=${fmtHexArr(B.data)}, C=${fmtHexArr(C.data)}`,
                outHex: `left=${fmtHexArr(left.data)}, right=${fmtHexArr(right.data)}`,
            });
        });

        it("mulMatrix : identity property A·I == A and I·A == A", async function () {
            t++;
            // A 2x2
            const A = [
                await qInt(2),
                await qInt(3),
                await qInt(5),
                await qInt(7),
            ];
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            await touchGas(harness, "mulMatrixHarness", [2n, 2n, A, I.rows, I.cols, I.data]);
            const gas1 = await estimateGas(harness, "mulMatrixHarness", [2n, 2n, A, I.rows, I.cols, I.data]);
            const AI = asMatrix(await harness.mulMatrixHarness(2n, 2n, A, I.rows, I.cols, I.data));

            await touchGas(harness, "mulMatrixHarness", [I.rows, I.cols, I.data, 2n, 2n, A]);
            const gas2 = await estimateGas(harness, "mulMatrixHarness", [I.rows, I.cols, I.data, 2n, 2n, A]);
            const IA = asMatrix(await harness.mulMatrixHarness(I.rows, I.cols, I.data, 2n, 2n, A));

            expect(await harness.matricesExactEqual(2n, 2n, A, AI.rows, AI.cols, AI.data)).to.equal(true);
            expect(await harness.matricesExactEqual(2n, 2n, A, IA.rows, IA.cols, IA.data)).to.equal(true);

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Multiplies a matrix by the identity on both sides and confirms invariance of A.",
                gas: `${gas1} / ${gas2}`,
                shapeIn: "A:2x2, I:2x2",
                shapeOut: "2x2",
                inHex: `A=${fmtHexArr(A)}, I=${fmtHexArr(I.data)}`,
                outHex: `AI=${fmtHexArr(AI.data)}, IA=${fmtHexArr(IA.data)}`,
            });
        });

        it("mulMatrix : A·0 = 0 and 0·A = 0", async function () {
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

            const zeroLeft = asMatrix(await harness.zerosHarness(colsA, 2n)); // 3x2
            const zeroRight = asMatrix(await harness.zerosHarness(rowsA, colsA)); // 2x3

            // A·0 (2x3 · 3x2 → 2x2)
            await touchGas(harness, "mulMatrixHarness", [rowsA, colsA, A, zeroLeft.rows, zeroLeft.cols, zeroLeft.data]);
            const gas1 = await estimateGas(harness, "mulMatrixHarness", [rowsA, colsA, A, zeroLeft.rows, zeroLeft.cols, zeroLeft.data]);
            const AZ = asMatrix(await harness.mulMatrixHarness(rowsA, colsA, A, zeroLeft.rows, zeroLeft.cols, zeroLeft.data));

            const zero = await qInt(0);
            for (const v of AZ.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }
            t++;

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Multiplies A by a zero matrix on the right and confirms the result is all zeros.",
                gas: gas1,
                shapeIn: "A:2x3, 0:3x2",
                shapeOut: `${AZ.rows}x${AZ.cols}`,
                inHex: `A=${fmtHexArr(A)}, 0=${fmtHexArr(zeroLeft.data)}`,
                outHex: fmtHexArr(AZ.data),
            });

            // 0·A (2x3 zero · 3x3? keep 0:2x3 · A:3x2)
            const zeroMatLeft = asMatrix(await harness.zerosHarness(rowsA, colsA)); // 2x3
            const A3x2 = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];
            await touchGas(harness, "mulMatrixHarness", [zeroMatLeft.rows, zeroMatLeft.cols, zeroMatLeft.data, 3n, 2n, A3x2]);
            const gas2 = await estimateGas(harness, "mulMatrixHarness", [zeroMatLeft.rows, zeroMatLeft.cols, zeroMatLeft.data, 3n, 2n, A3x2]);
            const ZA = asMatrix(await harness.mulMatrixHarness(zeroMatLeft.rows, zeroMatLeft.cols, zeroMatLeft.data, 3n, 2n, A3x2));

            for (const v of ZA.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Multiplies a zero matrix on the left by A and ensures the result is still all zeros.",
                gas: gas2,
                shapeIn: "0:2x3, A:3x2",
                shapeOut: `${ZA.rows}x${ZA.cols}`,
                inHex: `0=${fmtHexArr(zeroMatLeft.data)}, A=${fmtHexArr(A3x2)}`,
                outHex: fmtHexArr(ZA.data),
            });
        });

        it("mulMatrix : non-square 2×3 · 3×4 multiplication", async function () {
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

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Multiplies a 2x3 matrix by a 3x4 matrix and validates each of the 8 output entries.",
                gas,
                shapeIn: "A:2x3, B:3x4",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}`,
                outHex: fmtHexArr(C.data),
            });
        });

        it("mulMatrix : vector inner product (1×N · N×1)", async function () {
            t++;
            // row: [1,2,3] as 1x3
            const row = [await qInt(1), await qInt(2), await qInt(3)];
            // col: [4,5,6] as 3x1
            const col = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "mulMatrixHarness", [1n, 3n, row, 3n, 1n, col]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [1n, 3n, row, 3n, 1n, col]);
            const result = asMatrix(await harness.mulMatrixHarness(1n, 3n, row, 3n, 1n, col));

            expect(result.rows).to.equal(1n);
            expect(result.cols).to.equal(1n);

            const expectedDot = await qInt(32); // 1*4 + 2*5 + 3*6 = 32
            expect(result.data[0].toLowerCase()).to.equal(expectedDot.toLowerCase());

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Performs a 1x3 · 3x1 inner product and checks the resulting 1x1 scalar equals the dot product.",
                gas,
                shapeIn: "1x3 · 3x1",
                shapeOut: `${result.rows}x${result.cols}`,
                inHex: `row=${fmtHexArr(row)}, col=${fmtHexArr(col)}`,
                outHex: fmtHexArr(result.data),
            });
        });

        it("mulMatrix : vector outer product (N×1 · 1×N)", async function () {
            t++;
            // col: [1,2,3] as 3x1
            const col = [await qInt(1), await qInt(2), await qInt(3)];
            // row: [4,5,6] as 1x3
            const row = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "mulMatrixHarness", [3n, 1n, col, 1n, 3n, row]);
            const gas = await estimateGas(harness, "mulMatrixHarness", [3n, 1n, col, 1n, 3n, row]);
            const result = asMatrix(await harness.mulMatrixHarness(3n, 1n, col, 1n, 3n, row));

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

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Computes the outer product of a 3x1 and 1x3 vector and verifies the resulting 3x3 rank-1 matrix.",
                gas,
                shapeIn: "3x1 · 1x3",
                shapeOut: `${result.rows}x${result.cols}`,
                inHex: `col=${fmtHexArr(col)}, row=${fmtHexArr(row)}`,
                outHex: fmtHexArr(result.data),
            });
        });

        it("mulMatrix : stability with tiny entries (no underflow, result stays small but non-zero)", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;

            // tiny = 1 / 1000
            const tiny = await harness.qFromFrac(1n, 1000n);

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

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Multiplies two tiny-valued matrices and checks outputs remain small, positive, and non-zero (no underflow).",
                gas,
                shapeIn: `${rows}x${cols} · ${rows}x${cols}`,
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(A)}, B=${fmtHexArr(B)}, tiny=${tiny}`,
                outHex: fmtHexArr(C.data),
            });
        });

        it("mulMatrix : zero row in A produces zero row in A·B", async function () {
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

            printBlock({
                t,
                method: "mulMatrixHarness",
                explanation:
                    "Uses a 3x4 matrix with a fully zero last row and checks the corresponding row of A·B remains exactly zero.",
                gas,
                shapeIn: "A:3x4, B:4x2",
                shapeOut: `${C.rows}x${C.cols}`,
                inHex: `A=${fmtHexArr(Adata)}, B=${fmtHexArr(Bdata)}`,
                outHex: fmtHexArr(C.data),
            });
        });
    });

    // =========================================================
    // Matrix-Vector multiplication
    // =========================================================

    describe("Matrix Vector multiplication", function () {
        it("mulMatrixVector : 3x3 * 3x1 produces correct 3x1 result", async function () {
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

            printBlock({
                t,
                method: "mulMatrixVectorHarness",
                explanation:
                    "Multiplies a 3×3 dense quad matrix with a 3×1 column vector and validates rowwise dot-products.",
                gas,
                shapeIn: "A:3x3, x:3x1",
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(x)}`,
                outHex: fmtHexArr(out.data),
            });
        });

        it("mulMatrixVector : shape mismatch reverts", async function () {
            t++;

            const A = [
                await qInt(1), await qInt(2), await qInt(3),
                await qInt(4), await qInt(5), await qInt(6),
                await qInt(7), await qInt(8), await qInt(9),
            ]; // 3×3

            const xBad = [
                await qInt(1),
                await qInt(2),
            ]; // 2×1

            await touchGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 2n, 1n, xBad]);
            const gas = await estimateGas(harness, "mulMatrixVectorHarness", [3n, 3n, A, 2n, 1n, xBad]);

            await expect(
                harness.mulMatrixVectorHarness(3n, 3n, A, 2n, 1n, xBad)
            ).to.be.revertedWith("MatrixMaster: mulMatrix dims a.cols != b.rows");

            printBlock({
                t,
                method: "mulMatrixVectorHarness",
                explanation:
                    "Ensures A.cols == x.rows is required for matrix–vector multiplication; mismatched dims revert.",
                gas,
                shapeIn: "A:3x3, x:2x1",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(xBad)}`,
                outHex: "-",
            });
        });

        it("mulMatrixVector : multiplying by zero vector yields zero output", async function () {
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
            expect(out.data.every(v => v.toLowerCase() === zero.toLowerCase())).to.equal(true);

            printBlock({
                t,
                method: "mulMatrixVectorHarness",
                explanation:
                    "Verifies that A·0 = 0 holds for quad-precision matrices: multiplying any 4×4 matrix by a zero vector returns a zero column vector.",
                gas,
                shapeIn: "A:4x4, x:4x1",
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: `A=${fmtHexArr(A)}, x=${fmtHexArr(xZero)}`,
                outHex: fmtHexArr(out.data),
            });
        });
    });

    // =========================================================
    // Dot product
    // =========================================================
    describe("Dot product", function () {
        it("dot : basic 3-element vectors", async function () {
            t++;

            const a = [await qInt(1), await qInt(2), await qInt(3)];
            const b = [await qInt(4), await qInt(5), await qInt(6)];

            await touchGas(harness, "dotHarness", [3n, 1n, a, 3n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [3n, 1n, a, 3n, 1n, b]);
            const out = await harness.dotHarness(3n, 1n, a, 3n, 1n, b);

            const expected = await qInt(32);

            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            printBlock({
                t,
                method: "dotHarness",
                explanation: "Computes dot product for 3-element vectors.",
                gas,
                shapeIn: "3x1 · 3x1", // Update label
                shapeOut: "scalar",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: out,
            });
        });

        it("dot : negative entries", async function () {
            t++;

            const a = [await qInt(-2), await qInt(3)];
            const b = [await qInt(5), await qInt(-4)];

            await touchGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const out = await harness.dotHarness(2n, 1n, a, 2n, 1n, b);

            const expected = await qInt(-22);

            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            printBlock({
                t,
                method: "dotHarness",
                explanation: "Handles negative entries correctly.",
                gas,
                shapeIn: "2x1 · 2x1",
                shapeOut: "scalar",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: out,
            });
        });

        it("dot : length mismatch reverts", async function () {
            t++;

            const a = [await qInt(1), await qInt(2)];
            const b = [await qInt(3)];

            await touchGas(harness, "dotHarness", [2n, 1n, a, 1n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 1n, 1n, b]);

            await expect(
                harness.dotHarness(2n, 1n, a, 1n, 1n, b)
            ).to.be.revertedWith("MatrixMaster: dot length mismatch");

            printBlock({
                t,
                method: "dotHarness",
                explanation: "Rejects mismatched vector lengths for dot product.",
                gas,
                shapeIn: "2x1 · 1x1",
                shapeOut: "revert",
                inHex: `a=${fmtHexArr(a)}, b=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });
    });
});