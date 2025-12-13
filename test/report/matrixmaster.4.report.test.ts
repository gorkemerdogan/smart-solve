// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests cover the transpose operation, which swaps the rows and columns of a matrix.
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;

    // matrix comparison (helper)
    matricesExactEqual(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<boolean>;

    // creation (helper)
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    createIdentityMatrixHarness(n: bigint): Promise<[bigint, bigint, string[]]>;
    randomMatrixHarness(rows: bigint, cols: bigint, seed: string): Promise<[bigint, bigint, string[]]>;

    // element access
    setHarness(rows: bigint, cols: bigint, dataFlat: string[], row: bigint, col: bigint, val: string): Promise<[bigint, bigint, string[]]>;

    // transpose
    transposeHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;
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
        return `[${arr.slice(0, 3).join(", ")}, ..., ${arr.slice(-3).join(", ")}] (len=${arr.length})`;
    }
    return `[${arr.join(", ")}]`;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("MatrixMaster — Transpose", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    const qInt = async (n: number | string | bigint): Promise<string> =>
        await harness.qFromInt(BigInt(n));

    // --------------------------------------------------------
    //  Transpose
    // --------------------------------------------------------

    describe("Section: Transpose", function () {

        it("Test 1: transpose 2x3 → 3x2", async function () {
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

            await touchGas(harness, "transposeHarness", [rows, cols, vals]);
            const gas = await estimateGas(harness, "transposeHarness", [rows, cols, vals]);

            const tMat = asMatrix(await harness.transposeHarness(rows, cols, vals));
            expect(tMat.rows).to.equal(3n);
            expect(tMat.cols).to.equal(2n);

            // Expected 3x2: [[1,4],[2,5],[3,6]]
            const expected = [
                await qInt(1),
                await qInt(4),
                await qInt(2),
                await qInt(5),
                await qInt(3),
                await qInt(6),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(tMat.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Transposes a 2x3 matrix into 3x2 and checks entries follow A(i,j) → AT(j,i).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("Test 2: transpose 10x10 random matrix", async function () {
            t++;
            const n = 10n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("transpose-10x10"));

            const A = asMatrix(await harness.randomMatrixHarness(n, n, seed));

            await touchGas(harness, "transposeHarness", [A.rows, A.cols, A.data]);
            const gas = await estimateGas(harness, "transposeHarness", [A.rows, A.cols, A.data]);

            const AT = asMatrix(await harness.transposeHarness(A.rows, A.cols, A.data));

            expect(AT.rows).to.equal(n);
            expect(AT.cols).to.equal(n);

            const rows = Number(n);
            const cols = Number(n);
            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idxA = i * cols + j;
                    const idxAT = j * rows + i;
                    expect(AT.data[idxAT].toLowerCase()).to.equal(A.data[idxA].toLowerCase());
                }
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Transposes a 10x10 random matrix and checks AT(j,i) = A(i,j) for all entries.",
                gas,
                shapeIn: `${A.rows}x${A.cols}`,
                shapeOut: `${AT.rows}x${AT.cols}`,
                inHex: fmtHexArr(A.data),
                outHex: fmtHexArr(AT.data),
            });
        });

        it("Test 3: transpose 40x40 random matrix", async function () {
            t++;
            const n = 40n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("transpose-40x40"));

            const A = asMatrix(await harness.randomMatrixHarness(n, n, seed));

            await touchGas(harness, "transposeHarness", [A.rows, A.cols, A.data]);
            const gas = await estimateGas(harness, "transposeHarness", [A.rows, A.cols, A.data]);

            const AT = asMatrix(await harness.transposeHarness(A.rows, A.cols, A.data));

            expect(AT.rows).to.equal(n);
            expect(AT.cols).to.equal(n);

            const dim = Number(n);
            for (let i = 0; i < dim; ++i) {
                for (let j = 0; j < dim; ++j) {
                    const idxA = i * dim + j;
                    const idxAT = j * dim + i;
                    expect(AT.data[idxAT].toLowerCase()).to.equal(A.data[idxA].toLowerCase());
                }
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Runs transpose on a 40x40 random matrix and validates full i,j ↔ j,i mapping.",
                gas,
                shapeIn: `${A.rows}x${A.cols}`,
                shapeOut: `${AT.rows}x${AT.cols}`,
                inHex: fmtHexArr(A.data),
                outHex: fmtHexArr(AT.data),
            });
        });

        it("Test 4: transpose 100x100 sparse matrix with single non-zero", async function () {
            t++;
            const rows = 100n;
            const cols = 100n;

            // Start with a 100x100 zero matrix.
            const zeroMat = asMatrix(await harness.zerosHarness(rows, cols));

            const one = await qInt(1);
            const zero = await qInt(0);

            // Pick some non-trivial position (10, 70) to set = 1
            const r: bigint = 10n;
            const c: bigint = 70n;

            // setHarness(rows, cols, data, row, col, value)
            const A = asMatrix(
                await harness.setHarness(zeroMat.rows, zeroMat.cols, zeroMat.data, r, c, one),
            );

            await touchGas(harness, "transposeHarness", [A.rows, A.cols, A.data]);
            const gas = await estimateGas(harness, "transposeHarness", [A.rows, A.cols, A.data]);

            const AT = asMatrix(await harness.transposeHarness(A.rows, A.cols, A.data));

            expect(AT.rows).to.equal(cols);
            expect(AT.cols).to.equal(rows);

            const dim = Number(rows);

            // Expect A(r,c) = 1 → AT(c,r) = 1, everything else 0.
            const rNum = Number(r);
            const cNum = Number(c);

            for (let i = 0; i < dim; ++i) {
                for (let j = 0; j < dim; ++j) {
                    const idxAT = i * dim + j;
                    const v = AT.data[idxAT];

                    if (i === cNum && j === rNum) {
                        expect(v.toLowerCase()).to.equal(one.toLowerCase()); // The one moved to (c,r)
                    } else {
                        expect(v.toLowerCase()).to.equal(zero.toLowerCase());
                    }
                }
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Transposes a 100x100 matrix with a single non-zero and confirms (r,c) → (c,r) and zeros elsewhere.",
                gas,
                shapeIn: `${A.rows}x${A.cols}`,
                shapeOut: `${AT.rows}x${AT.cols}`,
                inHex: fmtHexArr(A.data),
                outHex: fmtHexArr(AT.data),
            });
        });

        it("Test 5: transpose(transpose(A)) = A (involution)", async function () {
            t++;
            const vals = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ]; // 2x3
            const rows = 2n;
            const cols = 3n;

            await touchGas(harness, "transposeHarness", [rows, cols, vals]);
            const gas = await estimateGas(harness, "transposeHarness", [rows, cols, vals]);

            const t1 = asMatrix(await harness.transposeHarness(rows, cols, vals));
            const t2 = asMatrix(await harness.transposeHarness(t1.rows, t1.cols, t1.data));

            expect(t2.rows).to.equal(rows);
            expect(t2.cols).to.equal(cols);
            expect(t2.data.length).to.equal(vals.length);

            for (let i = 0; i < vals.length; ++i) {
                expect(t2.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Applies transpose twice and confirms the involution property returns the original matrix.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${t2.rows}x${t2.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(t2.data),
            });
        });

        it("Test 6: row vector 1xN becomes column vector Nx1", async function () {
            t++;
            const rows = 1n;
            const cols = 4n;
            const vals = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];

            await touchGas(harness, "transposeHarness", [rows, cols, vals]);
            const gas = await estimateGas(harness, "transposeHarness", [rows, cols, vals]);

            const tMat = asMatrix(await harness.transposeHarness(rows, cols, vals));

            expect(tMat.rows).to.equal(4n);
            expect(tMat.cols).to.equal(1n);
            for (let i = 0; i < vals.length; ++i) {
                expect(tMat.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Transposes a 1x4 row vector into a 4x1 column vector while preserving flat order.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("Test 7: column vector Nx1 becomes row vector 1xN", async function () {
            t++;
            const rows = 4n;
            const cols = 1n;
            const vals = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];

            await touchGas(harness, "transposeHarness", [rows, cols, vals]);
            const gas = await estimateGas(harness, "transposeHarness", [rows, cols, vals]);

            const tMat = asMatrix(await harness.transposeHarness(rows, cols, vals));

            expect(tMat.rows).to.equal(1n);
            expect(tMat.cols).to.equal(4n);
            for (let i = 0; i < vals.length; ++i) {
                expect(tMat.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Transposes a 4x1 column vector into a 1x4 row vector, keeping the same data sequence.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("Test 8: 1x1 matrix is invariant under transpose", async function () {
            t++;
            const rows = 1n;
            const cols = 1n;
            const vals = [await qInt(13)];

            await touchGas(harness, "transposeHarness", [rows, cols, vals]);
            const gas = await estimateGas(harness, "transposeHarness", [rows, cols, vals]);

            const tMat = asMatrix(await harness.transposeHarness(rows, cols, vals));

            expect(tMat.rows).to.equal(rows);
            expect(tMat.cols).to.equal(cols);
            expect(tMat.data.length).to.equal(1);
            expect(tMat.data[0].toLowerCase()).to.equal(vals[0].toLowerCase());

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Checks that a 1x1 matrix is unchanged by transpose (shape and value).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("Test 9: identity matrix transpose Iᵀ = I", async function () {
            t++;
            const n = 3n;
            const I = asMatrix(await harness.createIdentityMatrixHarness(n));

            await touchGas(harness, "transposeHarness", [I.rows, I.cols, I.data]);
            const gas = await estimateGas(harness, "transposeHarness", [I.rows, I.cols, I.data]);

            const IT = asMatrix(await harness.transposeHarness(I.rows, I.cols, I.data));

            expect(
                await harness.matricesExactEqual(I.rows, I.cols, I.data, IT.rows, IT.cols, IT.data),
            ).to.equal(true);

            printBlockMatrix({
                t,
                method: "transposeHarness",
                explanation: "Confirms that transposing an identity matrix leaves it unchanged (Iᵀ = I).",
                gas,
                shapeIn: `${I.rows}x${I.cols}`,
                shapeOut: `${IT.rows}x${IT.cols}`,
                inHex: fmtHexArr(I.data),
                outHex: fmtHexArr(IT.data),
            });
        });
    });
});