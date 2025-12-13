// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests implements determinant and inverse critical, critical for solving systems of equations and performing transformations.
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;

    // matrix comparison (helper)
    matricesExactEqual(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<boolean>;

    // creation (helper)
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    createIdentityMatrixHarness(n: bigint): Promise<[bigint, bigint, string[]]>;
    fromDiagonalHarness(diag: string[]): Promise<[bigint, bigint, string[]]>;

    // elementwise arithmetic (helper)
    addHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // matrix multiplication (helper)
    mulMatrixHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;

    // determinant & inverse
    detHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<string>;
    inverseHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;
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

describe("MatrixMaster (library) : determinant & inverse over ABDK quad", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    const qInt = async (n: number | string | bigint): Promise<string> =>
        harness.qFromInt(BigInt(n));

    // ------------------------------------------------------------
    //  Section 1: Determinant
    // ------------------------------------------------------------

    describe("Section 1: determinant (det)", function () {
        it("Test 1: small known 2x2 determinant", async function () {
            t++;
            // A = [[1,2],[3,4]] -> det = -2
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);
            const expected = await qInt(-2);

            expect(det.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Runs LU-based determinant on a simple 2x2 matrix with a known closed-form value.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 2: singular matrix returns 0", async function () {
            t++;
            // Rows linearly dependent: [[1,2],[2,4]] -> det = 0
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(2),
                await qInt(4),
            ];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);
            const zero = await qInt(0);

            expect(det.toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Detects linear dependence in rows and returns an exact zero determinant.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 3: pivoting matrix requiring row swap", async function () {
            t++;
            // A = [[0,1],[1,0]] -> det = -1, and requires pivot swap
            const A = [
                await qInt(0),
                await qInt(1),
                await qInt(1),
                await qInt(0),
            ];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);
            const expected = await qInt(-1);

            expect(det.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Forces LU pivoting via row swap and verifies determinant sign tracking.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 4: non-square matrix reverts", async function () {
            t++;
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ]; // 2x3

            await touchGas(harness, "detHarness", [2n, 3n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 3n, A]);

            await expect(harness.detHarness(2n, 3n, A)).to.be.revertedWith("MatrixMaster: matrix must be square");

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Confirms determinant is only defined for n×n matrices and rejects rectangular input.",
                gas,
                shapeIn: "2x3",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("Test 5: 1x1 matrix returns the single entry", async function () {
            t++;
            const val = await qInt(7);
            const A = [val];

            await touchGas(harness, "detHarness", [1n, 1n, A]);
            const gas = await estimateGas(harness, "detHarness", [1n, 1n, A]);
            const det = await harness.detHarness(1n, 1n, A);

            expect(det.toLowerCase()).to.equal(val.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Checks 1x1 case where det([a]) = a exactly.",
                gas,
                shapeIn: "1x1",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 6: identity matrix has determinant 1", async function () {
            t++;
            const n = 3n;
            const I = asMatrix(await harness.createIdentityMatrixHarness(n));

            await touchGas(harness, "detHarness", [I.rows, I.cols, I.data]);
            const gas = await estimateGas(harness, "detHarness", [I.rows, I.cols, I.data]);
            const det = await harness.detHarness(I.rows, I.cols, I.data);
            const one = await qInt(1);

            expect(det.toLowerCase()).to.equal(one.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Computes det(I_n) for n=3 and verifies it equals one as expected.",
                gas,
                shapeIn: `${I.rows}x${I.cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(I.data),
                outHex: `det=${det}`,
            });
        });

        it("Test 7: all-zero matrix has determinant 0", async function () {
            t++;
            const n = 3n;
            const Z = asMatrix(await harness.zerosHarness(n, n));

            await touchGas(harness, "detHarness", [Z.rows, Z.cols, Z.data]);
            const gas = await estimateGas(harness, "detHarness", [Z.rows, Z.cols, Z.data]);
            const det = await harness.detHarness(Z.rows, Z.cols, Z.data);
            const zero = await qInt(0);

            expect(det.toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Confirms that a 3x3 zero matrix has a determinant exactly equal to zero.",
                gas,
                shapeIn: `${Z.rows}x${Z.cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(Z.data),
                outHex: `det=${det}`,
            });
        });

        it("Test 8: upper triangular matrix equals product of diagonal entries", async function () {
            t++;
            // A = [[2,1,2],[0,3,4],[0,0,4]]; det = 2*3*4 = 24
            const A = [
                await qInt(2),
                await qInt(1),
                await qInt(2),
                await qInt(0),
                await qInt(3),
                await qInt(4),
                await qInt(0),
                await qInt(0),
                await qInt(4),
            ];

            await touchGas(harness, "detHarness", [3n, 3n, A]);
            const gas = await estimateGas(harness, "detHarness", [3n, 3n, A]);
            const det = await harness.detHarness(3n, 3n, A);
            const expected = await qInt(24);

            expect(det.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Runs determinant on an upper triangular matrix and checks it equals the product of its diagonal.",
                gas,
                shapeIn: "3x3 (upper triangular)",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 9: near-singular matrix with tiny pivot yields small but non-zero determinant", async function () {
            t++;
            // Construct epsilon = 1 / 1000 in quad
            const one = await qInt(1);
            const eps = await harness.qFromFrac(1n, 1000n);

            // Build 1 + eps via 1x1 addHarness
            const onePlusEpsMat = asMatrix(
                await harness.addHarness(1n, 1n, [one], 1n, 1n, [eps]),
            );
            const onePlusEps = onePlusEpsMat.data[0];

            // A = [[1, 1],
            //      [1, 1+eps]]
            const A = [one, one, one, onePlusEps];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);

            const zeroQ = await qInt(0);
            const zeroBI = BigInt(zeroQ);
            const detBI = BigInt(det);
            const oneBI = BigInt(one);

            // For this construction, det(A) ≈ eps: strictly positive & much smaller than 1
            expect(detBI).to.be.gt(zeroBI);
            expect(detBI).to.be.lt(oneBI);

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Uses A=[[1,1],[1,1+eps]] with eps<<1 and checks determinant is small but non-zero, exercising tiny pivot handling.",
                gas,
                shapeIn: "2x2 (near-singular)",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}, eps=${eps}`,
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 2: Inverse
    // ------------------------------------------------------------

    describe("Section 2: inverse", function () {
        it("Test 10: 2x2 upper-triangular with det=1, A·A^-1 = I", async function () {
            t++;
            // A = [[1,1],[0,1]]; inverse = [[1,-1],[0,1]] all integers
            const A = [
                await qInt(1),
                await qInt(1),
                await qInt(0),
                await qInt(1),
            ];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);
            const inv = asMatrix(await harness.inverseHarness(2n, 2n, A));

            // Check A·A^-1 == I
            const prod = asMatrix(await harness.mulMatrixHarness(2n, 2n, A, inv.rows, inv.cols, inv.data),);
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            expect(
                await harness.matricesExactEqual(
                    prod.rows,
                    prod.cols,
                    prod.data,
                    I.rows,
                    I.cols,
                    I.data,
                ),
            ).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts an upper-triangular matrix and verifies the product yields the identity.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${inv.rows}x${inv.cols}`,
                inHex: fmtHexArr(A),
                outHex: `inv=${fmtHexArr(inv.data)}`,
            });
        });

        it("Test 11: 3x3 permutation matrix with multiple pivots", async function () {
            t++;
            // A = permutation matrix:
            //     [0 1 0]
            //     [0 0 1]
            //     [1 0 0]
            // A^-1 = Aᵀ, and A·A^-1 = I
            const zero = await qInt(0);
            const one = await qInt(1);

            const Adata = [
                zero, one, zero,
                zero, zero, one,
                one, zero, zero,
            ];

            await touchGas(harness, "inverseHarness", [3n, 3n, Adata]);
            const gas = await estimateGas(harness, "inverseHarness", [3n, 3n, Adata]);
            const invA = asMatrix(await harness.inverseHarness(3n, 3n, Adata));

            // Check A·A^-1 = I_3
            const prod = asMatrix(
                await harness.mulMatrixHarness(
                    3n,
                    3n,
                    Adata,
                    invA.rows,
                    invA.cols,
                    invA.data,
                ),
            );
            const I = asMatrix(await harness.createIdentityMatrixHarness(3n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts a 3x3 permutation matrix that forces multiple pivot choices and confirms A·A^-1 = I_3.",
                gas,
                shapeIn: "3x3 (permutation)",
                shapeOut: `${invA.rows}x${invA.cols}`,
                inHex: fmtHexArr(Adata),
                outHex: `invA=${fmtHexArr(invA.data)}, prod=${fmtHexArr(prod.data)}`,
            });
        });

        it("Test 12: pivoting case (row swap) works", async function () {
            t++;
            // A = [[0,1],[1,0]]; A^-1 = A
            const A = [
                await qInt(0),
                await qInt(1),
                await qInt(1),
                await qInt(0),
            ];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);
            const inv = asMatrix(await harness.inverseHarness(2n, 2n, A));

            // inverse should equal A exactly
            expect(await harness.matricesExactEqual(2n, 2n, A, inv.rows, inv.cols, inv.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Gauss–Jordan pivoting on a swap matrix and checks A^-1 equals A itself.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${inv.rows}x${inv.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(inv.data),
            });
        });

        it("Test 13: singular matrix reverts", async function () {
            t++;
            // [[1,2],[2,4]] singular
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(2),
                await qInt(4),
            ];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);

            await expect(harness.inverseHarness(2n, 2n, A)).to.be.revertedWith("MatrixMaster: singular matrix");

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Attempts to invert a rank-deficient matrix and ensures a singularity revert.",
                gas,
                shapeIn: "2x2",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("Test 14: non-square matrix reverts", async function () {
            t++;
            const A = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
                await qInt(5),
                await qInt(6),
            ]; // 2x3

            await touchGas(harness, "inverseHarness", [2n, 3n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 3n, A]);

            await expect(harness.inverseHarness(2n, 3n, A)).to.be.revertedWith("MatrixMaster: matrix must be square");

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Verifies inversion is only allowed for square matrices and rejects 2x3 input.",
                gas,
                shapeIn: "2x3",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("Test 15: identity matrix is its own inverse", async function () {
            t++;
            const n = 3n;
            const I = asMatrix(await harness.createIdentityMatrixHarness(n));

            await touchGas(harness, "inverseHarness", [I.rows, I.cols, I.data]);
            const gas = await estimateGas(harness, "inverseHarness", [I.rows, I.cols, I.data]);
            const invI = asMatrix(await harness.inverseHarness(I.rows, I.cols, I.data));

            expect(await harness.matricesExactEqual(I.rows, I.cols, I.data, invI.rows, invI.cols, invI.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts a 3x3 identity matrix and ensures the result is bitwise-identical to I.",
                gas,
                shapeIn: `${I.rows}x${I.cols}`,
                shapeOut: `${invI.rows}x${invI.cols}`,
                inHex: fmtHexArr(I.data),
                outHex: fmtHexArr(invI.data),
            });
        });

        it("Test 16: 1x1 matrix inverse behaves as scalar reciprocal", async function () {
            t++;
            const a = await qInt(2); // [2]
            const A = [a];

            await touchGas(harness, "inverseHarness", [1n, 1n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [1n, 1n, A]);
            const inv = asMatrix(await harness.inverseHarness(1n, 1n, A));

            // Check A·A^-1 = [1]
            const prod = asMatrix(await harness.mulMatrixHarness(1n, 1n, A, inv.rows, inv.cols, inv.data));
            const one = await qInt(1);

            expect(prod.rows).to.equal(1n);
            expect(prod.cols).to.equal(1n);
            expect(prod.data[0].toLowerCase()).to.equal(one.toLowerCase());

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts 1x1 matrix [a] and checks [a]·[a]^-1 = [1], matching scalar reciprocal semantics.",
                gas,
                shapeIn: "1x1",
                shapeOut: `${inv.rows}x${inv.cols}`,
                inHex: fmtHexArr(A),
                outHex: `inv=${fmtHexArr(inv.data)}, prod=${fmtHexArr(prod.data)}`,
            });
        });

        it("Test 17: diagonal matrix inverse yields reciprocal diagonal (via product check)", async function () {
            t++;
            // diag = [2,4] so inverse diag should be [1/2,1/4]
            const d0 = await qInt(2);
            const d1 = await qInt(4);
            const diag = [d0, d1];

            const D = asMatrix(await harness.fromDiagonalHarness(diag));

            await touchGas(harness, "inverseHarness", [D.rows, D.cols, D.data]);
            const gas = await estimateGas(harness, "inverseHarness", [D.rows, D.cols, D.data]);
            const invD = asMatrix(
                await harness.inverseHarness(D.rows, D.cols, D.data),
            );
            const prod = asMatrix(await harness.mulMatrixHarness(D.rows, D.cols, D.data, invD.rows, invD.cols, invD.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts 2x2 diagonal matrix and verifies via D·D^-1 = I that diagonal entries act as reciprocals.",
                gas,
                shapeIn: `${D.rows}x${D.cols}`,
                shapeOut: `${invD.rows}x${invD.cols}`,
                inHex: fmtHexArr(D.data),
                outHex: `invD=${fmtHexArr(invD.data)}, prod=${fmtHexArr(prod.data)}`,
            });
        });

        it("Test 18: inverse(inverse(A)) == A for simple integer 2x2 matrix", async function () {
            t++;
            // A = [[1,1],[0,1]] with integer inverse [[1,-1],[0,1]]
            const A = [
                await qInt(1),
                await qInt(1),
                await qInt(0),
                await qInt(1),
            ];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);
            const invA = asMatrix(await harness.inverseHarness(2n, 2n, A));
            const invInvA = asMatrix(await harness.inverseHarness(invA.rows, invA.cols, invA.data));

            expect(await harness.matricesExactEqual(2n, 2n, A, invInvA.rows, invInvA.cols, invInvA.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Applies inverse operator twice to A and verifies inverse(inverse(A)) restores A exactly.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${invInvA.rows}x${invInvA.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(invInvA.data),
            });
        });
    });
});