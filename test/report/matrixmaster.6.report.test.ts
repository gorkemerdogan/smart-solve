// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Test suite validating MatrixMaster determinant and inverse operations using ABDK quad precision (bytes16).
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

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

describe("MatrixMaster : Determinant & Inverse", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    beforeEach(async () => {
        harness = await newHarness();
    });

    // --- quad constructors ---

    const qInt = async (n: number | string | bigint): Promise<string> =>
        harness.qFromInt(BigInt(n));

    // ------------------------------------------------------------
    //  Section 9: Determinant
    // ------------------------------------------------------------

    describe("Section 9: Determinant (det)", function () {
        it("Test 1: small known 2x2 determinant", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);
            const expected = await qInt(-2);

            expect(det.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Runs LU-based determinant on a 2x2 matrix.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 2: singular matrix returns 0", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(2), await qInt(4)];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);
            const zero = await qInt(0);

            expect(det.toLowerCase()).to.equal(zero.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Detects linear dependence (Singular Matrix).",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 3: pivoting matrix requiring row swap", async function () {
            t++;

            const A = [await qInt(0), await qInt(1), await qInt(1), await qInt(0)];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);
            const expected = await qInt(-1);

            expect(det.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Forces LU pivoting (row swap) and verifies sign flip.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 4: non-square matrix reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];

            await expect(harness.detHarness(2n, 3n, A)).to.be.revertedWith("MatrixMaster: matrix must be square");

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Rejects rectangular input.",
                gas: "Revert",
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
                explanation: "det([a]) = a",
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
                explanation: "det(Identity) = 1",
                gas,
                shapeIn: "3x3",
                shapeOut: "scalar",
                inHex: "Identity",
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
                explanation: "det(ZeroMatrix) = 0",
                gas,
                shapeIn: "3x3",
                shapeOut: "scalar",
                inHex: "Zeros",
                outHex: `det=${det}`,
            });
        });

        it("Test 8: upper triangular matrix equals product of diagonal entries", async function () {
            t++;
            // A = [[2,1,2],[0,3,4],[0,0,4]]; det = 2*3*4 = 24
            const A = [
                await qInt(2), await qInt(1), await qInt(2),
                await qInt(0), await qInt(3), await qInt(4),
                await qInt(0), await qInt(0), await qInt(4),
            ];

            await touchGas(harness, "detHarness", [3n, 3n, A]);
            const gas = await estimateGas(harness, "detHarness", [3n, 3n, A]);
            const det = await harness.detHarness(3n, 3n, A);
            const expected = await qInt(24);

            expect(det.toLowerCase()).to.equal(expected.toLowerCase());

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Verifies det = product(diagonal) for triangular matrix.",
                gas,
                shapeIn: "3x3 (triangular)",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("Test 9: near-singular matrix with tiny pivot", async function () {
            t++;
            const one = await qInt(1);
            const eps = await harness.qFromFrac(1n, 1000n); // 0.001

            // 1 + eps
            const onePlusEpsMat = asMatrix(await harness.addHarness(1n, 1n, [one], 1n, 1n, [eps]));
            const onePlusEps = onePlusEpsMat.data[0];

            // A = [[1, 1], [1, 1+eps]] -> det = eps
            const A = [one, one, one, onePlusEps];

            await touchGas(harness, "detHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "detHarness", [2n, 2n, A]);
            const det = await harness.detHarness(2n, 2n, A);

            // Convert to standard JS numbers (floats)
            const detNum = Number(await harness.toFloat(det));
            const epsNum = Number(await harness.toFloat(eps));

            // Define tolerance range
            const delta = 0.000001; // tolerance chosen to accommodate quad to float conversion error
            const min = epsNum - delta;
            const max = epsNum + delta;

            // Use within for safe floating-point comparison
            expect(detNum).to.be.within(min, max);

            printBlockMatrix({
                t,
                method: "detHarness",
                explanation: "Near-singular matrix returns small non-zero determinant.",
                gas,
                shapeIn: "2x2 (near-singular)",
                shapeOut: "scalar",
                inHex: `eps=${eps}`,
                outHex: `det=${det}`,
            });
        });
    });

    // ------------------------------------------------------------
    //  Section 10: Inverse
    // ------------------------------------------------------------

    describe("Section 10: Inverse", function () {
        it("Test 10: 2x2 upper-triangular A·A^-1 = I", async function () {
            t++;
            // A = [[1,1],[0,1]] -> inverse = [[1,-1],[0,1]]
            const A = [await qInt(1), await qInt(1), await qInt(0), await qInt(1)];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);
            const inv = asMatrix(await harness.inverseHarness(2n, 2n, A));

            // Check A·A^-1 == I
            const prod = asMatrix(await harness.mulMatrixHarness(2n, 2n, A, inv.rows, inv.cols, inv.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts upper-triangular matrix; verifies A * A^-1 = Identity.",
                gas,
                shapeIn: "2x2",
                shapeOut: "2x2",
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(inv.data),
            });
        });

        it("Test 11: 3x3 permutation matrix", async function () {
            t++;
            // A = Permutation -> Inverse is Transpose
            const zero = await qInt(0);
            const one = await qInt(1);
            const Adata = [zero, one, zero, zero, zero, one, one, zero, zero];

            await touchGas(harness, "inverseHarness", [3n, 3n, Adata]);
            const gas = await estimateGas(harness, "inverseHarness", [3n, 3n, Adata]);
            const invA = asMatrix(await harness.inverseHarness(3n, 3n, Adata));

            const prod = asMatrix(await harness.mulMatrixHarness(3n, 3n, Adata, invA.rows, invA.cols, invA.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(3n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverts 3x3 permutation matrix (multi-pivot check).",
                gas,
                shapeIn: "3x3",
                shapeOut: "3x3",
                inHex: fmtHexArr(Adata),
                outHex: fmtHexArr(invA.data),
            });
        });

        it("Test 12: pivoting case (row swap)", async function () {
            t++;
            // A = [[0,1],[1,0]] -> Inverse = A
            const A = [await qInt(0), await qInt(1), await qInt(1), await qInt(0)];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);
            const inv = asMatrix(await harness.inverseHarness(2n, 2n, A));

            expect(await harness.matricesExactEqual(2n, 2n, A, inv.rows, inv.cols, inv.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Verifies Gauss-Jordan pivoting on swap matrix.",
                gas,
                shapeIn: "2x2",
                shapeOut: "2x2",
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(inv.data),
            });
        });

        it("Test 13: singular matrix reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(2), await qInt(4)];

            await expect(harness.inverseHarness(2n, 2n, A)).to.be.revertedWith("MatrixMaster: singular matrix");

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Rejects rank-deficient (singular) matrix.",
                gas: "Revert",
                shapeIn: "2x2",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("Test 14: non-square matrix reverts", async function () {
            t++;
            const A = [await qInt(1), await qInt(2), await qInt(3), await qInt(4), await qInt(5), await qInt(6)];

            await expect(harness.inverseHarness(2n, 3n, A)).to.be.revertedWith("MatrixMaster: matrix must be square");

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Rejects non-square input.",
                gas: "Revert",
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

            await touchGas(harness, "inverseHarness", [n, n, I.data]);
            const gas = await estimateGas(harness, "inverseHarness", [n, n, I.data]);
            const invI = asMatrix(await harness.inverseHarness(n, n, I.data));

            expect(await harness.matricesExactEqual(n, n, I.data, n, n, invI.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "inv(Identity) = Identity.",
                gas,
                shapeIn: "3x3",
                shapeOut: "3x3",
                inHex: "Identity",
                outHex: "Identity",
            });
        });

        it("Test 16: 1x1 matrix inverse is scalar reciprocal", async function () {
            t++;
            const a = await qInt(2);
            const A = [a];

            await touchGas(harness, "inverseHarness", [1n, 1n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [1n, 1n, A]);
            const inv = asMatrix(await harness.inverseHarness(1n, 1n, A));
            const prod = asMatrix(await harness.mulMatrixHarness(1n, 1n, A, 1n, 1n, inv.data));

            expect(prod.data[0]).to.equal(await qInt(1));

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "1x1 inverse behaves as scalar reciprocal (1/a).",
                gas,
                shapeIn: "1x1",
                shapeOut: "1x1",
                inHex: "[2]",
                outHex: fmtHexArr(inv.data),
            });
        });

        it("Test 17: diagonal matrix inverse", async function () {
            t++;
            const diag = [await qInt(2), await qInt(4)];
            const D = asMatrix(await harness.fromDiagonalHarness(diag));

            await touchGas(harness, "inverseHarness", [2n, 2n, D.data]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, D.data]);
            const invD = asMatrix(await harness.inverseHarness(2n, 2n, D.data));
            const prod = asMatrix(await harness.mulMatrixHarness(2n, 2n, D.data, 2n, 2n, invD.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            expect(await harness.matricesExactEqual(2n, 2n, prod.data, 2n, 2n, I.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "Inverse of diagonal matrix contains reciprocals of diagonal elements.",
                gas,
                shapeIn: "2x2",
                shapeOut: "2x2",
                inHex: fmtHexArr(D.data),
                outHex: fmtHexArr(invD.data),
            });
        });

        it("Test 18: double inverse restores original", async function () {
            t++;
            const A = [await qInt(1), await qInt(1), await qInt(0), await qInt(1)];

            await touchGas(harness, "inverseHarness", [2n, 2n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [2n, 2n, A]);
            const invA = asMatrix(await harness.inverseHarness(2n, 2n, A));
            const invInvA = asMatrix(await harness.inverseHarness(2n, 2n, invA.data));

            expect(await harness.matricesExactEqual(2n, 2n, A, 2n, 2n, invInvA.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "inverseHarness",
                explanation: "inv(inv(A)) = A.",
                gas,
                shapeIn: "2x2",
                shapeOut: "2x2",
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(invInvA.data),
            });
        });
    });
});