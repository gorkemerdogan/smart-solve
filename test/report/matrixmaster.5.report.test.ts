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
    // Determinant
    // =========================================================

    describe("determinant (det)", function () {
        it("det : small known 2x2 determinant", async function () {
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

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Runs LU-based determinant on a simple 2x2 matrix with a known closed-form value.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("det : singular matrix returns 0", async function () {
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

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Detects linear dependence in rows and returns an exact zero determinant.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("det : pivoting matrix requiring row swap", async function () {
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

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Forces LU pivoting via row swap and verifies determinant sign tracking.",
                gas,
                shapeIn: "2x2",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("det : non-square matrix reverts", async function () {
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

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Confirms determinant is only defined for n×n matrices and rejects rectangular input.",
                gas,
                shapeIn: "2x3",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("det : 1×1 matrix returns the single entry", async function () {
            t++;
            const val = await qInt(7);
            const A = [val];

            await touchGas(harness, "detHarness", [1n, 1n, A]);
            const gas = await estimateGas(harness, "detHarness", [1n, 1n, A]);
            const det = await harness.detHarness(1n, 1n, A);

            expect(det.toLowerCase()).to.equal(val.toLowerCase());

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Checks the degenerate 1x1 case where det([a]) = a exactly.",
                gas,
                shapeIn: "1x1",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("det : identity matrix has determinant 1", async function () {
            t++;
            const n = 3n;
            const I = asMatrix(await harness.createIdentityMatrixHarness(n));

            await touchGas(harness, "detHarness", [I.rows, I.cols, I.data]);
            const gas = await estimateGas(harness, "detHarness", [I.rows, I.cols, I.data]);
            const det = await harness.detHarness(I.rows, I.cols, I.data);
            const one = await qInt(1);

            expect(det.toLowerCase()).to.equal(one.toLowerCase());

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Computes det(Iₙ) for n=3 and verifies it equals one as expected.",
                gas,
                shapeIn: `${I.rows}x${I.cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(I.data),
                outHex: `det=${det}`,
            });
        });

        it("det : all-zero matrix has determinant 0", async function () {
            t++;
            const n = 3n;
            const Z = asMatrix(await harness.zerosHarness(n, n));

            await touchGas(harness, "detHarness", [Z.rows, Z.cols, Z.data]);
            const gas = await estimateGas(harness, "detHarness", [Z.rows, Z.cols, Z.data]);
            const det = await harness.detHarness(Z.rows, Z.cols, Z.data);
            const zero = await qInt(0);

            expect(det.toLowerCase()).to.equal(zero.toLowerCase());

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Confirms that a 3x3 zero matrix has a determinant exactly equal to zero.",
                gas,
                shapeIn: `${Z.rows}x${Z.cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(Z.data),
                outHex: `det=${det}`,
            });
        });

        it("det : upper triangular matrix equals product of diagonal entries", async function () {
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

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Runs determinant on an upper triangular matrix and checks it equals the product of its diagonal.",
                gas,
                shapeIn: "3x3 (upper triangular)",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}`,
            });
        });

        it("det : near-singular matrix with tiny pivot yields small but non-zero determinant", async function () {
            t++;
            // Construct epsilon = 1 / 1000 in quad
            const one = await qInt(1);
            const big = await qInt(1000);
            const eps = await harness.qFromFrac(1n, 1000n);

            // Build 1 + eps via 1x1 addHarness
            const onePlusEpsMat = asMatrix(await harness.addHarness(1n, 1n, [one], 1n, 1n, [eps]));
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

            // For this construction, det(A) ≈ eps:
            //  - strictly positive
            //  - much smaller than 1
            expect(detBI).to.be.gt(zeroBI);
            expect(detBI).to.be.lt(oneBI);

            printBlock({
                t,
                method: "detHarness",
                explanation:
                    "Uses A=[[1,1],[1,1+ε]] with ε<<1 and checks determinant is small but non-zero, exercising tiny pivot handling.",
                gas,
                shapeIn: "2x2 (near-singular)",
                shapeOut: "scalar",
                inHex: fmtHexArr(A),
                outHex: `det=${det}, eps=${eps}`,
            });
        });
    });

    // =========================================================
    // Inverse
    // =========================================================

    describe("inverse", function () {
        it("inverse : 2x2 upper-triangular with det=1, A·A⁻¹ = I", async function () {
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

            // Check A·A⁻¹ == I
            const prod = asMatrix(await harness.mulMatrixHarness(2n, 2n, A, inv.rows, inv.cols, inv.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Inverts an upper-triangular matrix and verifies the product yields the identity.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${inv.rows}x${inv.cols}`,
                inHex: fmtHexArr(A),
                outHex: `inv=${fmtHexArr(inv.data)}`,
            });
        });

        it("inverse : 3x3 permutation matrix with multiple pivots", async function () {
            t++;
            // A = permutation matrix:
            // [0 1 0]
            // [0 0 1]
            // [1 0 0]
            // A⁻¹ = Aᵀ, and A·A⁻¹ = I
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

            // Check A·A⁻¹ = I₃
            const prod = asMatrix(await harness.mulMatrixHarness(3n, 3n, Adata, invA.rows, invA.cols, invA.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(3n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Inverts a 3x3 permutation matrix that forces multiple pivot choices and confirms A·A⁻¹ = I₃.",
                gas,
                shapeIn: "3x3 (permutation)",
                shapeOut: `${invA.rows}x${invA.cols}`,
                inHex: fmtHexArr(Adata),
                outHex: `invA=${fmtHexArr(invA.data)}, prod=${fmtHexArr(prod.data)}`,
            });
        });

        it("inverse : pivoting case (row swap) works", async function () {
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

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Exercises Gauss–Jordan pivoting on a swap matrix and checks A⁻¹ equals A itself.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${inv.rows}x${inv.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(inv.data),
            });
        });

        it("inverse : singular matrix reverts", async function () {
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

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Attempts to invert a rank-deficient matrix and ensures a singularity revert.",
                gas,
                shapeIn: "2x2",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("inverse : non-square matrix reverts", async function () {
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

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Verifies inversion is only allowed for square matrices and rejects 2x3 input.",
                gas,
                shapeIn: "2x3",
                shapeOut: "revert",
                inHex: fmtHexArr(A),
                outHex: "-",
            });
        });

        it("inverse : identity matrix is its own inverse", async function () {
            t++;
            const n = 3n;
            const I = asMatrix(await harness.createIdentityMatrixHarness(n));

            await touchGas(harness, "inverseHarness", [I.rows, I.cols, I.data]);
            const gas = await estimateGas(harness, "inverseHarness", [I.rows, I.cols, I.data]);
            const invI = asMatrix(await harness.inverseHarness(I.rows, I.cols, I.data));

            expect(await harness.matricesExactEqual(I.rows, I.cols, I.data, invI.rows, invI.cols, invI.data)).to.equal(true);

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Inverts a 3x3 identity matrix and ensures the result is bitwise-identical to I.",
                gas,
                shapeIn: `${I.rows}x${I.cols}`,
                shapeOut: `${invI.rows}x${invI.cols}`,
                inHex: fmtHexArr(I.data),
                outHex: fmtHexArr(invI.data),
            });
        });

        it("inverse : 1×1 matrix inverse behaves as scalar reciprocal", async function () {
            t++;
            const a = await qInt(2); // [2]
            const A = [a];

            await touchGas(harness, "inverseHarness", [1n, 1n, A]);
            const gas = await estimateGas(harness, "inverseHarness", [1n, 1n, A]);
            const inv = asMatrix(await harness.inverseHarness(1n, 1n, A));

            // Check A·A⁻¹ = [1]
            const prod = asMatrix(await harness.mulMatrixHarness(1n, 1n, A, inv.rows, inv.cols, inv.data));
            const one = await qInt(1);

            expect(prod.rows).to.equal(1n);
            expect(prod.cols).to.equal(1n);
            expect(prod.data[0].toLowerCase()).to.equal(one.toLowerCase());

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Inverts a 1x1 matrix [a] and checks [a]·[a]⁻¹ = [1], matching scalar reciprocal semantics.",
                gas,
                shapeIn: "1x1",
                shapeOut: `${inv.rows}x${inv.cols}`,
                inHex: fmtHexArr(A),
                outHex: `inv=${fmtHexArr(inv.data)}, prod=${fmtHexArr(prod.data)}`,
            });
        });

        it("inverse : diagonal matrix inverse yields reciprocal diagonal (via product check)", async function () {
            t++;
            // diag = [2,4] so inverse diag should be [1/2,1/4]
            const d0 = await qInt(2);
            const d1 = await qInt(4);
            const diag = [d0, d1];

            const D = asMatrix(await harness.fromDiagonalHarness(diag));

            await touchGas(harness, "inverseHarness", [D.rows, D.cols, D.data]);
            const gas = await estimateGas(harness, "inverseHarness", [D.rows, D.cols, D.data]);
            const invD = asMatrix(await harness.inverseHarness(D.rows, D.cols, D.data));
            const prod = asMatrix(await harness.mulMatrixHarness(D.rows, D.cols, D.data, invD.rows, invD.cols, invD.data));
            const I = asMatrix(await harness.createIdentityMatrixHarness(2n));

            expect(await harness.matricesExactEqual(prod.rows, prod.cols, prod.data, I.rows, I.cols, I.data)).to.equal(true);

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Inverts a 2x2 diagonal matrix and verifies via D·D⁻¹ = I that diagonal entries act as reciprocals.",
                gas,
                shapeIn: `${D.rows}x${D.cols}`,
                shapeOut: `${invD.rows}x${invD.cols}`,
                inHex: fmtHexArr(D.data),
                outHex: `invD=${fmtHexArr(invD.data)}, prod=${fmtHexArr(prod.data)}`,
            });
        });

        it("inverse : inverse(inverse(A)) == A for simple integer 2x2 matrix", async function () {
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

            printBlock({
                t,
                method: "inverseHarness",
                explanation:
                    "Applies the inverse operator twice to A and verifies inverse(inverse(A)) restores A exactly.",
                gas,
                shapeIn: "2x2",
                shapeOut: `${invInvA.rows}x${invInvA.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(invInvA.data),
            });
        });
    });

    // =========================================================
    // Norm & Normalize
    // =========================================================
    describe("Norm & Normalize", function () {

        // --- Norm ---
        it("norm : computes Euclidean norm of 3-4-5 triangle", async function () {
            t++;
            // Vector [3, 4]^T -> Norm should be 5
            const v = [await qInt(3), await qInt(4)];

            await touchGas(harness, "normHarness", [2n, 1n, v]);
            const gas = await estimateGas(harness, "normHarness", [2n, 1n, v]);
            const out = await harness.normHarness(2n, 1n, v);

            const expected = await qInt(5);

            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            printBlock({
                t,
                method: "normHarness",
                explanation: "Computes ||v|| for vector [3, 4].",
                gas,
                shapeIn: "2x1",
                shapeOut: "scalar",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: out,
            });
        });

        it("norm : reverts on row vector", async function () {
            t++;
            // Pass as Row Vector (1 row, 2 cols) -> Should revert
            const v = [await qInt(1), await qInt(1)];

            await touchGas(harness, "normHarness", [1n, 2n, v]);
            const gas = await estimateGas(harness, "normHarness", [1n, 2n, v]);

            await expect(harness.normHarness(1n, 2n, v)).to.be.revertedWith("MatrixMaster: norm requires column vector");

            printBlock({
                t,
                method: "normHarness",
                explanation: "norm : reverts on row vector.",
                gas,
                shapeIn: "1x1",
                shapeOut: "scalar",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: "-",
            });
        });

        // --- Normalize ---
        it("normalize : scales vector to unit length", async function () {
            t++;
            const v = [await qInt(3), await qInt(0), await qInt(4)];

            await touchGas(harness, "normalizeHarness", [3n, 1n, v]);
            const gas = await estimateGas(harness, "normalizeHarness", [3n, 1n, v]);
            const result = await harness.normalizeHarness(3n, 1n, v);

            const data = result.data || result[2];

            // expected = [3/5, 0, 4/5]
            const five = await qInt(5);
            const ex0 = await harness.qFromFrac(3n, 5n);
            const ex1 = await qInt(0);
            const ex2 = await harness.qFromFrac(4n, 5n);

            expect(data[0].toLowerCase()).to.equal(ex0.toLowerCase());
            expect(data[1].toLowerCase()).to.equal(ex1.toLowerCase());
            expect(data[2].toLowerCase()).to.equal(ex2.toLowerCase());

            printBlock({
                t,
                method: "normalizeHarness",
                explanation: "Normalizes [3,0,4] into unit vector (3/5, 0, 4/5) using quad math.",
                gas,
                shapeIn: "3x1",
                shapeOut: "3x1",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: fmtHexArr(data),
            });
        });

        it("normalize : reverts on zero vector", async function () {
            t++;
            // Norm is 0, cannot divide by zero -> revert
            const v = [await qInt(0), await qInt(0)];

            await touchGas(harness, "normalizeHarness", [2n, 1n, v]);
            const gas = await estimateGas(harness, "normalizeHarness", [2n, 1n, v]);

            await expect(harness.normalizeHarness(2n, 1n, v)).to.be.revertedWith("MatrixMaster: cannot normalize zero vector");

            printBlock({
                t,
                method: "normHarness",
                explanation: "norm : reverts on row vector.",
                gas,
                shapeIn: "1x1",
                shapeOut: "scalar",
                inHex: `v=${fmtHexArr(v)}`,
                outHex: "-",
            });
        });
    });

    // =========================================================
    // Utilities: Random & Converged
    // =========================================================
    describe("Utils: Random & Converged", function () {

        // =========================================================
        // Random Vector
        // =========================================================

        it("randomVector : generates deterministic n x 1 vector", async function () {
            t++;
            const n = 5n;
            const seed = ethers.ZeroHash;

            await touchGas(harness, "randomVectorHarness", [n, seed]);
            const gas = await estimateGas(harness, "randomVectorHarness", [n, seed]);

            const res1 = await harness.randomVectorHarness(n, seed);
            const res2 = await harness.randomVectorHarness(n, seed);

            const [rows1, cols1, data1] = res1;
            const [rows2, cols2, data2] = res2;

            expect(rows1).to.equal(n);
            expect(cols1).to.equal(1n);
            expect(rows2).to.equal(n);
            expect(cols2).to.equal(1n);
            expect(data1).to.deep.equal(data2); // deterministic for fixed seed

            printBlock({
                t,
                method: "randomVectorHarness",
                explanation: "Deterministic generation with fixed seed, returns a stable 5x1 vector.",
                gas,
                shapeIn: `n=${n}`,
                shapeOut: "5x1",
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(data1),
            });
        });

        it("randomVector : different seeds produce different vectors", async function () {
            t++;
            const n = 5n;
            const seedA = ethers.ZeroHash;
            const seedB = ethers.keccak256(ethers.toUtf8Bytes("another"));

            await touchGas(harness, "randomVectorHarness", [n, seedA]);
            await touchGas(harness, "randomVectorHarness", [n, seedB]);

            const gasA = await estimateGas(harness, "randomVectorHarness", [n, seedA]);
            const gasB = await estimateGas(harness, "randomVectorHarness", [n, seedB]);
            const gas = gasA + gasB;

            const resA = await harness.randomVectorHarness(n, seedA);
            const resB = await harness.randomVectorHarness(n, seedB);

            const [, , dataA] = resA;
            const [, , dataB] = resB;

            expect(dataA).to.not.deep.equal(dataB); // different seed -> different vector

            printBlock({
                t,
                method: "randomVectorHarness",
                explanation: "Different seeds must yield different pseudo-random vectors.",
                gas,
                shapeIn: `n=${n}`,
                shapeOut: "5x1",
                inHex: `seedA=${seedA}, seedB=${seedB}`,
                outHex: "vectors differ",
            });
        });

        // =========================================================
        // Converged
        // =========================================================

        it("hasConverged : detects convergence within tolerance", async function () {
            t++;

            // vOld = [0, 0], vNew = [1, 0]
            // ||vNew - vOld||₂ = 1
            const vOld = [await qInt(0), await qInt(0)];
            const vNew = [await qInt(1), await qInt(0)];

            // tolPass = 2  → 1 < 2  => true
            // tolFail = 1  → 1 < 1  => false (strict inequality)
            const tolPass = await qInt(2);
            const tolFail = await qInt(1);

            await touchGas(harness, "hasConvergedHarness", [2n, 1n, vNew, 2n, 1n, vOld, tolPass]);
            const gas = await estimateGas(harness, "hasConvergedHarness", [2n, 1n, vNew, 2n, 1n, vOld, tolPass]);
            const yes = await harness.hasConvergedHarness(2n, 1n, vNew, 2n, 1n, vOld, tolPass);
            const no = await harness.hasConvergedHarness(2n, 1n, vNew, 2n, 1n, vOld, tolFail);

            expect(yes).to.equal(true);
            expect(no).to.equal(false);

            printBlock({
                t,
                method: "hasConvergedHarness",
                explanation: "Checks whether ||vNew - vOld||₂ < tol (here 1 < 2 is true, 1 < 1 is false).",
                gas,
                shapeIn: "2x1, 2x1",
                shapeOut: "bool",
                inHex: `tolPass=${tolPass}, tolFail=${tolFail}`,
                outHex: yes ? "true" : "false",
            });
        });

        it("hasConverged : reverts on row vectors", async function () {
            const vRow = [await qInt(1), await qInt(2)];
            const tol = await qInt(1);

            await expect(
                harness.hasConvergedHarness(
                    1n, 2n, vRow,
                    1n, 2n, vRow,
                    tol,
                ),
            ).to.be.revertedWith("MatrixMaster: converged requires vectors");
        });
    });

    // =========================================================
    // Power Iteration
    // =========================================================
    describe("PowerIteration", function () {
        it("powerIteration : diagonal 2x2 matrix returns dominant eigenvalue 5", async function () {
            t++;

            const five = await qInt(5);
            const two = await qInt(2);

            // A = [ [5,0],
            //       [0,2] ]
            const A = [five, qInt(0), qInt(0), two];

            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1000000n); // 1e-6

            await touchGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(
                2n, 2n, A, seed, tol
            );

            const expected = five
            const diff = BigInt(lambda) > BigInt(expected)
                ? BigInt(lambda) - BigInt(expected)
                : BigInt(expected) - BigInt(lambda);

            expect(diff).to.be.lt(BigInt(tol));

            printBlock({
                t,
                method: "powerIterationHarness",
                explanation: "Diagonal matrix diag(5,2) -> eigenvalue = 5.",
                gas,
                shapeIn: "2x2",
                shapeOut: "eigenvalue + eigenvector",
                inHex: `A=${A}`,
                outHex: `lambda=${lambda}, x=${fmtHexArr(xData)}`
            });
        });

        it("powerIteration : symmetric 2x2 matrix returns dominant eigenvalue 4", async function () {
            t++;

            const three = await qInt(3);
            const one = await qInt(1);

            // A = [3,1; 1,3]
            const A = [three, one, one, three];

            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1000000n);

            await touchGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [2n, 2n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(
                2n, 2n, A, seed, tol
            );

            const expectedEigen = await qInt(4);
            const diff = BigInt(lambda) > BigInt(expectedEigen)
                ? BigInt(lambda) - BigInt(expectedEigen)
                : BigInt(expectedEigen) - BigInt(lambda);

            expect(diff).to.be.lt(BigInt(tol));

            printBlock({
                t,
                method: "powerIterationHarness",
                explanation: "Symmetric matrix [[3,1],[1,3]] -> dominant eigenvalue = 4.",
                gas,
                shapeIn: "2x2",
                shapeOut: "eigenvalue + eigenvector",
                inHex: `A=${A}`,
                outHex: `lambda=${lambda}, x=${fmtHexArr(xData)}`
            });
        });

        it("powerIteration : non-symmetric 3x3 matrix returns dominant eigenvalue 3", async function () {
            t++;

            const two = await qInt(2);
            const three = await qInt(3);
            const one = await qInt(1);
            const zero = qInt(0);

            // A =
            // [2 1 0
            //  0 3 1
            //  0 0 1]
            const A = [
                two, one, zero,
                zero, three, one,
                zero, zero, one
            ];

            const seed = ethers.ZeroHash;
            const tol = await harness.qFromFrac(1n, 1000000n);

            await touchGas(harness, "powerIterationHarness", [3n, 3n, A, seed, tol]);
            const gas = await estimateGas(harness, "powerIterationHarness", [3n, 3n, A, seed, tol]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(3n, 3n, A, seed, tol);

            const expectedEigen = three.toLowerCase();
            const diff = BigInt(lambda) > BigInt(expectedEigen)
                ? BigInt(lambda) - BigInt(expectedEigen)
                : BigInt(expectedEigen) - BigInt(lambda);

            printBlock({
                t,
                method: "powerIterationHarness",
                explanation: "Upper-triangular 3x3 -> dominant eigenvalue = 3.",
                gas,
                shapeIn: "3x3",
                shapeOut: "eigenvalue + eigenvector",
                inHex: `A=${A}`,
                outHex: `lambda=${lambda}, x=${fmtHexArr(xData)}`
            });
        });
    });
});