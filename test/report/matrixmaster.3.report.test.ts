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
    // Transpose
    // =========================================================

    describe("Transpose", function () {
        it("transpose : 2x3 → 3x2", async function () {
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

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Swaps rows and columns while preserving numeric values in the new orientation.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("transpose : 10x10 random matrix", async function () {
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
                    expect(AT.data[idxAT].toLowerCase()).to.equal(
                        A.data[idxA].toLowerCase(),
                    );
                }
            }

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Transposes a 10x10 random matrix and checks every entry satisfies AT(j,i)=A(i,j).",
                gas,
                shapeIn: `${A.rows}x${A.cols}`,
                shapeOut: `${AT.rows}x${AT.cols}`,
                inHex: fmtHexArr(A.data),
                outHex: fmtHexArr(AT.data),
            });
        });

        it("transpose : 40x40 random matrix", async function () {
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
                    expect(AT.data[idxAT].toLowerCase()).to.equal(
                        A.data[idxA].toLowerCase(),
                    );
                }
            }

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Runs transpose on a 40x40 dense random matrix and validates full i,j ↔ j,i mapping.",
                gas,
                shapeIn: `${A.rows}x${A.cols}`,
                shapeOut: `${AT.rows}x${AT.cols}`,
                inHex: fmtHexArr(A.data),
                outHex: fmtHexArr(AT.data),
            });
        });

        it("transpose : 100x100 sparse matrix with single non-zero (stress shape)", async function () {
            t++;
            const rows = 100n;
            const cols = 100n;

            // Start with a big 100x100 zero matrix (cheap compared to randomMatrix).
            const zeroMat = asMatrix(await harness.zerosHarness(rows, cols));

            const one = await qInt(1);
            const zero = await qInt(0);

            // Pick some non-trivial position (10, 70) to set = 1
            const r: bigint = 10n;
            const c: bigint = 70n;

            // setHarness(rows, cols, data, row, col, value)
            const A = asMatrix(await harness.setHarness(zeroMat.rows, zeroMat.cols, zeroMat.data, r, c, one));

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
                        // The one moved to (c,r)
                        expect(v.toLowerCase()).to.equal(one.toLowerCase());
                    } else {
                        expect(v.toLowerCase()).to.equal(zero.toLowerCase());
                    }
                }
            }

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Stress-transposes a 100x100 matrix with a single non-zero entry and verifies the (r,c) → (c,r) mapping and shape, without expensive random fill.",
                gas,
                shapeIn: `${A.rows}x${A.cols}`,
                shapeOut: `${AT.rows}x${AT.cols}`,
                inHex: fmtHexArr(A.data),
                outHex: fmtHexArr(AT.data),
            });
        });

        it("transpose(transpose(A)) == A", async function () {
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

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Applies transpose twice and confirms the involution property restores the original.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${t2.rows}x${t2.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(t2.data),
            });
        });

        it("transpose : row vector 1xN becomes column vector Nx1", async function () {
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

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Transposes a 1x4 row vector into a 4x1 column vector while preserving the entry order.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("transpose : column vector Nx1 becomes row vector 1xN", async function () {
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

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Transposes a 4x1 column vector into a 1x4 row vector, keeping the flat encoding unchanged.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });

        it("transpose : 1x1 matrix is invariant", async function () {
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

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Checks that transposing a 1x1 matrix is a no-op on both shape and underlying quad value.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${tMat.rows}x${tMat.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(tMat.data),
            });
        });
    });

    // =========================================================
    // Elementwise arithmetic
    // =========================================================

    describe("Elementwise arithmetic", function () {
        it("add : A+B elementwise", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const b = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "addHarness", [rows, cols, a, rows, cols, b]);

            const gas = await estimateGas(harness, "addHarness", [rows, cols, a, rows, cols, b]);
            const c = asMatrix(await harness.addHarness(rows, cols, a, rows, cols, b));
            const expected = [await qInt(6), await qInt(8), await qInt(10), await qInt(12)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "addHarness",
                explanation:
                    "Performs elementwise addition of two same-shaped matrices and checks sum entries.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("add : shape mismatch reverts", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const b = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];

            await touchGas(harness, "addHarness", [2n, 2n, a, 1n, 4n, b]);
            const gas = await estimateGas(harness, "addHarness", [2n, 2n, a, 1n, 4n, b]);
            await expect(harness.addHarness(2n, 2n, a, 1n, 4n, b)).to.be.revertedWith("MatrixMaster: shape mismatch");

            printBlock({
                t,
                method: "addHarness",
                explanation:
                    "Rejects addition when operand matrices have incompatible dimension layouts.",
                gas,
                shapeIn: "A:2x2, B:1x4",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });

        it("sub : A−B elementwise", async function () {
            t++;
            const a = [
                await qInt(6),
                await qInt(8),
                await qInt(10),
                await qInt(12),
            ];
            const b = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "subHarness", [rows, cols, a, rows, cols, b]);
            const gas = await estimateGas(harness, "subHarness", [rows, cols, a, rows, cols, b]);
            const c = asMatrix(await harness.subHarness(rows, cols, a, rows, cols, b));
            const expected = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "subHarness",
                explanation:
                    "Performs elementwise subtraction and ensures each entry is A(i,j)−B(i,j).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("sub : shape mismatch reverts", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const b = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];

            await touchGas(harness, "subHarness", [2n, 2n, a, 1n, 4n, b]);
            const gas = await estimateGas(harness, "subHarness", [2n, 2n, a, 1n, 4n, b]);
            await expect(harness.subHarness(2n, 2n, a, 1n, 4n, b)).to.be.revertedWith("MatrixMaster: shape mismatch");

            printBlock({
                t,
                method: "subHarness",
                explanation:
                    "Ensures subtraction requires identical shapes and fails on mismatched sizes.",
                gas,
                shapeIn: "A:2x2, B:1x4",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });

        it("mulScalar : scalar multiply applied to all elements", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const k = await qInt(3);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, k));
            const expected = [await qInt(3), await qInt(6), await qInt(9), await qInt(12)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "mulScalarHarness",
                explanation:
                    "Scales every matrix entry by a quad scalar and validates uniform scaling.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("mulScalar : multiply by 0 wipes all entries", async function () {
            t++;
            const a = [
                await qInt(1),
                await qInt(-2),
                await qInt(3),
                await qInt(-4),
            ];
            const k = await qInt(0);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, k));

            // Treat both +0 and -0 encodings as valid zeros.
            for (const v of c.data) {
                expect(isQuadZero(v)).to.equal(true);
            }

            printBlock({
                t,
                method: "mulScalarHarness",
                explanation:
                    "Confirms multiplying by zero produces a matrix whose entries are quad zeros (±0).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("mulScalar : multiply by −1 negates all entries", async function () {
            t++;
            const a = [
                await qInt(1),
                await qInt(-2),
                await qInt(3),
                await qInt(-4),
            ];
            const k = await qInt(-1);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, k));
            const expected = [await qInt(-1), await qInt(2), await qInt(-3), await qInt(4)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "mulScalarHarness",
                explanation:
                    "Scales by −1 and checks every entry is sign-flipped while magnitudes are preserved.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("mulScalar : tiny scalar in (0,1) shrinks magnitudes", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const a = [
                await qInt(10),
                await qInt(20),
                await qInt(30),
                await qInt(40),
            ];

            // tiny = 1 / 1000
            const tiny = await harness.qFromFrac(1n, 1000n);

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, tiny]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, tiny]);
            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, tiny));

            const zeroQ = await qInt(0);
            const zeroBI = BigInt(zeroQ);

            // For positive entries, 0 <= a*s < a should hold
            for (let i = 0; i < a.length; ++i) {
                const aBI = BigInt(a[i]);
                const cBI = BigInt(c.data[i]);

                expect(cBI).to.be.gte(zeroBI);
                expect(cBI).to.be.lt(aBI);
            }

            printBlock({
                t,
                method: "mulScalarHarness",
                explanation:
                    "Constructs a tiny scalar s=1/1000 and verifies that multiplying by s reduces magnitudes.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, tiny=${tiny}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("divScalar : scalar divide applied to all elements", async function () {
            t++;
            const two = await qInt(2);
            const four = await qInt(4);
            const six = await qInt(6);
            const eight = await qInt(8);

            const a = [two, four, six, eight];
            const k = await qInt(2);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "divScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "divScalarHarness", [rows, cols, a, k]);
            const c = asMatrix(await harness.divScalarHarness(rows, cols, a, k));

            const one = await qInt(1);
            const three = await qInt(3);
            const expected = [one, await qInt(2), three, await qInt(4)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "divScalarHarness",
                explanation:
                    "Divides each entry by a non-zero scalar and checks the resulting quad ratios.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("divScalar : division by zero scalar reverts", async function () {
            t++;
            const vals = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];
            const zero = await qInt(0);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "divScalarHarness", [rows, cols, vals, zero]);
            const gas = await estimateGas(harness, "divScalarHarness", [rows, cols, vals, zero]);
            await expect(harness.divScalarHarness(rows, cols, vals, zero)).to.be.revertedWith("MatrixMaster: division by zero");

            printBlock({
                t,
                method: "divScalarHarness",
                explanation:
                    "Confirms division by an exact zero scalar is rejected to avoid NaN-like states.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("divScalar : divide by negative scalar flips signs", async function () {
            t++;
            const two = await qInt(2);
            const four = await qInt(4);
            const six = await qInt(6);
            const eight = await qInt(8);

            const a = [two, four, six, eight];
            const k = await qInt(-2);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "divScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "divScalarHarness", [rows, cols, a, k]);
            const c = asMatrix(await harness.divScalarHarness(rows, cols, a, k));

            const expected = [await qInt(-1), await qInt(-2), await qInt(-3), await qInt(-4)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "divScalarHarness",
                explanation:
                    "Divides by a negative scalar and checks the resulting entries have flipped signs.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("divScalar : divide by tiny scalar in (0,1) grows magnitudes", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const a = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];

            // tiny = 1 / 1000
            const tiny = await harness.qFromFrac(1n, 1000n);

            await touchGas(harness, "divScalarHarness", [rows, cols, a, tiny]);
            const gas = await estimateGas(harness, "divScalarHarness", [rows, cols, a, tiny]);
            const c = asMatrix(await harness.divScalarHarness(rows, cols, a, tiny));

            const zeroQ = await qInt(0);
            const zeroBI = BigInt(zeroQ);

            // For positive entries, dividing by tiny in (0,1) should yield strictly larger magnitudes
            for (let i = 0; i < a.length; ++i) {
                const aBI = BigInt(a[i]);
                const cBI = BigInt(c.data[i]);

                expect(aBI).to.be.gt(zeroBI);
                expect(cBI).to.be.gt(aBI);
            }

            printBlock({
                t,
                method: "divScalarHarness",
                explanation:
                    "Divides by a very small positive scalar and checks that outputs are magnified.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, tiny=${tiny}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("add : handles negative entries correctly", async function () {
            t++;
            const a = [
                await qInt(-1),
                await qInt(-2),
                await qInt(3),
                await qInt(4),
            ];
            const b = [
                await qInt(5),
                await qInt(-6),
                await qInt(-3),
                await qInt(2),
            ];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "addHarness", [rows, cols, a, rows, cols, b]);
            const gas = await estimateGas(harness, "addHarness", [rows, cols, a, rows, cols, b]);
            const c = asMatrix(await harness.addHarness(rows, cols, a, rows, cols, b));
            const expected = [await qInt(4), await qInt(-8), await qInt(0), await qInt(6)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "addHarness",
                explanation:
                    "Adds matrices containing mixed positive and negative entries and checks signed results.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("add : A + 0 = A", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const A = [
                await qInt(2),
                await qInt(-3),
                await qInt(5),
                await qInt(7),
            ];

            const zeroMat = asMatrix(await harness.zerosHarness(rows, cols));
            await touchGas(harness, "addHarness", [rows, cols, A, zeroMat.rows, zeroMat.cols, zeroMat.data]);
            const gas = await estimateGas(harness, "addHarness", [rows, cols, A, zeroMat.rows, zeroMat.cols, zeroMat.data]);
            const sum = asMatrix(await harness.addHarness(rows, cols, A, zeroMat.rows, zeroMat.cols, zeroMat.data));

            expect(await harness.matricesExactEqual(rows, cols, A, sum.rows, sum.cols, sum.data)).to.equal(true);

            printBlock({
                t,
                method: "addHarness",
                explanation:
                    "Verifies that adding a zero matrix is a no-op and preserves A exactly (A+0=A).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${sum.rows}x${sum.cols}`,
                inHex: `A=${fmtHexArr(A)}, 0=${fmtHexArr(zeroMat.data)}`,
                outHex: fmtHexArr(sum.data),
            });
        });

        it("sub : supports negative entries on both operands", async function () {
            t++;
            const a = [
                await qInt(-5),
                await qInt(2),
                await qInt(0),
                await qInt(7),
            ];
            const b = [
                await qInt(3),
                await qInt(-4),
                await qInt(1),
                await qInt(-2),
            ];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "subHarness", [rows, cols, a, rows, cols, b]);
            const gas = await estimateGas(harness, "subHarness", [rows, cols, a, rows, cols, b]);
            const c = asMatrix(await harness.subHarness(rows, cols, a, rows, cols, b));
            const expected = [await qInt(-8), await qInt(6), await qInt(-1), await qInt(9)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "subHarness",
                explanation:
                    "Subtracts matrices with mixed signs and checks sign-sensitive differences.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("sub : A − A = 0", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const A = [
                await qInt(3),
                await qInt(-2),
                await qInt(7),
                await qInt(0),
            ];

            await touchGas(harness, "subHarness", [rows, cols, A, rows, cols, A]);
            const gas = await estimateGas(harness, "subHarness", [rows, cols, A, rows, cols, A]);
            const diff = asMatrix(await harness.subHarness(rows, cols, A, rows, cols, A));

            const zero = await qInt(0);
            for (const v of diff.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "subHarness",
                explanation:
                    "Subtracts a matrix from itself and confirms all entries cancel exactly to zero.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${diff.rows}x${diff.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(diff.data),
            });
        });
    });
});