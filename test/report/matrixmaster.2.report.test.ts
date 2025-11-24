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
    // Element Access
    // =========================================================

    describe("Element access (get, set)", function () {
        it("get : reads correct element", async function () {
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
            const row = 1n;
            const col = 2n;

            await touchGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const gas = await estimateGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const v = await harness.getHarness(rows, cols, vals, row, col); // row=1,col=2 -> 6
            expect(v.toLowerCase()).to.equal(vals[5].toLowerCase());

            printBlock({
                t,
                method: "getHarness",
                explanation:
                    "Fetches a single quad entry by row/col indices from a flat row-major matrix.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: v,
            });
        });

        it("get : out-of-bounds access reverts", async function () {
            t++;
            const vals = [
                await qInt(1),
                await qInt(2),
                await qInt(3),
                await qInt(4),
            ];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "getHarness", [rows, cols, vals, 2n, 0n]);
            const gas1 = await estimateGas(harness, "getHarness", [rows, cols, vals, 2n, 0n]);
            await expect(harness.getHarness(rows, cols, vals, 2n, 0n)).to.be.revertedWith("MatrixMaster: index out of bounds");  // row=2 out of range

            printBlock({
                t,
                method: "getHarness",
                explanation:
                    "Checks that reading with an out-of-range row index triggers bounds protection.",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });

            await touchGas(harness, "getHarness", [rows, cols, vals, 0n, 2n]);
            const gas2 = await estimateGas(harness, "getHarness", [rows, cols, vals, 0n, 2n]);

            await expect(harness.getHarness(rows, cols, vals, 0n, 2n)).to.be.revertedWith("MatrixMaster: index out of bounds"); // col=2 out of range

            printBlock({
                t,
                method: "getHarness",
                explanation:
                    "Checks that reading with an out-of-range column index reverts via bounds check.",
                gas: gas2,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("set : writes correct element", async function () {
            t++;
            // 2x2 all zeros
            const zero = await qInt(0);
            const one = await qInt(1);
            const init = [zero, zero, zero, zero];

            const rows = 2n;
            const cols = 2n;
            const row = 1n;
            const col = 0n;

            await touchGas(harness, "setHarness", [rows, cols, init, row, col, one]);
            const gas = await estimateGas(harness, "setHarness", [rows, cols, init, row, col, one]);
            const out = asMatrix(await harness.setHarness(rows, cols, init, row, col, one));
            expect(out.rows).to.equal(rows);
            expect(out.cols).to.equal(cols);

            // Expect only (1,0) to be 1
            expect(out.data[0].toLowerCase()).to.equal(zero.toLowerCase());
            expect(out.data[1].toLowerCase()).to.equal(zero.toLowerCase());
            expect(out.data[2].toLowerCase()).to.equal(one.toLowerCase()); // row1,col0
            expect(out.data[3].toLowerCase()).to.equal(zero.toLowerCase());

            printBlock({
                t,
                method: "setHarness",
                explanation:
                    "Mutates a single matrix cell in-place and verifies only that entry is modified.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: fmtHexArr(init),
                outHex: fmtHexArr(out.data),
            });
        });

        it("set : out-of-bounds write reverts", async function () {
            t++;
            const zero = await qInt(0);
            const init = [zero, zero, zero, zero];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "setHarness", [rows, cols, init, 2n, 0n, zero]);
            const gas1 = await estimateGas(harness, "setHarness", [rows, cols, init, 2n, 0n, zero]);
            await expect(harness.setHarness(rows, cols, init, 2n, 0n, zero)).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlock({
                t,
                method: "setHarness",
                explanation:
                    "Verifies that assigning outside valid row indices reverts with an explicit error.",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(init),
                outHex: "-",
            });

            await touchGas(harness, "setHarness", [rows, cols, init, 0n, 2n, zero]);
            const gas2 = await estimateGas(harness, "setHarness", [rows, cols, init, 0n, 2n, zero]);
            await expect(harness.setHarness(rows, cols, init, 0n, 2n, zero)).to.be.revertedWith("MatrixMaster: index out of bounds");

            printBlock({
                t,
                method: "setHarness",
                explanation:
                    "Verifies that assigning outside valid column indices also reverts safely.",
                gas: gas2,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(init),
                outHex: "-",
            });
        });

        it("get : retrieves corner elements (0,0) and (last,last)", async function () {
            t++;
            const rows = 3n;
            const cols = 3n;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }

            // (0,0)
            await touchGas(harness, "getHarness", [rows, cols, vals, 0n, 0n]);
            const gas1 = await estimateGas(harness, "getHarness", [rows, cols, vals, 0n, 0n]);
            const topLeft = await harness.getHarness(rows, cols, vals, 0n, 0n);
            expect(topLeft.toLowerCase()).to.equal(vals[0].toLowerCase());

            printBlock({
                t,
                method: "getHarness",
                explanation:
                    "Accesses the top-left corner element (0,0) in a 3x3 matrix and checks correct decoding.",
                gas: gas1,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: topLeft,
            });

            // (rows-1, cols-1)
            const lastRow = rows - 1n;
            const lastCol = cols - 1n;

            await touchGas(harness, "getHarness", [rows, cols, vals, lastRow, lastCol]);
            const gas2 = await estimateGas(harness, "getHarness", [rows, cols, vals, lastRow, lastCol]);
            const bottomRight = await harness.getHarness(rows, cols, vals, lastRow, lastCol);
            const expectedIdx = Number(lastRow * cols + lastCol);
            expect(bottomRight.toLowerCase()).to.equal(
                vals[expectedIdx].toLowerCase(),
            );

            printBlock({
                t,
                method: "getHarness",
                explanation:
                    "Accesses the bottom-right corner element (rows-1, cols-1) and verifies flat-index mapping.",
                gas: gas2,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: bottomRight,
            });
        });

        it("get : random interior element inside a larger 10x10 matrix", async function () {
            t++;
            const rows = 10n;
            const cols = 10n;
            const vals: string[] = [];
            for (let i = 1; i <= 100; ++i) {
                vals.push(await qInt(i));
            }

            const row = 7n;
            const col = 3n; // deterministic but “random-looking” interior cell
            const expectedIdx = Number(row * cols + col);
            const expectedVal = vals[expectedIdx];

            await touchGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const gas = await estimateGas(harness, "getHarness", [rows, cols, vals, row, col]);
            const v = await harness.getHarness(rows, cols, vals, row, col);
            expect(v.toLowerCase()).to.equal(expectedVal.toLowerCase());

            printBlock({
                t,
                method: "getHarness",
                explanation:
                    "Indexes an interior element of a 10x10 matrix and validates large-grid index arithmetic.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "scalar",
                inHex: fmtHexArr(vals),
                outHex: v,
            });
        });

        it("set : mutation is reflected on a subsequent get() call", async function () {
            t++;
            const zero = await qInt(0);
            const one = await qInt(1);

            const rows = 3n;
            const cols = 3n;
            const init = Array.from({ length: 9 }, () => zero);

            const row = 2n;
            const col = 1n;

            await touchGas(harness, "setHarness", [rows, cols, init, row, col, one]);
            const gas = await estimateGas(harness, "setHarness", [rows, cols, init, row, col, one]);
            const out = asMatrix(await harness.setHarness(rows, cols, init, row, col, one));

            const readBack = await harness.getHarness(out.rows, out.cols, out.data, row, col);
            expect(readBack.toLowerCase()).to.equal(one.toLowerCase());

            printBlock({
                t,
                method: "setHarness",
                explanation:
                    "Writes a single cell using set() and immediately reads it back via get() to confirm persistence.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${out.rows}x${out.cols}`,
                inHex: fmtHexArr(init),
                outHex: fmtHexArr(out.data),
            });
        });
    });

    // =========================================================
    // Slice & Reshape
    // =========================================================

    describe("Slice & reshape", function () {
        it("slice : centered slice", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 4n;
            const rowStart = 1n;
            const rowEnd = 3n;
            const colStart = 1n;
            const colEnd = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));
            expect(s.rows).to.equal(2n);
            expect(s.cols).to.equal(2n);

            const expected = [await qInt(6), await qInt(7), await qInt(10), await qInt(11)];

            expect(s.data.length).to.equal(4);
            for (let i = 0; i < 4; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation:
                    "Extracts an interior sub-block and checks row-major layout is preserved correctly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : full row slice", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const rows = 3n;
            const cols = 4n;
            await touchGas(harness, "sliceHarness", [rows, cols, vals, 1n, 2n, 0n, 4n]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 1n, 2n, 0n, 4n]);

            // row 1 only, all cols -> [5,6,7,8]
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, 1n, 2n, 0n, 4n));
            expect(s.rows).to.equal(1n);
            expect(s.cols).to.equal(4n);

            const expected = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];
            for (let i = 0; i < 4; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation:
                    "Slices a single full row and verifies contiguous row extraction semantics.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : invalid or out-of-range bounds revert", async function () {
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 2n;
            const cols = 3n;

            // rowStart >= rowEnd
            {
                t++;
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 1n, 1n, 0n, 2n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 1n, 1n, 0n, 2n]);

                await expect(harness.sliceHarness(rows, cols, vals, 1n, 1n, 0n, 2n)).to.be.revertedWith("MatrixMaster: invalid slice range");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation:
                        "Ensures start/end row indices must form a strictly positive slice window.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // colStart >= colEnd
            {
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 1n, 2n, 2n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 1n, 2n, 2n]);
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 1n, 2n, 2n)).to.be.revertedWith("MatrixMaster: invalid slice range");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation:
                        "Ensures start/end column indices also must define a non-empty valid range.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // rowEnd > rows
            {
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 3n, 0n, 2n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 3n, 0n, 2n]);
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 3n, 0n, 2n)).to.be.revertedWith("MatrixMaster: slice out of bounds");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation:
                        "Guards against slice windows that extend beyond the matrix row dimension.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }

            // colEnd > cols
            {
                await touchGas(harness, "sliceHarness", [rows, cols, vals, 0n, 2n, 0n, 4n]);
                const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, 0n, 2n, 0n, 4n]);
                await expect(harness.sliceHarness(rows, cols, vals, 0n, 2n, 0n, 4n)).to.be.revertedWith("MatrixMaster: slice out of bounds");

                printBlock({
                    t,
                    method: "sliceHarness",
                    explanation:
                        "Guards against slice windows that overflow the matrix column dimension.",
                    gas,
                    shapeIn: `${rows}x${cols}`,
                    shapeOut: "revert",
                    inHex: fmtHexArr(vals),
                    outHex: "-",
                });
            }
        });

        it("slice : full column slice", async function () {
            t++;
            // 3x4 matrix [1..12]
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 4n;

            // take full column 2 (zero-based) => [3,7,11]
            const rowStart = 0n;
            const rowEnd = 3n;
            const colStart = 2n;
            const colEnd = 3n;

            const expected = [
                await qInt(3),
                await qInt(7),
                await qInt(11),
            ];

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(3n);
            expect(s.cols).to.equal(1n);
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation:
                    "Selects a single full column and ensures the resulting 3x1 view matches the expected entries.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("reshape : reshape 2x6 → 3x4 preserves order", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 12; ++i) {
                vals.push(await qInt(i));
            }

            const srcRows = 2n;
            const srcCols = 6n;
            const dstRows = 3n;
            const dstCols = 4n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const r = asMatrix(await harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols));

            expect(r.rows).to.equal(dstRows);
            expect(r.cols).to.equal(dstCols);
            expect(r.data.length).to.equal(12);

            // reshape does not change data order
            for (let i = 0; i < 12; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation:
                    "Changes matrix shape while reusing the same flat data array and order of entries.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("reshape : mismatched area reverts", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const srcRows = 2n;
            const srcCols = 3n;
            const dstRows = 4n;
            const dstCols = 2n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);

            // 2x3 -> area=6; 4x2 -> area=8 => revert
            await expect(harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols)).to.be.revertedWith("MatrixMaster: reshape area mismatch");

            printBlock({
                t,
                method: "reshapeHarness",
                explanation:
                    "Prevents reshapes that would change total element count and corrupt matrix data.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("slice : full matrix slice returns an identical layout", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 2n;
            const cols = 3n;

            const rowStart = 0n;
            const rowEnd = rows;
            const colStart = 0n;
            const colEnd = cols;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(rows);
            expect(s.cols).to.equal(cols);
            expect(s.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation:
                    "Slices the entire matrix domain and checks it is bitwise identical to the original layout.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : last row boundary slices", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 3n;

            // last row only: [7,8,9]
            const rowStart = 2n;
            const rowEnd = 3n;
            const colStart = 0n;
            const colEnd = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(1n);
            expect(s.cols).to.equal(3n);
            const expected = [await qInt(7), await qInt(8), await qInt(9)];
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation:
                    "Slices the last row of a 3x3 matrix to ensure upper-bound row indices behave correctly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });
        });

        it("slice : last column boundary slices", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 9; ++i) {
                vals.push(await qInt(i));
            }
            const rows = 3n;
            const cols = 3n;

            // last column only: [3,6,9]
            const rowStart = 0n;
            const rowEnd = 3n;
            const colStart = 2n;
            const colEnd = 3n;

            await touchGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const gas = await estimateGas(harness, "sliceHarness", [rows, cols, vals, rowStart, rowEnd, colStart, colEnd]);
            const s = asMatrix(await harness.sliceHarness(rows, cols, vals, rowStart, rowEnd, colStart, colEnd));

            expect(s.rows).to.equal(3n);
            expect(s.cols).to.equal(1n);
            const expected = [await qInt(3), await qInt(6), await qInt(9)];
            for (let i = 0; i < expected.length; ++i) {
                expect(s.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "sliceHarness",
                explanation:
                    "Slices the last column of a 3x3 matrix, checking column upper-bounds are handled cleanly.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${s.rows}x${s.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(s.data),
            });

        });

        it("reshape : MxN to 1x(MN) vector", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const srcRows = 2n;
            const srcCols = 3n;
            const dstRows = 1n;
            const dstCols = 6n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const r = asMatrix(await harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols));

            expect(r.rows).to.equal(dstRows);
            expect(r.cols).to.equal(dstCols);
            expect(r.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation:
                    "Flattens a 2x3 matrix into a 1x6 row vector, preserving the original row-major order.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("reshape : 1x(MN) vector back to MxN", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const srcRows = 1n;
            const srcCols = 6n;
            const dstRows = 2n;
            const dstCols = 3n;

            await touchGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const gas = await estimateGas(harness, "reshapeHarness", [srcRows, srcCols, vals, dstRows, dstCols]);
            const r = asMatrix(await harness.reshapeHarness(srcRows, srcCols, vals, dstRows, dstCols));

            expect(r.rows).to.equal(dstRows);
            expect(r.cols).to.equal(dstCols);
            expect(r.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation:
                    "Takes a 1x6 vector and reshapes it back to 2x3 while keeping the flat data untouched.",
                gas,
                shapeIn: `${srcRows}x${srcCols}`,
                shapeOut: `${r.rows}x${r.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r.data),
            });
        });

        it("reshape : double reshape returns to the original shape", async function () {
            t++;
            const vals: string[] = [];
            for (let i = 1; i <= 6; ++i) {
                vals.push(await qInt(i));
            }
            const rows0 = 2n;
            const cols0 = 3n;
            const rows1 = 3n;
            const cols1 = 2n;

            await touchGas(harness, "reshapeHarness", [rows0, cols0, vals, rows1, cols1]);
            const gas = await estimateGas(harness, "reshapeHarness", [rows0, cols0, vals, rows1, cols1]);
            const r1 = asMatrix(await harness.reshapeHarness(rows0, cols0, vals, rows1, cols1));
            const r2 = asMatrix(await harness.reshapeHarness(r1.rows, r1.cols, r1.data, rows0, cols0));

            expect(r2.rows).to.equal(rows0);
            expect(r2.cols).to.equal(cols0);
            expect(r2.data.length).to.equal(vals.length);
            for (let i = 0; i < vals.length; ++i) {
                expect(r2.data[i].toLowerCase()).to.equal(vals[i].toLowerCase());
            }

            printBlock({
                t,
                method: "reshapeHarness",
                explanation:
                    "Reshapes 2x3→3x2→2x3 and verifies the final matrix is identical to the original.",
                gas,
                shapeIn: `${rows0}x${cols0}`,
                shapeOut: `${r2.rows}x${r2.cols}`,
                inHex: fmtHexArr(vals),
                outHex: fmtHexArr(r2.data),
            });
        });
    });
});