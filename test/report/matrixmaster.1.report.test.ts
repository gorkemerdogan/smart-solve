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
    // Creation
    // =========================================================

    describe("Creation", function () {
        it("zeros : returns correct all-zero matrix", async function () {
            t++;
            const rows = 2n;
            const cols = 3n;
            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(6);

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "zerosHarness",
                explanation:
                    "Creates a dense rows×cols matrix fully initialized with quad-precision zeros.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("zeros : very large 100x100 dense matrix", async function () {
            t++;
            const rows = 100n;
            const cols = 100n;

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "zerosHarness",
                explanation:
                    "Stress-allocates a 100x100 dense matrix and verifies every entry is an exact quad zero.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("ones : returns correct all-one matrix", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            await touchGas(harness, "onesHarness", [rows, cols]);
            const gas = await estimateGas(harness, "onesHarness", [rows, cols]);

            const m = asMatrix(await harness.onesHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(4);

            const one = await qInt(1);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(one.toLowerCase());
            }

            printBlock({
                t,
                method: "onesHarness",
                explanation:
                    "Allocates a rows×cols matrix filled with quad-precision ones for baseline checks.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("Identity Matrix : produces valid identity", async function () {
            t++;
            const n = 3n;
            await touchGas(harness, "createIdentityMatrixHarness", [n]);
            const gas = await estimateGas(harness, "createIdentityMatrixHarness", [n]);

            const m = asMatrix(await harness.createIdentityMatrixHarness(n));
            const zero = await qInt(0);
            const one = await qInt(1);

            expect(m.rows).to.equal(n);
            expect(m.cols).to.equal(n);
            expect(m.data.length).to.equal(9);

            const rows = Number(m.rows);
            const cols = Number(m.cols);

            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected = i === j ? one : zero;
                    expect(m.data[idx].toLowerCase()).to.equal(expected.toLowerCase());
                }
            }

            printBlock({
                t,
                method: "createIdentityMatrixHarness",
                explanation:
                    "Builds a square identity matrix with ones on the diagonal and strict zeros elsewhere.",
                gas,
                shapeIn: `n=${n.toString()}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("Identity Matrix : n = 0 reverts", async function () {
            t++;
            const n = 0n;
            await touchGas(harness, "createIdentityMatrixHarness", [n]);
            const gas = await estimateGas(harness, "createIdentityMatrixHarness", [n]);

            await expect(harness.createIdentityMatrixHarness(n)).to.be.revertedWith(
                "MatrixMaster: n must be > 0",
            );

            printBlock({
                t,
                method: "createIdentityMatrixHarness",
                explanation:
                    "Ensures identity construction rejects zero-sized matrices via dimension guard.",
                gas,
                shapeIn: `n=${n.toString()}`,
                shapeOut: "revert",
                inHex: "-",
                outHex: "-",
            });
        });

        it("fromDiagonal : diagonal assembled correctly", async function () {
            t++;
            const d0 = await qInt(1);
            const d1 = await qInt(2);
            const d2 = await qInt(3);
            const diag = [d0, d1, d2];

            await touchGas(harness, "fromDiagonalHarness", [diag]);
            const gas = await estimateGas(harness, "fromDiagonalHarness", [diag]);
            const m = asMatrix(await harness.fromDiagonalHarness(diag));
            const zero = await qInt(0);

            expect(m.rows).to.equal(3n);
            expect(m.cols).to.equal(3n);

            const rows = Number(m.rows);
            const cols = Number(m.cols);

            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected =
                        i === j ? [d0, d1, d2][i].toLowerCase() : zero.toLowerCase();
                    expect(m.data[idx].toLowerCase()).to.equal(expected);
                }
            }

            printBlock({
                t,
                method: "fromDiagonalHarness",
                explanation:
                    "Lifts a 1D quad array into a square diagonal matrix with zeros off the diagonal.",
                gas,
                shapeIn: `diag length=${diag.length}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: fmtHexArr(diag),
                outHex: fmtHexArr(m.data),
            });
        });

        it("fromDiagonal : empty diag reverts", async function () {
            t++;
            const diag: string[] = [];
            await touchGas(harness, "fromDiagonalHarness", [diag]);
            const gas = await estimateGas(harness, "fromDiagonalHarness", [diag]);

            await expect(harness.fromDiagonalHarness(diag)).to.be.revertedWith(
                "MatrixMaster: empty diagonal",
            );

            printBlock({
                t,
                method: "fromDiagonalHarness",
                explanation:
                    "Checks that constructing a diagonal matrix from an empty vector is rejected.",
                gas,
                shapeIn: "diag length=0",
                shapeOut: "revert",
                inHex: "[]",
                outHex: "-",
            });
        });

        it("fromDiagonal : supports negative and large-magnitude diagonal entries", async function () {
            t++;
            const d0 = await qInt(-5);
            const d1 = await qInt(0);
            const d2 = await qInt(42);
            const d3 = await qInt(1_000_000);
            const diag = [d0, d1, d2, d3];

            await touchGas(harness, "fromDiagonalHarness", [diag]);
            const gas = await estimateGas(harness, "fromDiagonalHarness", [diag]);
            const m = asMatrix(await harness.fromDiagonalHarness(diag));
            const zero = await qInt(0);

            expect(m.rows).to.equal(4n);
            expect(m.cols).to.equal(4n);

            const rows = Number(m.rows);
            const cols = Number(m.cols);
            for (let i = 0; i < rows; ++i) {
                for (let j = 0; j < cols; ++j) {
                    const idx = i * cols + j;
                    const expected =
                        i === j ? diag[i].toLowerCase() : zero.toLowerCase();
                    expect(m.data[idx].toLowerCase()).to.equal(expected);
                }
            }

            printBlock({
                t,
                method: "fromDiagonalHarness",
                explanation:
                    "Builds a diagonal matrix with mixed negative, zero, and large-magnitude quad entries.",
                gas,
                shapeIn: `diag length=${diag.length}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: fmtHexArr(diag),
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomMatrix : shape and determinism (same seed yields same matrix)", async function () {
            t++;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("seed-123"));
            const rows = 3n;
            const cols = 4n;
            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m1 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));
            const m2 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            expect(m1.rows).to.equal(rows);
            expect(m1.cols).to.equal(cols);
            expect(m1.data.length).to.equal(12);
            expect(m2.rows).to.equal(rows);
            expect(m2.cols).to.equal(cols);

            // same seed => identical data
            expect(m1.data.length).to.equal(m2.data.length);
            for (let i = 0; i < m1.data.length; ++i) {
                expect(m1.data[i]).to.equal(m2.data[i]);
            }

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation:
                    "Generates pseudo-random quad entries in [0,1) and verifies deterministic seeding.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m1.data),
            });
        });

        it("randomMatrix : different seeds give different data (with high probability)", async function () {
            t++;
            const seed1 = ethers.keccak256(ethers.toUtf8Bytes("seed-A"));
            const seed2 = ethers.keccak256(ethers.toUtf8Bytes("seed-B"));
            const rows = 2n;
            const cols = 3n;
            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed1]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed1]);

            const m1 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed1));
            const m2 = asMatrix(await harness.randomMatrixHarness(rows, cols, seed2));

            expect(m1.data.length).to.equal(6);
            expect(m2.data.length).to.equal(6);

            const same = m1.data.every((v, i) => v === m2.data[i]);
            expect(same).to.equal(false);

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation:
                    "Uses two different seeds and checks that generated patterns diverge as expected.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed1=${seed1}, seed2=${seed2}`,
                outHex: `m1=${fmtHexArr(m1.data)}, m2=${fmtHexArr(m2.data)}`,
            });
        });

        it("randomMatrix : all values lie in [0,1) range", async function () {
            t++;
            const rows = 3n;
            const cols = 3n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("range-check"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));
            const zeroQ = await qInt(0);
            const oneQ = await qInt(1);
            const zeroBI = BigInt(zeroQ);
            const oneBI = BigInt(oneQ);

            for (const v of m.data) {
                const bi = BigInt(v);
                expect(bi).to.be.gte(zeroBI);
                expect(bi).to.be.lt(oneBI);
            }

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation:
                    "Samples a small 3x3 random matrix and enforces every encoded quad lies in [0,1).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomMatrix : vector shapes 1xN are supported", async function () {
            t++;
            const rows = 1n;
            const cols = 5n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-1xN"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation:
                    "Generates a 1xN random row vector for use as simple quad noise or weights.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomMatrix : vector shapes Nx1 are supported", async function () {
            t++;
            const rows = 5n;
            const cols = 1n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-Nx1"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));

            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation:
                    "Generates an Nx1 random column vector suitable for stochastic gradient toy examples.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomMatrix : structure / diversity: non-trivial variety across entries", async function () {
            t++;
            const rows = 4n;
            const cols = 4n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-diversity"));

            await touchGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomMatrixHarness", [rows, cols, seed]);
            const m = asMatrix(await harness.randomMatrixHarness(rows, cols, seed));
            const zeroQ = await qInt(0);

            const distinct = new Set(m.data);
            const hasNonZero = m.data.some((v) => v.toLowerCase() !== zeroQ.toLowerCase());

            // Not a strict proof, just a sanity check that the generator is not degenerate.
            expect(distinct.size).to.be.greaterThan(1);
            expect(hasNonZero).to.equal(true);

            printBlock({
                t,
                method: "randomMatrixHarness",
                explanation:
                    "Samples a 4x4 random matrix and verifies it exhibits non-trivial value diversity.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("zeros : large 50x50 dense matrix", async function () {
            t++;
            const rows = 50n;
            const cols = 50n;

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "zerosHarness",
                explanation:
                    "Allocates a relatively large 50x50 dense matrix and validates all entries are exact zeros.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("zeros : vector shapes 1xN return all-zero vectors", async function () {
            t++;
            const rows = 1n;
            const cols = 5n;

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "zerosHarness",
                explanation:
                    "Builds a 1xN row vector of quad zeros and confirms it behaves like a zero row.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("zeros : vector shapes  Nx1 return all-zero vectors", async function () {
            t++;
            const rows = 5n;
            const cols = 1n;

            await touchGas(harness, "zerosHarness", [rows, cols]);
            const gas = await estimateGas(harness, "zerosHarness", [rows, cols]);

            const m = asMatrix(await harness.zerosHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            const zero = await qInt(0);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlock({
                t,
                method: "zerosHarness",
                explanation:
                    "Allocates an Nx1 column vector of zeros and checks every entry is exactly zero.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("ones : large 40x40 dense matrix", async function () {
            t++;
            const rows = 40n;
            const cols = 40n;

            await touchGas(harness, "onesHarness", [rows, cols]);
            const gas = await estimateGas(harness, "onesHarness", [rows, cols]);

            const m = asMatrix(await harness.onesHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            const one = await qInt(1);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(one.toLowerCase());
            }

            printBlock({
                t,
                method: "onesHarness",
                explanation:
                    "Constructs a larger 40x40 matrix of quad ones to exercise allocation and fill loops.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("ones : vector shapes 1xN return all-one vectors", async function () {
            t++;
            const rows = 1n;
            const cols = 4n;

            await touchGas(harness, "onesHarness", [rows, cols]);
            const gas = await estimateGas(harness, "onesHarness", [rows, cols]);

            const m = asMatrix(await harness.onesHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);

            const one = await qInt(1);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(one.toLowerCase());
            }

            printBlock({
                t,
                method: "onesHarness",
                explanation:
                    "Creates a 1xN row vector of ones for simple broadcast-style operations.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("ones : vector shapes  Nx1 return all-one vectors", async function () {
            t++;
            const rows = 4n;
            const cols = 1n;

            await touchGas(harness, "onesHarness", [rows, cols]);
            const gas = await estimateGas(harness, "onesHarness", [rows, cols]);

            const m = asMatrix(await harness.onesHarness(rows, cols));
            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);

            const one = await qInt(1);
            for (const v of m.data) {
                expect(v.toLowerCase()).to.equal(one.toLowerCase());
            }

            printBlock({
                t,
                method: "onesHarness",
                explanation:
                    "Creates an Nx1 column vector of ones to mimic simple column-bias terms.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: "-",
                outHex: fmtHexArr(m.data),
            });
        });

        it("Identity Matrix : transpose(identity) = identity", async function () {
            t++;
            const n = 3n;

            const I = asMatrix(await harness.createIdentityMatrixHarness(n));

            await touchGas(harness, "transposeHarness", [I.rows, I.cols, I.data]);
            const gas = await estimateGas(harness, "transposeHarness", [I.rows, I.cols, I.data]);
            const IT = asMatrix(await harness.transposeHarness(I.rows, I.cols, I.data));

            // I^T should be bitwise identical to I
            expect(await harness.matricesExactEqual(I.rows, I.cols, I.data, IT.rows, IT.cols, IT.data)).to.equal(true);

            printBlock({
                t,
                method: "transposeHarness",
                explanation:
                    "Confirms that transposing an identity matrix leaves it unchanged (Iᵀ = I).",
                gas,
                shapeIn: `${I.rows}x${I.cols}`,
                shapeOut: `${IT.rows}x${IT.cols}`,
                inHex: fmtHexArr(I.data),
                outHex: fmtHexArr(IT.data),
            });
        });
    });
});