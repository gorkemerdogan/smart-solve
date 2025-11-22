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

    // matrix comparison helpers
    matricesExactEqual(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<boolean>;

    // creation
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    onesHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;
    createIdentityMatrixHarness(n: bigint): Promise<[bigint, bigint, string[]]>;
    fromDiagonalHarness(diag: string[]): Promise<[bigint, bigint, string[]]>;
    randomUniformHarness(rows: bigint, cols: bigint, seed: string): Promise<[bigint, bigint, string[]]>;

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
    hasConvergedHarness(prevEig: string, currEig: string, tol: string): Promise<boolean>;
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

        it("randomUniform : shape and determinism (same seed yields same matrix)", async function () {
            t++;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("seed-123"));
            const rows = 3n;
            const cols = 4n;
            await touchGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomUniformHarness", [rows, cols, seed]);

            const m1 = asMatrix(await harness.randomUniformHarness(rows, cols, seed));
            const m2 = asMatrix(await harness.randomUniformHarness(rows, cols, seed));

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
                method: "randomUniformHarness",
                explanation:
                    "Generates pseudo-random quad entries in [0,1) and verifies deterministic seeding.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m1.data),
            });
        });

        it("randomUniform : different seeds give different data (with high probability)", async function () {
            t++;
            const seed1 = ethers.keccak256(ethers.toUtf8Bytes("seed-A"));
            const seed2 = ethers.keccak256(ethers.toUtf8Bytes("seed-B"));
            const rows = 2n;
            const cols = 3n;
            await touchGas(harness, "randomUniformHarness", [rows, cols, seed1]);
            const gas = await estimateGas(harness, "randomUniformHarness", [rows, cols, seed1]);

            const m1 = asMatrix(await harness.randomUniformHarness(rows, cols, seed1));
            const m2 = asMatrix(await harness.randomUniformHarness(rows, cols, seed2));

            expect(m1.data.length).to.equal(6);
            expect(m2.data.length).to.equal(6);

            const same = m1.data.every((v, i) => v === m2.data[i]);
            expect(same).to.equal(false);

            printBlock({
                t,
                method: "randomUniformHarness",
                explanation:
                    "Uses two different seeds and checks that generated patterns diverge as expected.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m1.rows}x${m1.cols}`,
                inHex: `seed1=${seed1}, seed2=${seed2}`,
                outHex: `m1=${fmtHexArr(m1.data)}, m2=${fmtHexArr(m2.data)}`,
            });
        });

        it("randomUniform : all values lie in [0,1) range", async function () {
            t++;
            const rows = 3n;
            const cols = 3n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("range-check"));

            await touchGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomUniformHarness", [rows, cols, seed]);

            const m = asMatrix(await harness.randomUniformHarness(rows, cols, seed));
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
                method: "randomUniformHarness",
                explanation:
                    "Samples a small 3x3 random matrix and enforces every encoded quad lies in [0,1).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomUniform : vector shapes 1xN are supported", async function () {
            t++;
            const rows = 1n;
            const cols = 5n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-1xN"));

            await touchGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const m = asMatrix(await harness.randomUniformHarness(rows, cols, seed));

            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            printBlock({
                t,
                method: "randomUniformHarness",
                explanation:
                    "Generates a 1xN random row vector for use as simple quad noise or weights.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomUniform : vector shapes Nx1 are supported", async function () {
            t++;
            const rows = 5n;
            const cols = 1n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-Nx1"));

            await touchGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const m = asMatrix(await harness.randomUniformHarness(rows, cols, seed));

            expect(m.rows).to.equal(rows);
            expect(m.cols).to.equal(cols);
            expect(m.data.length).to.equal(Number(rows * cols));

            printBlock({
                t,
                method: "randomUniformHarness",
                explanation:
                    "Generates an Nx1 random column vector suitable for stochastic gradient toy examples.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${m.rows}x${m.cols}`,
                inHex: `seed=${seed}`,
                outHex: fmtHexArr(m.data),
            });
        });

        it("randomUniform : structure / diversity: non-trivial variety across entries", async function () {
            t++;
            const rows = 4n;
            const cols = 4n;
            const seed = ethers.keccak256(ethers.toUtf8Bytes("ru-diversity"));

            await touchGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const gas = await estimateGas(harness, "randomUniformHarness", [rows, cols, seed]);
            const m = asMatrix(await harness.randomUniformHarness(rows, cols, seed));
            const zeroQ = await qInt(0);

            const distinct = new Set(m.data);
            const hasNonZero = m.data.some((v) => v.toLowerCase() !== zeroQ.toLowerCase());

            // Not a strict proof, just a sanity check that the generator is not degenerate.
            expect(distinct.size).to.be.greaterThan(1);
            expect(hasNonZero).to.equal(true);

            printBlock({
                t,
                method: "randomUniformHarness",
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

            const A = asMatrix(await harness.randomUniformHarness(n, n, seed));

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

            const A = asMatrix(await harness.randomUniformHarness(n, n, seed));

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

            // Start with a big 100x100 zero matrix (cheap compared to randomUniform).
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

            // Build a small scalar s = 1 / 1000 using divScalar on a 1x1 matrix [1]
            const big = await qInt(1000);
            const one = await qInt(1);
            const tinyMat = asMatrix(
                await harness.divScalarHarness(1n, 1n, [one], big),
            );
            const tiny = tinyMat.data[0];

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

            // tiny = 1 / 1000 again
            const big = await qInt(1000);
            const one = await qInt(1);
            const tinyMat = asMatrix(
                await harness.divScalarHarness(1n, 1n, [one], big),
            );
            const tiny = tinyMat.data[0];

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

            // Build a tiny scalar s = 1 / 1000 using divScalar on a 1x1 matrix [1]
            const one = await qInt(1);
            const big = await qInt(1000);
            const tinyMat = asMatrix(
                await harness.divScalarHarness(1n, 1n, [one], big),
            );
            const tiny = tinyMat.data[0];

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

    describe("Matrix multiplication", function () {
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

            // FIX: Swap dimensions to 3 rows, 1 col
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

            // FIX: Swap dimensions to 2 rows, 1 col
            await touchGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 2n, 1n, b]);
            const out = await harness.dotHarness(2n, 1n, a, 2n, 1n, b);

            const expected = await qInt(-22);

            expect(out.toLowerCase()).to.equal(expected.toLowerCase());

            // ... printBlock update ...
        });

        it("dot : length mismatch reverts", async function () {
            t++;

            const a = [await qInt(1), await qInt(2)];
            const b = [await qInt(3)];

            // FIX: Swap dimensions so cols=1. 
            // Now it will pass the first require, and fail the second (length) require.
            await touchGas(harness, "dotHarness", [2n, 1n, a, 1n, 1n, b]);
            const gas = await estimateGas(harness, "dotHarness", [2n, 1n, a, 1n, 1n, b]);

            await expect(
                harness.dotHarness(2n, 1n, a, 1n, 1n, b)
            ).to.be.revertedWith("MatrixMaster: dot length mismatch");

            // ... printBlock update ...
        });
    });

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
            const tinyMat = asMatrix(await harness.divScalarHarness(1n, 1n, [one], big));
            const eps = tinyMat.data[0];

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
            const ex0 = (await harness.divScalarHarness(1n, 1n, [await qInt(3)], five))[2][0];
            const ex1 = await qInt(0);
            const ex2 = (await harness.divScalarHarness(1n, 1n, [await qInt(4)], five))[2][0];

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
});