// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

/**
 * @title  MatrixMaster: Matrix Library using ABDK Math Quad (bytes16)
 * @notice Provides comprehensive utilities for matrix operations using the ABDK bytes16/quad fixed-point math type.
 *         Tests covers element-wise arithmetic operations on matrices, including addition, subtraction,
 *         scalar multiplication, scalar division. 
 */

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    // quad helpers
    qFromInt(n: bigint): Promise<string>;
    qFromUInt(n: bigint): Promise<string>;
    qFromFrac(n: bigint, m: bigint): Promise<string>;

    // matrix comparison (helper)
    matricesExactEqual(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<boolean>;

    // creation (helper)
    zerosHarness(rows: bigint, cols: bigint): Promise<[bigint, bigint, string[]]>;

    // elementwise arithmetic
    addHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;
    subHarness(aRows: bigint, aCols: bigint, aData: string[], bRows: bigint, bCols: bigint, bData: string[]): Promise<[bigint, bigint, string[]]>;
    mulScalarHarness(rows: bigint, cols: bigint, dataFlat: string[], k: string): Promise<[bigint, bigint, string[]]>;
    divScalarHarness(rows: bigint, cols: bigint, dataFlat: string[], k: string): Promise<[bigint, bigint, string[]]>;
};

const QUAD_POS_ZERO = "0x00000000000000000000000000000000";
const QUAD_NEG_ZERO = "0x80000000000000000000000000000000";

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

function isQuadZero(hex: string): boolean {
    const h = hex.toLowerCase();
    return h === QUAD_POS_ZERO || h === QUAD_NEG_ZERO;
}

// ------------------------------------------------------------
//  Gas Estimation
// ------------------------------------------------------------

async function touchGas(harness: MatrixMasterHarness, method: string, args: any[]) {
    try {
        const data = harness.interface.encodeFunctionData(method, args);
        const [signer] = await ethers.getSigners();
        const to = await harness.getAddress();
        const tx = await signer.sendTransaction({ to, data });
        await tx.wait();
    } catch {
        // Ignored for reverts
    }
}

async function estimateGas(harness: MatrixMasterHarness, method: string, args: any[]) {
    try {
        const anyH = harness as any;
        if (anyH[method]?.estimateGas) {
            return (await anyH[method].estimateGas(...args)).toString();
        }
        const data = harness.interface.encodeFunctionData(method, args);
        const [signer] = await ethers.getSigners();
        const to = await harness.getAddress();
        const gas = await signer.estimateGas({ to, data });
        return gas.toString();
    } catch {
        return "revert";
    }
}

// ------------------------------------------------------------
//  Print Block
// ------------------------------------------------------------

function printBlock({
    t,
    method,
    explanation,
    gas,
    shapeIn,
    shapeOut,
    inHex,
    outHex,
}: any) {
    const sep = "-".repeat(60);
    console.log(
        `
        ${sep}
        Test ${t}
        Method: ${method}
        Explanation: ${explanation}
        Gas Usage: ${gas}
        Shape In: ${shapeIn}
        Shape Out: ${shapeOut}
        Input: ${inHex || "-"}
        Output: ${outHex || "-"}
`.trim(),
    );
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("MatrixMaster — Elementwise arithmetic", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

    // --------------------------------------------------------
    //  Quad helper wrappers
    // --------------------------------------------------------

    const qInt = async (n: number | bigint) => await harness.qFromInt(BigInt(n));

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const math = await MathLibFactory.deploy();
        await math.waitForDeployment();

        const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": await math.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();
    });

    // ------------------------------------------------------------
    //  Section 3: Elementwise Arithmetic (add / sub / mulScalar / divScalar)
    // ------------------------------------------------------------

    describe("Section 3: Elementwise arithmetic", function () {
        // ------------------------------
        //  add
        // ------------------------------

        it("Test 1: add A+B elementwise", async function () {
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
                explanation: "Performs elementwise addition of two same-shaped matrices and checks sum entries.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 2: add shape mismatch reverts", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const b = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];

            await touchGas(harness, "addHarness", [2n, 2n, a, 1n, 4n, b]);
            const gas = await estimateGas(harness, "addHarness", [2n, 2n, a, 1n, 4n, b]);

            await expect(harness.addHarness(2n, 2n, a, 1n, 4n, b)).to.be.revertedWith("MatrixMaster: shape mismatch");

            printBlock({
                t,
                method: "addHarness",
                explanation: "Rejects addition when operand matrices have incompatible dimension layouts.",
                gas,
                shapeIn: "A:2x2, B:1x4",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });

        it("Test 3: add handles negative entries", async function () {
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
            const expected = [
                await qInt(4),
                await qInt(-8),
                await qInt(0),
                await qInt(6),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "addHarness",
                explanation: "Adds matrices containing mixed positive and negative entries and checks signed results.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 4: add A+0=A (zero matrix identity)", async function () {
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
                explanation: "Verifies that adding a zero matrix is a no-op and preserves A exactly (A+0=A).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${sum.rows}x${sum.cols}`,
                inHex: `A=${fmtHexArr(A)}, 0=${fmtHexArr(zeroMat.data)}`,
                outHex: fmtHexArr(sum.data),
            });
        });

        // ------------------------------
        //  sub
        // ------------------------------

        it("Test 5: sub A−B elementwise", async function () {
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
            const expected = [
                await qInt(5),
                await qInt(6),
                await qInt(7),
                await qInt(8),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "subHarness",
                explanation: "Performs elementwise subtraction and ensures each entry is A(i,j)−B(i,j).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 6: sub shape mismatch reverts", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const b = [await qInt(5), await qInt(6), await qInt(7), await qInt(8)];

            await touchGas(harness, "subHarness", [2n, 2n, a, 1n, 4n, b]);
            const gas = await estimateGas(harness, "subHarness", [2n, 2n, a, 1n, 4n, b]);

            await expect(harness.subHarness(2n, 2n, a, 1n, 4n, b)).to.be.revertedWith("MatrixMaster: shape mismatch");

            printBlock({
                t,
                method: "subHarness",
                explanation: "Ensures subtraction requires identical shapes and fails on mismatched sizes.",
                gas,
                shapeIn: "A:2x2, B:1x4",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });

        it("Test 7: sub supports negative operands", async function () {
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
            const expected = [
                await qInt(-8),
                await qInt(6),
                await qInt(-1),
                await qInt(9),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "subHarness",
                explanation: "Subtracts matrices with mixed signs and checks sign-sensitive differences.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 8: sub A−A=0 (cancellation to zero)", async function () {
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
                explanation: "Subtracts a matrix from itself and confirms all entries cancel exactly to zero.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${diff.rows}x${diff.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(diff.data),
            });
        });

        // ------------------------------
        //  divScalar
        // ------------------------------

        it("Test 9: divScalar applies scalar division to all entries", async function () {
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
            const expected = [
                one,
                await qInt(2),
                three,
                await qInt(4),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "divScalarHarness",
                explanation: "Divides each entry by a non-zero scalar and checks the resulting quad ratios.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        // ------------------------------
        //  mulScalar
        // ------------------------------

        it("Test 10: mulScalar applies scalar multiplication to all entries", async function () {
            t++;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const k = await qInt(3);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, k]);

            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, k));
            const expected = [
                await qInt(3),
                await qInt(6),
                await qInt(9),
                await qInt(12),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "mulScalarHarness",
                explanation: "Scales every matrix entry by a quad scalar and validates uniform scaling.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 11: mulScalar by 0 wipes all entries (quad zeros)", async function () {
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
                explanation: "Confirms multiplying by zero produces a matrix whose entries are quad zeros (+-0).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 12: mulScalar by −1 negates all entries", async function () {
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
            const expected = [
                await qInt(-1),
                await qInt(2),
                await qInt(-3),
                await qInt(4),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "mulScalarHarness",
                explanation: "Scales by −1 and checks every entry is sign-flipped while magnitudes are preserved.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 13: mulScalar tiny scalar in (0,1) shrinks magnitudes", async function () {
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
                explanation: "Constructs a tiny scalar s=1/1000 and verifies that multiplying by s reduces magnitudes.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, tiny=${tiny}`,
                outHex: fmtHexArr(c.data),
            });
        });

        // ------------------------------
        //  divScalar edge cases
        // ------------------------------

        it("Test 14: divScalar division by zero scalar reverts", async function () {
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
                explanation: "Confirms division by an exact zero scalar is rejected to avoid NaN-like states.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: "revert",
                inHex: fmtHexArr(vals),
                outHex: "-",
            });
        });

        it("Test 15: divScalar by negative scalar flips signs", async function () {
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

            const expected = [
                await qInt(-1),
                await qInt(-2),
                await qInt(-3),
                await qInt(-4),
            ];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlock({
                t,
                method: "divScalarHarness",
                explanation: "Divides by a negative scalar and checks the resulting entries have flipped signs.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 16: divScalar by tiny scalar in (0,1) grows magnitudes", async function () {
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
                explanation: "Divides by a very small positive scalar and checks that outputs are magnified.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, tiny=${tiny}`,
                outHex: fmtHexArr(c.data),
            });
        });
    });
});