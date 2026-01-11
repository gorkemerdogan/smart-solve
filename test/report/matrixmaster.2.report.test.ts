// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "../test-utils";

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

            printBlockMatrix({
                t,
                method: "addHarness",
                explanation: "Performs elementwise addition A+B.",
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
            
            await expect(harness.addHarness(2n, 2n, a, 1n, 4n, b)).to.be.revertedWith("MatrixMaster: shape mismatch");

            printBlockMatrix({
                t, method: "addHarness",
                explanation: "Rejects addition when operand matrices have incompatible shapes (2x2 vs 1x4).",
                gas: "Revert",
                shapeIn: "A:2x2, B:1x4",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });

        it("Test 3: add handles negative entries", async function () {
            t++;
            const a = [await qInt(-1), await qInt(-2), await qInt(3), await qInt(4)];
            const b = [await qInt(5), await qInt(-6), await qInt(-3), await qInt(2)];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "addHarness", [rows, cols, a, rows, cols, b]);
            const gas = await estimateGas(harness, "addHarness", [rows, cols, a, rows, cols, b]);

            const c = asMatrix(await harness.addHarness(rows, cols, a, rows, cols, b));
            const expected = [await qInt(4), await qInt(-8), await qInt(0), await qInt(6)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "addHarness",
                explanation: "Adds matrices with mixed signs correctly.",
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
            const A = [await qInt(2), await qInt(-3), await qInt(5), await qInt(7)];
            const zeroMat = asMatrix(await harness.zerosHarness(rows, cols));

            await touchGas(harness, "addHarness", [rows, cols, A, zeroMat.rows, zeroMat.cols, zeroMat.data]);
            const gas = await estimateGas(harness, "addHarness", [rows, cols, A, zeroMat.rows, zeroMat.cols, zeroMat.data]);

            const sum = asMatrix(await harness.addHarness(rows, cols, A, zeroMat.rows, zeroMat.cols, zeroMat.data));

            expect(await harness.matricesExactEqual(rows, cols, A, sum.rows, sum.cols, sum.data)).to.equal(true);

            printBlockMatrix({
                t,
                method: "addHarness",
                explanation: "Verifies additive identity property A+0=A.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${sum.rows}x${sum.cols}`,
                inHex: `A=${fmtHexArr(A)}, 0=Zeros`,
                outHex: fmtHexArr(sum.data),
            });
        });

        // ------------------------------
        //  sub
        // ------------------------------

        it("Test 5: sub A−B elementwise", async function () {
            t++;
            const a = [await qInt(6), await qInt(8), await qInt(10), await qInt(12)];
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

            printBlockMatrix({
                t,
                method: "subHarness",
                explanation: "Performs elementwise subtraction A-B.",
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
            
            await expect(harness.subHarness(2n, 2n, a, 1n, 4n, b)).to.be.revertedWith("MatrixMaster: shape mismatch");

            printBlockMatrix({
                t,
                method: "subHarness",
                explanation: "Ensures subtraction fails on mismatched sizes.",
                gas: "Revert",
                shapeIn: "A:2x2, B:1x4",
                shapeOut: "revert",
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: "-",
            });
        });

        it("Test 7: sub supports negative operands", async function () {
            t++;
            const a = [await qInt(-5), await qInt(2), await qInt(0), await qInt(7)];
            const b = [await qInt(3), await qInt(-4), await qInt(1), await qInt(-2)];
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "subHarness", [rows, cols, a, rows, cols, b]);
            const gas = await estimateGas(harness, "subHarness", [rows, cols, a, rows, cols, b]);

            const c = asMatrix(await harness.subHarness(rows, cols, a, rows, cols, b));
            const expected = [await qInt(-8), await qInt(6), await qInt(-1), await qInt(9)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "subHarness",
                explanation: "Subtracts with mixed signs (e.g., 2 - (-4) = 6).",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, B=${fmtHexArr(b)}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 8: sub A−A=0 (cancellation)", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const A = [await qInt(3), await qInt(-2), await qInt(7), await qInt(0)];

            await touchGas(harness, "subHarness", [rows, cols, A, rows, cols, A]);
            const gas = await estimateGas(harness, "subHarness", [rows, cols, A, rows, cols, A]);

            const diff = asMatrix(await harness.subHarness(rows, cols, A, rows, cols, A));
            const zero = await qInt(0);
            
            for (const v of diff.data) {
                expect(v.toLowerCase()).to.equal(zero.toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "subHarness",
                explanation: "Self-subtraction A-A must equal Zero Matrix.",
                gas,
                shapeIn: `${rows}x${cols}`, shapeOut: `${diff.rows}x${diff.cols}`,
                inHex: fmtHexArr(A),
                outHex: fmtHexArr(diff.data),
            });
        });

        // ------------------------------
        //  divScalar / mulScalar
        // ------------------------------

        it("Test 9: divScalar applies scalar division", async function () {
            t++;
            const a = [await qInt(2), await qInt(4), await qInt(6), await qInt(8)];
            const k = await qInt(2);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "divScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "divScalarHarness", [rows, cols, a, k]);

            const c = asMatrix(await harness.divScalarHarness(rows, cols, a, k));
            const expected = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];

            for (let i = 0; i < expected.length; ++i) {
                expect(c.data[i].toLowerCase()).to.equal(expected[i].toLowerCase());
            }

            printBlockMatrix({
                t,
                method: "divScalarHarness",
                explanation: "Divides each entry by 2.0.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 10: mulScalar applies scalar multiplication", async function () {
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

            printBlockMatrix({
                t,
                method: "mulScalarHarness",
                explanation: "Multiplies each entry by 3.0.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 11: mulScalar by 0 wipes entries", async function () {
            t++;
            const a = [await qInt(1), await qInt(-2), await qInt(3), await qInt(-4)];
            const k = await qInt(0);
            const rows = 2n;
            const cols = 2n;

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, k]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, k]);

            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, k));

            for (const v of c.data) {
                expect(isQuadZero(v)).to.equal(true);
            }

            printBlockMatrix({
                t,
                method: "mulScalarHarness",
                explanation: "Multiplying by zero produces zero matrix.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 12: mulScalar by -1 flips signs", async function () {
            t++;
            const a = [await qInt(1), await qInt(-2), await qInt(3), await qInt(-4)];
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

            printBlockMatrix({
                t,
                method: "mulScalarHarness",
                explanation: "Multiplying by -1 negates all values.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 13: mulScalar tiny scalar shrinks magnitudes", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const a = [await qInt(10), await qInt(20), await qInt(30), await qInt(40)];
            const tiny = await harness.qFromFrac(1n, 1000n); // 0.001

            await touchGas(harness, "mulScalarHarness", [rows, cols, a, tiny]);
            const gas = await estimateGas(harness, "mulScalarHarness", [rows, cols, a, tiny]);

            const c = asMatrix(await harness.mulScalarHarness(rows, cols, a, tiny));

            for (let i = 0; i < a.length; ++i) {
                const origVal = await harness.toFloat(a[i]);
                const newVal = await harness.toFloat(c.data[i]);
                
                // New value should be approx 0.001 * Orig
                expect(newVal).to.be.lessThan(origVal);
                expect(newVal).to.be.greaterThan(0);
            }

            printBlockMatrix({
                t,
                method: "mulScalarHarness",
                explanation: "Multiplication by s=0.001 reduces magnitude.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, tiny=${tiny}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 14: divScalar division by zero reverts", async function () {
            t++;
            const vals = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const zero = await qInt(0);
                        
            await expect(harness.divScalarHarness(2n, 2n, vals, zero)).to.be.revertedWith("MatrixMaster: division by zero");

            printBlockMatrix({
                t,
                method: "divScalarHarness",
                explanation: "Rejects division by zero scalar.",
                gas: "Revert",
                shapeIn: "2x2",
                shapeOut: "revert",
                inHex: fmtHexArr(vals), outHex: "-",
            });
        });

        it("Test 15: divScalar by negative scalar flips signs", async function () {
            t++;
            const a = [await qInt(2), await qInt(4), await qInt(6), await qInt(8)];
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

            printBlockMatrix({
                t,
                method: "divScalarHarness",
                explanation: "Division by -2.0 halves magnitude and flips sign.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, k=${k}`,
                outHex: fmtHexArr(c.data),
            });
        });

        it("Test 16: divScalar by tiny scalar grows magnitudes", async function () {
            t++;
            const rows = 2n;
            const cols = 2n;
            const a = [await qInt(1), await qInt(2), await qInt(3), await qInt(4)];
            const tiny = await harness.qFromFrac(1n, 1000n); // 0.001

            await touchGas(harness, "divScalarHarness", [rows, cols, a, tiny]);
            const gas = await estimateGas(harness, "divScalarHarness", [rows, cols, a, tiny]);

            const c = asMatrix(await harness.divScalarHarness(rows, cols, a, tiny));

            // Use Floating Point comparison for magnitude check
            for (let i = 0; i < a.length; ++i) {
                const origVal = await harness.toFloat(a[i]);
                const newVal = await harness.toFloat(c.data[i]);

                // Division by 0.001 should multiply value by 1000
                expect(newVal).to.be.greaterThan(origVal);
            }

            printBlockMatrix({
                t,
                method: "divScalarHarness",
                explanation: "Division by small scalar (0.001) magnifies values.",
                gas,
                shapeIn: `${rows}x${cols}`,
                shapeOut: `${c.rows}x${c.cols}`,
                inHex: `A=${fmtHexArr(a)}, tiny=${tiny}`,
                outHex: fmtHexArr(c.data),
            });
        });
    });
});