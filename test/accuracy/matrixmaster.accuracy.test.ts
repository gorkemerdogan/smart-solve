// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;

    transposeHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;
    mulMatrixHarness(
        aRows: bigint, aCols: bigint, aData: string[],
        bRows: bigint, bCols: bigint, bData: string[]
    ): Promise<[bigint, bigint, string[]]>;
    mulSparseMatrixVectorHarness(
        aRows: bigint,
        aCols: bigint,
        rowPtr: bigint[],
        colInd: bigint[],
        values: string[],
        xRows: bigint,
        xData: string[]
    ): Promise<[bigint, bigint, string[]]>;
    detHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<string>;
    inverseHarness(rows: bigint, cols: bigint, dataFlat: string[]): Promise<[bigint, bigint, string[]]>;
    powerIterationHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[],
        seed: string,
        tol: string
    ): Promise<[string, bigint, bigint, string[]]>;
    powerIterationWithIterHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[],
        seed: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, bigint, string[], bigint]>;
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function vecInfNorm(v: bigint[]): bigint {
    let m = 0n;
    for (const x of v) {
        const ax = absBigInt(x);
        if (ax > m) m = ax;
    }
    return m;
}

function subVec(a: bigint[], b: bigint[]): bigint[] {
    return a.map((x, i) => x - b[i]);
}

function flatten<T>(m: T[][]): T[] {
    return m.flat();
}

function matMulScaled(A: bigint[][], B: bigint[][]): bigint[][] {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;
    const out: bigint[][] = Array.from({ length: rows }, () => Array(cols).fill(0n));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let s = 0n;
            for (let k = 0; k < inner; k++) {
                s += (A[i][k] * B[k][j]) / SCALE;
            }
            out[i][j] = s;
        }
    }
    return out;
}

function transposeScaled(A: bigint[][]): bigint[][] {
    const rows = A.length;
    const cols = A[0].length;
    const T: bigint[][] = Array.from({ length: cols }, () => Array(rows).fill(0n));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

function fmtVec(v: bigint[]): string {
    return `[${v.map(formatScaledInt).join(", ")}]`;
}

function fmtMatFlat(v: bigint[], rows: number, cols: number): string {
    const parts: string[] = [];
    for (let i = 0; i < rows; i++) {
        const row = v.slice(i * cols, (i + 1) * cols);
        parts.push(`[${row.map(formatScaledInt).join(", ")}]`);
    }
    return `[${parts.join(", ")}]`;
}

async function qVecFromInts(h: MatrixMasterHarness, vals: Array<number | bigint>): Promise<string[]> {
    return Promise.all(vals.map(v => h.qFromInt(v)));
}

async function qVecFromFracs(
    h: MatrixMasterHarness,
    vals: Array<[number | bigint, number | bigint]>
): Promise<string[]> {
    return Promise.all(vals.map(([n, d]) => h.qFromFrac(n, d)));
}

async function scaledVec(h: MatrixMasterHarness, vals: string[]): Promise<bigint[]> {
    return Promise.all(vals.map(v => h.toFloat(v)));
}

function printMatrixBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expected: string;
    output: string;
    errorNorm: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected: ${args.expected}`);
    console.log(`Output: ${args.output}`);
    console.log(`Error Norm (inf): ${args.errorNorm}`);
    console.log("------------------------------------------------------------");
}

function printScalarBlock(args: {
    t: string;
    method: string;
    explanation: string;
    input: string;
    expected: string;
    output: string;
    absError: string;
}) {
    console.log("------------------------------------------------------------");
    console.log(`Test: ${args.t}`);
    console.log(`Method: ${args.method}`);
    console.log(`Explanation: ${args.explanation}`);
    console.log(`Input: ${args.input}`);
    console.log(`Expected: ${args.expected}`);
    console.log(`Output: ${args.output}`);
    console.log(`Absolute Error: ${args.absError}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("MatrixMaster Library - Numerical Accuracy Tests", function () {
    let harness: MatrixMasterHarness;
    let TOL_1E_9: string;
    let SEED: string;

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const MatrixFactory = await ethers.getContractFactory("MatrixMaster");
        const matrixLib = await MatrixFactory.deploy();
        await matrixLib.waitForDeployment();

        const HF = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: {
                MathLib: await mathlib.getAddress(),
            },
        });

        harness = (await HF.deploy()) as unknown as MatrixMasterHarness;
        TOL_1E_9 = await harness.qFromFrac(1, 1_000_000_000);
        SEED = ethers.id("matrixmaster-accuracy-seed");
    });

    // ------------------------------------------------------------
    // Section 1: Transpose
    // ------------------------------------------------------------

    describe("Section 1: Transpose correctness", function () {
        let testNo = 0;

        it(`Test 1.${++testNo}: transpose produces the exact transposed matrix`, async function () {
            // A = [[1,2,3],[4,5,6]]
            const A = [
                [1n * SCALE, 2n * SCALE, 3n * SCALE],
                [4n * SCALE, 5n * SCALE, 6n * SCALE],
            ];
            const expected = transposeScaled(A);

            const Adata = await qVecFromInts(harness, [1, 2, 3, 4, 5, 6]);
            const [, , out] = await harness.transposeHarness(2n, 3n, Adata);
            const outScaled = await scaledVec(harness, out);

            const expectedFlat = flatten(expected);
            const err = vecInfNorm(subVec(outScaled, expectedFlat));

            printMatrixBlock({
                t: `1.${testNo}`,
                method: "transpose",
                explanation: "Transpose should swap rows and columns exactly.",
                input: "A=[[1,2,3],[4,5,6]]",
                expected: fmtMatFlat(expectedFlat, 3, 2),
                output: fmtMatFlat(outScaled, 3, 2),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 2: Dense matrix multiplication
    // ------------------------------------------------------------

    describe("Section 2: Dense matrix multiplication", function () {
        let testNo = 0;

        it(`Test 2.${++testNo}: dense matrix multiplication matches exact result`, async function () {
            // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
            // AB = [[19,22],[43,50]]
            const A = [
                [1n * SCALE, 2n * SCALE],
                [3n * SCALE, 4n * SCALE],
            ];
            const B = [
                [5n * SCALE, 6n * SCALE],
                [7n * SCALE, 8n * SCALE],
            ];
            const expected = matMulScaled(A, B);

            const Adata = await qVecFromInts(harness, [1, 2, 3, 4]);
            const Bdata = await qVecFromInts(harness, [5, 6, 7, 8]);

            const [, , out] = await harness.mulMatrixHarness(2n, 2n, Adata, 2n, 2n, Bdata);
            const outScaled = await scaledVec(harness, out);

            const expectedFlat = flatten(expected);
            const err = vecInfNorm(subVec(outScaled, expectedFlat));

            printMatrixBlock({
                t: `2.${testNo}`,
                method: "dense matmul",
                explanation: "Dense matrix multiplication should reproduce the exact product on a small benchmark.",
                input: "A=[[1,2],[3,4]], B=[[5,6],[7,8]]",
                expected: fmtMatFlat(expectedFlat, 2, 2),
                output: fmtMatFlat(outScaled, 2, 2),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 3: Sparse matrix × vector multiplication
    // ------------------------------------------------------------

    describe("Section 3: Sparse matrix-vector multiplication", function () {
        let testNo = 0;

        it(`Test 3.${++testNo}: sparse matrix-vector multiplication matches exact result`, async function () {
            // A = [[10,0,0],[0,20,0],[0,0,30]], x=[1,2,3], y=[10,40,90]
            const expected = [10n * SCALE, 40n * SCALE, 90n * SCALE];

            const rowPtr = [0n, 1n, 2n, 3n];
            const colInd = [0n, 1n, 2n];
            const values = await qVecFromInts(harness, [10, 20, 30]);
            const xData = await qVecFromInts(harness, [1, 2, 3]);

            const [, , out] = await harness.mulSparseMatrixVectorHarness(
                3n, 3n, rowPtr, colInd, values, 3n, xData
            );
            const outScaled = await scaledVec(harness, out);

            const err = vecInfNorm(subVec(outScaled, expected));

            printMatrixBlock({
                t: `3.${testNo}`,
                method: "sparse matvec",
                explanation: "Sparse matrix-vector multiplication should reproduce the exact diagonal action.",
                input: "A=diag(10,20,30), x=[1,2,3]",
                expected: fmtVec(expected),
                output: fmtVec(outScaled),
                errorNorm: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 4: Determinant
    // ------------------------------------------------------------

    describe("Section 4: Determinant accuracy", function () {
        let testNo = 0;

        it(`Test 4.${++testNo}: determinant is exact on a 2x2 benchmark`, async function () {
            // det [[1,2],[3,4]] = -2
            const Adata = await qVecFromInts(harness, [1, 2, 3, 4]);
            const out = await harness.detHarness(2n, 2n, Adata);
            const outScaled = await harness.toFloat(out);

            const expected = -2n * SCALE;
            const err = absBigInt(outScaled - expected);

            printScalarBlock({
                t: `4.${testNo}`,
                method: "det",
                explanation: "Determinant should match the exact closed-form value on a 2x2 matrix.",
                input: "A=[[1,2],[3,4]]",
                expected: formatScaledInt(expected),
                output: formatScaledInt(outScaled),
                absError: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });

        it(`Test 4.${++testNo}: determinant is exact on a triangular matrix`, async function () {
            // det = product of diagonal = 2*3*4 = 24
            const Adata = await qVecFromInts(harness, [2, 1, 1, 0, 3, 2, 0, 0, 4]);
            const out = await harness.detHarness(3n, 3n, Adata);
            const outScaled = await harness.toFloat(out);

            const expected = 24n * SCALE;
            const err = absBigInt(outScaled - expected);

            printScalarBlock({
                t: `4.${testNo}`,
                method: "det",
                explanation: "Determinant of an upper-triangular matrix should equal the product of diagonal entries.",
                input: "A=[[2,1,1],[0,3,2],[0,0,4]]",
                expected: formatScaledInt(expected),
                output: formatScaledInt(outScaled),
                absError: formatScaledInt(err),
            });

            expect(err).to.equal(0n);
        });
    });

    // ------------------------------------------------------------
    // Section 5: Inverse
    // ------------------------------------------------------------

    describe("Section 5: Inverse accuracy", function () {
        let testNo = 0;

        it(`Test 5.${++testNo}: inverse matches exact 2x2 inverse`, async function () {
            // A = [[4,7],[2,6]], A^{-1} = 1/10 * [[6,-7],[-2,4]]
            const expected = [
                [600000000000n, -700000000000n],
                [-200000000000n, 400000000000n],
            ];

            const Adata = await qVecFromInts(harness, [4, 7, 2, 6]);
            const [, , out] = await harness.inverseHarness(2n, 2n, Adata);
            const outScaled = await scaledVec(harness, out);

            const expectedFlat = flatten(expected);
            const err = vecInfNorm(subVec(outScaled, expectedFlat));

            printMatrixBlock({
                t: `5.${testNo}`,
                method: "inverse",
                explanation: "Inverse should match the exact 2x2 analytical inverse.",
                input: "A=[[4,7],[2,6]]",
                expected: fmtMatFlat(expectedFlat, 2, 2),
                output: fmtMatFlat(outScaled, 2, 2),
                errorNorm: formatScaledInt(err),
            });

            expect(err < 10n).to.equal(true);
        });

        it(`Test 5.${++testNo}: A * A^{-1} is approximately identity`, async function () {
            const A = [
                [4n * SCALE, 7n * SCALE],
                [2n * SCALE, 6n * SCALE],
            ];
            const I = [
                [1n * SCALE, 0n],
                [0n, 1n * SCALE],
            ];

            const Adata = await qVecFromInts(harness, [4, 7, 2, 6]);
            const [, , invRaw] = await harness.inverseHarness(2n, 2n, Adata);
            const invScaledFlat = await scaledVec(harness, invRaw);

            const inv = [
                [invScaledFlat[0], invScaledFlat[1]],
                [invScaledFlat[2], invScaledFlat[3]],
            ];

            const prod = matMulScaled(A, inv);
            const prodFlat = flatten(prod);
            const expectedFlat = flatten(I);
            const err = vecInfNorm(subVec(prodFlat, expectedFlat));

            printMatrixBlock({
                t: `5.${testNo}`,
                method: "inverse identity check",
                explanation: "Multiplying a matrix by its inverse should recover the identity matrix up to roundoff.",
                input: "A=[[4,7],[2,6]]",
                expected: fmtMatFlat(expectedFlat, 2, 2),
                output: fmtMatFlat(prodFlat, 2, 2),
                errorNorm: formatScaledInt(err),
            });

            expect(err < 100n).to.equal(true);
        });
    });

    // ------------------------------------------------------------
    // Section 6: Power iteration
    // ------------------------------------------------------------

    describe("Section 6: Power iteration accuracy", function () {
        let testNo = 0;

        it(`Test 6.${++testNo}: power iteration recovers dominant eigenvalue of diagonal matrix`, async function () {
            // A = diag(5,2,1), dominant eigenvalue = 5
            const Adata = await qVecFromInts(harness, [
                5, 0, 0,
                0, 2, 0,
                0, 0, 1
            ]);

            const [lambda, xRows, xCols, xData] = await harness.powerIterationHarness(
                3n, 3n, Adata, SEED, TOL_1E_9
            );

            const lambdaScaled = await harness.toFloat(lambda);
            const expected = 5n * SCALE;
            const err = absBigInt(lambdaScaled - expected);

            const xScaled = await scaledVec(harness, xData);

            printScalarBlock({
                t: `6.${testNo}.1`,
                method: "power iteration eigenvalue",
                explanation: "Power iteration should recover the dominant eigenvalue on a diagonal matrix.",
                input: "A=diag(5,2,1)",
                expected: formatScaledInt(expected),
                output: formatScaledInt(lambdaScaled),
                absError: formatScaledInt(err),
            });

            console.log(`Eigenvector shape: ${xRows.toString()}x${xCols.toString()}`);
            console.log(`Eigenvector approx: ${fmtVec(xScaled)}`);

            expect(err < 100_000n).to.equal(true); // 1e-7
        });

        it(`Test 6.${++testNo}: power iteration converges faster with a looser tolerance`, async function () {
            // Same diagonal benchmark
            const Adata = await qVecFromInts(harness, [
                5, 0, 0,
                0, 2, 0,
                0, 0, 1
            ]);

            const tolLoose = await harness.qFromFrac(1, 1_000_000);      // 1e-6
            const tolTight = await harness.qFromFrac(1, 1_000_000_000);  // 1e-9

            const [, , , , itLoose] = await harness.powerIterationWithIterHarness(
                3n, 3n, Adata, SEED, tolLoose, 500n
            );

            const [, , , , itTight] = await harness.powerIterationWithIterHarness(
                3n, 3n, Adata, SEED, tolTight, 500n
            );

            console.log("------------------------------------------------------------");
            console.log(`Test: 6.${testNo}.2`);
            console.log("Method: power iteration iteration-count sensitivity");
            console.log("Explanation: A looser convergence tolerance should require no more iterations than a tighter tolerance.");
            console.log(`Iterations (tol=1e-6): ${itLoose.toString()}`);
            console.log(`Iterations (tol=1e-9): ${itTight.toString()}`);
            console.log("------------------------------------------------------------");

            expect(itLoose <= itTight).to.equal(true);
        });
    });
});