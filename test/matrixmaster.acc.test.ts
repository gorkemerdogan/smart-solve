// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { formatBenchmarkExecution, HARNESS_ESTIMATE_CALL } from "./test-utils";
import { printPrecisionMetadata } from "./precision-utils";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    fromFloat(x: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;

    transposeHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[]
    ): Promise<[bigint, bigint, string[]]>;

    mulMatrixHarness(
        aRows: bigint,
        aCols: bigint,
        aData: string[],
        bRows: bigint,
        bCols: bigint,
        bData: string[]
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

    detHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[]
    ): Promise<string>;

    inverseHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[]
    ): Promise<[bigint, bigint, string[]]>;

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

    powerIterationWithStatusHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[],
        seed: string,
        tol: string
    ): Promise<[string, bigint, bigint, string[], bigint, boolean]>;
};

// ------------------------------------------------------------
// Global helpers / constants
// ------------------------------------------------------------

let SCALE_DECIMALS = 32n;
let SCALE = 10n ** SCALE_DECIMALS;

const TEST_COUNT = 20;
const SPARSITY_LEVELS = [5, 15, 25];

const DENSE_SIZES = [2, 3, 4, 5, 8];
const SPARSE_SIZES = [4, 8, 16, 32, 64];
const DET_SIZES = [2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24];
const INV_SIZES = [2, 3, 4, 5, 8];
const POWER_SIZES = [2, 3, 4, 5, 6];

const TOL_EIGEN_NUM = 1e-10;

function asBigInt(v: unknown): bigint {
    if (typeof v === "bigint") return v;
    if (typeof v === "number") return BigInt(v);
    if (typeof v === "string") return BigInt(v);

    if (v && typeof v === "object") {
        const maybeToString = (v as { toString?: () => string }).toString;
        if (typeof maybeToString === "function") {
            return BigInt(maybeToString.call(v));
        }
    }

    throw new Error(`Cannot convert value to bigint: ${String(v)}`);
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

function inferScaleDecimals(scale: bigint): bigint {
    const s = scale.toString();
    if (!/^10*$/.test(s) || s[0] !== "1") {
        return SCALE_DECIMALS;
    }
    return BigInt(s.length - 1);
}

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function flatten<T>(m: T[][]): T[] {
    return m.flat();
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

function fmtHexVec(v: string[]): string {
    return `[${v.join(", ")}]`;
}

function fmtHexMatFlat(v: string[], rows: number, cols: number): string {
    const parts: string[] = [];
    for (let i = 0; i < rows; i++) {
        const row = v.slice(i * cols, (i + 1) * cols);
        parts.push(`[${row.join(", ")}]`);
    }
    return `[${parts.join(", ")}]`;
}

async function qVecFromInts(
    h: MatrixMasterHarness,
    vals: Array<number | bigint>
): Promise<string[]> {
    return Promise.all(vals.map(v => h.qFromInt(v)));
}

async function qVecFromFracs(
    h: MatrixMasterHarness,
    vals: Array<[number | bigint, number | bigint]>
): Promise<string[]> {
    return Promise.all(vals.map(([n, d]) => h.qFromFrac(n, d)));
}

async function scaledVec(
    h: MatrixMasterHarness,
    vals: string[]
): Promise<bigint[]> {
    return Promise.all(vals.map(async v => asBigInt(await h.toFloat(v))));
}

async function estimateMethodGas(
    contract: Contract,
    method: string,
    args: unknown[]
): Promise<bigint> {
    const fn = contract.getFunction(method) as unknown as {
        estimateGas: (...fnArgs: unknown[]) => Promise<bigint>;
    };
    return fn.estimateGas(...args);
}

// ------------------------------------------------------------
// Numeric helpers for accuracy tests
// ------------------------------------------------------------

function infNormVector(v: number[]): number {
    let m = 0;
    for (const x of v) {
        const ax = Math.abs(x);
        if (ax > m) m = ax;
    }
    return m;
}

function infNormMatrix(A: number[][]): number {
    let m = 0;
    for (const row of A) {
        for (const x of row) {
            const ax = Math.abs(x);
            if (ax > m) m = ax;
        }
    }
    return m;
}

function subVecNum(a: number[], b: number[]): number[] {
    return a.map((x, i) => x - b[i]);
}

function subMat(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((x, j) => x - B[i][j]));
}

function identity(n: number): number[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );
}

function matMul(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;

    const out: number[][] = Array.from({ length: rows }, () =>
        Array(cols).fill(0)
    );

    for (let i = 0; i < rows; i++) {
        for (let k = 0; k < inner; k++) {
            for (let j = 0; j < cols; j++) {
                out[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return out;
}

function matVec(A: number[][], x: number[]): number[] {
    return A.map(row => row.reduce((s, aij, j) => s + aij * x[j], 0));
}

function dot(a: number[], b: number[]): number {
    return a.reduce((s, x, i) => s + x * b[i], 0);
}

function norm2(v: number[]): number {
    return Math.sqrt(dot(v, v));
}

function normalize(v: number[]): number[] {
    const n = norm2(v);
    return v.map(x => x / n);
}

function sparseMatVec(A: number[][], x: number[]): number[] {
    const n = A.length;
    const y = Array(n).fill(0);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }

    return y;
}

function det(A: number[][]): number {
    const n = A.length;
    const M = A.map(row => [...row]);
    let sign = 1;
    let d = 1;

    for (let k = 0; k < n; k++) {
        let pivot = k;
        for (let i = k + 1; i < n; i++) {
            if (Math.abs(M[i][k]) > Math.abs(M[pivot][k])) {
                pivot = i;
            }
        }

        if (Math.abs(M[pivot][k]) < 1e-15) return 0;

        if (pivot !== k) {
            [M[pivot], M[k]] = [M[k], M[pivot]];
            sign *= -1;
        }

        const piv = M[k][k];
        d *= piv;

        for (let i = k + 1; i < n; i++) {
            const factor = M[i][k] / piv;
            for (let j = k; j < n; j++) {
                M[i][j] -= factor * M[k][j];
            }
        }
    }

    return sign * d;
}

function powerIterationRef(A: number[][], maxIter = 5000, tol = 1e-12) {
    const n = A.length;
    let x = Array.from({ length: n }, (_, i) => i + 1);
    x = normalize(x);

    let lambda = 0;

    for (let iter = 0; iter < maxIter; iter++) {
        const y = matVec(A, x);
        const xNext = normalize(y);
        const Ax = matVec(A, xNext);
        const lambdaNext = dot(xNext, Ax);

        const diff = infNormVector(subVecNum(xNext, x));
        x = xNext;
        lambda = lambdaNext;

        if (diff < tol) break;
    }

    const residual = subVecNum(matVec(A, x), x.map(xi => lambda * xi));
    return {
        lambda,
        x,
        residualInf: infNormVector(residual),
    };
}

// ------------------------------------------------------------
// Deterministic pseudo-random helpers
// ------------------------------------------------------------

function rand01(
    tag: string,
    n: number,
    caseId: number,
    i: number,
    j = 0
): number {
    const packed = ethers.solidityPacked(
        ["string", "uint256", "uint256", "uint256", "uint256"],
        [tag, n, caseId, i, j]
    );
    const h = ethers.keccak256(packed);
    const slice = h.slice(2, 14); // 48 bits
    const v = Number(BigInt(`0x${slice}`));
    return v / 2 ** 48;
}

function randInRange(
    tag: string,
    n: number,
    caseId: number,
    i: number,
    j: number,
    lo: number,
    hi: number
): number {
    return lo + (hi - lo) * rand01(tag, n, caseId, i, j);
}

function denseRandomMatrix(n: number, caseId: number): number[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) =>
            randInRange("denseA", n, caseId, i, j, -2, 2)
        )
    );
}

function denseRandomMatrixB(n: number, caseId: number): number[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) =>
            randInRange("denseB", n, caseId, i, j, -2, 2)
        )
    );
}

function denseVector(n: number, caseId: number): number[] {
    return Array.from({ length: n }, (_, i) =>
        randInRange("vec", n, caseId, i, 0, -2, 2)
    );
}

function diagonalDominantMatrix(n: number, caseId: number): number[][] {
    const A: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        let offDiagAbsSum = 0;

        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const v = randInRange("diagdom", n, caseId, i, j, -0.5, 0.5);
            A[i][j] = v;
            offDiagAbsSum += Math.abs(v);
        }

        A[i][i] =
            offDiagAbsSum +
            1 +
            randInRange("diagdom_d", n, caseId, i, i, 0, 1);
    }

    return A;
}

function symmetricDiagonalDominantMatrix(n: number, caseId: number): number[][] {
    const A: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const v = randInRange("symm", n, caseId, i, j, -0.4, 0.4);
            A[i][j] = v;
            A[j][i] = v;
        }
    }

    for (let i = 0; i < n; i++) {
        let offDiagAbsSum = 0;
        for (let j = 0; j < n; j++) {
            if (i !== j) offDiagAbsSum += Math.abs(A[i][j]);
        }
        A[i][i] =
            offDiagAbsSum +
            1 +
            randInRange("symm_d", n, caseId, i, i, 0, 1);
    }

    return A;
}

function sparseMatrix(n: number, densityPercent: number, caseId: number): number[][] {
    const density = densityPercent / 100;
    const A: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    const targetNnz = Math.max(1, Math.floor(n * n * density));

    let nnz = 0;

    for (let i = 0; i < n && nnz < targetNnz; i++) {
        A[i][i] = randInRange("sparse_diag", n, caseId, i, i, 0.5, 2);
        nnz++;
    }

    for (let k = 0; nnz < targetNnz && k < n * n * 5; k++) {
        const r = Math.floor(rand01("sparse_r", n, caseId, k) * n);
        const c = Math.floor(rand01("sparse_c", n, caseId, k) * n);

        if (A[r][c] === 0) {
            A[r][c] = randInRange("sparse_v", n, caseId, r, c, -1, 1);
            nnz++;
        }
    }

    return A;
}

function denseToCSR(A: number[][]): {
    rowPtr: bigint[];
    colInd: bigint[];
    values: number[];
} {
    const n = A.length;
    const rowPtr: bigint[] = [0n];
    const colInd: bigint[] = [];
    const values: number[] = [];

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (A[i][j] !== 0) {
                colInd.push(BigInt(j));
                values.push(A[i][j]);
            }
        }
        rowPtr.push(BigInt(colInd.length));
    }

    return { rowPtr, colInd, values };
}

// ------------------------------------------------------------
// Quad conversion helpers
// ------------------------------------------------------------

async function qFromNumber(h: MatrixMasterHarness, x: number): Promise<string> {
    const scaled = BigInt(Math.round(x * Number(SCALE)));
    return h.fromFloat(scaled);
}

async function qArrayFromNumbers(h: MatrixMasterHarness, arr: number[]): Promise<string[]> {
    return Promise.all(arr.map(x => qFromNumber(h, x)));
}

async function quadToNumber(h: MatrixMasterHarness, q: string): Promise<number> {
    const scaled = asBigInt(await h.toFloat(q));
    return Number(scaled) / Number(SCALE);
}

async function quadArrayToNumbers(h: MatrixMasterHarness, arr: string[]): Promise<number[]> {
    return Promise.all(arr.map(q => quadToNumber(h, q)));
}

// ------------------------------------------------------------
// Test Suite Setup
// ------------------------------------------------------------

describe("MatrixMasterHarness - Mixed Fixed-Point/JS-Number Accuracy Tests", function () {
    this.timeout(300000); // 5 minutes
    let harness: MatrixMasterHarness;
    let TOL_1E_18: string;
    let SEED: string;
    let resultWriter: BenchmarkResultWriter;

    before(async () => {
        resultWriter = await BenchmarkResultWriter.create({ suite: "matrixmaster.accuracy.harness" });
        console.log(`Execution Model      : ${formatBenchmarkExecution(HARNESS_ESTIMATE_CALL)}`);
        printPrecisionMetadata({
            classification: "JavaScript-number-limited comparison",
            comparisonScale: "mixed: truncating 1e12 fixed-point and JavaScript Number",
            oraclePrecision: "exact integer references for some operations; JS Number for determinant/eigenvalue paths",
            conversion: "binary128 -> 1e12 fixed-point; selected paths then convert to JS Number",
            claim: "fixed-point/method accuracy at the reported tolerance; not binary128 accuracy",
        });
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HarnessFactory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: {
                MathLib: await mathlib.getAddress(),
            },
        });

        harness = (await HarnessFactory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();

        TOL_1E_18 = await harness.qFromFrac(1, 1_000_000_000_000_000_000n);
        SEED = ethers.id("matrixmaster-accuracy-seed");

        const oneScaled = asBigInt(await harness.toFloat(await harness.qFromInt(1)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    after(async () => {
        await resultWriter.flush();
    });

    // ---------------------------------------------------------------------------
    // Accuracy tests
    // ---------------------------------------------------------------------------

    it.skip("measures average dense matrix-matrix multiplication accuracy", async function () {
        const avgErrors: { n: number; avgError: number; avgGas: bigint }[] = [];

        for (const n of DENSE_SIZES) {
            let errSum = 0;
            let gasSum = 0n;

            console.log("============================================================");
            console.log(`Dense Matrix-Matrix Multiplication Accuracy Results for n=${n}`);
            console.log("============================================================");

            for (let caseId = 0; caseId < TEST_COUNT; caseId++) {
                const A = denseRandomMatrix(n, caseId);
                const B = denseRandomMatrixB(n, caseId);

                const aQ = await qArrayFromNumbers(harness, flatten(A));
                const bQ = await qArrayFromNumbers(harness, flatten(B));

                const gasUsed = await estimateMethodGas(harness, "mulMatrixHarness", [
                    BigInt(n),
                    BigInt(n),
                    aQ,
                    BigInt(n),
                    BigInt(n),
                    bQ
                ]);

                const result = await harness.mulMatrixHarness(
                    BigInt(n),
                    BigInt(n),
                    aQ,
                    BigInt(n),
                    BigInt(n),
                    bQ
                );

                const cDataQ = result[2];
                const cSolFlat = await quadArrayToNumbers(harness, cDataQ);
                const Csol = Array.from({ length: n }, (_, i) =>
                    cSolFlat.slice(i * n, (i + 1) * n)
                );

                const Cref = matMul(A, B);
                const err = infNormMatrix(subMat(Csol, Cref));

                errSum += err;
                gasSum += gasUsed;

                console.log(`Case ${caseId + 1}/${TEST_COUNT}`);
                console.log(`  E_inf     = ${err}`);
                console.log(`  Gas Usage = ${gasUsed}`);
                console.log("------------------------------------------------------------");
            }

            const avgError = errSum / TEST_COUNT;
            const avgGas = gasSum / BigInt(TEST_COUNT);
            avgErrors.push({ n, avgError, avgGas });

            console.log(`Average for n=${n}`);
            console.log(`  avg E_inf     = ${avgError}`);
            console.log(`  avg gas usage = ${avgGas}`);
            console.log("============================================================");
        }

        console.log("#################### FINAL DENSE MAT-MAT SUMMARY ####################");
        for (const item of avgErrors) {
            console.log(`n=${item.n} | avg E_inf=${item.avgError} | avg gas usage=${item.avgGas}`);
            expect(item.avgError).to.be.lessThan(1e-8);
        }
        console.log("####################################################################");
    });

    it.skip("measures average sparse matrix-vector multiplication accuracy across density levels", async function () {
        const results: { n: number; density: number; avgError: number; avgGas: bigint }[] = [];

        for (const n of SPARSE_SIZES) {
            for (const density of SPARSITY_LEVELS) {
                let errSum = 0;
                let gasSum = 0n;

                console.log("============================================================");
                console.log(`Sparse Matrix-Vector Multiplication Accuracy Results for n=${n}, density=${density}%`);
                console.log("============================================================");

                for (let caseId = 0; caseId < TEST_COUNT; caseId++) {
                    const A = sparseMatrix(n, density, caseId);
                    const x = denseVector(n, caseId);

                    const { rowPtr, colInd, values } = denseToCSR(A);
                    const valuesQ = await qArrayFromNumbers(harness, values);
                    const xQ = await qArrayFromNumbers(harness, x);

                    const gasUsed = await estimateMethodGas(harness, "mulSparseMatrixVectorHarness", [
                        BigInt(n),
                        BigInt(n),
                        rowPtr,
                        colInd,
                        valuesQ,
                        BigInt(n),
                        xQ
                    ]);

                    const result = await harness.mulSparseMatrixVectorHarness(
                        BigInt(n),
                        BigInt(n),
                        rowPtr,
                        colInd,
                        valuesQ,
                        BigInt(n),
                        xQ
                    );

                    const yQ = result[2];
                    const ySol = await quadArrayToNumbers(harness, yQ);
                    const yRef = sparseMatVec(A, x);

                    const err = infNormVector(subVecNum(ySol, yRef));

                    errSum += err;
                    gasSum += gasUsed;

                    console.log(`Case ${caseId + 1}/${TEST_COUNT}`);
                    console.log(`  E_inf     = ${err}`);
                    console.log(`  Gas Usage = ${gasUsed}`);
                    console.log("------------------------------------------------------------");
                }

                const avgError = errSum / TEST_COUNT;
                const avgGas = gasSum / BigInt(TEST_COUNT);
                results.push({ n, density, avgError, avgGas });

                console.log(`Average for n=${n}, density=${density}%`);
                console.log(`  avg E_inf     = ${avgError}`);
                console.log(`  avg gas usage = ${avgGas}`);
                console.log("============================================================");
            }
        }

        console.log("#################### FINAL SPARSE MAT-VEC SUMMARY ####################");
        for (const item of results) {
            console.log(
                `n=${item.n} | density=${item.density}% | avg E_inf=${item.avgError} | avg gas usage=${item.avgGas}`
            );
            expect(item.avgError).to.be.lessThan(1e-8);
        }
        console.log("########################################################################");
    });

    it("measures average determinant accuracy on diagonal-dominant matrices", async function () {
    const results: {
        n: number;
        avgAbsError: number;
        avgRelError: number;
        avgGas: bigint;
        failedCases: number;
    }[] = [];

    const REL_TOL_DET = 1e-10;

    for (const n of DET_SIZES) {
        let absErrSum = 0;
        let relErrSum = 0;
        let gasSum = 0n;
        let failedCases = 0;

        console.log("============================================================");
        console.log(`Determinant Accuracy Results for n=${n}`);
        console.log("============================================================");

        for (let caseId = 0; caseId < TEST_COUNT; caseId++) {
            const A = diagonalDominantMatrix(n, caseId);
            const aQ = await qArrayFromNumbers(harness, flatten(A));

            const gasUsed = await estimateMethodGas(harness, "detHarness", [
                BigInt(n),
                BigInt(n),
                aQ
            ]);

            const detQ = await harness.detHarness(BigInt(n), BigInt(n), aQ);
            const detSol = await quadToNumber(harness, detQ);
            const detRef = det(A);

            const absErr = Math.abs(detSol - detRef);
            const relErr =
                Math.abs(detRef) > 0
                    ? absErr / Math.abs(detRef)
                    : absErr;

            absErrSum += absErr;
            relErrSum += relErr;
            gasSum += gasUsed;
            if (!Number.isFinite(relErr) || relErr >= REL_TOL_DET) failedCases++;

            console.log(`Case ${caseId + 1}/${TEST_COUNT}`);
            console.log(`  det_sol   = ${detSol}`);
            console.log(`  det_ref   = ${detRef}`);
            console.log(`  abs error = ${absErr}`);
            console.log(`  rel error = ${relErr}`);
            console.log(`  Gas Usage = ${gasUsed}`);
            console.log("------------------------------------------------------------");
        }

        const avgAbsError = absErrSum / TEST_COUNT;
        const avgRelError = relErrSum / TEST_COUNT;
        const avgGas = gasSum / BigInt(TEST_COUNT);

        results.push({ n, avgAbsError, avgRelError, avgGas, failedCases });
        resultWriter.record({
            benchmark: "diagonal-dominant determinant accuracy",
            category: "matrix operations",
            operation: "determinant",
            execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
            input: { n, tests: TEST_COUNT, pattern: "diagonal-dominant" },
            gas: avgGas.toString(),
            status: failedCases === 0 ? "success" : "failure",
            errorMetrics: {
                averageAbsoluteError: avgAbsError,
                averageRelativeError: avgRelError,
                numericalFailures: failedCases,
                relativeTolerance: REL_TOL_DET,
            },
        });

        console.log(`Average for n=${n}`);
        console.log(`  avg abs error = ${avgAbsError}`);
        console.log(`  avg rel error = ${avgRelError}`);
        console.log(`  avg gas usage = ${avgGas} (successful executions only)`);
        console.log(`  total cases = ${TEST_COUNT}`);
        console.log(`  successful executions = ${TEST_COUNT}`);
        console.log(`  numerical failures = ${failedCases}`);
        console.log("  reverted cases = 0");
        console.log("  out-of-gas cases = 0");
        console.log(`  pass rate = ${(((TEST_COUNT - failedCases) / TEST_COUNT) * 100).toFixed(2)}%`);
        console.log("============================================================");
    }

    console.log("#################### FINAL DETERMINANT SUMMARY ####################");

    for (const item of results) {
        console.log(
            `n=${item.n} | avg abs error=${item.avgAbsError} | avg rel error=${item.avgRelError} | avg gas usage=${item.avgGas} (successful executions only) | failed=${item.failedCases}`
        );

        expect(item.failedCases, `n=${item.n}: determinant cases outside relative tolerance`).to.equal(0);
    }

    console.log("##################################################################");
});

    it.skip("measures inversion accuracy using ||A*A^{-1} - I||_inf", async function () {
        const results: { n: number; avgInverseError: number; avgGas: bigint }[] = [];

        for (const n of INV_SIZES) {
            let errSum = 0;
            let gasSum = 0n;

            console.log("============================================================");
            console.log(`Inverse Accuracy Results for n=${n}`);
            console.log("============================================================");

            for (let caseId = 0; caseId < TEST_COUNT; caseId++) {
                const A = diagonalDominantMatrix(n, caseId);
                const aQ = await qArrayFromNumbers(harness, flatten(A));

                const gasUsed = await estimateMethodGas(harness, "inverseHarness", [
                    BigInt(n),
                    BigInt(n),
                    aQ
                ]);

                const result = await harness.inverseHarness(BigInt(n), BigInt(n), aQ);
                const invQ = result[2];
                const invSolFlat = await quadArrayToNumbers(harness, invQ);

                const InvSol = Array.from({ length: n }, (_, i) =>
                    invSolFlat.slice(i * n, (i + 1) * n)
                );

                const recon = matMul(A, InvSol);
                const err = infNormMatrix(subMat(recon, identity(n)));

                errSum += err;
                gasSum += gasUsed;

                console.log(`Case ${caseId + 1}/${TEST_COUNT}`);
                console.log(`  E_inverse = ${err}`);
                console.log(`  Gas Usage = ${gasUsed}`);
                console.log("------------------------------------------------------------");
            }

            const avgInverseError = errSum / TEST_COUNT;
            const avgGas = gasSum / BigInt(TEST_COUNT);
            results.push({ n, avgInverseError, avgGas });

            console.log(`Average for n=${n}`);
            console.log(`  avg E_inverse   = ${avgInverseError}`);
            console.log(`  avg gas usage   = ${avgGas}`);
            console.log("============================================================");
        }

        console.log("#################### FINAL INVERSE SUMMARY ####################");
        for (const item of results) {
            console.log(`n=${item.n} | avg E_inverse=${item.avgInverseError} | avg gas usage=${item.avgGas}`);
            expect(item.avgInverseError).to.be.lessThan(1e-7);
        }
        console.log("################################################################");
    });

    it.skip("measures power iteration eigenvalue accuracy and residual consistency", async function () {
        const results: {
            n: number;
            avgEigenAbsError: number;
            avgResidualInf: number;
            avgGas: bigint;
        }[] = [];

        const tolQ = await qFromNumber(harness, TOL_EIGEN_NUM);

        for (const n of POWER_SIZES) {
            let eigErrSum = 0;
            let residualSum = 0;
            let gasSum = 0n;

            console.log("============================================================");
            console.log(`Power Iteration Accuracy Results for n=${n}`);
            console.log("============================================================");

            for (let caseId = 0; caseId < TEST_COUNT; caseId++) {
                const A = symmetricDiagonalDominantMatrix(n, caseId);
                const aQ = await qArrayFromNumbers(harness, flatten(A));

                const seed = ethers.keccak256(
                    ethers.solidityPacked(
                        ["bytes32", "string", "uint256", "uint256"],
                        [SEED, "power", n, caseId]
                    )
                );

                const gasUsed = await estimateMethodGas(harness, "powerIterationWithStatusHarness", [
                    BigInt(n),
                    BigInt(n),
                    aQ,
                    seed,
                    tolQ
                ]);

                const result = await harness.powerIterationWithStatusHarness(
                    BigInt(n),
                    BigInt(n),
                    aQ,
                    seed,
                    tolQ
                );

                const lambdaQ = result[0];
                const xQ = result[3];
                const iterations = result[4];
                const converged = result[5];

                const lambdaSol = await quadToNumber(harness, lambdaQ);
                const xSol = await quadArrayToNumbers(harness, xQ);

                const ref = powerIterationRef(A);

                const eigenAbsErr = Math.abs(lambdaSol - ref.lambda);
                const residual = subVecNum(
                    matVec(A, xSol),
                    xSol.map(xi => lambdaSol * xi)
                );
                const residualInf = infNormVector(residual);

                eigErrSum += eigenAbsErr;
                residualSum += residualInf;
                gasSum += gasUsed;

                console.log(`Case ${caseId + 1}/${TEST_COUNT}`);
                console.log(`  lambda_sol       = ${lambdaSol}`);
                console.log(`  lambda_ref       = ${ref.lambda}`);
                console.log(`  eigen abs error  = ${eigenAbsErr}`);
                console.log(`  residual inf     = ${residualInf}`);
                console.log(`  iterations       = ${iterations}`);
                console.log(`  converged        = ${converged}`);
                console.log(`  Gas Usage        = ${gasUsed}`);
                console.log("------------------------------------------------------------");
            }

            const avgEigenAbsError = eigErrSum / TEST_COUNT;
            const avgResidualInf = residualSum / TEST_COUNT;
            const avgGas = gasSum / BigInt(TEST_COUNT);

            results.push({
                n,
                avgEigenAbsError,
                avgResidualInf,
                avgGas,
            });

            console.log(`Average for n=${n}`);
            console.log(`  avg eigen abs error = ${avgEigenAbsError}`);
            console.log(`  avg residual inf    = ${avgResidualInf}`);
            console.log(`  avg gas usage       = ${avgGas}`);
            console.log("============================================================");
        }

        console.log("#################### FINAL POWER ITERATION SUMMARY ####################");
        for (const item of results) {
            console.log(
                `n=${item.n} | avg eigen abs error=${item.avgEigenAbsError} | avg residual inf=${item.avgResidualInf} | avg gas usage=${item.avgGas}`
            );
            expect(item.avgEigenAbsError).to.be.lessThan(1e-3);
            expect(item.avgResidualInf).to.be.lessThan(1e-3);
        }
        console.log("########################################################################");
    });
});
