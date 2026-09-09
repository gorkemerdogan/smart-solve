// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import {
    countExecutionFailure,
    emptyExecutionFailureCounts,
    estimateGas,
    printBlockMatrix,
    printExecutionFeasibilitySummary,
    touchGas,
    HARNESS_ESTIMATE_CALL,
} from "./test-utils";
import { BenchmarkResultWriter, benchmarkExecutionRecord } from "./benchmark-results";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromInt(n: bigint): Promise<string>;
    qFromUInt(n: bigint): Promise<string>;
    qFromFrac(num: bigint, den: bigint): Promise<string>;
    toFloat(x: string): Promise<bigint>;

    mulMatrixHarness(
        aRows: bigint,
        aCols: bigint,
        aData: string[],
        bRows: bigint,
        bCols: bigint,
        bData: string[]
    ): Promise<[bigint, bigint, string[]]>;

    mulMatrixVectorHarness(
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
};

// ------------------------------------------------------------
// Constants & Helpers
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;
const Q_SCALE = 1_000_000n;

const REL_SCALE_DECIMALS = 12n;
const REL_SCALE = 10n ** REL_SCALE_DECIMALS;

const DENSE_SEED = ethers.keccak256(ethers.toUtf8Bytes("matrixmaster-part2-dense-fixed-seed"));
const SPARSE_SEED = ethers.keccak256(ethers.toUtf8Bytes("matrixmaster-part2-sparse-fixed-seed"));

// The n=96 full-band plus n=128 half/full-band CSR inputs characterize the
// 30M-gas feasibility boundary. They remain reproducible behind this explicit
// opt-in flag.
const RUN_HEAVY_MATRIX_SCALABILITY = process.env.RUN_HEAVY_MATRIX_SCALABILITY === "1";

function formatScaledInt(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / SCALE;
    const fracPart = abs % SCALE;
    const fracStr = fracPart.toString().padStart(Number(SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function formatRelativeScaled(v: bigint): string {
    const neg = v < 0n;
    const abs = neg ? -v : v;
    const intPart = abs / REL_SCALE;
    const fracPart = abs % REL_SCALE;
    const fracStr = fracPart.toString().padStart(Number(REL_SCALE_DECIMALS), "0");
    return `${neg ? "-" : ""}${intPart.toString()}.${fracStr}`.replace(/\.?0+$/, "");
}

function toGasBigInt(gas: unknown): bigint {
    if (typeof gas === "bigint") return gas;
    return BigInt(gas as string);
}

function headTail(arr: (string | bigint)[], limit = 3): string {
    if (arr.length <= limit * 2) return `[${arr.join(", ")}]`;
    return `[${arr.slice(0, limit).join(", ")}, ..., ${arr.slice(-limit).join(", ")}]`;
}

async function qFromNumber(h: MatrixMasterHarness, x: number): Promise<string> {
    if (Number.isInteger(x)) {
        return h.qFromInt(BigInt(x));
    }
    const scaled = Math.round(x * Number(Q_SCALE));
    return h.qFromFrac(BigInt(scaled), Q_SCALE);
}

async function qArrayFromNumbers(h: MatrixMasterHarness, arr: number[]): Promise<string[]> {
    return Promise.all(arr.map(v => qFromNumber(h, v)));
}

async function fromQuadArray(h: MatrixMasterHarness, arr: string[]): Promise<bigint[]> {
    return Promise.all(arr.map(v => h.toFloat(v)));
}

function keccakToBigInt(seed: string, ...parts: (string | number)[]): bigint {
    const packed = ethers.solidityPacked(
        ["bytes32", ...parts.map(() => "string")],
        [seed, ...parts.map(v => String(v))]
    );
    return BigInt(ethers.keccak256(packed));
}

function randIntFromKeccak(
    seed: string,
    min: number,
    max: number,
    ...parts: (string | number)[]
): number {
    const span = BigInt(max - min + 1);
    const r = keccakToBigInt(seed, ...parts) % span;
    return Number(r) + min;
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

// ------------------------------------------------------------
// Deterministic Random Builders
// ------------------------------------------------------------

function makeDenseRandomKeccak(n: number, seed: string, caseId: number, tag: "A" | "B"): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;

            let v = randIntFromKeccak(seed, -3, 3, "dense", caseId, tag, i, j);
            if (v === 0) v = 1;

            A[i][j] = v;
            rowSum += Math.abs(v);
        }

        const diagBoost = randIntFromKeccak(seed, 8, 14, "dense-diag", caseId, tag, i);
        A[i][i] = rowSum + diagBoost;
    }

    return A;
}

function flatten(mat: number[][]): number[] {
    return mat.flat();
}

function makeDenseVectorKeccak(n: number, seed: string, caseId: number): number[] {
    return Array.from({ length: n }, (_, i) => {
        let v = randIntFromKeccak(seed, -5, 5, "vector", caseId, i);
        if (v === 0) v = 1;
        return v;
    });
}

function buildSparseBandCSRKeccak(
    n: number,
    bandwidth: number,
    seed: string,
    caseId: number
): {
    rowPtr: bigint[];
    colInd: bigint[];
    values: number[];
    vector: number[];
} {
    const rowPtr: bigint[] = [0n];
    const colInd: bigint[] = [];
    const values: number[] = [];
    const vector = makeDenseVectorKeccak(n, seed, caseId);

    for (let i = 0; i < n; i++) {
        for (let j = Math.max(0, i - bandwidth); j <= Math.min(n - 1, i + bandwidth); j++) {
            colInd.push(BigInt(j));

            let v: number;
            if (i === j) {
                v = randIntFromKeccak(seed, 2, 9, "sparse-diag", caseId, i, j);
            } else {
                v = randIntFromKeccak(seed, -3, 3, "sparse-offdiag", caseId, i, j);
                if (v === 0) v = 1;
            }

            values.push(v);
        }
        rowPtr.push(BigInt(colInd.length));
    }

    return { rowPtr, colInd, values, vector };
}

// ------------------------------------------------------------
// Expected Value Helpers
// ------------------------------------------------------------

function denseMatMulExpected(A: number[][], B: number[][]): number[][] {
    const n = A.length;
    const C = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

function sparseMatVecExpected(
    n: number,
    rowPtr: bigint[],
    colInd: bigint[],
    values: number[],
    x: number[]
): number[] {
    const y = new Array<number>(n).fill(0);

    for (let i = 0; i < n; i++) {
        let sum = 0;
        const start = Number(rowPtr[i]);
        const end = Number(rowPtr[i + 1]);

        for (let p = start; p < end; p++) {
            sum += values[p] * x[Number(colInd[p])];
        }

        y[i] = sum;
    }

    return y;
}

function denseMatVecExpected(A: number[][], x: number[]): number[] {
    const n = A.length;
    const y = new Array<number>(n).fill(0);

    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        y[i] = sum;
    }

    return y;
}

function toScaled12FromIntegerArray(arr: number[]): bigint[] {
    return arr.map(v => BigInt(v) * SCALE);
}

function toScaled12FromIntegerMatrix(mat: number[][]): bigint[] {
    return mat.flat().map(v => BigInt(v) * SCALE);
}

function computeErrorStats(actual: bigint[], expected: bigint[]) {
    let maxAbsError = 0n;
    let sumAbsError = 0n;

    let maxRelErrorScaled = 0n;
    let sumRelErrorScaled = 0n;
    let relCount = 0n;

    for (let i = 0; i < actual.length; i++) {
        const absErr = absBigInt(actual[i] - expected[i]);

        if (absErr > maxAbsError) maxAbsError = absErr;
        sumAbsError += absErr;

        if (expected[i] !== 0n) {
            const relErrScaled = (absErr * REL_SCALE) / absBigInt(expected[i]);
            if (relErrScaled > maxRelErrorScaled) maxRelErrorScaled = relErrScaled;
            sumRelErrorScaled += relErrScaled;
            relCount += 1n;
        }
    }

    const avgAbsError =
        actual.length > 0 ? sumAbsError / BigInt(actual.length) : 0n;

    const avgRelErrorScaled =
        relCount > 0n ? sumRelErrorScaled / relCount : 0n;

    const residual = maxAbsError;

    return {
        maxAbsError,
        avgAbsError,
        maxRelErrorScaled,
        avgRelErrorScaled,
        residual,
    };
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("MatrixMasterHarness - Gas Growth Tests (Multiplication)", function () {
    let harness: MatrixMasterHarness;
    let t = 0;
    let resultWriter: BenchmarkResultWriter;

    function recordMatrixCase(args: {
        benchmark: string;
        operation: string;
        input: Record<string, string | number>;
        gas?: bigint;
        failureKind?: "revert" | "out-of-gas" | "failure";
        metrics?: Record<string, string>;
    }): void {
        resultWriter.record({
            benchmark: args.benchmark,
            category: "matrix operations",
            operation: args.operation,
            execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
            input: args.input,
            ...(args.gas === undefined ? {} : { gas: args.gas.toString() }),
            status: args.failureKind === undefined ? "success" : "failure",
            ...(args.failureKind === undefined ? {} : { failureKind: args.failureKind }),
            ...(args.metrics === undefined ? {} : { errorMetrics: args.metrics }),
        });
    }

    before(async function () {
        resultWriter = await BenchmarkResultWriter.create({ suite: "matrixmaster.multiplication.harness" });
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": await mathlib.getAddress() },
        });

        harness = (await Factory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();
    });

    after(async function () {
        await resultWriter.flush();
    });

    // ------------------------------------------------------------
    // Section 1: Dense Matrix-Matrix Multiplication
    // ------------------------------------------------------------

    describe("Section 1: Dense Matrix-Matrix Multiplication", function () {
        const DIM_CASES = [2, 4, 8, 12, 16, 18, 19];
        let localIdx = 0;

        for (const n of DIM_CASES) {
            for (let caseNo = 1; caseNo <= 5; caseNo++) {
                const sub = `1.${++localIdx}`;

                it(`${sub} Dense matmul gas growth for n=${n} case=${caseNo}`, async function () {
                    t++;

                    const A = makeDenseRandomKeccak(n, DENSE_SEED, 1000 + n * 10 + caseNo, "A");
                    const B = makeDenseRandomKeccak(n, DENSE_SEED, 2000 + n * 10 + caseNo, "B");

                    const Aq = await qArrayFromNumbers(harness, flatten(A));
                    const Bq = await qArrayFromNumbers(harness, flatten(B));

                    await touchGas(harness, "mulMatrixHarness", [
                        BigInt(n), BigInt(n), Aq,
                        BigInt(n), BigInt(n), Bq
                    ]);

                    const gas = await estimateGas(harness, "mulMatrixHarness", [
                        BigInt(n), BigInt(n), Aq,
                        BigInt(n), BigInt(n), Bq
                    ]);

                    if (gas === "revert") {
                        printBlockMatrix({
                            t,
                            method: "matrix-matrix multiplication",
                            explanation: `Gas growth with respect to matrix dimension for deterministic keccak-generated dense ${n}x${n} × ${n}x${n} multiplication.`,
                            gas: "revert",
                            shapeIn: `A=${n}x${n}, B=${n}x${n}`,
                            shapeOut: "reverted",
                            inHex: `pattern=dense_random_keccak, seed=${DENSE_SEED}, case=${caseNo}`,
                            outHex: "Reverted",
                            expectedDec: "Reverted",
                            outDec: "Reverted",
                            maxAbsError: "-",
                            avgAbsError: "-",
                            maxRelError: "-",
                            avgRelError: "-",
                            residual: "-",
                        });
                        recordMatrixCase({
                            benchmark: "dense matrix-matrix gas growth",
                            operation: "matrix-matrix multiplication",
                            input: { n, case: caseNo, pattern: "dense_random_keccak" },
                            failureKind: "revert",
                        });
                        return;
                    }

                    const [outRows, outCols, out] = await harness.mulMatrixHarness(
                        BigInt(n), BigInt(n), Aq,
                        BigInt(n), BigInt(n), Bq
                    );

                    const actualScaled12 = await fromQuadArray(harness, out);
                    const expectedScaled12 = toScaled12FromIntegerMatrix(denseMatMulExpected(A, B));

                    const {
                        maxAbsError,
                        avgAbsError,
                        maxRelErrorScaled,
                        avgRelErrorScaled,
                        residual,
                    } = computeErrorStats(actualScaled12, expectedScaled12);

                    printBlockMatrix({
                        t,
                        method: "matrix-matrix multiplication",
                        explanation: `Gas growth and numerical accuracy with respect to matrix dimension for deterministic keccak-generated dense ${n}x${n} × ${n}x${n} multiplication.`,
                        gas,
                        shapeIn: `A=${n}x${n}, B=${n}x${n}`,
                        shapeOut: `${outRows}x${outCols}`,
                        inHex: `pattern=dense_random_keccak, seed=${DENSE_SEED}, case=${caseNo}`,
                        outHex: headTail(out),
                        expectedDec: headTail(expectedScaled12.map(formatScaledInt)),
                        outDec: headTail(actualScaled12.map(formatScaledInt)),
                        maxAbsError: formatScaledInt(maxAbsError),
                        avgAbsError: formatScaledInt(avgAbsError),
                        maxRelError: formatRelativeScaled(maxRelErrorScaled),
                        avgRelError: formatRelativeScaled(avgRelErrorScaled),
                        residual: formatScaledInt(residual),
                    });
                    recordMatrixCase({
                        benchmark: "dense matrix-matrix gas growth",
                        operation: "matrix-matrix multiplication",
                        input: { n, case: caseNo, pattern: "dense_random_keccak" },
                        gas: toGasBigInt(gas),
                        metrics: {
                            maxAbsError: maxAbsError.toString(), avgAbsError: avgAbsError.toString(),
                            maxRelErrorScaled: maxRelErrorScaled.toString(), avgRelErrorScaled: avgRelErrorScaled.toString(), residual: residual.toString(),
                        },
                    });

                    expect(actualScaled12.length).to.equal(expectedScaled12.length);
                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });

    // ------------------------------------------------------------
    // Section 2: Sparse Matrix-Vector Multiplication
    // ------------------------------------------------------------

    describe("Section 2: Sparse Matrix-Vector Multiplication", function () {
        const SIZE_CASES = [8, 16, 32, 64, 96, 128];
        const BANDWIDTH_CASES = [0, 1, 2, 4, 8, 16, 32, 64];
        let localIdx = 0;

        for (const n of SIZE_CASES) {
            for (const bandwidth of BANDWIDTH_CASES) {
                for (let caseNo = 1; caseNo <= 5; caseNo++) {
                    const sub = `2.${++localIdx}`;
                    const isFeasibilityCase =
                        (n >= 96 && bandwidth === 64) ||
                        (n === 128 && bandwidth === 32);
                    const titlePrefix = isFeasibilityCase ? "[opt-in feasibility] " : "";

                    it(`${titlePrefix}${sub} Sparse matvec gas vs nnz at n=${n}, bw=${bandwidth}, case=${caseNo}`, async function () {
                        if (isFeasibilityCase && !RUN_HEAVY_MATRIX_SCALABILITY) {
                            this.skip();
                        }

                        t++;

                        const failures = emptyExecutionFailureCounts();
                        let successful = 0;

                        try {
                        const sparse = buildSparseBandCSRKeccak(
                            n,
                            bandwidth,
                            SPARSE_SEED,
                            5000 + n * 100 + bandwidth * 10 + caseNo
                        );

                        const valuesQ = await qArrayFromNumbers(harness, sparse.values);
                        const xQ = await qArrayFromNumbers(harness, sparse.vector);

                        await touchGas(harness, "mulSparseMatrixVectorHarness", [
                            BigInt(n), BigInt(n), sparse.rowPtr, sparse.colInd, valuesQ, BigInt(n), xQ,
                        ]);

                        const gas = await estimateGas(harness, "mulSparseMatrixVectorHarness", [
                            BigInt(n), BigInt(n), sparse.rowPtr, sparse.colInd, valuesQ, BigInt(n), xQ,
                        ]);

                        const [outRows, outCols, out] = await harness.mulSparseMatrixVectorHarness(
                            BigInt(n), BigInt(n), sparse.rowPtr, sparse.colInd, valuesQ, BigInt(n), xQ
                        );

                        const actualScaled12 = await fromQuadArray(harness, out);
                        const expectedScaled12 = toScaled12FromIntegerArray(
                            sparseMatVecExpected(
                                n,
                                sparse.rowPtr,
                                sparse.colInd,
                                sparse.values,
                                sparse.vector
                            )
                        );

                        const {
                            maxAbsError,
                            avgAbsError,
                            maxRelErrorScaled,
                            avgRelErrorScaled,
                            residual,
                        } = computeErrorStats(actualScaled12, expectedScaled12);

                        printBlockMatrix({
                            t,
                            method: "sparse matrix-vector multiplication",
                            explanation: `Gas growth and numerical accuracy with respect to number of nonzero entries (nnz) using deterministic keccak-generated banded sparse structure with bandwidth=${bandwidth} at matrix size ${n}x${n}.`,
                            gas,
                            shapeIn: `A=${n}x${n}, x=${n}x1`,
                            shapeOut: `${outRows}x${outCols}`,
                            inHex: `pattern=band-${bandwidth}, nnz=${sparse.colInd.length}, seed=${SPARSE_SEED}, case=${caseNo}`,
                            outHex: headTail(out),
                            expectedDec: headTail(expectedScaled12.map(formatScaledInt)),
                            outDec: headTail(actualScaled12.map(formatScaledInt)),
                            maxAbsError: formatScaledInt(maxAbsError),
                            avgAbsError: formatScaledInt(avgAbsError),
                            maxRelError: formatRelativeScaled(maxRelErrorScaled),
                            avgRelError: formatRelativeScaled(avgRelErrorScaled),
                            residual: formatScaledInt(residual),
                        });
                        recordMatrixCase({
                            benchmark: "sparse matrix-vector gas growth",
                            operation: "sparse matrix-vector multiplication",
                            input: { n, bandwidth, nnz: sparse.colInd.length, case: caseNo },
                            gas: toGasBigInt(gas),
                            metrics: {
                                maxAbsError: maxAbsError.toString(), avgAbsError: avgAbsError.toString(),
                                maxRelErrorScaled: maxRelErrorScaled.toString(), avgRelErrorScaled: avgRelErrorScaled.toString(), residual: residual.toString(),
                            },
                        });

                        expect(actualScaled12.length).to.equal(expectedScaled12.length);
                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                        successful++;
                        } catch (error) {
                            if (!isFeasibilityCase) throw error;
                            const kind = countExecutionFailure(failures, error);
                            recordMatrixCase({
                                benchmark: "sparse matrix-vector feasibility boundary",
                                operation: "sparse matrix-vector multiplication",
                                input: { n, bandwidth, case: caseNo },
                                failureKind: kind,
                                metrics: { error: error instanceof Error ? error.message : String(error) },
                            });
                            console.log(
                                `Feasibility case | sparse matvec n=${n}, bw=${bandwidth}, case=${caseNo} | status=${kind}`,
                            );
                        }

                        if (isFeasibilityCase) {
                            printExecutionFeasibilitySummary({
                                label: `sparse matvec n=${n}, bw=${bandwidth} under the configured 30M block gas limit`,
                                total: 1,
                                successful,
                                failures,
                            });
                            expect(successful + failures.failed).to.equal(1);
                            expect(failures.revert, "unexpected feasibility reverts").to.equal(0);
                            expect(failures.failure, "unexpected feasibility failures").to.equal(0);
                            expect(failures["out-of-gas"]).to.be.within(0, 1);
                        }
                    });
                }
            }
        }
    });

    // ------------------------------------------------------------
    // Section 3: Dense Matrix-Vector Multiplication
    // ------------------------------------------------------------

    describe("Section 3: Dense Matrix-Vector Multiplication", function () {
        const DIM_CASES = [2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 82];
        let localIdx = 0;

        for (const n of DIM_CASES) {
            for (let caseNo = 1; caseNo <= 5; caseNo++) {
                const sub = `3.${++localIdx}`;

                it(`${sub} Dense matvec gas growth for n=${n} case=${caseNo}`, async function () {
                    t++;

                    const A = makeDenseRandomKeccak(
                        n,
                        DENSE_SEED,
                        30000 + n * 10 + caseNo,
                        "A"
                    );
                    const x = makeDenseVectorKeccak(
                        n,
                        DENSE_SEED,
                        40000 + n * 10 + caseNo
                    );

                    const Aq = await qArrayFromNumbers(harness, flatten(A));
                    const xq = await qArrayFromNumbers(harness, x);

                    await touchGas(harness, "mulMatrixVectorHarness", [
                        BigInt(n), BigInt(n), Aq,
                        BigInt(n), 1n, xq
                    ]);

                    const gas = await estimateGas(harness, "mulMatrixVectorHarness", [
                        BigInt(n), BigInt(n), Aq,
                        BigInt(n), 1n, xq
                    ]);

                    if (gas === "revert") {
                        printBlockMatrix({
                            t,
                            method: "dense matrix-vector multiplication",
                            explanation: `Gas growth with respect to matrix dimension for deterministic keccak-generated dense ${n}x${n} × ${n}x1 multiplication.`,
                            gas: "revert",
                            shapeIn: `A=${n}x${n}, x=${n}x1`,
                            shapeOut: "reverted",
                            inHex: `pattern=dense_random_keccak, seed=${DENSE_SEED}, case=${caseNo}`,
                            outHex: "Reverted",
                            expectedDec: "Reverted",
                            outDec: "Reverted",
                            maxAbsError: "-",
                            avgAbsError: "-",
                            maxRelError: "-",
                            avgRelError: "-",
                            residual: "-",
                        });
                        recordMatrixCase({
                            benchmark: "dense matrix-vector gas growth",
                            operation: "dense matrix-vector multiplication",
                            input: { n, case: caseNo, pattern: "dense_random_keccak" },
                            failureKind: "revert",
                        });
                        return;
                    }

                    const [outRows, outCols, out] = await harness.mulMatrixVectorHarness(
                        BigInt(n), BigInt(n), Aq,
                        BigInt(n), 1n, xq
                    );

                    const actualScaled12 = await fromQuadArray(harness, out);
                    const expectedScaled12 = toScaled12FromIntegerArray(
                        denseMatVecExpected(A, x)
                    );

                    const {
                        maxAbsError,
                        avgAbsError,
                        maxRelErrorScaled,
                        avgRelErrorScaled,
                        residual,
                    } = computeErrorStats(actualScaled12, expectedScaled12);

                    printBlockMatrix({
                        t,
                        method: "dense matrix-vector multiplication",
                        explanation: `Gas growth and numerical accuracy with respect to matrix dimension for deterministic keccak-generated dense ${n}x${n} × ${n}x1 multiplication.`,
                        gas,
                        shapeIn: `A=${n}x${n}, x=${n}x1`,
                        shapeOut: `${outRows}x${outCols}`,
                        inHex: `pattern=dense_random_keccak, seed=${DENSE_SEED}, case=${caseNo}`,
                        outHex: headTail(out),
                        expectedDec: headTail(expectedScaled12.map(formatScaledInt)),
                        outDec: headTail(actualScaled12.map(formatScaledInt)),
                        maxAbsError: formatScaledInt(maxAbsError),
                        avgAbsError: formatScaledInt(avgAbsError),
                        maxRelError: formatRelativeScaled(maxRelErrorScaled),
                        avgRelError: formatRelativeScaled(avgRelErrorScaled),
                        residual: formatScaledInt(residual),
                    });
                    recordMatrixCase({
                        benchmark: "dense matrix-vector gas growth",
                        operation: "dense matrix-vector multiplication",
                        input: { n, case: caseNo, pattern: "dense_random_keccak" },
                        gas: toGasBigInt(gas),
                        metrics: {
                            maxAbsError: maxAbsError.toString(), avgAbsError: avgAbsError.toString(),
                            maxRelErrorScaled: maxRelErrorScaled.toString(), avgRelErrorScaled: avgRelErrorScaled.toString(), residual: residual.toString(),
                        },
                    });

                    expect(actualScaled12.length).to.equal(expectedScaled12.length);
                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                });
            }
        }
    });
});
