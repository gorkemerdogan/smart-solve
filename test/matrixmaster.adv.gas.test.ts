// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";
import { touchGas, estimateGas, printBlockMatrix } from "./test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromInt(n: bigint): Promise<string>;
    qFromUInt(n: bigint): Promise<string>;
    qFromFrac(num: bigint, den: bigint): Promise<string>;
    toFloat(x: string): Promise<bigint>;

    transposeHarness(
        rows: bigint,
        cols: bigint,
        dataFlat: string[]
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

    mulMatrixHarness(
        aRows: bigint,
        aCols: bigint,
        aData: string[],
        bRows: bigint,
        bCols: bigint,
        bData: string[]
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
};

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;
const Q_SCALE = 1_000_000n;

const REL_SCALE_DECIMALS = 12n;
const REL_SCALE = 10n ** REL_SCALE_DECIMALS;

let SIZE_CASES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20];

const CASES_PER_SIZE_ADVANCED = 10; // 8 sizes * 2 = 16 per advanced op
const CASES_PER_SIZE_POWER = 10;    // 8 sizes * 2 = 16 for power vs size
const CASES_PER_PATTERN_POWER = 10; // 5 patterns * 2 = 10 for power by pattern

const ADV_SEED = ethers.keccak256(
    ethers.toUtf8Bytes("matrix-advanced-keccak-seed-v2")
);
const POWER_MATRIX_SEED = ethers.keccak256(
    ethers.toUtf8Bytes("matrix-power-matrix-keccak-seed-v2")
);
const POWER_INIT_SEED = ethers.keccak256(
    ethers.toUtf8Bytes("matrix-power-init-keccak-seed-v2")
);

const POWER_MAX_ITER = 250n;

// ------------------------------------------------------------
// Formatting Helpers
// ------------------------------------------------------------

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

function idx(n: number, i: number, j: number): number {
    return i * n + j;
}

function flatten(mat: number[][]): number[] {
    return mat.flat();
}

function headTail(arr: (string | bigint)[], limit = 3): string {
    if (arr.length <= limit * 2) return `[${arr.join(", ")}]`;
    return `[${arr.slice(0, limit).join(", ")}, ..., ${arr.slice(-limit).join(", ")}]`;
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

function absNumber(x: number): number {
    return x < 0 ? -x : x;
}

function infNorm(v: bigint[]): bigint {
    let m = 0n;
    for (const x of v) {
        const ax = absBigInt(x);
        if (ax > m) m = ax;
    }
    return m;
}

// ------------------------------------------------------------
// Quad Helpers
// ------------------------------------------------------------

async function qFromNumber(h: MatrixMasterHarness, x: number): Promise<string> {
    if (Number.isInteger(x)) {
        return h.qFromInt(BigInt(x));
    }
    const scaled = Math.round(x * Number(Q_SCALE));
    return h.qFromFrac(BigInt(scaled), Q_SCALE);
}

async function qArrayFromNumbers(
    h: MatrixMasterHarness,
    arr: number[]
): Promise<string[]> {
    return Promise.all(arr.map(v => qFromNumber(h, v)));
}

async function fromQuad(
    h: MatrixMasterHarness,
    q: string
): Promise<bigint> {
    return h.toFloat(q);
}

async function fromQuadArray(
    h: MatrixMasterHarness,
    arr: string[]
): Promise<bigint[]> {
    return Promise.all(arr.map(v => h.toFloat(v)));
}

// ------------------------------------------------------------
// Deterministic keccak RNG
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Matrix Builders
// ------------------------------------------------------------

function makeUpperTriangularKeccak(
    n: number,
    seed: string,
    caseId: number
): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            if (i === j) {
                A[i][j] = randIntFromKeccak(seed, 2, 9, "utri-diag", caseId, i, j);
            } else {
                A[i][j] = randIntFromKeccak(seed, -3, 3, "utri-offdiag", caseId, i, j);
            }
        }
    }

    return A;
}

function transposeExpected(A: number[][]): number[][] {
    const rows = A.length;
    const cols = A[0].length;
    const T = Array.from({ length: cols }, () => Array(rows).fill(0));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            T[j][i] = A[i][j];
        }
    }

    return T;
}

function determinantUpperTriangular(A: number[][]): bigint {
    let det = 1n;
    for (let i = 0; i < A.length; i++) {
        det *= BigInt(A[i][i]);
    }
    return det;
}

function identityScaledFlat(n: number): bigint[] {
    const out: bigint[] = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            out.push(i === j ? SCALE : 0n);
        }
    }
    return out;
}

function multiplyIntegerMatrixWithScaledMatrix(
    A: number[][],
    Bscaled: bigint[],
    n: number
): bigint[] {
    const out = new Array<bigint>(n * n).fill(0n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0n;
            for (let k = 0; k < n; k++) {
                sum += BigInt(A[i][k]) * Bscaled[idx(n, k, j)];
            }
            out[idx(n, i, j)] = sum;
        }
    }

    return out;
}

function makeSymmetricMatrixKeccak(
    n: number,
    seed: string,
    caseId: number,
    pattern: "identity" | "diagDominant" | "clustered" | "weighted" | "mixed"
): number[][] {
    const A = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let v = 0;

            if (pattern === "identity") {
                v = 0;
            } else if (pattern === "clustered") {
                v = randIntFromKeccak(seed, 1, 2, "clustered", caseId, i, j);
            } else if (pattern === "weighted") {
                v = randIntFromKeccak(seed, 1, 4, "weighted", caseId, i, j);
            } else if (pattern === "mixed") {
                v = randIntFromKeccak(seed, 1, 3, "mixed", caseId, i, j);
            } else {
                v = randIntFromKeccak(seed, 1, 2, "diagDominant", caseId, i, j);
            }

            A[i][j] = v;
            A[j][i] = v;
        }
    }

    for (let i = 0; i < n; i++) {
        let rowAbsSum = 0;
        for (let j = 0; j < n; j++) {
            if (i !== j) rowAbsSum += absNumber(A[i][j]);
        }

        if (pattern === "identity") {
            A[i][i] = 1;
        } else if (pattern === "clustered") {
            A[i][i] = rowAbsSum + randIntFromKeccak(seed, 4, 7, "clustered-diag", caseId, i);
        } else if (pattern === "weighted") {
            A[i][i] = rowAbsSum + randIntFromKeccak(seed, 8, 12, "weighted-diag", caseId, i);
        } else if (pattern === "mixed") {
            A[i][i] = rowAbsSum + randIntFromKeccak(seed, 5, 9, "mixed-diag", caseId, i);
        } else {
            A[i][i] = rowAbsSum + randIntFromKeccak(seed, 3, 5, "diagDominant-diag", caseId, i);
        }
    }

    return A;
}

// ------------------------------------------------------------
// Expected / Residual Helpers
// ------------------------------------------------------------

function integerMatrixToScaledFlat(A: number[][]): bigint[] {
    return A.flat().map(v => BigInt(v) * SCALE);
}

function determinantExpectedScaled(A: number[][]): bigint {
    return determinantUpperTriangular(A) * SCALE;
}

function matVecScaled12(A: number[][], xScaled: bigint[]): bigint[] {
    const n = A.length;
    const out = new Array<bigint>(n).fill(0n);

    for (let i = 0; i < n; i++) {
        let sum = 0n;
        for (let j = 0; j < n; j++) {
            sum += BigInt(A[i][j]) * xScaled[j];
        }
        out[i] = sum;
    }

    return out;
}

function lambdaTimesVectorScaled12(
    lambdaScaled: bigint,
    xScaled: bigint[]
): bigint[] {
    return xScaled.map(v => (lambdaScaled * v) / SCALE);
}

type ErrorStats = {
    maxAbsError: bigint;
    avgAbsError: bigint;
    maxRelErrorScaled: bigint;
    avgRelErrorScaled: bigint;
    residual: bigint;
    normalizedResidualScaled: bigint;
    withinTolCount: bigint;
    exceededTolCount: bigint;
    worstIndex: number;
    absTol: bigint;
};

function computeErrorStats(
    actual: bigint[],
    expected: bigint[],
    absTol: bigint
): ErrorStats {
    let maxAbsError = 0n;
    let sumAbsError = 0n;

    let maxRelErrorScaled = 0n;
    let sumRelErrorScaled = 0n;
    let relCount = 0n;

    let withinTolCount = 0n;
    let exceededTolCount = 0n;
    let worstIndex = -1;

    const diff: bigint[] = [];

    for (let i = 0; i < actual.length; i++) {
        const absErr = absBigInt(actual[i] - expected[i]);
        diff.push(absErr);

        if (absErr > maxAbsError) {
            maxAbsError = absErr;
            worstIndex = i;
        }

        sumAbsError += absErr;

        if (absErr <= absTol) withinTolCount += 1n;
        else exceededTolCount += 1n;

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
    const denom = infNorm(expected);
    const normalizedResidualScaled =
        denom === 0n ? 0n : (infNorm(diff) * REL_SCALE) / denom;

    return {
        maxAbsError,
        avgAbsError,
        maxRelErrorScaled,
        avgRelErrorScaled,
        residual,
        normalizedResidualScaled,
        withinTolCount,
        exceededTolCount,
        worstIndex,
        absTol,
    };
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("MatrixMasterHarness - Gas and Accuracy Tests (Advanced Ops + Power Iteration)", function () {
    let harness: MatrixMasterHarness;
    let t = 0;
    let powerSeed: string;

    before(async function () {
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const Factory = await ethers.getContractFactory("MatrixMasterHarness", {
            libraries: {
                "contracts/libraries/MathLib.sol:MathLib": await mathlib.getAddress(),
            },
        });

        harness = (await Factory.deploy()) as unknown as MatrixMasterHarness;
        await harness.waitForDeployment();

        powerSeed = POWER_INIT_SEED;
    });

    // ------------------------------------------------------------
    // Section 4: Matrix size vs gas (transpose / determinant / inversion)
    // ------------------------------------------------------------

    describe("Section 4: Matrix size vs gas", function () {
        describe("Section 4.1: Transpose", function () {
            let localIdx = 0;
            const ABS_TOL_TRANSPOSE = 1n;

            SIZE_CASES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 29, 32, 64, 128, 154];

            for (const n of SIZE_CASES) {
                for (let caseNo = 1; caseNo <= CASES_PER_SIZE_ADVANCED; caseNo++) {
                    const sub = `4.1.${++localIdx}`;

                    it(`${sub} Transpose gas growth for n=${n}, case=${caseNo}`, async function () {
                        t++;

                        const A = makeUpperTriangularKeccak(
                            n,
                            ADV_SEED,
                            1000 + n * 10 + caseNo
                        );
                        const Aq = await qArrayFromNumbers(harness, flatten(A));

                        await touchGas(harness, "transposeHarness", [BigInt(n), BigInt(n), Aq]);
                        const gas = await estimateGas(harness, "transposeHarness", [BigInt(n), BigInt(n), Aq]);

                        const [outRows, outCols, outHex] =
                            await harness.transposeHarness(BigInt(n), BigInt(n), Aq);
                        const outDec = await fromQuadArray(harness, outHex);

                        const expectedScaled = integerMatrixToScaledFlat(transposeExpected(A));
                        const stats = computeErrorStats(outDec, expectedScaled, ABS_TOL_TRANSPOSE);

                        printBlockMatrix({
                            t,
                            method: "transpose",
                            explanation: `Gas growth and numerical accuracy of transpose operation for keccak-generated upper-triangular matrix of size n=${n}.`,
                            gas,
                            shapeIn: `${n}x${n}`,
                            shapeOut: `${outRows}x${outCols}`,
                            inHex: `pattern=upper-triangular-keccak, seed=${ADV_SEED}, case=${caseNo}`,
                            outHex: headTail(outHex),
                            expectedDec: headTail(expectedScaled.map(formatScaledInt)),
                            outDec: headTail(outDec.map(formatScaledInt)),
                            maxAbsError: formatScaledInt(stats.maxAbsError),
                            avgAbsError: formatScaledInt(stats.avgAbsError),
                            maxRelError: formatRelativeScaled(stats.maxRelErrorScaled),
                            avgRelError: formatRelativeScaled(stats.avgRelErrorScaled),
                            residual: formatScaledInt(stats.residual),
                            normalizedResidual: formatRelativeScaled(stats.normalizedResidualScaled),
                            tolerance: formatScaledInt(stats.absTol),
                            withinTol: `${stats.withinTolCount}/${outDec.length}`,
                            exceededTol: `${stats.exceededTolCount}/${outDec.length}`,
                            worstCase: `${stats.worstIndex}`,
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    });
                }
            }
        });

        describe("Section 4.2: Determinant", function () {

            let localIdx = 0;
            const ABS_TOL_DET = 10n;

            SIZE_CASES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 21, 24, 26, 27, 28, 29];

            for (const n of SIZE_CASES) {
                for (let caseNo = 1; caseNo <= CASES_PER_SIZE_ADVANCED; caseNo++) {
                    const sub = `4.2.${++localIdx}`;

                    it(`${sub} Determinant gas growth for n=${n}, case=${caseNo}`, async function () {
                        t++;

                        const A = makeUpperTriangularKeccak(
                            n,
                            ADV_SEED,
                            2000 + n * 10 + caseNo
                        );
                        const Aq = await qArrayFromNumbers(harness, flatten(A));

                        await touchGas(harness, "detHarness", [BigInt(n), BigInt(n), Aq]);
                        const gas = await estimateGas(harness, "detHarness", [BigInt(n), BigInt(n), Aq]);

                        const detHex = await harness.detHarness(BigInt(n), BigInt(n), Aq);
                        const detDec = await fromQuad(harness, detHex);

                        const expectedScaled = determinantExpectedScaled(A);
                        const stats = computeErrorStats([detDec], [expectedScaled], ABS_TOL_DET);

                        printBlockMatrix({
                            t,
                            method: "determinant",
                            explanation: `Gas growth and numerical accuracy of determinant computation for keccak-generated upper-triangular matrix of size n=${n}.`,
                            gas,
                            shapeIn: `${n}x${n}`,
                            shapeOut: "1x1",
                            inHex: `pattern=upper-triangular-keccak, seed=${ADV_SEED}, case=${caseNo}`,
                            outHex: detHex,
                            expectedDec: formatScaledInt(expectedScaled),
                            outDec: formatScaledInt(detDec),
                            maxAbsError: formatScaledInt(stats.maxAbsError),
                            avgAbsError: formatScaledInt(stats.avgAbsError),
                            maxRelError: formatRelativeScaled(stats.maxRelErrorScaled),
                            avgRelError: formatRelativeScaled(stats.avgRelErrorScaled),
                            residual: formatScaledInt(stats.residual),
                            normalizedResidual: formatRelativeScaled(stats.normalizedResidualScaled),
                            tolerance: formatScaledInt(stats.absTol),
                            withinTol: `${stats.withinTolCount}/1`,
                            exceededTol: `${stats.exceededTolCount}/1`,
                            worstCase: `${stats.worstIndex}`,
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    });
                }
            }
        });

        describe("Section 4.2: Determinant Accuracy", function () {
            this.timeout(300000); // 5 minutes

            const ABS_TOL_DET = 10n;

            const SIZE_CASES = [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                14, 16, 18, 20, 21, 24, 26, 27, 28, 29
            ];

            it("measures average determinant accuracy and gas growth on upper-triangular keccak matrices", async function () {
                const results: { n: number; avgAbsError: bigint; avgGas: bigint }[] = [];

                for (const n of SIZE_CASES) {
                    let errSum = 0n;
                    let gasSum = 0n;

                    console.log("============================================================");
                    console.log(`Determinant Accuracy Results for n=${n}`);
                    console.log("============================================================");

                    for (let caseNo = 1; caseNo <= CASES_PER_SIZE_ADVANCED; caseNo++) {
                        const A = makeUpperTriangularKeccak(
                            n,
                            ADV_SEED,
                            2000 + n * 10 + caseNo
                        );

                        const Aq = await qArrayFromNumbers(harness, flatten(A));

                        await touchGas(harness, "detHarness", [
                            BigInt(n),
                            BigInt(n),
                            Aq
                        ]);

                        const gas = await estimateGas(harness, "detHarness", [
                            BigInt(n),
                            BigInt(n),
                            Aq
                        ]);

                        const detHex = await harness.detHarness(
                            BigInt(n),
                            BigInt(n),
                            Aq
                        );

                        const detDec = await fromQuad(harness, detHex);
                        const expectedScaled = determinantExpectedScaled(A);

                        const absErr =
                            detDec >= expectedScaled
                                ? detDec - expectedScaled
                                : expectedScaled - detDec;

                        errSum += absErr;
                        gasSum += toGasBigInt(gas);

                        console.log(`Case ${caseNo}/${CASES_PER_SIZE_ADVANCED}`);
                        console.log(`  det_sol   = ${formatScaledInt(detDec)}`);
                        console.log(`  det_ref   = ${formatScaledInt(expectedScaled)}`);
                        console.log(`  abs error = ${formatScaledInt(absErr)}`);
                        console.log(`  Gas Usage = ${toGasBigInt(gas)}`);
                        console.log("------------------------------------------------------------");

                        const stats = computeErrorStats(
                            [detDec],
                            [expectedScaled],
                            ABS_TOL_DET
                        );

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                        expect(
                            stats.maxAbsError <= stats.absTol,
                            `n=${n}, case=${caseNo}: determinant absolute error exceeds tolerance`
                        ).to.equal(true);
                    }

                    const avgAbsError = errSum / BigInt(CASES_PER_SIZE_ADVANCED);
                    const avgGas = gasSum / BigInt(CASES_PER_SIZE_ADVANCED);

                    results.push({
                        n,
                        avgAbsError,
                        avgGas
                    });

                    console.log(`Average for n=${n}`);
                    console.log(`  avg abs error = ${formatScaledInt(avgAbsError)}`);
                    console.log(`  avg gas usage = ${avgGas} (successful executions only)`);
                    console.log(`  total cases = ${CASES_PER_SIZE_ADVANCED}`);
                    console.log(`  successful cases = ${CASES_PER_SIZE_ADVANCED}`);
                    console.log("  failed/reverted/out-of-gas cases = 0/0/0");
                    console.log("  pass rate = 100.00%");
                    console.log("============================================================");
                }

                console.log("#################### FINAL DETERMINANT SUMMARY ####################");

                for (const item of results) {
                    console.log(
                        `n=${item.n} | avg abs error=${formatScaledInt(item.avgAbsError)} | avg gas usage=${item.avgGas} (successful executions only) | failures=0`
                    );
                }

                console.log("##################################################################");
            });
        });

        describe("Section 4.3: Inversion", function () {
            let localIdx = 0;
            const ABS_TOL_INV = 100n;

            SIZE_CASES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 21];

            for (const n of SIZE_CASES) {
                for (let caseNo = 1; caseNo <= CASES_PER_SIZE_ADVANCED; caseNo++) {
                    const sub = `4.3.${++localIdx}`;

                    it(`${sub} Inversion gas growth for n=${n}, case=${caseNo}`, async function () {
                        t++;

                        const A = makeUpperTriangularKeccak(
                            n,
                            ADV_SEED,
                            3000 + n * 10 + caseNo
                        );
                        const Aq = await qArrayFromNumbers(harness, flatten(A));

                        await touchGas(harness, "inverseHarness", [BigInt(n), BigInt(n), Aq]);
                        const gas = await estimateGas(harness, "inverseHarness", [BigInt(n), BigInt(n), Aq]);

                        const [outRows, outCols, invHex] =
                            await harness.inverseHarness(BigInt(n), BigInt(n), Aq);
                        const invDec = await fromQuadArray(harness, invHex);

                        const aInvScaled = multiplyIntegerMatrixWithScaledMatrix(A, invDec, n);
                        const identityScaled = identityScaledFlat(n);

                        const stats = computeErrorStats(aInvScaled, identityScaled, ABS_TOL_INV);

                        printBlockMatrix({
                            t,
                            method: "inverse",
                            explanation: `Gas growth and numerical accuracy of matrix inversion for keccak-generated upper-triangular matrix of size n=${n}. Accuracy is evaluated using the residual of A·A^{-1} against I.`,
                            gas,
                            shapeIn: `${n}x${n}`,
                            shapeOut: `${outRows}x${outCols}`,
                            inHex: `pattern=upper-triangular-keccak, seed=${ADV_SEED}, case=${caseNo}`,
                            outHex: headTail(invHex),
                            expectedDec: headTail(identityScaled.map(formatScaledInt)),
                            outDec: `inv=${headTail(invDec.map(formatScaledInt))} | Ainv=${headTail(aInvScaled.map(formatScaledInt))}`,
                            maxAbsError: formatScaledInt(stats.maxAbsError),
                            avgAbsError: formatScaledInt(stats.avgAbsError),
                            maxRelError: formatRelativeScaled(stats.maxRelErrorScaled),
                            avgRelError: formatRelativeScaled(stats.avgRelErrorScaled),
                            residual: formatScaledInt(stats.residual),
                            normalizedResidual: formatRelativeScaled(stats.normalizedResidualScaled),
                            tolerance: formatScaledInt(stats.absTol),
                            withinTol: `${stats.withinTolCount}/${aInvScaled.length}`,
                            exceededTol: `${stats.exceededTolCount}/${aInvScaled.length}`,
                            worstCase: `${stats.worstIndex}`,
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    });
                }
            }
        });
    });

    // ------------------------------------------------------------
    // Section 5: Power Iteration
    // ------------------------------------------------------------

    describe("Section 5: Power Iteration", function () {
        describe("Section 5.1: Gas vs matrix dimension", function () {
            let localIdx = 0;
            const ABS_TOL_POWER = 1000n;

            SIZE_CASES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

            for (const n of SIZE_CASES) {
                for (let caseNo = 1; caseNo <= CASES_PER_SIZE_POWER; caseNo++) {
                    const sub = `5.1.${++localIdx}`;

                    it(`${sub} Power iteration gas growth for n=${n}, case=${caseNo}`, async function () {
                        t++;

                        const A = makeSymmetricMatrixKeccak(
                            n,
                            POWER_MATRIX_SEED,
                            4000 + n * 10 + caseNo,
                            "diagDominant"
                        );
                        const Aq = await qArrayFromNumbers(harness, flatten(A));
                        const tol = await harness.qFromFrac(1n, 1_000_000n); // 1e-6

                        await touchGas(harness, "powerIterationWithIterHarness", [
                            BigInt(n),
                            BigInt(n),
                            Aq,
                            powerSeed,
                            tol,
                            POWER_MAX_ITER
                        ]);

                        const gas = await estimateGas(harness, "powerIterationWithIterHarness", [
                            BigInt(n),
                            BigInt(n),
                            Aq,
                            powerSeed,
                            tol,
                            POWER_MAX_ITER
                        ]);

                        const [lambdaHex, xRows, xCols, xData, iterCount] =
                            await harness.powerIterationWithIterHarness(
                                BigInt(n),
                                BigInt(n),
                                Aq,
                                powerSeed,
                                tol,
                                POWER_MAX_ITER
                            );

                        const lambdaDec = await fromQuad(harness, lambdaHex);
                        const xDec = await fromQuadArray(harness, xData);

                        const axScaled = matVecScaled12(A, xDec);
                        const lambdaXScaled = lambdaTimesVectorScaled12(lambdaDec, xDec);

                        const stats = computeErrorStats(axScaled, lambdaXScaled, ABS_TOL_POWER);

                        printBlockMatrix({
                            t,
                            method: "power iteration",
                            explanation: `Gas growth and numerical accuracy with respect to matrix dimension for dominant eigenpair approximation using keccak-generated diagDominant symmetric matrix. Accuracy is evaluated with the residual of Ax-λx.`,
                            gas,
                            shapeIn: `${n}x${n}`,
                            shapeOut: `${xRows}x${xCols}`,
                            inHex: `pattern=diagDominant, matrixSeed=${POWER_MATRIX_SEED}, initSeed=fixed, tol=1e-6, case=${caseNo}, iter=${iterCount}`,
                            outHex: `lambda=${lambdaHex}, eigenvector=${headTail(xData)}`,
                            expectedDec: headTail(lambdaXScaled.map(formatScaledInt)),
                            outDec: `lambda=${formatScaledInt(lambdaDec)} | x=${headTail(xDec.map(formatScaledInt))} | Ax=${headTail(axScaled.map(formatScaledInt))}`,
                            maxAbsError: formatScaledInt(stats.maxAbsError),
                            avgAbsError: formatScaledInt(stats.avgAbsError),
                            maxRelError: formatRelativeScaled(stats.maxRelErrorScaled),
                            avgRelError: formatRelativeScaled(stats.avgRelErrorScaled),
                            residual: formatScaledInt(stats.residual),
                            normalizedResidual: formatRelativeScaled(stats.normalizedResidualScaled),
                            tolerance: formatScaledInt(stats.absTol),
                            withinTol: `${stats.withinTolCount}/${axScaled.length}`,
                            exceededTol: `${stats.exceededTolCount}/${axScaled.length}`,
                            worstCase: `${stats.worstIndex}`,
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    });
                }
            }
        });

        describe("Section 5.2: Gas by matrix pattern", function () {
            const PATTERNS: Array<"identity" | "diagDominant" | "clustered" | "weighted" | "mixed"> = [
                "identity",
                "diagDominant",
                "clustered",
                "weighted",
                "mixed",
            ];

            const n = 8;
            const ABS_TOL_POWER = 1000n;
            let localIdx = 0;

            for (const pattern of PATTERNS) {
                for (let caseNo = 1; caseNo <= CASES_PER_PATTERN_POWER; caseNo++) {
                    const sub = `5.2.${++localIdx}`;

                    it(`${sub} Power iteration gas sensitivity for ${pattern}, case=${caseNo}`, async function () {
                        t++;

                        const A = makeSymmetricMatrixKeccak(
                            n,
                            POWER_MATRIX_SEED,
                            5000 + PATTERNS.indexOf(pattern) * 100 + caseNo,
                            pattern
                        );
                        const Aq = await qArrayFromNumbers(harness, flatten(A));
                        const tol = await harness.qFromFrac(1n, 1_000_000n); // 1e-6

                        await touchGas(harness, "powerIterationWithIterHarness", [
                            BigInt(n),
                            BigInt(n),
                            Aq,
                            powerSeed,
                            tol,
                            POWER_MAX_ITER
                        ]);

                        const gas = await estimateGas(harness, "powerIterationWithIterHarness", [
                            BigInt(n),
                            BigInt(n),
                            Aq,
                            powerSeed,
                            tol,
                            POWER_MAX_ITER
                        ]);

                        const [lambdaHex, xRows, xCols, xData, iterCount] =
                            await harness.powerIterationWithIterHarness(
                                BigInt(n),
                                BigInt(n),
                                Aq,
                                powerSeed,
                                tol,
                                POWER_MAX_ITER
                            );

                        const lambdaDec = await fromQuad(harness, lambdaHex);
                        const xDec = await fromQuadArray(harness, xData);

                        const axScaled = matVecScaled12(A, xDec);
                        const lambdaXScaled = lambdaTimesVectorScaled12(lambdaDec, xDec);

                        const stats = computeErrorStats(axScaled, lambdaXScaled, ABS_TOL_POWER);

                        printBlockMatrix({
                            t,
                            method: "power iteration",
                            explanation: `Gas sensitivity and numerical accuracy of power iteration under different keccak-generated symmetric matrix patterns at fixed dimension n=${n}. Accuracy is evaluated with the residual of Ax-λx.`,
                            gas,
                            shapeIn: `${n}x${n}`,
                            shapeOut: `${xRows}x${xCols}`,
                            inHex: `pattern=${pattern}, matrixSeed=${POWER_MATRIX_SEED}, initSeed=fixed, tol=1e-6, case=${caseNo}, iter=${iterCount}`,
                            outHex: `lambda=${lambdaHex}, eigenvector=${headTail(xData)}`,
                            expectedDec: headTail(lambdaXScaled.map(formatScaledInt)),
                            outDec: `lambda=${formatScaledInt(lambdaDec)} | x=${headTail(xDec.map(formatScaledInt))} | Ax=${headTail(axScaled.map(formatScaledInt))}`,
                            maxAbsError: formatScaledInt(stats.maxAbsError),
                            avgAbsError: formatScaledInt(stats.avgAbsError),
                            maxRelError: formatRelativeScaled(stats.maxRelErrorScaled),
                            avgRelError: formatRelativeScaled(stats.avgRelErrorScaled),
                            residual: formatScaledInt(stats.residual),
                            normalizedResidual: formatRelativeScaled(stats.normalizedResidualScaled),
                            tolerance: formatScaledInt(stats.absTol),
                            withinTol: `${stats.withinTolCount}/${axScaled.length}`,
                            exceededTol: `${stats.exceededTolCount}/${axScaled.length}`,
                            worstCase: `${stats.worstIndex}`,
                        });

                        expect(toGasBigInt(gas) > 0n).to.equal(true);
                    });
                }
            }
        });
    });
});
