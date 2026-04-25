// SPDX-License-Identifier: MIT
import {expect} from "chai";
import {ethers} from "hardhat";
import type {Contract} from "ethers";
import {touchGas, estimateGas, printBlockRegular, printBlockMatrix} from "./test-utils";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type MatrixMasterHarness = Contract & {
    qFromFrac(num: bigint, den: bigint): Promise<string>;
    toFloat(x: string): Promise<bigint>;

    addHarness(
        aRows: bigint,
        aCols: bigint,
        aData: string[],
        bRows: bigint,
        bCols: bigint,
        bData: string[]
    ): Promise<[bigint, bigint, string[]]>;

    subHarness(
        aRows: bigint,
        aCols: bigint,
        aData: string[],
        bRows: bigint,
        bCols: bigint,
        bData: string[]
    ): Promise<[bigint, bigint, string[]]>;
};

// ------------------------------------------------------------
// Constants & Types
// ------------------------------------------------------------

const SCALE_DECIMALS = 12n;
const SCALE = 10n ** SCALE_DECIMALS;

// Input random generation precision
const Q_SCALE_DECIMALS = 18n;
const Q_SCALE = 10n ** Q_SCALE_DECIMALS;

// Conversion from 1e18-scale expected values to 1e12-scale comparison values
const COMPARE_DOWN_SCALE = 10n ** (Q_SCALE_DECIMALS - SCALE_DECIMALS); // 1e6

// Relative error display scale
const REL_SCALE_DECIMALS = 12n;
const REL_SCALE = 10n ** REL_SCALE_DECIMALS;

type MatrixPattern = "dense" | "banded" | "identity" | "sparse";
type OpType = "add" | "sub";

type AddSubCase = {
    sub: string;
    group: "set1" | "set2";
    op: OpType;
    n: number;
    patternA: MatrixPattern;
    patternB: MatrixPattern;
    label: string;
    caseId: number;
};

// ------------------------------------------------------------
// Generic Helpers
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

function headTail(arr: (string | bigint)[], limit = 3): string {
    if (arr.length <= limit * 2) return `[${arr.join(", ")}]`;
    return `[${arr.slice(0, limit).join(", ")}, ..., ${arr.slice(-limit).join(", ")}]`;
}

function absBigInt(x: bigint): bigint {
    return x < 0n ? -x : x;
}

async function fromQuadArray(h: MatrixMasterHarness, arr: string[]): Promise<bigint[]> {
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

function randBigIntFromKeccak(
    seed: string,
    min: bigint,
    max: bigint,
    ...parts: (string | number)[]
): bigint {
    const span = max - min + 1n;
    const r = keccakToBigInt(seed, ...parts) % span;
    return min + r;
}

// ------------------------------------------------------------
// Fixed-point random generation with 18 decimals
// ------------------------------------------------------------

function makeSignedScaledValue(
    seed: string,
    caseId: number,
    tag: "A" | "B",
    i: number,
    j: number,
    variant: string
): bigint {
    const intPart = BigInt(
        randIntFromKeccak(seed, -9, 9, "int", caseId, tag, i, j, variant)
    );

    const fracPart = randBigIntFromKeccak(
        seed,
        0n,
        Q_SCALE - 1n,
        "frac",
        caseId,
        tag,
        i,
        j,
        variant
    );

    if (intPart < 0n) return intPart * Q_SCALE - fracPart;
    if (intPart > 0n) return intPart * Q_SCALE + fracPart;

    const signBit = randIntFromKeccak(seed, 0, 1, "sign", caseId, tag, i, j, variant);
    return signBit === 0 ? fracPart : -fracPart;
}

function makeNonZeroSignedScaledValue(
    seed: string,
    caseId: number,
    tag: "A" | "B",
    i: number,
    j: number,
    variant: string
): bigint {
    let v = makeSignedScaledValue(seed, caseId, tag, i, j, variant);
    if (v === 0n) v = 1n; // 1e-18
    return v;
}

async function qArrayFromScaledBigints(
    h: MatrixMasterHarness,
    arr: bigint[]
): Promise<string[]> {
    return Promise.all(arr.map(v => h.qFromFrac(v, Q_SCALE)));
}

// ------------------------------------------------------------
// Matrix builders
// ------------------------------------------------------------

function makeRandomMatrixByPattern(
    n: number,
    pattern: MatrixPattern,
    seed: string,
    caseId: number,
    tag: "A" | "B"
): bigint[] {
    const A = new Array<bigint>(n * n).fill(0n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const key = `case-${caseId}-${tag}-${i}-${j}`;
            const p = idx(n, i, j);

            if (pattern === "identity") {
                A[p] = i === j ? Q_SCALE : 0n;
                continue;
            }

            if (pattern === "banded") {
                const d = Math.abs(i - j);
                A[p] =
                    d <= 1
                        ? makeNonZeroSignedScaledValue(seed, caseId, tag, i, j, `${key}-banded`)
                        : 0n;
                continue;
            }

            if (pattern === "sparse") {
                const include =
                    i === j
                        ? true
                        : randIntFromKeccak(seed, 0, 99, `${key}-mask`) < 20;

                A[p] = include
                    ? makeNonZeroSignedScaledValue(seed, caseId, tag, i, j, `${key}-sparse`)
                    : 0n;
                continue;
            }

            A[p] = makeNonZeroSignedScaledValue(seed, caseId, tag, i, j, `${key}-dense`);
        }
    }

    return A;
}

// ------------------------------------------------------------
// Accuracy helpers
// ------------------------------------------------------------

function scale18To12(x: bigint): bigint {
    return x / COMPARE_DOWN_SCALE;
}

function expectedAddScaled12(A: bigint[], B: bigint[]): bigint[] {
    return A.map((a, i) => scale18To12(a + B[i]));
}

function expectedSubScaled12(A: bigint[], B: bigint[]): bigint[] {
    return A.map((a, i) => scale18To12(a - B[i]));
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

    // Reporting a single scalar as infinity-norm style residual.
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
// Case builders
// ------------------------------------------------------------

function buildPairCases(
    group: "set1" | "set2",
    prefix: string,
    n: number,
    patternA: MatrixPattern,
    patternB: MatrixPattern,
    perOpCount: number,
    baseCaseId: number
): AddSubCase[] {
    const addCases: AddSubCase[] = Array.from({length: perOpCount}, (_, k) => ({
        sub: `${prefix}.A${k + 1}`,
        group,
        op: "add",
        n,
        patternA,
        patternB,
        label: `${patternA}+${patternB}`,
        caseId: baseCaseId + k,
    }));

    const subCases: AddSubCase[] = Array.from({length: perOpCount}, (_, k) => ({
        sub: `${prefix}.S${k + 1}`,
        group,
        op: "sub",
        n,
        patternA,
        patternB,
        label: `${patternA}-${patternB}`,
        caseId: baseCaseId + perOpCount + k,
    }));

    return [...addCases, ...subCases];
}

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

describe("MatrixMasterHarness - Gas and Accuracy Tests (Add/Sub)", function () {
    let harness: MatrixMasterHarness;
    let t = 0;

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
    });

    describe("Section 1: Matrix Add/Sub", function () {
        const RNG_SEED = ethers.keccak256(
            ethers.toUtf8Bytes("matrix-add-sub-fixed-seed-dec18-v3")
        );

        const ALL_CASES: AddSubCase[] = [
            // ------------------------------------------------
            // Set-1: n = 8
            // Each pair: 10 add + 10 sub
            // ------------------------------------------------
            ...buildPairCases("set1", "1.1", 8, "dense", "banded", 10, 1000),
            ...buildPairCases("set1", "1.2", 8, "dense", "identity", 10, 2000),
            ...buildPairCases("set1", "1.3", 8, "dense", "sparse", 10, 3000),
            ...buildPairCases("set1", "1.4", 8, "sparse", "identity", 10, 4000),

            // ------------------------------------------------
            // Set-2: n = 4,8,16,32,64,88
            // dense&sparse, 5 add + 5 sub for each size
            // ------------------------------------------------
            ...buildPairCases("set2", "2.1", 4, "dense", "sparse", 5, 5000),
            ...buildPairCases("set2", "2.2", 8, "dense", "sparse", 5, 6000),
            ...buildPairCases("set2", "2.3", 16, "dense", "sparse", 5, 7000),
            ...buildPairCases("set2", "2.4", 32, "dense", "sparse", 5, 8000),
            ...buildPairCases("set2", "2.5", 64, "dense", "sparse", 5, 9000),
            ...buildPairCases("set2", "2.6", 80, "dense", "sparse", 5, 10000),
            ...buildPairCases("set2", "2.7", 92, "dense", "sparse", 5, 11000),
        ];

        for (const c of ALL_CASES) {
            it(`${c.sub} Matrix ${c.op} gas and accuracy for ${c.label} at n=${c.n}`, async function () {
                t++;

                const A = makeRandomMatrixByPattern(c.n, c.patternA, RNG_SEED, c.caseId, "A");
                const B = makeRandomMatrixByPattern(c.n, c.patternB, RNG_SEED, c.caseId, "B");

                const Aq = await qArrayFromScaledBigints(harness, A);
                const Bq = await qArrayFromScaledBigints(harness, B);

                if (c.op === "add") {
                    await touchGas(harness, "addHarness", [
                        BigInt(c.n), BigInt(c.n), Aq,
                        BigInt(c.n), BigInt(c.n), Bq,
                    ]);

                    const gas = await estimateGas(harness, "addHarness", [
                        BigInt(c.n), BigInt(c.n), Aq,
                        BigInt(c.n), BigInt(c.n), Bq,
                    ]);

                    const [, , out] = await harness.addHarness(
                        BigInt(c.n), BigInt(c.n), Aq,
                        BigInt(c.n), BigInt(c.n), Bq
                    );

                    const actualScaled12 = await fromQuadArray(harness, out);
                    const expectedScaled12 = expectedAddScaled12(A, B);

                    const {
                        maxAbsError,
                        avgAbsError,
                        maxRelErrorScaled,
                        avgRelErrorScaled,
                    } = computeErrorStats(actualScaled12, expectedScaled12);

                    printBlockMatrix({
                        t,
                        method: "matrix add",
                        explanation:
                            `Gas and numerical accuracy of matrix addition for deterministic keccak-generated ${c.label} operands at dimension n=${c.n} using 18-decimal random values.`,
                        gas,
                        inHex: `group=${c.group}, op=add, pattern=${c.label}, shape=${c.n}x${c.n}, seed=${RNG_SEED}, caseId=${c.caseId}`,
                        outHex: headTail(out),
                        expectedDec: headTail(expectedScaled12.map(formatScaledInt)),
                        outDec: headTail(actualScaled12.map(formatScaledInt)),
                        maxAbsError: formatScaledInt(maxAbsError),
                        avgAbsError: formatScaledInt(avgAbsError),
                        maxRelError: formatRelativeScaled(maxRelErrorScaled),
                        avgRelError: formatRelativeScaled(avgRelErrorScaled),
                    });

                    expect(actualScaled12.length).to.equal(expectedScaled12.length);
                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                } else {
                    await touchGas(harness, "subHarness", [
                        BigInt(c.n), BigInt(c.n), Aq,
                        BigInt(c.n), BigInt(c.n), Bq,
                    ]);

                    const gas = await estimateGas(harness, "subHarness", [
                        BigInt(c.n), BigInt(c.n), Aq,
                        BigInt(c.n), BigInt(c.n), Bq,
                    ]);

                    const [, , out] = await harness.subHarness(
                        BigInt(c.n), BigInt(c.n), Aq,
                        BigInt(c.n), BigInt(c.n), Bq
                    );

                    const actualScaled12 = await fromQuadArray(harness, out);
                    const expectedScaled12 = expectedSubScaled12(A, B);

                    const {
                        maxAbsError,
                        avgAbsError,
                        maxRelErrorScaled,
                        avgRelErrorScaled,
                        residual,
                    } = computeErrorStats(actualScaled12, expectedScaled12);

                    printBlockMatrix({
                        t,
                        method: "matrix sub",
                        explanation:
                            `Gas and numerical accuracy of matrix subtraction for deterministic keccak-generated ${c.label} operands at dimension n=${c.n} using 18-decimal random values.`,
                        gas,
                        inHex: `group=${c.group}, op=sub, pattern=${c.label}, shape=${c.n}x${c.n}, seed=${RNG_SEED}, caseId=${c.caseId}`,
                        outHex: headTail(out),
                        expectedDec: headTail(expectedScaled12.map(formatScaledInt)),
                        outDec: headTail(actualScaled12.map(formatScaledInt)),
                        maxAbsError: formatScaledInt(maxAbsError),
                        avgAbsError: formatScaledInt(avgAbsError),
                        maxRelError: formatRelativeScaled(maxRelErrorScaled),
                        avgRelError: formatRelativeScaled(avgRelErrorScaled),
                    });

                    expect(actualScaled12.length).to.equal(expectedScaled12.length);
                    expect(toGasBigInt(gas) > 0n).to.equal(true);
                }
            });
        }
    });
});