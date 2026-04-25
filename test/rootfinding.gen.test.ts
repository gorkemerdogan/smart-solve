// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------

type RootFindingHarness = Contract & {
    qFromInt(x: number | bigint): Promise<string>;
    qFromUInt(x: number | bigint): Promise<string>;
    qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
    toFloat(q: string): Promise<unknown>;
    rootFindingBisection(
        target: string,
        fSelector: string,
        a: string,
        b: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, boolean, string]>;
    rootFindingNewton(
        target: string,
        fSelector: string,
        dfTarget: string,
        dfSelector: string,
        x0: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, boolean, string]>;
    rootFindingSecant(
        target: string,
        fSelector: string,
        x0: string,
        x1: string,
        tol: string,
        maxIter: bigint
    ): Promise<[string, bigint, boolean, string]>;
};

// ------------------------------------------------------------
// Global numeric helpers
// ------------------------------------------------------------

let SCALE_DECIMALS = 18n;
let SCALE = 10n ** SCALE_DECIMALS;

const FIXED_SEED = 37n;
const NUMBER_OF_TESTS = 90;

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

function inferScaleDecimals(scale: bigint): bigint {
    const s = scale.toString();
    if (!/^10*$/.test(s) || s[0] !== "1") return SCALE_DECIMALS;
    return BigInt(s.length - 1);
}

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

function scaledAbsError(actual: bigint, expected: bigint): bigint {
    return absBigInt(actual - expected);
}

function scaledRelError(actual: bigint, expected: bigint): bigint {
    const num = absBigInt(actual - expected);
    const den = absBigInt(expected);
    if (den === 0n) return 0n;
    return (num * SCALE) / den;
}

async function outScaled(harness: RootFindingHarness, q: string): Promise<bigint> {
    const raw = await harness.toFloat(q);
    return asBigInt(raw);
}

function avgBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => a + b, 0n) / BigInt(values.length);
}

function minBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => (a < b ? a : b));
}

function maxBigInt(values: bigint[]): bigint {
    if (values.length === 0) return 0n;
    return values.reduce((a, b) => (a > b ? a : b));
}

// ------------------------------------------------------------
// Fixed-seed pseudo-random helpers
// ------------------------------------------------------------

function randomWord(tag: string): bigint {
    const encoded = ethers.solidityPacked(
        ["uint256", "string"],
        [FIXED_SEED, tag]
    );
    return BigInt(ethers.keccak256(encoded));
}

function randomScaledBetween(tag: string, minScaled: bigint, maxScaled: bigint): bigint {
    if (maxScaled <= minScaled) return minScaled;
    const span = maxScaled - minScaled + 1n;
    return minScaled + (randomWord(tag) % span);
}

// ------------------------------------------------------------
// Benchmark config & summary
// ------------------------------------------------------------

type MethodName = "Bisection" | "Newton" | "Secant";

type MethodSummary = {
    convergedCount: number;
    absErrorsConv: bigint[];
    relErrorsConv: bigint[];
    residualsConv: bigint[];
    iterationsAll: bigint[];
    gasesAll: bigint[];
};

type BenchmarkConfig = {
    name: string;
    fSelector: string;
    dfSelector: string;
    expectedRootScaled: bigint;

    maxIterBisection: bigint;
    maxIterNewton: bigint;
    maxIterSecant: bigint;

    makeBisectionBracket: (tag: string) => { aScaled: bigint; bScaled: bigint };
    makeNewtonInitialGuess: (tag: string) => bigint;
    makeSecantInitialGuesses: (tag: string) => { x0Scaled: bigint; x1Scaled: bigint };
};

function createSummary(): MethodSummary {
    return {
        convergedCount: 0,
        absErrorsConv: [],
        relErrorsConv: [],
        residualsConv: [],
        iterationsAll: [],
        gasesAll: [],
    };
}

function printMethodSummary(
    benchmarkName: string,
    method: MethodName,
    s: MethodSummary,
    totalTests: number
) {
    console.log("============================================================");
    console.log(`Benchmark            : ${benchmarkName}`);
    console.log(`Method               : ${method}`);
    console.log(`Number of Tests      : ${totalTests}`);
    console.log(`Converged Tests      : ${s.convergedCount}`);
    console.log(`Average Abs. Error   : ${formatScaledInt(avgBigInt(s.absErrorsConv))}`);
    console.log(`Average Rel. Error   : ${formatScaledInt(avgBigInt(s.relErrorsConv))}`);
    console.log(`Average Residual     : ${formatScaledInt(avgBigInt(s.residualsConv))}`);
    console.log(`Average Iterations   : ${avgBigInt(s.iterationsAll).toString()}`);
    console.log(`Min Gas              : ${minBigInt(s.gasesAll).toString()}`);
    console.log(`Average Gas          : ${avgBigInt(s.gasesAll).toString()}`);
    console.log(`Max Gas              : ${maxBigInt(s.gasesAll).toString()}`);
    console.log("============================================================");
}

// ------------------------------------------------------------
// Per-case logging helpers
// ------------------------------------------------------------

function printCaseHeader(benchmarkName: string, caseNo: number) {
    console.log("------------------------------------------------------------");
    console.log(`Benchmark            : ${benchmarkName}`);
    console.log(`Case                 : ${caseNo}`);
    console.log("------------------------------------------------------------");
}

function printCaseResult(args: {
    benchmarkName: string;
    caseNo: number;
    method: MethodName;
    initText: string;
    converged: boolean;
    iterations: bigint;
    gas: bigint;
    rootScaled: bigint;
    expectedRootScaled: bigint;
    residualScaled: bigint;
}) {
    const absErr = scaledAbsError(args.rootScaled, args.expectedRootScaled);
    const relErr = scaledRelError(args.rootScaled, args.expectedRootScaled);

    console.log(`Benchmark            : ${args.benchmarkName}`);
    console.log(`Case                 : ${args.caseNo}`);
    console.log(`Method               : ${args.method}`);
    console.log(`Initialization       : ${args.initText}`);
    console.log(`Converged            : ${args.converged ? "Yes" : "No"}`);
    console.log(`Iterations           : ${args.iterations.toString()}`);
    console.log(`Estimated Gas        : ${args.gas.toString()}`);
    console.log(`Expected Root        : ${formatScaledInt(args.expectedRootScaled)}`);
    console.log(`Computed Root        : ${formatScaledInt(args.rootScaled)}`);
    console.log(`Absolute Error       : ${formatScaledInt(absErr)}`);
    console.log(`Relative Error       : ${formatScaledInt(relErr)}`);
    console.log(`Residual             : ${formatScaledInt(args.residualScaled)}`);
    console.log("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Test suite
// ------------------------------------------------------------

describe("RootFinding Library - Multi-Case Accuracy & Gas Benchmarks", function () {
    this.timeout(0); // Infinity minutes
    let harness: RootFindingHarness;
    let target: string;

    let TOL_1E_30: string;

    let selX2Minus4: string;
    let selDf2x: string;
    let selCubic: string;
    let selDfCubic: string;
    let selQuartic: string;
    let selDfQuartic: string;

    const qInt = async (x: number | bigint) => await harness.qFromInt(x);
    const qFrac = async (num: number | bigint, den: number | bigint) => await harness.qFromFrac(num, den);

    async function scaledFromDecimal(
        integerPart: bigint,
        fractionalDigits: string
    ): Promise<bigint> {
        const frac = fractionalDigits.replace(/_+/g, "");
        const den = 10n ** BigInt(frac.length);
        const num = integerPart * den + BigInt(frac);
        const q = await harness.qFromFrac(num, den);
        return await outScaled(harness, q);
    }

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HF = await ethers.getContractFactory("RootFindingHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HF.deploy()) as unknown as RootFindingHarness;
        await harness.waitForDeployment();
        target = await harness.getAddress();

        selX2Minus4 = harness.interface.getFunction("f_x2_minus_4")!.selector;
        selDf2x = harness.interface.getFunction("df_2x")!.selector;

        selCubic = harness.interface.getFunction("f_cubic")!.selector;
        selDfCubic = harness.interface.getFunction("df_cubic")!.selector;

        selQuartic = harness.interface.getFunction("f_quartic")!.selector;
        selDfQuartic = harness.interface.getFunction("df_quartic")!.selector;

        TOL_1E_30 = await qFrac(1n, 1_000_000_000_000_000_000_000_000n);

        const oneScaled = asBigInt(await harness.toFloat(await qInt(1n)));
        SCALE = oneScaled;
        SCALE_DECIMALS = inferScaleDecimals(oneScaled);
    });

    async function runBenchmark(cfg: BenchmarkConfig) {
        const summaries: Record<MethodName, MethodSummary> = {
            Bisection: createSummary(),
            Newton: createSummary(),
            Secant: createSummary(),
        };

        for (let i = 1; i <= NUMBER_OF_TESTS; i++) {
            const tag = `${cfg.name}-case-${i}`;

            printCaseHeader(cfg.name, i);

            // --------------------------------------------------------
            // Bisection
            // --------------------------------------------------------

            {
                const { aScaled, bScaled } = cfg.makeBisectionBracket(tag);
                const a = await qFrac(aScaled, SCALE);
                const b = await qFrac(bScaled, SCALE);

                const gas = await harness
                    .getFunction("rootFindingBisection")
                    .estimateGas(target, cfg.fSelector, a, b, TOL_1E_30, cfg.maxIterBisection);

                const [root, iterations, converged, fAtRoot] =
                    await harness.rootFindingBisection(
                        target,
                        cfg.fSelector,
                        a,
                        b,
                        TOL_1E_30,
                        cfg.maxIterBisection
                    );

                const rootScaled = await outScaled(harness, root);
                const residualScaled = absBigInt(await outScaled(harness, fAtRoot));

                summaries.Bisection.iterationsAll.push(iterations);
                summaries.Bisection.gasesAll.push(asBigInt(gas));

                if (converged) {
                    summaries.Bisection.convergedCount++;
                    summaries.Bisection.absErrorsConv.push(
                        scaledAbsError(rootScaled, cfg.expectedRootScaled)
                    );
                    summaries.Bisection.relErrorsConv.push(
                        scaledRelError(rootScaled, cfg.expectedRootScaled)
                    );
                    summaries.Bisection.residualsConv.push(residualScaled);
                }

                printCaseResult({
                    benchmarkName: cfg.name,
                    caseNo: i,
                    method: "Bisection",
                    initText: `a=${formatScaledInt(aScaled)}, b=${formatScaledInt(bScaled)}`,
                    converged,
                    iterations,
                    gas: asBigInt(gas),
                    rootScaled,
                    expectedRootScaled: cfg.expectedRootScaled,
                    residualScaled,
                });
            }

            // --------------------------------------------------------
            // Newton
            // --------------------------------------------------------

            {
                const x0Scaled = cfg.makeNewtonInitialGuess(tag);
                const x0 = await qFrac(x0Scaled, SCALE);

                const gas = await harness
                    .getFunction("rootFindingNewton")
                    .estimateGas(
                        target,
                        cfg.fSelector,
                        target,
                        cfg.dfSelector,
                        x0,
                        TOL_1E_30,
                        cfg.maxIterNewton
                    );

                const [root, iterations, converged, fAtRoot] =
                    await harness.rootFindingNewton(
                        target,
                        cfg.fSelector,
                        target,
                        cfg.dfSelector,
                        x0,
                        TOL_1E_30,
                        cfg.maxIterNewton
                    );

                const rootScaled = await outScaled(harness, root);
                const residualScaled = absBigInt(await outScaled(harness, fAtRoot));

                summaries.Newton.iterationsAll.push(iterations);
                summaries.Newton.gasesAll.push(asBigInt(gas));

                if (converged) {
                    summaries.Newton.convergedCount++;
                    summaries.Newton.absErrorsConv.push(
                        scaledAbsError(rootScaled, cfg.expectedRootScaled)
                    );
                    summaries.Newton.relErrorsConv.push(
                        scaledRelError(rootScaled, cfg.expectedRootScaled)
                    );
                    summaries.Newton.residualsConv.push(residualScaled);
                }

                printCaseResult({
                    benchmarkName: cfg.name,
                    caseNo: i,
                    method: "Newton",
                    initText: `x0=${formatScaledInt(x0Scaled)}`,
                    converged,
                    iterations,
                    gas: asBigInt(gas),
                    rootScaled,
                    expectedRootScaled: cfg.expectedRootScaled,
                    residualScaled,
                });
            }

            // --------------------------------------------------------
            // Secant
            // --------------------------------------------------------

            {
                const { x0Scaled, x1Scaled } = cfg.makeSecantInitialGuesses(tag);
                const x0 = await qFrac(x0Scaled, SCALE);
                const x1 = await qFrac(x1Scaled, SCALE);

                const gas = await harness
                    .getFunction("rootFindingSecant")
                    .estimateGas(target, cfg.fSelector, x0, x1, TOL_1E_30, cfg.maxIterSecant);

                const [root, iterations, converged, fAtRoot] =
                    await harness.rootFindingSecant(
                        target,
                        cfg.fSelector,
                        x0,
                        x1,
                        TOL_1E_30,
                        cfg.maxIterSecant
                    );

                const rootScaled = await outScaled(harness, root);
                const residualScaled = absBigInt(await outScaled(harness, fAtRoot));

                summaries.Secant.iterationsAll.push(iterations);
                summaries.Secant.gasesAll.push(asBigInt(gas));

                if (converged) {
                    summaries.Secant.convergedCount++;
                    summaries.Secant.absErrorsConv.push(
                        scaledAbsError(rootScaled, cfg.expectedRootScaled)
                    );
                    summaries.Secant.relErrorsConv.push(
                        scaledRelError(rootScaled, cfg.expectedRootScaled)
                    );
                    summaries.Secant.residualsConv.push(residualScaled);
                }

                printCaseResult({
                    benchmarkName: cfg.name,
                    caseNo: i,
                    method: "Secant",
                    initText: `x0=${formatScaledInt(x0Scaled)}, x1=${formatScaledInt(x1Scaled)}`,
                    converged,
                    iterations,
                    gas: asBigInt(gas),
                    rootScaled,
                    expectedRootScaled: cfg.expectedRootScaled,
                    residualScaled,
                });
            }
        }

        printMethodSummary(cfg.name, "Bisection", summaries.Bisection, NUMBER_OF_TESTS);
        printMethodSummary(cfg.name, "Newton", summaries.Newton, NUMBER_OF_TESTS);
        printMethodSummary(cfg.name, "Secant", summaries.Secant, NUMBER_OF_TESTS);

        return summaries;
    }

    it("should run quadratic benchmark: f(x)=x^2-4", async function () {
        const root = 2n * SCALE;

        const summaries = await runBenchmark({
            name: "Quadratic benchmark: f(x)=x^2-4",
            fSelector: selX2Minus4,
            dfSelector: selDf2x,
            expectedRootScaled: root,
            maxIterBisection: 1130n,
            maxIterNewton: 1130n,
            maxIterSecant: 1130n,

            makeBisectionBracket: (_tag: string) => {
                return {
                    aScaled: 1n * SCALE,
                    bScaled: 3n * SCALE,
                };
            },

            makeNewtonInitialGuess: (tag: string) => {
                const side = randomWord(`${tag}-newton-side`) % 2n;
                const offset = randomScaledBetween(`${tag}-newton-offset`, SCALE / 20n, SCALE / 2n);
                return side === 0n ? root - offset : root + offset;
            },

            makeSecantInitialGuesses: (tag: string) => {
                const leftOffset = randomScaledBetween(`${tag}-sec-left`, SCALE / 20n, 4n * SCALE / 10n);
                const rightOffset = randomScaledBetween(`${tag}-sec-right`, SCALE / 20n, 4n * SCALE / 10n);
                return {
                    x0Scaled: root - leftOffset,
                    x1Scaled: root + rightOffset,
                };
            },
        });

        expect(summaries.Bisection.convergedCount).to.be.greaterThan(0);
        expect(summaries.Newton.convergedCount).to.be.greaterThan(0);
        expect(summaries.Secant.convergedCount).to.be.greaterThan(0);
    });

    it("should run cubic benchmark: f(x)=x^3-x-2", async function () {
        const cubicRoot = await scaledFromDecimal(1n, "521379706805");
        const root = cubicRoot;

        const summaries = await runBenchmark({
            name: "Cubic benchmark: f(x)=x^3-x-2",
            fSelector: selCubic,
            dfSelector: selDfCubic,
            expectedRootScaled: root,
            maxIterBisection: 1130n,
            maxIterNewton: 1130n,
            maxIterSecant: 1130n,

            makeBisectionBracket: (_tag: string) => {
                return {
                    aScaled: 1n * SCALE,
                    bScaled: 2n * SCALE,
                };
            },

            makeNewtonInitialGuess: (tag: string) => {
                const side = randomWord(`${tag}-newton-side`) % 2n;
                const offset = randomScaledBetween(`${tag}-newton-offset`, SCALE / 20n, 6n * SCALE / 10n);
                return side === 0n ? root - offset : root + offset;
            },

            makeSecantInitialGuesses: (tag: string) => {
                const leftOffset = randomScaledBetween(`${tag}-sec-left`, SCALE / 20n, 3n * SCALE / 10n);
                const rightOffset = randomScaledBetween(`${tag}-sec-right`, SCALE / 20n, 3n * SCALE / 10n);
                return {
                    x0Scaled: root - leftOffset,
                    x1Scaled: root + rightOffset,
                };
            },
        });

        expect(summaries.Bisection.convergedCount).to.be.greaterThan(0);
        expect(summaries.Newton.convergedCount).to.be.greaterThan(0);
        expect(summaries.Secant.convergedCount).to.be.greaterThan(0);
    });

    it("should run quartic benchmark: f(x)=x^4-10", async function () {
        const quarticRoot = await scaledFromDecimal(1n, "778279410038");
        const root = quarticRoot;

        const summaries = await runBenchmark({
            name: "Quartic benchmark: f(x)=x^4-10",
            fSelector: selQuartic,
            dfSelector: selDfQuartic,
            expectedRootScaled: root,
            maxIterBisection: 1130n,
            maxIterNewton: 1130n,
            maxIterSecant: 1130n,

            makeBisectionBracket: (_tag: string) => {
                return {
                    aScaled: 1n * SCALE,
                    bScaled: 2n * SCALE,
                };
            },

            makeNewtonInitialGuess: (tag: string) => {
                const side = randomWord(`${tag}-newton-side`) % 2n;
                const offset = randomScaledBetween(`${tag}-newton-offset`, SCALE / 20n, 5n * SCALE / 10n);
                return side === 0n ? root - offset : root + offset;
            },

            makeSecantInitialGuesses: (tag: string) => {
                const leftOffset = randomScaledBetween(`${tag}-sec-left`, SCALE / 20n, 2n * SCALE / 10n);
                const rightOffset = randomScaledBetween(`${tag}-sec-right`, SCALE / 20n, 2n * SCALE / 10n);
                return {
                    x0Scaled: root - leftOffset,
                    x1Scaled: root + rightOffset,
                };
            },
        });

        expect(summaries.Bisection.convergedCount).to.be.greaterThan(0);
        expect(summaries.Newton.convergedCount).to.be.greaterThan(0);
        expect(summaries.Secant.convergedCount).to.be.greaterThan(0);
    });
});