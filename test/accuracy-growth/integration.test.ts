// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

describe("IntegrationHarness - Accuracy & Gas Benchmark Suite", function () {
    this.timeout(300000); // 5 minutes
    // ------------------------------------------------------------
    // Types
    // ------------------------------------------------------------

    type IntegrationHarness = Contract & {
        qAdd(a: string, b: string): Promise<string>;
        qSub(a: string, b: string): Promise<string>;
        qMul(a: string, b: string): Promise<string>;
        qDiv(a: string, b: string): Promise<string>;
        qNeg(a: string): Promise<string>;

        qFromInt(x: number | bigint): Promise<string>;
        qFromFrac(num: number | bigint, den: number | bigint): Promise<string>;
        toFloat(q: string): Promise<unknown>;
        fromFloat(n: bigint): Promise<string>;
        PI(): Promise<string>;

        trapezoidal(target: string, selector: string, a: string, b: string, n: bigint): Promise<string>;
        simpson13(target: string, selector: string, a: string, b: string, n: bigint): Promise<string>;
        simpson38(target: string, selector: string, a: string, b: string, n: bigint): Promise<string>;
    };

    type MethodName = "trapezoidal" | "simpson13" | "simpson38";
    type FunctionKey = "f_one" | "f_linear" | "f_square" | "f_cube" | "f_sin" | "f_inv" | "f_piecewise";

    interface IntervalSpec {
        label: string;
        a: number;
        b: number;
    }

    interface QuadIntervalSpec {
        label: string;
        a: string;
        b: string;
        aNum: number;
        bNum: number;
    }

    interface BenchmarkFunction {
        key: FunctionKey;
        label: string;
        selectorName: string;
        intervals: IntervalSpec[];
        exactIntegral: (a: number, b: number) => number;
    }

    interface TestRecord {
        method: MethodName;
        func: FunctionKey;
        funcLabel: string;
        intervalLabel: string;
        a: number;
        b: number;
        n: bigint;
        expected: number;
        actual: number;
        absError: number;
        relError: number;
        estimatedGas: bigint;
        gasUsed: bigint;
    }

    // ------------------------------------------------------------
    // State
    // ------------------------------------------------------------

    let harness: IntegrationHarness;
    let target: string;

    const SCALE = 1_000_000_000_000n; // 1e12
    const N_VALUES = [12n, 24n, 48n, 96n, 192n];
    const METHODS: MethodName[] = ["trapezoidal", "simpson13", "simpson38"];

    let SIN_INTERVALS_QUAD: QuadIntervalSpec[] = [];

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

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

    async function toQuad(x: number): Promise<string> {
        const scaled = BigInt(Math.round(x * Number(SCALE)));
        return await harness.fromFloat(scaled);
    }

    async function quadToNumber(q: string): Promise<number> {
        const scaledRaw = await harness.toFloat(q);
        const scaled = asBigInt(scaledRaw);
        return Number(scaled) / Number(SCALE);
    }

    function computeRelativeError(actual: number, expected: number): number {
        const denom = Math.abs(expected);
        if (denom === 0) {
            return Math.abs(actual);
        }
        return Math.abs(actual - expected) / denom;
    }

    function mean(values: number[]): number {
        if (values.length === 0) return 0;
        return values.reduce((acc, v) => acc + v, 0) / values.length;
    }

    function meanBigInt(values: bigint[]): bigint {
        if (values.length === 0) return 0n;
        const total = values.reduce((acc, v) => acc + v, 0n);
        return total / BigInt(values.length);
    }

    function minBigInt(values: bigint[]): bigint {
        return values.reduce((min, v) => (v < min ? v : min), values[0]);
    }

    function maxBigInt(values: bigint[]): bigint {
        return values.reduce((max, v) => (v > max ? v : max), values[0]);
    }

    function minNumber(values: number[]): number {
        return Math.min(...values);
    }

    function maxNumber(values: number[]): number {
        return Math.max(...values);
    }

    function exactPiecewiseIntegral(a: number, b: number): number {
        if (a < 0 || b > 2 || a > b) {
            throw new Error(`Invalid piecewise interval: [${a}, ${b}]`);
        }

        if (b <= 1) {
            return b - a;
        }

        if (a >= 1) {
            return 3 * (b - a);
        }

        return (1 - a) + 3 * (b - 1);
    }

    function printOverallSummary(records: TestRecord[]): void {
        const absErrors = records.map((r) => r.absError);
        const relErrors = records.map((r) => r.relError);
        const estimatedGases = records.map((r) => r.estimatedGas);
        const gasUsedValues = records.map((r) => r.gasUsed);

        console.log("============================================================");
        console.log("OVERALL SUMMARY");
        console.log("============================================================");
        console.log(`Total Tests           : ${records.length}`);
        console.log(`Average Abs. Error    : ${mean(absErrors)}`);
        console.log(`Average Rel. Error    : ${mean(relErrors)}`);
        console.log(`Min Abs. Error        : ${minNumber(absErrors)}`);
        console.log(`Max Abs. Error        : ${maxNumber(absErrors)}`);
        console.log(`Average Estimated Gas : ${meanBigInt(estimatedGases).toString()}`);
        console.log(`Average Gas Used      : ${meanBigInt(gasUsedValues).toString()}`);
        console.log(`Min Estimated Gas     : ${minBigInt(estimatedGases).toString()}`);
        console.log(`Max Estimated Gas     : ${maxBigInt(estimatedGases).toString()}`);
        console.log(`Min Gas Used          : ${minBigInt(gasUsedValues).toString()}`);
        console.log(`Max Gas Used          : ${maxBigInt(gasUsedValues).toString()}`);
    }

    function printMethodSummaries(records: TestRecord[]): void {
        for (const method of METHODS) {
            const subset = records.filter((r) => r.method === method);

            const absErrors = subset.map((r) => r.absError);
            const relErrors = subset.map((r) => r.relError);
            const estimatedGases = subset.map((r) => r.estimatedGas);
            const gasUsedValues = subset.map((r) => r.gasUsed);

            console.log("============================================================");
            console.log(`SUMMARY - ${method}`);
            console.log("============================================================");
            console.log(`Total Tests           : ${subset.length}`);
            console.log(`Average Abs. Error    : ${mean(absErrors)}`);
            console.log(`Average Rel. Error    : ${mean(relErrors)}`);
            console.log(`Min Abs. Error        : ${minNumber(absErrors)}`);
            console.log(`Max Abs. Error        : ${maxNumber(absErrors)}`);
            console.log(`Average Estimated Gas : ${meanBigInt(estimatedGases).toString()}`);
            console.log(`Average Gas Used      : ${meanBigInt(gasUsedValues).toString()}`);
            console.log(`Min Estimated Gas     : ${minBigInt(estimatedGases).toString()}`);
            console.log(`Max Estimated Gas     : ${maxBigInt(estimatedGases).toString()}`);
            console.log(`Min Gas Used          : ${minBigInt(gasUsedValues).toString()}`);
            console.log(`Max Gas Used          : ${maxBigInt(gasUsedValues).toString()}`);
        }
    }

    function printRecord(record: TestRecord): void {
        console.log("------------------------------------------------------------");
        console.log(`Method        : ${record.method}`);
        console.log(`Function      : ${record.funcLabel}`);
        console.log(`Interval      : ${record.intervalLabel}`);
        console.log(`n             : ${record.n.toString()}`);
        console.log(`Expected      : ${record.expected}`);
        console.log(`Actual        : ${record.actual}`);
        console.log(`Abs. Error    : ${record.absError}`);
        console.log(`Rel. Error    : ${record.relError}`);
        console.log(`Estimated Gas : ${record.estimatedGas.toString()}`);
        console.log(`Gas Used      : ${record.gasUsed.toString()}`);
    }

    // ------------------------------------------------------------
    // Benchmark sets
    // ------------------------------------------------------------

    const COMMON_INTERVALS: IntervalSpec[] = [
        { label: "[0,1]", a: 0, b: 1 },
        { label: "[0,5]", a: 0, b: 5 },
        { label: "[-1,1]", a: -1, b: 1 },
        { label: "[-2,3]", a: -2, b: 3 },
        { label: "[1,6]", a: 1, b: 6 },
    ];

    const INV_INTERVALS: IntervalSpec[] = [
        { label: "[1,2]", a: 1, b: 2 },
        { label: "[1,5]", a: 1, b: 5 },
        { label: "[2,10]", a: 2, b: 10 },
        { label: "[-5,-1]", a: -5, b: -1 },
        { label: "[-2,-0.5]", a: -2, b: -0.5 },
    ];

    const PIECEWISE_INTERVALS: IntervalSpec[] = [
        { label: "[0,1]", a: 0, b: 1 },
        { label: "[1,2]", a: 1, b: 2 },
        { label: "[0,2]", a: 0, b: 2 },
        { label: "[0,1.5]", a: 0, b: 1.5 },
        { label: "[0.5,2]", a: 0.5, b: 2 },
    ];

    const FUNCTIONS: BenchmarkFunction[] = [
        {
            key: "f_one",
            label: "f(x)=1",
            selectorName: "f_one",
            intervals: COMMON_INTERVALS,
            exactIntegral: (a, b) => b - a,
        },
        {
            key: "f_linear",
            label: "f(x)=x",
            selectorName: "f_linear",
            intervals: COMMON_INTERVALS,
            exactIntegral: (a, b) => 0.5 * (b * b - a * a),
        },
        {
            key: "f_square",
            label: "f(x)=x^2",
            selectorName: "f_square",
            intervals: COMMON_INTERVALS,
            exactIntegral: (a, b) => (b ** 3 - a ** 3) / 3,
        },
        {
            key: "f_cube",
            label: "f(x)=x^3",
            selectorName: "f_cube",
            intervals: COMMON_INTERVALS,
            exactIntegral: (a, b) => (b ** 4 - a ** 4) / 4,
        },
        {
            key: "f_sin",
            label: "f(x)=sin(x)",
            selectorName: "f_sin",
            intervals: [],
            exactIntegral: (a, b) => -Math.cos(b) + Math.cos(a),
        },
        {
            key: "f_inv",
            label: "f(x)=1/x",
            selectorName: "f_inv",
            intervals: INV_INTERVALS,
            exactIntegral: (a, b) => Math.log(Math.abs(b)) - Math.log(Math.abs(a)),
        },
        {
            key: "f_piecewise",
            label: "f(x)=piecewise",
            selectorName: "f_piecewise",
            intervals: PIECEWISE_INTERVALS,
            exactIntegral: (a, b) => exactPiecewiseIntegral(a, b),
        },
    ];

    // ------------------------------------------------------------
    // Deploy
    // ------------------------------------------------------------

    before(async function () {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const mathlib = await MathLibFactory.deploy();
        await mathlib.waitForDeployment();

        const HarnessFactory = await ethers.getContractFactory("IntegrationHarness", {
            libraries: { MathLib: await mathlib.getAddress() },
        });

        harness = (await HarnessFactory.deploy()) as unknown as IntegrationHarness;
        await harness.waitForDeployment();
        target = await harness.getAddress();

        const zero = await harness.qFromInt(0n);
        const two = await harness.qFromInt(2n);
        const four = await harness.qFromInt(4n);
        const five = await harness.qFromInt(5n);
        const pi = await harness.PI();

        const halfPi = await harness.qDiv(pi, two);
        const twoPi = await harness.qMul(pi, two);
        const minusPi = await harness.qNeg(pi);
        const quarterPi = await harness.qDiv(pi, four);
        const fivePi = await harness.qMul(pi, five);
        const fivePiOverFour = await harness.qDiv(fivePi, four);

        SIN_INTERVALS_QUAD = [
            { label: "[0,pi/2]", a: zero, b: halfPi, aNum: 0, bNum: Math.PI / 2 },
            { label: "[0,pi]", a: zero, b: pi, aNum: 0, bNum: Math.PI },
            { label: "[0,2pi]", a: zero, b: twoPi, aNum: 0, bNum: 2 * Math.PI },
            { label: "[-pi,pi]", a: minusPi, b: pi, aNum: -Math.PI, bNum: Math.PI },
            { label: "[pi/4,5pi/4]", a: quarterPi, b: fivePiOverFour, aNum: Math.PI / 4, bNum: 5 * Math.PI / 4 },
        ];
    });

    // ------------------------------------------------------------
    // Main suite
    // ------------------------------------------------------------

    it("should execute the full deterministic accuracy and gas benchmark suite", async function () {
        const records: TestRecord[] = [];
        const [signer] = await ethers.getSigners();

        for (const fn of FUNCTIONS) {
            const selector = harness.interface.getFunction(fn.selectorName)?.selector;
            if (!selector) {
                throw new Error(`Selector not found for ${fn.selectorName}`);
            }

            if (fn.key === "f_sin") {
                for (const interval of SIN_INTERVALS_QUAD) {
                    for (const n of N_VALUES) {
                        for (const method of METHODS) {
                            const expected = fn.exactIntegral(interval.aNum, interval.bNum);

                            const txRequest = await harness.getFunction(method).populateTransaction(
                                target,
                                selector,
                                interval.a,
                                interval.b,
                                n
                            );

                            txRequest.to = target;

                            const estimatedGas = await ethers.provider.estimateGas({
                                ...txRequest,
                                from: await signer.getAddress(),
                            });

                            const response = await signer.sendTransaction({
                                ...txRequest,
                                gasLimit: estimatedGas + 100_000n,
                            });

                            const receipt = await response.wait();
                            if (!receipt) {
                                throw new Error("Transaction receipt is null");
                            }

                            const gasUsed = receipt.gasUsed;
                            const result = await harness[method](target, selector, interval.a, interval.b, n);
                            const actual = await quadToNumber(result);

                            const record: TestRecord = {
                                method,
                                func: fn.key,
                                funcLabel: fn.label,
                                intervalLabel: interval.label,
                                a: interval.aNum,
                                b: interval.bNum,
                                n,
                                expected,
                                actual,
                                absError: Math.abs(actual - expected),
                                relError: computeRelativeError(actual, expected),
                                estimatedGas,
                                gasUsed,
                            };

                            records.push(record);
                            printRecord(record);
                        }
                    }
                }
            } else {
                for (const interval of fn.intervals) {
                    const qa = await toQuad(interval.a);
                    const qb = await toQuad(interval.b);

                    for (const n of N_VALUES) {
                        for (const method of METHODS) {
                            const expected = fn.exactIntegral(interval.a, interval.b);

                            const txRequest = await harness.getFunction(method).populateTransaction(
                                target,
                                selector,
                                qa,
                                qb,
                                n
                            );

                            txRequest.to = target;

                            const estimatedGas = await ethers.provider.estimateGas({
                                ...txRequest,
                                from: await signer.getAddress(),
                            });

                            const response = await signer.sendTransaction({
                                ...txRequest,
                                gasLimit: estimatedGas + 100_000n,
                            });

                            const receipt = await response.wait();
                            if (!receipt) {
                                throw new Error("Transaction receipt is null");
                            }

                            const gasUsed = receipt.gasUsed;
                            const result = await harness[method](target, selector, qa, qb, n);
                            const actual = await quadToNumber(result);

                            const record: TestRecord = {
                                method,
                                func: fn.key,
                                funcLabel: fn.label,
                                intervalLabel: interval.label,
                                a: interval.a,
                                b: interval.b,
                                n,
                                expected,
                                actual,
                                absError: Math.abs(actual - expected),
                                relError: computeRelativeError(actual, expected),
                                estimatedGas,
                                gasUsed,
                            };

                            records.push(record);
                            printRecord(record);
                        }
                    }
                }
            }
        }

        printOverallSummary(records);
        printMethodSummaries(records);

        expect(records.length).to.equal(7 * 5 * 5 * 3); // 525 tests
    });
});