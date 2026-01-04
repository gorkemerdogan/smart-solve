// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { ZeroHash, type Contract } from "ethers";
import { touchGas, estimateGas, printBlockRegular } from "../test-utils";
import { numeric } from "../../typechain-types/facets";

// ------------------------------------------------------------
//  Types & Constants
// ------------------------------------------------------------

type TrigHarness = Contract & {
    fromFloat(x: bigint): Promise<string>;
    toFloat(q: string): Promise<bigint>;
    sin(x: string): Promise<string>;
    cos(x: string): Promise<string>;
    tan(x: string): Promise<string>;
    cot(x: string): Promise<string>;
    asin(x: string): Promise<string>;
    acos(x: string): Promise<string>;
    atan(x: string): Promise<string>;
    mul(a: string, b: string): Promise<string>;
    add(a: string, b: string): Promise<string>;
    QPI(): Promise<string>;
    QHALF_PI(): Promise<string>;
    isNaN(x: string): Promise<boolean>;
};

const SCALE = 1e12;

// ------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------

async function toQuad(h: TrigHarness, x: number): Promise<string> {
    const scaled = Math.round(x * SCALE);
    return h.fromFloat(BigInt(scaled));
}

async function fromQuad(h: TrigHarness, q: string): Promise<number> {
    const scaled: bigint = await h.toFloat(q);
    return Number(scaled) / SCALE;
}

// Rounds a numerical result to 12 decimal places and removes trailing zeros for clean reporting
function formatDec(val: number): string {
    if (Object.is(val, NaN)) return "NaN";
    return parseFloat(val.toFixed(12)).toString();
}

function flipSign(hex: string): string {
    const bn = BigInt(hex);
    const flipped = bn ^ (1n << 127n);
    return "0x" + flipped.toString(16).padStart(32, "0");
}

// function approx(a: number, b: number, tol = 1e-10) {
//     return Math.abs(a - b) <= tol;
// }

function approx(a: number, b: number, tol = 1e-11) {
    if (a === b) return true;
    const diff = Math.abs(a - b);
    return diff / (Math.abs(a) + Math.abs(b)) < tol;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Trigonometry — Report (HEX + DEC + Identities + NaN)", function () {
    let harness: TrigHarness;
    let t = 0;

    let QZERO: string;
    let QONE: string;

    before(async () => {
        const MathLibFactory = await ethers.getContractFactory("MathLib");
        const math = await MathLibFactory.deploy();
        await math.waitForDeployment();
        const mathAddr = await math.getAddress();

        const HF = await ethers.getContractFactory("TrigonometryHarness", {
            libraries: { "contracts/libraries/MathLib.sol:MathLib": mathAddr },
        });

        harness = (await HF.deploy()) as unknown as TrigHarness;
        await harness.waitForDeployment();

        QZERO = await harness.fromFloat(0n);
        QONE  = await harness.fromFloat(BigInt(SCALE));
    });

    // ------------------------------------------------------------
    //  Exact Hex Tests
    // ------------------------------------------------------------

    it("Test 1: HEX sin(0)", async function () {
        t++;
        const q0 = await toQuad(harness, 0);

        await touchGas(harness, "sin", [q0]);
        const gas = await estimateGas(harness, "sin", [q0]);
        const r = (await harness.sin(q0)).toLowerCase();

        const expectedDec = await fromQuad(harness, QZERO);
        const outDec = await fromQuad(harness, r);

        printBlockRegular({
            t,
            method: "sin",
            explanation: "sin(0) = 0.0",
            inHex: "x=0",
            expectedHex: QZERO,
            outHex: r,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(outDec),
            gas,
        });
        expect(r).to.equal(QZERO);
    });

    it("Test 2: HEX cos(0)", async function () {
        t++;
        const q0 = await toQuad(harness, 0);

        await touchGas(harness, "cos", [q0]);
        const gas = await estimateGas(harness, "cos", [q0]);
        const r = (await harness.cos(q0)).toLowerCase();

        const expectedDec = await fromQuad(harness, QONE);
        const outDec = await fromQuad(harness, r);

        printBlockRegular({
            t,
            method: "cos",
            explanation: "cos(0) = +1.0",
            inHex: "x=0",
            expectedHex: QONE,
            outHex: r,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(outDec),
            gas,
        });
        expect(r).to.equal(QONE);
    });

    it("Test 3: HEX sin(pi/2)", async function () {
        t++;
        const q = await harness.QHALF_PI();

        await touchGas(harness, "sin", [q]);
        const gas = await estimateGas(harness, "sin", [q]);
        const r = (await harness.sin(q)).toLowerCase();

        const expectedDec = await fromQuad(harness, QONE);
        const outDec = await fromQuad(harness, r);

        printBlockRegular({
            t,
            method: "sin",
            explanation: "sin(π/2) = +1.0",
            inHex: "x=π/2",
            expectedHex: QONE,
            outHex: r,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(outDec),
            gas,
        });
        expect(r).to.equal(QONE);
    });

    it("Test 4: HEX cos(pi/2)", async function () {
        t++;
        const q = await harness.QHALF_PI();

        await touchGas(harness, "cos", [q]);
        const gas = await estimateGas(harness, "cos", [q]);
        const r = (await harness.cos(q)).toLowerCase();

        const expectedDec = await fromQuad(harness, QZERO);
        const outDec = await fromQuad(harness, r);

        printBlockRegular({
            t,
            method: "cos",
            explanation: "cos(π/2) = 0.0",
            inHex: "x=π/2",
            expectedHex: QZERO,
            outHex: r,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(outDec),
            gas,
        });
        expect(r).to.equal(QZERO);
    });

    it("Test 5: HEX sin(pi)", async function () {
        t++;
        const q = await harness.QPI();

        await touchGas(harness, "sin", [q]);
        const gas = await estimateGas(harness, "sin", [q]);
        const r = (await harness.sin(q)).toLowerCase();

        const expectedDec = await fromQuad(harness, QZERO);
        const outDec = await fromQuad(harness, r);

        printBlockRegular({
            t,
            method: "sin",
            explanation: "sin(π) = 0",
            inHex: "x=π",
            expectedHex: QZERO,
            outHex: r,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(outDec),
            gas,
        });
        expect(r).to.equal(QZERO);
    });

    it("Test 6: HEX cos(pi)", async function () {
        t++;
        const q = await harness.QPI();

        await touchGas(harness, "cos", [q]);
        const gas = await estimateGas(harness, "cos", [q]);
        const r = (await harness.cos(q)).toLowerCase();

        const QNEG_ONE = flipSign(QONE);
        const expectedDec = await fromQuad(harness, QNEG_ONE);
        const outDec = await fromQuad(harness, r);

        printBlockRegular({
            t,
            method: "cos",
            explanation: "cos(π) = −1.0",
            inHex: "x=π",
            expectedHex: QNEG_ONE,
            outHex: r,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(outDec),
            gas,
        });
        expect(r).to.equal(QNEG_ONE);
    });

    // ------------------------------------------------------------
    //  Decimal & Identities
    // ------------------------------------------------------------

    it("Test 7: HEX sin(-x) symmetry", async function () {
        t++;
        const x = 0.345;
        const qx = await toQuad(harness, x);
        const qn = await toQuad(harness, -x);

        const posHex = (await harness.sin(qx)).toLowerCase();
        const posDec = await fromQuad(harness, posHex);

        await touchGas(harness, "sin", [qn]);
        const gas = await estimateGas(harness, "sin", [qn]);
        const negHex = (await harness.sin(qn)).toLowerCase();
        const negDec = await fromQuad(harness, negHex);

        const expectedNegHex = flipSign(posHex);
        const expectedNegDec = -posDec;

        printBlockRegular({
            t,
            method: "sin",
            explanation: "Verifies sin(−x) is bit-wise sign-flipped sin(x), and numerically −sin(x).",
            inHex: `x=${x}`,
            expectedHex: expectedNegHex,
            outHex: negHex,
            expectedDec: formatDec(expectedNegDec),
            outDec: formatDec(negDec),
            gas,
        });
        expect(negHex).to.equal(expectedNegHex);
    });

    it("Test 8: DEC sin(x^2)", async function () {
        t++;
        const x = 0.75;
        const qx = await toQuad(harness, x);
        const qx2 = await harness.mul(qx, qx);

        await touchGas(harness, "sin", [qx2]);
        const gas = await estimateGas(harness, "sin", [qx2]);

        const qs = await harness.sin(qx2);
        const s = await fromQuad(harness, qs);
        const expected = Math.sin(x * x);

        printBlockRegular({
            t,
            method: "sin",
            explanation: "Evaluates sin(x²) in quad and compares to JS Math.sin(x*x) in double precision.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: qs,
            expectedDec: formatDec(expected),
            outDec: formatDec(s),
            gas,
        });
        expect(approx(s, expected)).to.be.true;
    });

    it("Test 9: DEC cos(x^2)", async function () {
        t++;
        const x = -1.12;
        const qx = await toQuad(harness, x);
        const qx2 = await harness.mul(qx, qx);

        await touchGas(harness, "cos", [qx2]);
        const gas = await estimateGas(harness, "cos", [qx2]);

        const qc = await harness.cos(qx2);
        const c = await fromQuad(harness, qc);
        const expected = Math.cos(x * x);

        printBlockRegular({
            t,
            method: "cos",
            explanation: "Computes cos(x²) in quad and checks closeness to Math.cos(x*x).",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: qc,
            expectedDec: formatDec(expected),
            outDec: formatDec(c),
            gas,
        });
        expect(approx(c, expected)).to.be.true;
    });

    it("Test 10: DEC sin^2 + cos^2 = 1", async function () {
        t++;
        const x = 1.234567;
        const qx = await toQuad(harness, x);

        await touchGas(harness, "sin", [qx]); // measure cost of one sin
        const gas = await estimateGas(harness, "sin", [qx]);

        const qs = await harness.sin(qx);
        const qc = await harness.cos(qx);

        const qs2 = await harness.mul(qs, qs);
        const qc2 = await harness.mul(qc, qc);

        const sumQ = await harness.add(qs2, qc2);
        const sum = await fromQuad(harness, sumQ);

        const expectedDec = await fromQuad(harness, QONE);

        printBlockRegular({
            t,
            method: "identity",
            explanation: "Validates sin²(x)+cos²(x)≈1 using quad arithmetic (gas reported for sin).",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: sumQ,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(sum),
            gas,
        });
        expect(approx(sum, 1)).to.be.true;
    });

    it("Test 11: DEC tan(x) ≈ sin(x)/cos(x)", async function () {
        t++;
        const x = 0.42;
        const qx = await toQuad(harness, x);

        await touchGas(harness, "tan", [qx]);
        const gas = await estimateGas(harness, "tan", [qx]);

        const tQ = await harness.tan(qx);
        const tDec = await fromQuad(harness, tQ);

        const s = await fromQuad(harness, await harness.sin(qx));
        const c = await fromQuad(harness, await harness.cos(qx));
        const expected = s / c;

        printBlockRegular({
            t,
            method: "tan",
            explanation: "tan(x) from library should numerically match sin(x)/cos(x) in quad.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: tQ,
            expectedDec: formatDec(expected),
            outDec: formatDec(tDec),
            gas,
        });
        expect(approx(tDec, expected)).to.be.true;
    });

    it("Test 12: DEC asin(sin(x))", async function () {
        t++;
        const x = 0.3;
        const qx = await toQuad(harness, x);
        const s = await harness.sin(qx);

        await touchGas(harness, "asin", [s]);
        const gas = await estimateGas(harness, "asin", [s]);
        const a = await fromQuad(harness, await harness.asin(s));

        printBlockRegular({
            t,
            method: "asin",
            explanation: "asin(sin(x)) should return x for |x|≤π/2.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: s,
            expectedDec: formatDec(x),
            outDec: formatDec(a),
            gas,
        });
        expect(approx(a, x)).to.be.true;
    });

    it("Test 13: DEC tan(atan(x))", async function () {
        t++;
        const x = 1.2345;
        const qx = await toQuad(harness, x);
        const qa = await harness.atan(qx);

        await touchGas(harness, "tan", [qa]);
        const gas = await estimateGas(harness, "tan", [qa]);

        const tQ = await harness.tan(qa);
        const tDec = await fromQuad(harness, tQ);

        printBlockRegular({
            t,
            method: "tan∘atan",
            explanation: "Confirms tan(atan(x)) numerically returns x within tolerance.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: tQ,
            expectedDec: formatDec(x),
            outDec: formatDec(tDec),
            gas,
        });
        expect(approx(tDec, x, 1e-6)).to.be.true;
    });

    it("Test 14: DEC asin(x)+acos(x)=π/2", async function () {
        t++;
        const x = 0.37;
        const qx = await toQuad(harness, x);

        await touchGas(harness, "asin", [qx]); // measure asin cost
        const gas = await estimateGas(harness, "asin", [qx]);

        const qa = await harness.asin(qx);
        const qb = await harness.acos(qx);

        const a = await fromQuad(harness, qa);
        const b = await fromQuad(harness, qb);
        const sum = a + b;

        const expectedDec = await fromQuad(harness, await harness.QHALF_PI());

        printBlockRegular({
            t,
            method: "identity",
            explanation: "asin(x)+acos(x) should sum to π/2 for |x|≤1.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: qa,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(sum),
            gas,
        });
        expect(approx(sum, Math.PI / 2)).to.be.true;
    });

    it("Test 15: DEC asin(-x)=-asin(x)", async function () {
        t++;
        const x = 0.4;
        const qx = await toQuad(harness, x);
        const qn = await toQuad(harness, -x);

        const aPos = await fromQuad(harness, await harness.asin(qx));

        await touchGas(harness, "asin", [qn]);
        const gas = await estimateGas(harness, "asin", [qn]);
        const qNeg = await harness.asin(qn);
        const aNeg = await fromQuad(harness, qNeg);

        printBlockRegular({
            t,
            method: "asin",
            explanation: "asin(−x) should be −asin(x).",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: qNeg,
            expectedDec: formatDec(-aPos),
            outDec: formatDec(aNeg),
            gas,
        });
        expect(approx(aNeg, -aPos)).to.be.true;
    });

    it("Test 16: DEC atan(-x)=-atan(x)", async function () {
        t++;
        const x = 0.9;
        const qx = await toQuad(harness, x);
        const qn = await toQuad(harness, -x);

        const aPos = await fromQuad(harness, await harness.atan(qx));

        await touchGas(harness, "atan", [qn]);
        const gas = await estimateGas(harness, "atan", [qn]);
        const qNeg = await harness.atan(qn);
        const aNeg = await fromQuad(harness, qNeg);

        printBlockRegular({
            t,
            method: "atan",
            explanation: "atan(−x) must mirror atan(x) with opposite sign.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: qNeg,
            expectedDec: formatDec(-aPos),
            outDec: formatDec(aNeg),
            gas,
        });
        expect(approx(aNeg, -aPos)).to.be.true;
    });

    it("Test 17: DEC atan(1)=π/4", async function () {
        t++;
        const q1 = await toQuad(harness, 1);

        await touchGas(harness, "atan", [q1]);
        const gas = await estimateGas(harness, "atan", [q1]);

        const qa = await harness.atan(q1);
        const a = await fromQuad(harness, qa);

        const expectedDec = Math.PI / 4;

        printBlockRegular({
            t,
            method: "atan",
            explanation: "atan(1) should yield π/4, a classic sanity check for inverse tangent.",
            inHex: "x=1",
            expectedHex: "~",
            outHex: qa,
            expectedDec: formatDec(expectedDec),
            outDec: formatDec(a),
            gas,
        });
        expect(approx(a, Math.PI / 4)).to.be.true;
    });

    // ------------------------------------------------------------
    //  Periodicity & Advanced
    // ------------------------------------------------------------

    it("Test 18: DEC sin periodic", async function () {
        t++;
        const x = 0.73;
        const k = 100;
        const base = await fromQuad(
            harness,
            await harness.sin(await toQuad(harness, x)),
        );
        const xp = x + 2 * Math.PI * k;
        const qxp = await toQuad(harness, xp);

        await touchGas(harness, "sin", [qxp]);
        const gas = await estimateGas(harness, "sin", [qxp]);
        const sp = await fromQuad(harness, await harness.sin(qxp));

        printBlockRegular({
            t,
            method: "sin",
            explanation: "Checks sin(x+2πk) remains numerically equal to sin(x) for large integer k.",
            inHex: `x=${x}, k=${k}`,
            expectedHex: "~",
            outHex: "~",
            expectedDec: formatDec(base),
            outDec: formatDec(sp),
            gas,
        });
        expect(approx(sp, base)).to.be.true;
    });

    it("Test 19: DEC cos periodic magnitude", async function () {
        t++;
        const x = -1.11;
        const k = 100;
        const base = await fromQuad(
            harness,
            await harness.cos(await toQuad(harness, x)),
        );
        const xp = x + 2 * Math.PI * k;
        const qxp = await toQuad(harness, xp);

        await touchGas(harness, "cos", [qxp]);
        const gas = await estimateGas(harness, "cos", [qxp]);
        const cp = await fromQuad(harness, await harness.cos(qxp));

        const baseAbs = Math.abs(base);
        const cpAbs = Math.abs(cp);

        printBlockRegular({
            t,
            method: "cos",
            explanation: "|cos(x+2πk)| should match |cos(x)| even for large k.",
            inHex: `x=${x}, k=${k}`,
            expectedHex: "~",
            outHex: "~",
            expectedDec: formatDec(baseAbs),
            outDec: formatDec(cpAbs),
            gas,
        });
        expect(approx(cpAbs, baseAbs)).to.be.true;
    });

    it("Test 20: DEC sin(x+π/2)=cos(x)", async function () {
        t++;
        const x = 0.6;
        const qx = await toQuad(harness, x);
        const qhp = await harness.QHALF_PI();
        const qShift = await harness.add(qx, qhp);

        await touchGas(harness, "sin", [qShift]);
        const gas = await estimateGas(harness, "sin", [qShift]);

        const sShift = await fromQuad(harness, await harness.sin(qShift));
        const cBase = await fromQuad(harness, await harness.cos(qx));

        printBlockRegular({
            t,
            method: "phase shift",
            explanation: "check if sin(x+π/2) aligns with cos(x).",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: "~",
            expectedDec: formatDec(cBase),
            outDec: formatDec(sShift),
            gas,
        });
        expect(approx(sShift, cBase)).to.be.true;
    });

    it("Test 21: DEC cos^2(x+π/2)=sin^2(x)", async function () {
        t++;
        const x = -0.8;
        const qx = await toQuad(harness, x);
        const qhp = await harness.QHALF_PI();
        const qShift = await harness.add(qx, qhp);

        await touchGas(harness, "cos", [qShift]);
        const gas = await estimateGas(harness, "cos", [qShift]);

        const cShift = await fromQuad(harness, await harness.cos(qShift));
        const sBase = await fromQuad(harness, await harness.sin(qx));

        const expected = sBase * sBase;
        const actual = cShift * cShift;

        printBlockRegular({
            t,
            method: "phase shift",
            explanation: "cos²(x+π/2) should equal sin²(x).",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: "~",
            expectedDec: formatDec(expected),
            outDec: formatDec(actual),
            gas,
        });
        expect(approx(actual, expected)).to.be.true;
    });

    it("Test 22: DEC atan(tan(x))", async function () {
        t++;
        const x = 0.7;
        const qx = await toQuad(harness, x);
        const qt = await harness.tan(qx);

        await touchGas(harness, "atan", [qt]);
        const gas = await estimateGas(harness, "atan", [qt]);

        const qa = await harness.atan(qt);
        const back = await fromQuad(harness, qa);

        printBlockRegular({
            t,
            method: "atan∘tan",
            explanation: "atan(tan(x)) should recover x as long as x stays within the branch.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: qa,
            expectedDec: formatDec(x),
            outDec: formatDec(back),
            gas,
        });
        expect(approx(back, x, 1e-6)).to.be.true;
    });

    it("Test 23: DEC derivative of sin", async function () {
        t++;
        const x = 0.5;
        const hStep = 1e-4;
        const qxp = await toQuad(harness, x + hStep);

        await touchGas(harness, "sin", [qxp]); // measure a single sin call
        const gas = await estimateGas(harness, "sin", [qxp]);

        const qxm = await toQuad(harness, x - hStep);
        const sPlus = await fromQuad(harness, await harness.sin(qxp));
        const sMinus = await fromQuad(harness, await harness.sin(qxm));

        const numerical = (sPlus - sMinus) / (2 * hStep);
        const cExact = await fromQuad(harness, await harness.cos(await toQuad(harness, x)));

        printBlockRegular({
            t,
            method: "derivative",
            explanation: "Verifies d/dx sin(x) ≈ cos(x) using small symmetric step h.",
            inHex: `x=${x}`,
            expectedHex: "~",
            outHex: "~",
            expectedDec: formatDec(cExact),
            outDec: formatDec(numerical),
            gas,
        });
        expect(approx(numerical, cExact, 1e-5)).to.be.true;
    });

    it("Test 24: HEX tan(pi/2)=NaN", async function () {
        t++;
        const q = await harness.QHALF_PI();

        await touchGas(harness, "tan", [q]);
        const gas = await estimateGas(harness, "tan", [q]);
        const r = await harness.tan(q);

        printBlockRegular({
            t,
            method: "tan",
            explanation: "tan(π/2) should not produce a finite number but an internal NaN sentinel.",
            inHex: "x=π/2",
            expectedHex: "NaN",
            outHex: r,
            expectedDec: "NaN",
            outDec: "NaN",
            gas,
        });
        expect(await harness.isNaN(r)).to.be.true;
    });

    it("Test 25: HEX cot(0)=NaN", async function () {
        t++;
        const q0 = await toQuad(harness, 0);

        await touchGas(harness, "cot", [q0]);
        const gas = await estimateGas(harness, "cot", [q0]);
        const r = await harness.cot(q0);

        printBlockRegular({
            t,
            method: "cot",
            explanation: "cot(0)=cos(0)/sin(0) should trigger NaN instead of infinite magnitude.",
            inHex: "x=0",
            expectedHex: "NaN",
            outHex: r,
            expectedDec: "NaN",
            outDec: "NaN",
            gas,
        });
        expect(await harness.isNaN(r)).to.be.true;
    });
});