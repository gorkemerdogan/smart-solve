// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import { ZeroHash, type Contract } from "ethers";

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

async function touchGas(harness: TrigHarness, method: string, args: any[]) {
    const data = harness.interface.encodeFunctionData(method, args);
    const [signer] = await ethers.getSigners();
    const to = await harness.getAddress();
    const tx = await signer.sendTransaction({ to, data });
    await tx.wait();
}

async function estimateGas(harness: TrigHarness, method: string, args: any[]) {
    const anyH = harness as any;
    if (anyH[method]?.estimateGas) {
        return (await anyH[method].estimateGas(...args)).toString();
    }
    const data = harness.interface.encodeFunctionData(method, args);
    const [signer] = await ethers.getSigners();
    const to = await harness.getAddress();
    const gas = await signer.estimateGas({ to, data });
    return gas.toString();
}

function printBlock({ t, method, explanation, gas, inHex, outHex, outDec }: any) {
    const sep = "-".repeat(60);
    const decLine = outDec ? `Output: ${outDec}` : "";
    
    console.log(`
        ${sep}
        Test ${t}
        Method: ${method}
        Explanation: ${explanation}
        Gas Usage: ${gas}
        Input: ${inHex}
        Output (hex): ${outHex}
        ${decLine}
`.trim());
}

// ------------------------------------------------------------
//  Math Helpers
// ------------------------------------------------------------

async function toQuad(h: TrigHarness, x: number): Promise<string> {
    const scaled = Math.round(x * SCALE);
    return h.fromFloat(BigInt(scaled));
}

async function fromQuad(h: TrigHarness, q: string): Promise<number> {
    const scaled: bigint = await h.toFloat(q);
    return Number(scaled) / SCALE;
}

function flipSign(hex: string): string {
    const bn = BigInt(hex);
    const flipped = bn ^ (1n << 127n);
    return "0x" + flipped.toString(16).padStart(32, "0");
}

function approx(a: number, b: number, tol = 1e-9) {
    return Math.abs(a - b) <= tol;
}

// ------------------------------------------------------------
//  Test Suite
// ------------------------------------------------------------

describe("Trigonometry — Report (HEX + DEC + Identities + NaN)", function () {
    let harness: TrigHarness;
    let t = 0;

    let QZERO = "0x00000000000000000000000000000000";
    let QONE = "0x3fff0000000000000000000000000000"

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

        printBlock({ t, method: "sin", explanation: "sin(0) must equal +0", gas, inHex: "x=0", outHex: r, outDec: "0" });
        expect(r).to.equal(QZERO);
    });

    it("Test 2: HEX cos(0)", async function () {
        t++;
        const q0 = await toQuad(harness, 0);

        await touchGas(harness, "cos", [q0]);
        const gas = await estimateGas(harness, "cos", [q0]);
        const r = (await harness.cos(q0)).toLowerCase();

        printBlock({ t, method: "cos", explanation: "cos(0) must equal +1", gas, inHex: "x=0", outHex: r, outDec: "1" });
        expect(r).to.equal(QONE);
    });

    it("Test 3: HEX sin(pi/2)", async function () {
        t++;
        const q = await harness.QHALF_PI();

        await touchGas(harness, "sin", [q]);
        const gas = await estimateGas(harness, "sin", [q]);
        const r = (await harness.sin(q)).toLowerCase();

        printBlock({ t, method: "sin", explanation: "sin(pi/2)=1", gas, inHex: "x=π/2", outHex: r, outDec: "1" });
        expect(r).to.equal(QONE);
    });

    it("Test 4: HEX cos(pi/2)", async function () {
        t++;
        const q = await harness.QHALF_PI();

        await touchGas(harness, "cos", [q]);
        const gas = await estimateGas(harness, "cos", [q]);
        const r = (await harness.cos(q)).toLowerCase();

        printBlock({ t, method: "cos", explanation: "cos(pi/2)=0", gas, inHex: "x=π/2", outHex: r, outDec: "0" });
        expect(r).to.equal(QZERO);
    });

    it("Test 5: HEX sin(pi)", async function () {
        t++;
        const q = await harness.QPI();

        await touchGas(harness, "sin", [q]);
        const gas = await estimateGas(harness, "sin", [q]);
        const r = (await harness.sin(q)).toLowerCase();

        printBlock({ t, method: "sin", explanation: "sin(pi)=0", gas, inHex: "x=π", outHex: r, outDec: "0" });
        expect(r).to.equal(QZERO);
    });

    it("Test 6: HEX cos(pi)", async function () {
        t++;
        const q = await harness.QPI();

        await touchGas(harness, "cos", [q]);
        const gas = await estimateGas(harness, "cos", [q]);
        const r = (await harness.cos(q)).toLowerCase();

        printBlock({ t, method: "cos", explanation: "cos(pi)=-1", gas, inHex: "x=π", outHex: r, outDec: "-1" });
        expect(r).to.equal("0xbfff0000000000000000000000000000");
    });

    // ------------------------------------------------------------
    //  Decimal & Identities
    // ------------------------------------------------------------

    it("Test 7: HEX sin(-x) symmetry", async function () {
        t++;
        const x = 0.345;
        const qx = await toQuad(harness, x);
        const qn = await toQuad(harness, -x);

        const pos = (await harness.sin(qx)).toLowerCase();
        
        await touchGas(harness, "sin", [qn]);
        const gas = await estimateGas(harness, "sin", [qn]);
        const neg = (await harness.sin(qn)).toLowerCase();
        
        const expectedNeg = flipSign(pos);
        const posHex = (await harness.sin(qx)).toLowerCase();
        const expectedNegDec = -(await fromQuad(harness, posHex));

        printBlock({ t, method: "sin", explanation: "sin(-x) = -sin(x)", gas, inHex: `x=${x}`, outHex: neg, outDec: expectedNegDec });
        expect(neg).to.equal(expectedNeg);
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

        printBlock({ t, method: "sin", explanation: "sin(x^2)", gas, inHex: `x=${x}`, outHex: qs, outDec: `${s}` });
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

        printBlock({ t, method: "cos", explanation: "cos(x^2)", gas, inHex: `x=${x}`, outHex: qc, outDec: `${c}` });
        expect(approx(c, expected)).to.be.true;
    });

    it("Test 10: DEC sin^2 + cos^2 = 1", async function () {
        t++;
        const x = 1.234567;
        const qx = await toQuad(harness, x);

        await touchGas(harness, "sin", [qx]); // Measure sin component
        const gas = await estimateGas(harness, "sin", [qx]);

        const qs = await harness.sin(qx);
        const qc = await harness.cos(qx);

        const qs2 = await harness.mul(qs, qs);
        const qc2 = await harness.mul(qc, qc);

        const sumQ = await harness.add(qs2, qc2);
        const sum = await fromQuad(harness, sumQ);

        printBlock({ t, method: "identity", explanation: "sin^2+cos^2≈1 (gas for sin)", gas, inHex: `x=${x}`, outHex: sumQ, outDec: `${sum}` });
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

        printBlock({ t, method: "tan", explanation: "tan≈sin/cos", gas, inHex: `x=${x}`, outHex: tQ, outDec: `${tDec}` });
        expect(approx(tDec, s / c)).to.be.true;
    });

    it("Test 12: DEC asin(sin(x))", async function () {
        t++;
        const x = 0.3;
        const qx = await toQuad(harness, x);
        const s = await harness.sin(qx);

        await touchGas(harness, "asin", [s]);
        const gas = await estimateGas(harness, "asin", [s]);
        const a = await fromQuad(harness, await harness.asin(s));

        printBlock({ t, method: "asin", explanation: "asin(sin(x))≈x", gas, inHex: `x=${x}`, outHex: s, outDec: `${a}` });
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

        printBlock({ t, method: "tan∘atan", explanation: "tan(atan(x))≈x", gas, inHex: `x=${x}`, outHex: tQ, outDec: `${tDec}` });
        expect(approx(tDec, x, 1e-6)).to.be.true;
    });

    it("Test 14: DEC asin(x)+acos(x)=π/2", async function () {
        t++;
        const x = 0.37;
        const qx = await toQuad(harness, x);

        await touchGas(harness, "asin", [qx]); // Measure asin component
        const gas = await estimateGas(harness, "asin", [qx]);

        const qa = await harness.asin(qx);
        const qb = await harness.acos(qx);

        const a = await fromQuad(harness, qa);
        const b = await fromQuad(harness, qb);
        const sum = a + b;

        printBlock({ t, method: "identity", explanation: "asin+acos=π/2 (gas for asin)", gas, inHex: `x=${x}`, outHex: qa, outDec: `${sum}` });
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
        const aNeg = await fromQuad(harness, await harness.asin(qn));

        printBlock({ t, method: "asin", explanation: "odd symmetry", gas, inHex: `x=${x}`, outHex: qn, outDec: `${aNeg}` });
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
        const aNeg = await fromQuad(harness, await harness.atan(qn));

        printBlock({ t, method: "atan", explanation: "odd symmetry", gas, inHex: `x=${x}`, outHex: qn, outDec: `${aNeg}` });
        expect(approx(aNeg, -aPos)).to.be.true;
    });

    it("Test 17: DEC atan(1)=π/4", async function () {
        t++;
        const q1 = await toQuad(harness, 1);

        await touchGas(harness, "atan", [q1]);
        const gas = await estimateGas(harness, "atan", [q1]);

        const qa = await harness.atan(q1);
        const a = await fromQuad(harness, qa);

        printBlock({ t, method: "atan", explanation: "atan(1)=π/4", gas, inHex: "x=1", outHex: qa, outDec: `${a}` });
        expect(approx(a, Math.PI / 4)).to.be.true;
    });

    // ------------------------------------------------------------
    //  Periodicity & Advanced
    // ------------------------------------------------------------

    it("Test 18: DEC sin periodic", async function () {
        t++;
        const x = 0.73;
        const k = 100;
        const base = await fromQuad(harness, await harness.sin(await toQuad(harness, x)));
        const xp = x + 2 * Math.PI * k;
        const qxp = await toQuad(harness, xp);

        await touchGas(harness, "sin", [qxp]);
        const gas = await estimateGas(harness, "sin", [qxp]);
        const sp = await fromQuad(harness, await harness.sin(qxp));

        printBlock({ t, method: "sin", explanation: "periodicity", gas, inHex: `x=${x},k=${k}`, outHex: "~", outDec: `${sp}` });
        expect(approx(sp, base)).to.be.true;
    });

    it("Test 19: DEC cos periodic magnitude", async function () {
        t++;
        const x = -1.11;
        const k = 80;
        const base = await fromQuad(harness, await harness.cos(await toQuad(harness, x)));
        const xp = x + 2 * Math.PI * k;
        const qxp = await toQuad(harness, xp);

        await touchGas(harness, "cos", [qxp]);
        const gas = await estimateGas(harness, "cos", [qxp]);
        const cp = await fromQuad(harness, await harness.cos(qxp));

        printBlock({ t, method: "cos", explanation: "periodic magnitude", gas, inHex: `x=${x},k=${k}`, outHex: "~", outDec: `${cp}` });
        expect(approx(Math.abs(cp), Math.abs(base))).to.be.true;
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

        printBlock({ t, method: "phase shift", explanation: "sin(x+π/2)=cos(x) (gas sin)", gas, inHex: `x=${x}`, outHex: "~", outDec: `${sShift}` });
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

        printBlock({ t, method: "phase shift", explanation: "cos^2(x+π/2)=sin^2(x) (gas cos)", gas, inHex: `x=${x}`, outHex: "~", outDec: `${cShift * cShift}` });
        expect(approx(cShift * cShift, sBase * sBase)).to.be.true;
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

        printBlock({ t, method: "atan∘tan", explanation: "atan(tan(x))≈x", gas, inHex: `x=${x}`, outHex: qa, outDec: `${back}` });
        expect(approx(back, x, 1e-6)).to.be.true;
    });

    it("Test 23: DEC derivative of sin", async function () {
        t++;
        const x = 0.5;
        const hStep = 1e-4;
        const qxp = await toQuad(harness, x + hStep);

        await touchGas(harness, "sin", [qxp]); // Measure one sin call
        const gas = await estimateGas(harness, "sin", [qxp]);

        const qxm = await toQuad(harness, x - hStep);
        const sPlus = await fromQuad(harness, await harness.sin(qxp));
        const sMinus = await fromQuad(harness, await harness.sin(qxm));

        const numerical = (sPlus - sMinus) / (2 * hStep);
        const cExact = await fromQuad(harness, await harness.cos(await toQuad(harness, x)));

        printBlock({ t, method: "derivative", explanation: "d/dx sin(x)≈cos(x) (gas sin)", gas, inHex: `x=${x}`, outHex: "~", outDec: `${numerical}` });
        expect(approx(numerical, cExact, 1e-5)).to.be.true;
    });

    it("Test 24: HEX tan(pi/2)=NaN", async function () {
        t++;
        const q = await harness.QHALF_PI();

        await touchGas(harness, "tan", [q]);
        const gas = await estimateGas(harness, "tan", [q]);
        const r = await harness.tan(q);

        printBlock({ t, method: "tan", explanation: "tan(π/2)=NaN", gas, inHex: "x=π/2", outHex: r, outDec: "NaN" });
        expect(await harness.isNaN(r)).to.be.true;
    });

    it("Test 25: HEX cot(0)=NaN", async function () {
        t++;
        const q0 = await toQuad(harness, 0);

        await touchGas(harness, "cot", [q0]);
        const gas = await estimateGas(harness, "cot", [q0]);
        const r = await harness.cot(q0);

        printBlock({ t, method: "cot", explanation: "cot(0)=NaN", gas, inHex: "x=0", outHex: r, outDec: "NaN" });
        expect(await harness.isNaN(r)).to.be.true;
    });
});