// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

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

/*──────────────────────────────────────────────────────────
    REPORT BLOCK (same as Integration + Polynomial)
──────────────────────────────────────────────────────────*/

function printBlock(
  n: number,
  method: string,
  explanation: string,
  gas: string,
  input: string,
  outHex: string,
  outDec: string
) {
  const sep = "-".repeat(60);
  console.log(
    `\n${sep}\nTest ${n}\nMethod: ${method}\nExplanation: ${explanation}\nGas Usage (estimate): ${gas}\nInput: ${input}\nOutput (hex): ${outHex}\nOutput: ${outDec}`
  );
}

/*──────────────────────────────────────────────────────────
    JS HELPERS (same semantics as your original)
──────────────────────────────────────────────────────────*/

const SCALE = 1e12;

// Convert JS float → quad input
async function toQuad(h: any, x: number): Promise<string> {
  const scaled = Math.round(x * SCALE);
  return h.fromFloat(BigInt(scaled));
}

async function fromQuad(h: any, q: string): Promise<number> {
  const scaled: bigint = await h.toFloat(q);
  return Number(scaled) / SCALE;
}

// Gas helpers (same as Integration style)
async function touchGas(h: any, method: string, args: any[]) {
  const data = h.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const tx = await signer.sendTransaction({
    to: await h.getAddress(),
    data
  });
  await tx.wait();
}

async function estimateGas(h: any, method: string, args: any[]) {
  const anyH = h as any;
  if (anyH[method]?.estimateGas) {
    return (await anyH[method].estimateGas(...args)).toString();
  }
  const data = h.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  return (await signer.estimateGas({ to: await h.getAddress(), data })).toString();
}

function flipSign(hex: string): string {
  const bn = BigInt(hex);
  const flipped = bn ^ (1n << 127n);
  return "0x" + flipped.toString(16).padStart(32, "0");
}

function approx(a: number, b: number, tol = 1e-9) {
  return Math.abs(a - b) <= tol;
}

/*──────────────────────────────────────────────────────────
    MAIN SUITE — EXACT STYLE MATCH WITH INTEGRATION REPORTS
──────────────────────────────────────────────────────────*/

describe("Trigonometry — Report (HEX + DEC + Identities + NaN)", function () {
  let h: TrigHarness;
  let t = 0;

  before(async () => {
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const math = await MathLibFactory.deploy();
    await math.waitForDeployment();

    const HF = await ethers.getContractFactory("TrigonometryHarness", {
      libraries: { MathLib: await math.getAddress() }
    });

    h = (await HF.deploy()) as unknown as TrigHarness;
    await h.waitForDeployment();
  });

  /*──────────────────────────────────────────────────────────
    1 — HEX: sin(0)
  ──────────────────────────────────────────────────────────*/
  it("1. HEX: sin(0)", async () => {
    t++;
    const q0 = await toQuad(h, 0);
    await touchGas(h, "sin", [q0]);
    const gas = await estimateGas(h, "sin", [q0]);
    const r = (await h.sin(q0)).toLowerCase();

    printBlock(
      t,
      "sin",
      "sin(0) must equal +0",
      gas,
      "x=0",
      r,
      "0"
    );

    expect(r).to.equal("0x00000000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
    2 — HEX: cos(0)
  ──────────────────────────────────────────────────────────*/
  it("2. HEX: cos(0)", async () => {
    t++;
    const q0 = await toQuad(h, 0);
    await touchGas(h, "cos", [q0]);
    const gas = await estimateGas(h, "cos", [q0]);
    const r = (await h.cos(q0)).toLowerCase();

    printBlock(
      t,
      "cos",
      "cos(0) must equal +1",
      gas,
      "x=0",
      r,
      "1"
    );

    expect(r).to.equal("0x3fff0000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
    3 — HEX: sin(pi/2)
  ──────────────────────────────────────────────────────────*/
  it("3. HEX: sin(pi/2)", async () => {
    t++;
    const q = await h.QHALF_PI();
    await touchGas(h, "sin", [q]);
    const gas = await estimateGas(h, "sin", [q]);
    const r = (await h.sin(q)).toLowerCase();

    printBlock(t, "sin", "sin(pi/2)=1", gas, "x=π/2", r, "1");
    expect(r).to.equal("0x3fff0000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
    4 — HEX: cos(pi/2)
  ──────────────────────────────────────────────────────────*/
  it("4. HEX: cos(pi/2)", async () => {
    t++;
    const q = await h.QHALF_PI();
    await touchGas(h, "cos", [q]);
    const gas = await estimateGas(h, "cos", [q]);
    const r = (await h.cos(q)).toLowerCase();

    printBlock(t, "cos", "cos(pi/2)=0", gas, "x=π/2", r, "0");
    expect(r).to.equal("0x00000000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
    5 — HEX: sin(pi)
  ──────────────────────────────────────────────────────────*/
  it("5. HEX: sin(pi)", async () => {
    t++;
    const q = await h.QPI();
    await touchGas(h, "sin", [q]);
    const gas = await estimateGas(h, "sin", [q]);
    const r = (await h.sin(q)).toLowerCase();

    printBlock(t, "sin", "sin(pi)=0", gas, "x=π", r, "0");
    expect(r).to.equal("0x00000000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
    6 — HEX: cos(pi)
  ──────────────────────────────────────────────────────────*/
  it("6. HEX: cos(pi)", async () => {
    t++;
    const q = await h.QPI();
    await touchGas(h, "cos", [q]);
    const gas = await estimateGas(h, "cos", [q]);
    const r = (await h.cos(q)).toLowerCase();

    printBlock(t, "cos", "cos(pi)=-1", gas, "x=π", r, "-1");
    expect(r).to.equal("0xbfff0000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
    7 — HEX: sin(-x) = -sin(x)
  ──────────────────────────────────────────────────────────*/
  it("7. HEX: sin(-x) symmetry", async () => {
    t++;
    const x = 0.345;
    const qx = await toQuad(h, x);
    const qn = await toQuad(h, -x);

    const pos = (await h.sin(qx)).toLowerCase();
    const neg = (await h.sin(qn)).toLowerCase();
    const expectedNeg = flipSign(pos);

    printBlock(t, "sin", "sin(-x) = -sin(x)", "~", `x=${x}`, neg, expectedNeg);
    expect(neg).to.equal(expectedNeg);
  });

  /*──────────────────────────────────────────────────────────
    8 — DEC: sin(x^2)
  ──────────────────────────────────────────────────────────*/
  it("8. DEC: sin(x^2)", async () => {
    t++;
    const x = 0.75;
    const qx = await toQuad(h, x);
    const qx2 = await h.mul(qx, qx);

    await touchGas(h, "sin", [qx2]);
    const gas = await estimateGas(h, "sin", [qx2]);

    const qs = await h.sin(qx2);
    const s = await fromQuad(h, qs);
    const expected = Math.sin(x * x);

    printBlock(t, "sin", "sin(x^2)", gas, `x=${x}`, qs, `${s}`);
    expect(approx(s, expected)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    9 — DEC: cos(x^2)
  ──────────────────────────────────────────────────────────*/
  it("9. DEC: cos(x^2)", async () => {
    t++;
    const x = -1.12;
    const qx = await toQuad(h, x);
    const qx2 = await h.mul(qx, qx);

    await touchGas(h, "cos", [qx2]);
    const gas = await estimateGas(h, "cos", [qx2]);

    const qc = await h.cos(qx2);
    const c = await fromQuad(h, qc);
    const expected = Math.cos(x * x);

    printBlock(t, "cos", "cos(x^2)", gas, `x=${x}`, qc, `${c}`);
    expect(approx(c, expected)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    10 — DEC: sin^2 + cos^2 = 1
  ──────────────────────────────────────────────────────────*/
  it("10. DEC: sin^2+cos^2", async () => {
    t++;
    const x = 1.234567;
    const qx = await toQuad(h, x);

    const qs = await h.sin(qx);
    const qc = await h.cos(qx);

    const qs2 = await h.mul(qs, qs);
    const qc2 = await h.mul(qc, qc);

    const sumQ = await h.add(qs2, qc2);
    const sum = await fromQuad(h, sumQ);

    printBlock(t, "identity", "sin^2+cos^2≈1", "~", `x=${x}`, sumQ, `${sum}`);
    expect(approx(sum, 1)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    11 — DEC: tan(x) ≈ sin(x)/cos(x)
  ──────────────────────────────────────────────────────────*/
  it("11. DEC: tan(x)", async () => {
    t++;
    const x = 0.42;
    const qx = await toQuad(h, x);

    await touchGas(h, "tan", [qx]);
    const gas = await estimateGas(h, "tan", [qx]);

    const tQ = await h.tan(qx);
    const tDec = await fromQuad(h, tQ);

    const s = await fromQuad(h, await h.sin(qx));
    const c = await fromQuad(h, await h.cos(qx));

    printBlock(t, "tan", "tan≈sin/cos", gas, `x=${x}`, tQ, `${tDec}`);
    expect(approx(tDec, s / c)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    12 — DEC: asin(sin(x))
  ──────────────────────────────────────────────────────────*/
  it("12. DEC: asin(sin(x))", async () => {
    t++;
    const x = 0.3;
    const qx = await toQuad(h, x);

    const s = await h.sin(qx);
    const a = await fromQuad(h, await h.asin(s));

    printBlock(t, "asin", "asin(sin(x))≈x", "~", `x=${x}`, s, `${a}`);
    expect(approx(a, x)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    13 — DEC: tan(atan(x))
  ──────────────────────────────────────────────────────────*/
  it("13. DEC: tan(atan(x))", async () => {
    t++;
    const x = 1.2345;
    const qx = await toQuad(h, x);

    const qa = await h.atan(qx);
    const tQ = await h.tan(qa);
    const tDec = await fromQuad(h, tQ);

    printBlock(t, "tan∘atan", "tan(atan(x))≈x", "~", `x=${x}`, tQ, `${tDec}`);
    expect(approx(tDec, x, 1e-6)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    14 — DEC: asin(x)+acos(x)=π/2
  ──────────────────────────────────────────────────────────*/
  it("14. DEC: asin+acos=π/2", async () => {
    t++;
    const x = 0.37;
    const qx = await toQuad(h, x);

    const qa = await h.asin(qx);
    const qb = await h.acos(qx);

    const a = await fromQuad(h, qa);
    const b = await fromQuad(h, qb);
    const sum = a + b;

    printBlock(t, "identity", "asin+acos=π/2", "~", `x=${x}`, qa, `${sum}`);
    expect(approx(sum, Math.PI / 2)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    15 — DEC: asin(-x)=-asin(x)
  ──────────────────────────────────────────────────────────*/
  it("15. DEC: asin(-x)", async () => {
    t++;
    const x = 0.4;
    const qx = await toQuad(h, x);
    const qn = await toQuad(h, -x);

    const aPos = await fromQuad(h, await h.asin(qx));
    const aNeg = await fromQuad(h, await h.asin(qn));

    printBlock(t, "asin", "odd symmetry", "~", `x=${x}`, qn, `${aNeg}`);
    expect(approx(aNeg, -aPos)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    16 — DEC: atan(-x)=-atan(x)
  ──────────────────────────────────────────────────────────*/
  it("16. DEC: atan(-x)", async () => {
    t++;
    const x = 0.9;
    const qx = await toQuad(h, x);
    const qn = await toQuad(h, -x);

    const aPos = await fromQuad(h, await h.atan(qx));
    const aNeg = await fromQuad(h, await h.atan(qn));

    printBlock(t, "atan", "odd symmetry", "~", `x=${x}`, qn, `${aNeg}`);
    expect(approx(aNeg, -aPos)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    17 — DEC: atan(1)=π/4
  ──────────────────────────────────────────────────────────*/
  it("17. DEC: atan(1)", async () => {
    t++;
    const q1 = await h.fromUInt(1);

    await touchGas(h, "atan", [q1]);
    const gas = await estimateGas(h, "atan", [q1]);

    const qa = await h.atan(q1);
    const a = await fromQuad(h, qa);

    printBlock(t, "atan", "atan(1)=π/4", gas, "x=1", qa, `${a}`);
    expect(approx(a, Math.PI / 4)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    18 — DEC: sin(x+2πk)
  ──────────────────────────────────────────────────────────*/
  it("18. DEC: sin periodic", async () => {
    t++;
    const x = 0.73;
    const k = 100;

    const base = await fromQuad(h, await h.sin(await toQuad(h, x)));
    const xp = x + 2 * Math.PI * k;

    const sp = await fromQuad(h, await h.sin(await toQuad(h, xp)));

    printBlock(t, "sin", "periodicity", "~", `x=${x},k=${k}`, "~", `${sp}`);
    expect(approx(sp, base)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    19 — DEC: cos periodic magnitude
  ──────────────────────────────────────────────────────────*/
  it("19. DEC: cos periodic magnitude", async () => {
    t++;
    const x = -1.11;
    const k = 80;

    const base = await fromQuad(h, await h.cos(await toQuad(h, x)));
    const xp = x + 2 * Math.PI * k;
    const cp = await fromQuad(h, await h.cos(await toQuad(h, xp)));

    printBlock(t, "cos", "periodic magnitude", "~", `x=${x},k=${k}`, "~", `${cp}`);
    expect(approx(Math.abs(cp), Math.abs(base))).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    20 — DEC: sin(x+π/2)=cos(x)
  ──────────────────────────────────────────────────────────*/
  it("20. DEC: sin(x+π/2)", async () => {
    t++;
    const x = 0.6;
    const qx = await toQuad(h, x);
    const qhp = await h.QHALF_PI();
    const qShift = await h.add(qx, qhp);

    const sShift = await fromQuad(h, await h.sin(qShift));
    const cBase = await fromQuad(h, await h.cos(qx));

    printBlock(t, "phase shift", "sin(x+π/2)=cos(x)", "~", `x=${x}`, "~", `${sShift}`);
    expect(approx(sShift, cBase)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    21 — DEC: cos^2(x+π/2)=sin^2(x)
  ──────────────────────────────────────────────────────────*/
  it("21. DEC: cos^2(x+π/2)", async () => {
    t++;
    const x = -0.8;
    const qx = await toQuad(h, x);
    const qhp = await h.QHALF_PI();
    const qShift = await h.add(qx, qhp);

    const cShift = await fromQuad(h, await h.cos(qShift));
    const sBase = await fromQuad(h, await h.sin(qx));

    printBlock(t, "phase shift", "cos^2(x+π/2)=sin^2(x)", "~", `x=${x}`, "~", `${cShift * cShift}`);
    expect(approx(cShift * cShift, sBase * sBase)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    22 — DEC: atan(tan(x))
  ──────────────────────────────────────────────────────────*/
  it("22. DEC: atan(tan(x))", async () => {
    t++;
    const x = 0.7;
    const qx = await toQuad(h, x);

    const qt = await h.tan(qx);
    const qa = await h.atan(qt);
    const back = await fromQuad(h, qa);

    printBlock(t, "atan∘tan", "atan(tan(x))≈x", "~", `x=${x}`, qa, `${back}`);
    expect(approx(back, x, 1e-6)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    23 — DEC: numerical derivative of sin
  ──────────────────────────────────────────────────────────*/
  it("23. DEC: derivative of sin", async () => {
    t++;
    const x = 0.5;
    const hStep = 1e-4;

    const qxp = await toQuad(h, x + hStep);
    const qxm = await toQuad(h, x - hStep);

    const sPlus = await fromQuad(h, await h.sin(qxp));
    const sMinus = await fromQuad(h, await h.sin(qxm));

    const numerical = (sPlus - sMinus) / (2 * hStep);
    const cExact = await fromQuad(h, await h.cos(await toQuad(h, x)));

    printBlock(
      t,
      "derivative",
      "d/dx sin(x)≈cos(x)",
      "~",
      `x=${x}`,
      "~",
      `${numerical}`
    );

    expect(approx(numerical, cExact, 1e-5)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    24 — HEX: tan(pi/2) → NaN
  ──────────────────────────────────────────────────────────*/
  it("24. HEX: tan(pi/2)=NaN", async () => {
    t++;
    const q = await h.QHALF_PI();
    const r = await h.tan(q);

    printBlock(t, "tan", "tan(π/2)=NaN", "~", "x=π/2", r, "NaN");
    expect(await h.isNaN(r)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
    25 — HEX: cot(0) → NaN
  ──────────────────────────────────────────────────────────*/
  it("25. HEX: cot(0)=NaN", async () => {
    t++;
    const q0 = await toQuad(h, 0);
    const r = await h.cot(q0);

    printBlock(t, "cot", "cot(0)=NaN", "~", "x=0", r, "NaN");
    expect(await h.isNaN(r)).to.be.true;
  });
});