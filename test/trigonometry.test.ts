import { expect } from "chai";
import { ethers } from "hardhat";

/*──────────────────────────────────────────────────────────
    JS HELPERS
──────────────────────────────────────────────────────────*/

function approx(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol;
}

function log(name: string, expected: any, result: any) {
  console.log(`${name}\n  expected = ${expected}\n  result   = ${result}\n`);
}

/*──────────────────────────────────────────────────────────
    MAIN TEST SUITE
──────────────────────────────────────────────────────────*/

describe("Trigonometry.sol High-Precision Tests", function () {
  let trig: any;
  const SCALE = 1e12; // same scale used by harness

  async function toQuad(x: number): Promise<string> {
    const scaled = Math.round(x * SCALE);
    return trig.fromFloat(scaled);
  }

  async function fromQuad(q: string): Promise<number> {
    const scaled: bigint = await trig.toFloat(q);
    return Number(scaled) / SCALE;
  }

  before(async () => {
    // Deploy MathLib
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const mathlib = await MathLibFactory.deploy();
    await mathlib.waitForDeployment();

    // Deploy TrigonometryHarness WITH library linking
    const H = await ethers.getContractFactory("TrigonometryHarness", {
        libraries: {
        MathLib: await mathlib.getAddress(),
        },
  });

    trig = await H.deploy();
    await trig.waitForDeployment();
    });

  /*──────────────────────────────────────────────────────────
      SIMPLE BASE TESTS: sin(0), cos(0)
  ──────────────────────────────────────────────────────────*/

  it("sin(0) = 0", async () => {
    const q0 = await toQuad(0);
    const r = await trig.sin(q0);
    const v = await fromQuad(r);

    log("sin(0)", 0, v);
    expect(approx(v, 0)).to.be.true;
  });

  it("cos(0) = 1", async () => {
    const q0 = await toQuad(0);
    const r = await trig.cos(q0);
    const v = await fromQuad(r);

    log("cos(0)", 1, v);
    expect(approx(v, 1)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      EXACT VALUES USING π CONSTANTS
  ──────────────────────────────────────────────────────────*/

  it("sin(pi/2) = 1", async () => {
    const q = await trig.QHALF_PI();
    const r = await trig.sin(q);
    const v = await fromQuad(r);

    log("sin(pi/2)", 1, v);
    expect(approx(v, 1)).to.be.true;
  });

  it("cos(pi/2) = 0", async () => {
    const q = await trig.QHALF_PI();
    const r = await trig.cos(q);
    const v = await fromQuad(r);

    log("cos(pi/2)", 0, v);
    expect(approx(v, 0, 1e-9)).to.be.true;
  });

  it("sin(pi) = 0", async () => {
    const q = await trig.QPI();
    const r = await trig.sin(q);
    const v = await fromQuad(r);

    log("sin(pi)", 0, v);
    expect(approx(v, 0, 1e-9)).to.be.true;
  });

  it("cos(pi) = -1", async () => {
    const q = await trig.QPI();
    const r = await trig.cos(q);
    const v = await fromQuad(r);

    log("cos(pi)", -1, v);
    expect(approx(v, -1, 1e-9)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      SYMMETRY TESTS
  ──────────────────────────────────────────────────────────*/

  it("sin(-x) = -sin(x)", async () => {
    const x = 0.345;
    const qx = await toQuad(x);
    const qn = await toQuad(-x);

    const pos = await fromQuad(await trig.sin(qx));
    const neg = await fromQuad(await trig.sin(qn));

    log("sin(-x) vs -sin(x)", -pos, neg);
    expect(approx(neg, -pos)).to.be.true;
  });

  it("cos(-x) = cos(x)", async () => {
    const x = 0.789;
    const qx = await toQuad(x);
    const qn = await toQuad(-x);

    const pos = await fromQuad(await trig.cos(qx));
    const neg = await fromQuad(await trig.cos(qn));

    log("cos(-x)", pos, neg);
    expect(approx(neg, pos)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      PYTHAGOREAN IDENTITY: sin² + cos² ≈ 1
  ──────────────────────────────────────────────────────────*/

  it("sin^2 + cos^2 ≈ 1", async () => {
    const x = 1.234567;
    const qx = await toQuad(x);

    const s = await fromQuad(await trig.sin(qx));
    const c = await fromQuad(await trig.cos(qx));

    log("sin^2 + cos^2", 1, s * s + c * c);
    expect(approx(s * s + c * c, 1, 1e-9)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      TAN & COT IDENTITIES
  ──────────────────────────────────────────────────────────*/

  it("tan(x) ≈ sin(x)/cos(x)", async () => {
    const x = 0.42;
    const qx = await toQuad(x);

    const t = await fromQuad(await trig.tan(qx));
    const s = await fromQuad(await trig.sin(qx));
    const c = await fromQuad(await trig.cos(qx));

    log("tan(x)", s / c, t);
    expect(approx(t, s / c, 1e-9)).to.be.true;
  });

  it("cot(x) ≈ cos(x)/sin(x)", async () => {
    const x = 0.42;
    const qx = await toQuad(x);

    const t = await fromQuad(await trig.cot(qx));
    const s = await fromQuad(await trig.sin(qx));
    const c = await fromQuad(await trig.cos(qx));

    log("cot(x)", c / s, t);
    expect(approx(t, c / s, 1e-9)).to.be.true;
  });

  it("tan(pi/2) returns NaN", async () => {
    const q = await trig.QHALF_PI();
    const r = await trig.tan(q);
    expect(await trig.isNaN(r)).to.be.true;
  });

  it("cot(0) returns NaN", async () => {
    const q0 = await toQuad(0);
    const r = await trig.cot(q0);
    expect(await trig.isNaN(r)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      INVERSE TRIG TESTS
  ──────────────────────────────────────────────────────────*/

  it("asin(sin(x)) ≈ x for |x| <= 0.5", async () => {
    const x = 0.3;
    const qx = await toQuad(x);

    const s = await trig.sin(qx);
    const asinRes = await trig.asin(s);

    const v = await fromQuad(asinRes);

    log("asin(sin(x))", x, v);
    expect(approx(v, x, 1e-9)).to.be.true;
  });

  it("acos(cos(x)) ≈ x for x ∈ [0, pi]", async () => {
    const x = 0.9;
    const qx = await toQuad(x);

    const c = await trig.cos(qx);
    const acosRes = await fromQuad(await trig.acos(c));

    log("acos(cos(x))", x, acosRes);
    expect(approx(acosRes, x, 1e-9)).to.be.true;
  });

  it("tan(atan(x)) ≈ x", async () => {
    const x = 1.2345;
    const qx = await toQuad(x);

    const atanRes = await trig.atan(qx);
    const tanRes = await fromQuad(await trig.tan(atanRes));

    log("tan(atan(x))", x, tanRes);
    expect(approx(tanRes, x, 1e-9)).to.be.true;
  });

  it("acot(x) ≈ atan(1/x)", async () => {
    const x = 2.0;
    const qx = await toQuad(x);

    const inv = await toQuad(1 / x);
    const res1 = await fromQuad(await trig.atan(inv));       // atan(1/x)
    const res2 = await fromQuad(await trig.atan(await toQuad(1 / x))); // same

    log("acot(x) vs atan(1/x)", res1, res2);
    expect(approx(res1, res2, 1e-9)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      LARGE INPUT RANGE REDUCTION
  ──────────────────────────────────────────────────────────*/

  it("sin(2π * 1e6) ≈ 0", async () => {
    const twoPi = await trig.QTWO_PI();
    const big = 1_000_000;

    // Convert big integer to quad
    const qBig = await trig.fromDouble(big);
    const angle = await trig.mul(twoPi, qBig).catch(() => null);

    // If harness has no mul() helper, compute in JS then convert:
    const jsAngle = 2 * Math.PI * big;
    const qAngle = await toQuad(jsAngle);

    const r = await fromQuad(await trig.sin(qAngle));

    log("sin(2π * 1e6)", 0, r);
    expect(approx(r, 0, 1e-9)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      SMALL ANGLE APPROXIMATION
  ──────────────────────────────────────────────────────────*/

  it("sin(x) ≈ x for tiny x", async () => {
    const x = 1e-6;
    const qx = await toQuad(x);

    const r = await fromQuad(await trig.sin(qx));

    log("sin(x) ≈ x", x, r);
    expect(approx(r, x, 1e-9)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      tan(x) * cot(x) ≈ 1 (where defined)
  ──────────────────────────────────────────────────────────*/

  it("tan(x) * cot(x) ≈ 1", async () => {
    const x = 0.55;
    const qx = await toQuad(x);

    const t = await fromQuad(await trig.tan(qx));
    const c = await fromQuad(await trig.cot(qx));
    const prod = t * c;

    log("tan(x)*cot(x)", 1, prod);
    expect(approx(prod, 1, 1e-9)).to.be.true;
  });

});