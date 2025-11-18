import { expect } from "chai";
import { ethers } from "hardhat";

/*──────────────────────────────────────────────────────────
    JS HELPERS
──────────────────────────────────────────────────────────*/

const SCALE = 1e12;

// Convert JS float → quad input using SCALE
async function toQuad(trig: any, x: number): Promise<string> {
  const scaled = Math.round(x * SCALE);
  return trig.fromFloat(scaled);
}

// Convert quad → JS float using SCALE
async function fromQuad(trig: any, q: string): Promise<number> {
  const scaled: bigint = await trig.toFloat(q);
  return Number(scaled) / SCALE;
}

// Hex logger
function logHex(name: string, expected: string | null, result: string) {
  console.log(`${name}\n  expected = ${expected}\n  result   = ${result}\n`);
}

// Decimal logger
function log(name: string, expected: any, result: any) {
  console.log(`${name}\n  expected = ${expected}\n  result   = ${result}\n`);
}

function approx(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol;
}

/*──────────────────────────────────────────────────────────
    MAIN TEST SUITE
──────────────────────────────────────────────────────────*/

describe("Trigonometry.sol HEX + DEC Tests", function () {
  let trig: any;

  before(async () => {
    // Deploy MathLib
    const MathLibFactory = await ethers.getContractFactory("MathLib");
    const mathlib = await MathLibFactory.deploy();
    await mathlib.waitForDeployment();

    // Deploy Harness
    const H = await ethers.getContractFactory("TrigonometryHarness", {
      libraries: {
        MathLib: await mathlib.getAddress(),
      },
    });

    trig = await H.deploy();
    await trig.waitForDeployment();
  });

  /*──────────────────────────────────────────────────────────
      BASELINE HEX TESTS
  ──────────────────────────────────────────────────────────*/

  it("HEX: sin(0) == 0x00...", async () => {
    const q0 = await toQuad(trig, 0);
    const r = await trig.sin(q0);

    const hex = r.toLowerCase();
    logHex("sin(0)", "0x00000000000000000000000000000000", hex);

    expect(hex).to.equal("0x00000000000000000000000000000000");
  });

  it("HEX: cos(0) == +1", async () => {
    const q0 = await toQuad(trig, 0);
    const r = await trig.cos(q0);

    const hex = r.toLowerCase();
    logHex("cos(0)", "0x3fff0000000000000000000000000000", hex);

    expect(hex).to.equal("0x3fff0000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
      π CONSTANT HEX TESTS
  ──────────────────────────────────────────────────────────*/

  it("HEX: sin(pi/2) == 1.0", async () => {
    const q = await trig.QHALF_PI();
    const r = await trig.sin(q);

    const hex = r.toLowerCase();
    logHex("sin(pi/2)", "0x3fff0000000000000000000000000000", hex);

    expect(hex).to.equal("0x3fff0000000000000000000000000000");
  });

  it("HEX: cos(pi/2) == 0", async () => {
    const q = await trig.QHALF_PI();
    const r = await trig.cos(q);

    const hex = r.toLowerCase();
    logHex("cos(pi/2)", "0x00000000000000000000000000000000", hex);

    expect(hex).to.equal("0x00000000000000000000000000000000");
  });

  it("HEX: sin(pi) == 0", async () => {
    const q = await trig.QPI();
    const r = await trig.sin(q);

    const hex = r.toLowerCase();
    logHex("sin(pi)", "0x00000000000000000000000000000000", hex);

    expect(hex).to.equal("0x00000000000000000000000000000000");
  });

  it("HEX: cos(pi) == -1", async () => {
    const q = await trig.QPI();
    const r = await trig.cos(q);

    const hex = r.toLowerCase();
    logHex("cos(pi)", "0xbfff0000000000000000000000000000", hex);

    expect(hex).to.equal("0xbfff0000000000000000000000000000");
  });

  /*──────────────────────────────────────────────────────────
      SYMMETRY HEX TESTS
  ──────────────────────────────────────────────────────────*/

  it("HEX: sin(-x) == -sin(x)", async () => {
    const x = 0.345;

    const qx = await toQuad(trig, x);
    const qn = await toQuad(trig, -x);

    const pos = (await trig.sin(qx)).toLowerCase(); // sin(+x)
    const neg = (await trig.sin(qn)).toLowerCase(); // sin(-x)

    const expectedNeg = flipSign(pos);

    logHex("sin(-x)", expectedNeg, neg);
    expect(neg).to.equal(expectedNeg);
  });

  function flipSign(hex: string): string {
    const bn = BigInt(hex);
    const flipped = bn ^ (1n << 127n);
    return "0x" + flipped.toString(16).padStart(32, "0");
  }

  /*──────────────────────────────────────────────────────────
      APPROX TESTS (DECIMAL)
  ──────────────────────────────────────────────────────────*/

  it("DEC: sin(x^2) produces a consistent quad result", async () => {
    const x = 0.75;
    const qx = await toQuad(trig, x);

    // x^2 (quad)
    const qx2 = await trig.mul(qx, qx);

    // sin(x^2) (quad)
    const qs = await trig.sin(qx2);

    // convert to JS just for display
    const s = await fromQuad(trig, qs);

    log("sin(x^2)", Math.sin(x * x), s);
    expect(approx(s, Math.sin(x * x), 1e-9)).to.be.true;
  });

  it("DEC: cos(x^2) produces a consistent quad result", async () => {
    const x = -1.12;
    const qx = await toQuad(trig, x);

    // x^2 (quad)
    const qx2 = await trig.mul(qx, qx);

    // cos(x^2) (quad)
    const qc = await trig.cos(qx2);

    // convert to JS for comparison
    const c = await fromQuad(trig, qc);

    log("cos(x^2)", Math.cos(x * x), c);
    expect(approx(c, Math.cos(x * x), 1e-9)).to.be.true;
  });

  it("DEC: sin^2 + cos^2 ≈ 1", async () => {
    const x = 1.234567;
    const qx = await toQuad(trig, x);

    // quad sin, quad cos
    const qs = await trig.sin(qx);
    const qc = await trig.cos(qx);

    // quad sin^2, quad cos^2
    const qs2 = await trig.mul(qs, qs);
    const qc2 = await trig.mul(qc, qc);

    // quad sum
    const sumQuad = await trig.add(qs2, qc2);

    // convert to decimal via SCALE
    const sum = await fromQuad(trig, sumQuad);

    log("sin^2+cos^2", 1, sum);
    expect(approx(sum, 1, 1e-9)).to.be.true;
  });

  it("DEC: tan(x) ≈ sin(x)/cos(x)", async () => {
    const x = 0.42;
    const qx = await toQuad(trig, x);

    const t = await fromQuad(trig, await trig.tan(qx));
    const s = await fromQuad(trig, await trig.sin(qx));
    const c = await fromQuad(trig, await trig.cos(qx));

    log("tan(x)", s / c, t);
    expect(approx(t, s / c, 1e-9)).to.be.true;
  });

  it("DEC: asin(sin(x)) ≈ x", async () => {
    const x = 0.3;
    const qx = await toQuad(trig, x);

    const s = await trig.sin(qx);
    const a = await fromQuad(trig, await trig.asin(s));

    log("asin(sin(x))", x, a);
    expect(approx(a, x, 1e-9)).to.be.true;
  });

  it("DEC: tan(atan(x)) ≈ x", async () => {
    const x = 1.2345;
    const qx = await toQuad(trig, x);

    const at = await trig.atan(qx);
    const t = await fromQuad(trig, await trig.tan(at));

    log("tan(atan(x))", x, t);
    // For composition errors we allow a slightly looser tolerance
    expect(approx(t, x, 1e-6)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      ACADEMIC STABILITY / IDENTITY TESTS (10 EXTRA)
  ──────────────────────────────────────────────────────────*/

  it("DEC: asin(x) + acos(x) ≈ π/2", async () => {
    const x = 0.37;
    const qx = await toQuad(trig, x);

    const qa = await trig.asin(qx);
    const qb = await trig.acos(qx);

    const a = await fromQuad(trig, qa);
    const b = await fromQuad(trig, qb);

    const sum = a + b;
    log("asin(x)+acos(x)", Math.PI / 2, sum);
    expect(approx(sum, Math.PI / 2, 1e-9)).to.be.true;
  });

  it("DEC: asin is odd → asin(-x) = -asin(x)", async () => {
    const x = 0.4;
    const qx = await toQuad(trig, x);
    const qn = await toQuad(trig, -x);

    const aPos = await fromQuad(trig, await trig.asin(qx));
    const aNeg = await fromQuad(trig, await trig.asin(qn));

    log("asin(-x)", -aPos, aNeg);
    expect(approx(aNeg, -aPos, 1e-9)).to.be.true;
  });

  it("DEC: atan is odd → atan(-x) = -atan(x)", async () => {
    const x = 0.9;
    const qx = await toQuad(trig, x);
    const qn = await toQuad(trig, -x);

    const aPos = await fromQuad(trig, await trig.atan(qx));
    const aNeg = await fromQuad(trig, await trig.atan(qn));

    log("atan(-x)", -aPos, aNeg);
    expect(approx(aNeg, -aPos, 1e-9)).to.be.true;
  });

  it("DEC: atan(1) ≈ π/4", async () => {
    // QONE is exposed as a public constant on the harness
    const q1 = await trig.QONE();
    const qa = await trig.atan(q1);

    const a = await fromQuad(trig, qa);
    log("atan(1)", Math.PI / 4, a);
    expect(approx(a, Math.PI / 4, 1e-9)).to.be.true;
  });

  it("DEC: periodicity sin(x + 2πk) ≈ sin(x)", async () => {
    const x = 0.73;
    const k = 100; // moderate k to avoid JS overflow in scaling

    const base = await fromQuad(trig, await trig.sin(await toQuad(trig, x)));

    const xp = x + 2 * Math.PI * k;
    const sp = await fromQuad(trig, await trig.sin(await toQuad(trig, xp)));

    log("sin(x+2πk)", base, sp);
    expect(approx(sp, base, 1e-9)).to.be.true;
  });

  it("DEC: periodicity cos(x + 2πk) preserves magnitude", async () => {
    const x = -1.11;
    const k = 80;

    const base = await fromQuad(
      trig,
      await trig.cos(await toQuad(trig, x)),
    );

    const xp = x + 2 * Math.PI * k;
    const cp = await fromQuad(
      trig,
      await trig.cos(await toQuad(trig, xp)),
    );

    log("cos(x+2πk) (mag)", Math.abs(base), Math.abs(cp));
    expect(approx(Math.abs(cp), Math.abs(base), 1e-9)).to.be.true;
  });

  it("DEC: phase shift sin(x + π/2) ≈ cos(x)", async () => {
    const x = 0.6;
    const qx = await toQuad(trig, x);
    const qHalfPi = await trig.QHALF_PI();

    const qShift = await trig.add(qx, qHalfPi);

    const sShift = await fromQuad(trig, await trig.sin(qShift));
    const cBase = await fromQuad(trig, await trig.cos(qx));

    log("sin(x+π/2)", cBase, sShift);
    expect(approx(sShift, cBase, 1e-9)).to.be.true;
  });

  it("DEC: phase shift cos(x + π/2)^2 ≈ sin(x)^2", async () => {
    const x = -0.8;
    const qx = await toQuad(trig, x);
    const qHalfPi = await trig.QHALF_PI();

    const qShift = await trig.add(qx, qHalfPi);

    const cShift = await fromQuad(trig, await trig.cos(qShift));
    const sBase = await fromQuad(trig, await trig.sin(qx));

    const left = cShift * cShift;
    const right = sBase * sBase;

    log("cos^2(x+π/2) vs sin^2(x)", right, left);
    expect(approx(left, right, 1e-9)).to.be.true;
  });

  it("DEC: atan(tan(x)) ≈ x for |x| < π/2", async () => {
    const x = 0.7;
    const qx = await toQuad(trig, x);

    const qt = await trig.tan(qx);
    const qa = await trig.atan(qt);

    const back = await fromQuad(trig, qa);

    log("atan(tan(x))", x, back);
    expect(approx(back, x, 1e-6)).to.be.true;
  });

  it("DEC: numerical derivative of sin ≈ cos (central difference)", async () => {
    const x = 0.5;
    const h = 1e-4;

    const qxPlus = await toQuad(trig, x + h);
    const qxMinus = await toQuad(trig, x - h);

    const sPlus = await fromQuad(trig, await trig.sin(qxPlus));
    const sMinus = await fromQuad(trig, await trig.sin(qxMinus));

    const numerical = (sPlus - sMinus) / (2 * h);
    const cExact = await fromQuad(trig, await trig.cos(await toQuad(trig, x)));

    log("d/dx sin(x)", cExact, numerical);
    expect(approx(numerical, cExact, 1e-5)).to.be.true;
  });

  /*──────────────────────────────────────────────────────────
      SPECIAL CASE HEX TESTS
  ──────────────────────────────────────────────────────────*/

  it("HEX: tan(pi/2) → NaN", async () => {
    const q = await trig.QHALF_PI();
    const r = await trig.tan(q);

    expect(await trig.isNaN(r)).to.be.true;
  });

  it("HEX: cot(0) → NaN", async () => {
    const q0 = await toQuad(trig, 0);
    const r = await trig.cot(q0);

    expect(await trig.isNaN(r)).to.be.true;
  });
});