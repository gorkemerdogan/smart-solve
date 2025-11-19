// SPDX-License-Identifier: MIT

import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

type Harness = Contract & {
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;
  evaluateHorners(coeffs: string[], x: string): Promise<string>;
  evaluateWithDerivative(coeffs: string[], x: string): Promise<{ px: string; dpx: string }>;
  derivative(coeffs: string[]): Promise<string[]>;
  add(a: string[], b: string[]): Promise<string[]>;
  sub(a: string[], b: string[]): Promise<string[]>;
  mulScalar(a: string[], k: string): Promise<string[]>;
  mul(a: string[], b: string[]): Promise<string[]>;
  integral(a: string[], C: string): Promise<string[]>;
  degree(a: string[]): Promise<bigint>;
  trimTrailingZeros(a: string[]): Promise<string[]>;
  syntheticDivide(coeffs: string[], root: string): Promise<[string[], string]>;
  evalHornerMonic(lowerCoeffs: string[], x: string): Promise<string>;
  getAddress(): Promise<string>;
};

// Deploy harness
async function deployHarness(): Promise<Harness> {
  const MathLibFactory = await ethers.getContractFactory("MathLib");
  const math = await MathLibFactory.deploy();
  await math.waitForDeployment();
  const mathAddr = await math.getAddress();

  const HarnessFactory = await ethers.getContractFactory(
    "PolynomialHarness",
    {
      libraries: {
        "contracts/libraries/MathLib.sol:MathLib": mathAddr,
      }
    }
  );

  const h = await HarnessFactory.deploy();
  await h.waitForDeployment();
  return h as unknown as Harness;
}

// --- utilities ---
async function qInt(h: Harness, n: number | string) { return h.qFromInt(BigInt(n)); }
async function qFrac(h: Harness, n: number | string, d: number | string) { return h.qFromFrac(BigInt(n), BigInt(d)); }

/** Send a real transaction so Hardhat gas-reporter records a gas row. */
async function touchGas(h: Harness, method: string, args: any[]) {
  const data = h.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await h.getAddress();
  const tx = await signer.sendTransaction({ to, data });
  await tx.wait();
}

/** Estimate gas for any method (pure/view included). */
async function estimateGas(h: Harness, method: string, args: any[]) {
  const anyH = h as any;
  if (anyH[method]?.estimateGas) {
    return (await anyH[method].estimateGas(...args)).toString();
  }
  const data = h.interface.encodeFunctionData(method, args);
  const [signer] = await ethers.getSigners();
  const to = await h.getAddress();
  const gas = await signer.estimateGas({ to, data });
  return gas.toString();
}

/** Print test output */
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

const fmtHexArr = (arr: string[]) => `[${arr.join(", ")}]`;
const fmtDecArr = (arr: (number | string)[]) => `[${arr.join(", ")}]`;

// --- JS numeric mirrors ---
function evalHornerNum(c: number[], x: number) { let y=c[c.length-1]; for(let i=c.length-2;i>=0;i--) y=y*x+c[i]; return y; }
function evalHornerWithDerivNum(c: number[], x: number){let p=c[c.length-1],dp=0;for(let i=c.length-2;i>=0;i--){dp=dp*x+p;p=p*x+c[i];}return [p,dp];}
function derivativeCoeffsNum(c:number[]){if(c.length<=1)return[0];const d=new Array(c.length-1);for(let i=1;i<c.length;i++)d[i-1]=c[i]*i;return d;}
function addNum(a:number[],b:number[]){const n=Math.max(a.length,b.length);const o=new Array(n).fill(0);for(let i=0;i<n;i++)o[i]=(a[i]??0)+(b[i]??0);return o;}
function subNum(a:number[],b:number[]){const n=Math.max(a.length,b.length);const o=new Array(n).fill(0);for(let i=0;i<n;i++)o[i]=(a[i]??0)-(b[i]??0);return o;}
function mulScalarNum(a:number[],k:number){return a.map(v=>v*k);}
function mulNum(a:number[],b:number[]){const o=new Array(a.length+b.length-1).fill(0);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)o[i+j]+=a[i]*b[j];return o;}
function integralNum(a:number[],C:number){const o=new Array(a.length+1).fill(0);o[0]=C;for(let i=0;i<a.length;i++)o[i+1]=a[i]/(i+1);return o;}
function degreeNum(a:number[]){for(let i=a.length-1;i>=0;i--)if(a[i]!==0)return i;return 0;}
function trimTrailingZerosNum(a:number[]){return a.slice(0,degreeNum(a)+1);}
function syntheticDivideNum(c:number[],r:number){const n=c.length;if(n===0)return{q:[0],r:0};if(n===1)return{q:[0],r:c[0]};const q=new Array(n-1).fill(0);q[n-2]=c[n-1];for(let i=n-1;i>1;i--)q[i-2]=c[i-1]+r*q[i-1];const rem=c[0]+r*q[0];return{q,r:rem};}
function evalHornerMonicNum(lower:number[],x:number){let acc=1;for(let i=lower.length-1;i>=0;i--)acc=acc*x+lower[i];return acc;}

// ===================== REPORT SUITE =====================
describe("Polynomial — Report (values + gas)", function () {
  let h: Harness; let t = 0;

  before(async () => {
    h = await deployHarness();
    expect(await h.getAddress()).to.be.properAddress;
  });

  it("evaluateHorners", async function () {
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    const xHex=await qInt(h,5);
    await touchGas(h,"evaluateHorners",[coeffs,xHex]);
    const gas = await estimateGas(h,"evaluateHorners",[coeffs,xHex]);
    const yHex=await h.evaluateHorners(coeffs,xHex);
    printBlock(
      t,
      "evaluateHorners",
      "Evaluates p(x) using Horner's method with ascending coefficients.",
      gas,
      "f(x)=3x^2+2x+5, x=5",
      `y=${yHex}`,
      `y=${evalHornerNum([5,2,3],5)}`
    );
  });

  it("evaluateWithDerivative", async function () {
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    const xHex=await qFrac(h,6,5);
    await touchGas(h,"evaluateWithDerivative",[coeffs,xHex]);
    const gas = await estimateGas(h,"evaluateWithDerivative",[coeffs,xHex]);
    const {px,dpx}=await h.evaluateWithDerivative(coeffs,xHex);
    const [pDec,dpDec]=evalHornerWithDerivNum([5,2,3],1.2);
    printBlock(
      t,
      "evaluateWithDerivative",
      "Computes p(x) and p’(x) in one pass via extended Horner.",
      gas,
      "f(x)=3x^2+2x+5, x=1.2",
      `p=${px}, p'=${dpx}`,
      `p=${pDec}, p'=${dpDec}`
    );
  });

  it("derivative", async function () {
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    await touchGas(h,"derivative",[coeffs]);
    const gas = await estimateGas(h,"derivative",[coeffs]);
    const dHex=await h.derivative(coeffs);
    printBlock(
      t,
      "derivative",
      "Returns coefficients of the first derivative (ascending powers).",
      gas,
      "[5,2,3]",
      fmtHexArr(dHex),
      fmtDecArr(derivativeCoeffsNum([5,2,3]))
    );
  });

  it("add", async function () {
    t++;
    const p1=[await qInt(h,2),await qInt(h,1)], p2=[await qInt(h,4),await qInt(h,3)];
    await touchGas(h,"add",[p1,p2]);
    const gas = await estimateGas(h,"add",[p1,p2]);
    const out=await h.add(p1,p2);
    printBlock(
      t,
      "add",
      "Coefficient-wise addition aligning by power index.",
      gas,
      "(x+2)+(3x+4)",
      fmtHexArr(out),
      fmtDecArr(addNum([2,1],[4,3]))
    );
  });

  it("sub", async function () {
    t++;
    const p1=[await qInt(h,2),await qInt(h,1)], p2=[await qInt(h,4),await qInt(h,3)];
    await touchGas(h,"sub",[p1,p2]);
    const gas = await estimateGas(h,"sub",[p1,p2]);
    const out=await h.sub(p1,p2);
    printBlock(
      t,
      "sub",
      "Coefficient-wise subtraction: first minus second.",
      gas,
      "(x+2)-(3x+4)",
      fmtHexArr(out),
      fmtDecArr(subNum([2,1],[4,3]))
    );
  });

  it("mulScalar", async function () {
    t++;
    const p=[await qInt(h,1),await qInt(h,3),await qInt(h,4)], k=await qInt(h,2);
    await touchGas(h,"mulScalar",[p,k]);
    const gas = await estimateGas(h,"mulScalar",[p,k]);
    const out=await h.mulScalar(p,k);
    printBlock(
      t,
      "mulScalar",
      "Scales all coefficients by a scalar k (handles k=0 too).",
      gas,
      "2*(4x^2+3x+1)",
      fmtHexArr(out),
      fmtDecArr(mulScalarNum([1,3,4],2))
    );
  });

  it("mul (convolution)", async function () {
    t++;
    const p1=[await qInt(h,1),await qInt(h,1)],p2=[await qInt(h,1),await qInt(h,1)];
    await touchGas(h,"mul",[p1,p2]);
    const gas = await estimateGas(h,"mul",[p1,p2]);
    const out=await h.mul(p1,p2);
    printBlock(
      t,
      "mul",
      "Polynomial multiplication via coefficient convolution.",
      gas,
      "(x+1)*(x+1)",
      fmtHexArr(out),
      fmtDecArr(mulNum([1,1],[1,1]))
    );
  });

  it("integral", async function () {
    t++;
    const p=[await qInt(h,12),await qInt(h,6)],C=await qInt(h,5);
    await touchGas(h,"integral",[p,C]);
    const gas = await estimateGas(h,"integral",[p,C]);
    const out=await h.integral(p,C);
    printBlock(
      t,
      "integral",
      "Indefinite integral coefficients with constant term C as q(0)=C.",
      gas,
      "∫(6x+12)dx+C",
      fmtHexArr(out),
      fmtDecArr(integralNum([12,6],5))
    );
  });

  it("degree", async function () {
    t++;
    const p = [await qInt(h,5), await qInt(h,2), await qInt(h,0), await qInt(h,0)];
    await touchGas(h, "degree", [p]);
    const gas = await estimateGas(h,"degree",[p]);
    const deg = await h.degree(p);
    printBlock(
      t,
      "degree",
      "Returns highest index i with non-zero coeff; empty/zero -> 0.",
      gas,
      "[5,2,0,0]",
      `deg=${deg}`,
      `deg=${degreeNum([5,2,0,0])}`
    );
  });

  it("trimTrailingZeros", async function () {
    t++;
    const p = [await qInt(h,5), await qInt(h,2), await qInt(h,0), await qInt(h,0)];
    await touchGas(h, "trimTrailingZeros", [p]);
    const gas = await estimateGas(h,"trimTrailingZeros",[p]);
    const trimmed = await h.trimTrailingZeros(p);
    printBlock(
      t,
      "trimTrailingZeros",
      "Removes highest-order zero coefficients to canonicalize length.",
      gas,
      "[5,2,0,0]",
      fmtHexArr(trimmed),
      fmtDecArr(trimTrailingZerosNum([5,2,0,0]))
    );
  });

  it("syntheticDivide", async function () {
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    const root=await qFrac(h,3,2);
    await touchGas(h,"syntheticDivide",[coeffs,root]);
    const gas = await estimateGas(h,"syntheticDivide",[coeffs,root]);
    const [qHex,rHex]=await h.syntheticDivide(coeffs,root);
    const {q:rQ,r:rR}=syntheticDivideNum([5,2,3],1.5);
    printBlock(
      t,
      "syntheticDivide",
      "Divides P(x) by (x - r) using synthetic division with ascending coeffs.",
      gas,
      "P=[5,2,3],(x-1.5)",
      `Q=${fmtHexArr(qHex)},R=${rHex}`,
      `Q=${fmtDecArr(rQ)},R=${rR}`
    );
  });

  it("evalHornerMonic", async function () {
    t++;
    const lower=[await qInt(h,2),await qFrac(h,1,2),await qInt(h,4)];
    const xHex=await qInt(h,5);
    await touchGas(h,"evalHornerMonic",[lower,xHex]);
    const gas = await estimateGas(h,"evalHornerMonic",[lower,xHex]);
    const yHex=await h.evalHornerMonic(lower,xHex);
    printBlock(
      t,
      "evalHornerMonic",
      "Evaluates a monic polynomial given only lower coefficients [a0..a_{n-1}].",
      gas,
      "x^3+4x^2+0.5x+2 at x=5",
      `y=${yHex}`,
      `y=${evalHornerMonicNum([2,0.5,4],5)}`
    );
  });
});

// ===================== EDGE CASES SUITE =====================
describe("Polynomial — Edge Cases (values + gas)", function () {
  let h: Harness; let t = 0;

  before(async () => {
    h = await deployHarness();
    expect(await h.getAddress()).to.be.properAddress;
  });

  // evaluateHorners
  it("evaluateHorners: empty coeffs -> 0", async function () {
    t++;
    const coeffs: string[] = [];
    const x = await qInt(h, 7);
    await touchGas(h, "evaluateHorners", [coeffs, x]);
    const gas = await estimateGas(h,"evaluateHorners",[coeffs,x]);
    const y = await h.evaluateHorners(coeffs, x);
    printBlock(t,"evaluateHorners (empty)","Empty polynomial evaluates to 0 for any x.",gas,"[] , x=7",`y=${y}`,`y=0`);
  });

  it("evaluateHorners: constant", async function () {
    t++;
    const coeffs = [await qInt(h, 7)];
    const x = await qInt(h, 123);
    await touchGas(h, "evaluateHorners", [coeffs, x]);
    const gas = await estimateGas(h,"evaluateHorners",[coeffs,x]);
    const y = await h.evaluateHorners(coeffs, x);
    printBlock(t,"evaluateHorners (const)","Constant polynomial returns the same value regardless of x.",gas,"[7], x=123",`y=${y}`,`y=7`);
  });

  it("evaluateHorners: negative x", async function () {
    t++;
    const coeffs=[await qInt(h,1),await qInt(h,2),await qInt(h,1)];
    const x = await qInt(h, -3);
    await touchGas(h, "evaluateHorners", [coeffs, x]);
    const gas = await estimateGas(h,"evaluateHorners",[coeffs,x]);
    const y = await h.evaluateHorners(coeffs, x);
    printBlock(t,"evaluateHorners (neg x)","Confirms Horner works with negative inputs.",gas,"[1,2,1], x=-3",`y=${y}`,`y=${evalHornerNum([1,2,1],-3)}`);
  });

  // evaluateWithDerivative
  it("evaluateWithDerivative: constant poly -> dp=0", async function () {
    t++;
    const coeffs=[await qInt(h,5)];
    const x = await qInt(h, 42);
    await touchGas(h, "evaluateWithDerivative", [coeffs, x]);
    const gas = await estimateGas(h,"evaluateWithDerivative",[coeffs,x]);
    const { px, dpx } = await h.evaluateWithDerivative(coeffs, x);
    printBlock(t,"evaluateWithDerivative (const)","Derivative of a constant is zero; p(x) equals the constant.",gas,"[5], x=42",`p=${px}, p'=${dpx}`,`p=5, p'=0`);
  });

  it("evaluateWithDerivative: linear", async function () {
    t++;
    const coeffs=[await qInt(h,3),await qInt(h,2)];
    const x = await qInt(h, 10);
    await touchGas(h, "evaluateWithDerivative", [coeffs, x]);
    const gas = await estimateGas(h,"evaluateWithDerivative",[coeffs,x]);
    const { px, dpx } = await h.evaluateWithDerivative(coeffs, x);
    const [pDec, dpDec] = evalHornerWithDerivNum([3,2], 10);
    printBlock(t,"evaluateWithDerivative (linear)","Linear polynomials have constant derivative equal to slope.",gas,"[3,2], x=10",`p=${px}, p'=${dpx}`,`p=${pDec}, p'=${dpDec}`);
  });

  // derivative
  it("derivative: empty -> [0]", async function () {
    t++;
    const coeffs: string[] = [];
    await touchGas(h, "derivative", [coeffs]);
    const gas = await estimateGas(h,"derivative",[coeffs]);
    const d = await h.derivative(coeffs);
    printBlock(t,"derivative (empty)","Empty polynomial derivative is defined as zero polynomial.",gas,"[]",fmtHexArr(d),fmtDecArr([0]));
  });

  it("derivative: zero middle terms", async function () {
    t++;
    const coeffs=[await qInt(h,0),await qInt(h,2),await qInt(h,0),await qInt(h,4)];
    await touchGas(h, "derivative", [coeffs]);
    const gas = await estimateGas(h,"derivative",[coeffs]);
    const d = await h.derivative(coeffs);
    printBlock(t,"derivative (zeros-middles)","Handles internal zeros correctly when multiplying by index.",gas,"[0,2,0,4]",fmtHexArr(d),fmtDecArr(derivativeCoeffsNum([0,2,0,4])));
  });

  // add
  it("add: different lengths", async function () {
    t++;
    const a=[await qInt(h,1)];
    const b=[await qInt(h,1),await qInt(h,2),await qInt(h,3)];
    await touchGas(h, "add", [a, b]);
    const gas = await estimateGas(h,"add",[a,b]);
    const out = await h.add(a,b);
    printBlock(t,"add (len mismatch)","Pads the shorter polynomial with zeros implicitly.",gas,"[1] + [1,2,3]",fmtHexArr(out),fmtDecArr(addNum([1],[1,2,3])));
  });

  it("add: empty + non-empty", async function () {
    t++;
    const a: string[] = [];
    const b=[await qInt(h,1),await qInt(h,2)];
    await touchGas(h, "add", [a, b]);
    const gas = await estimateGas(h,"add",[a,b]);
    const out = await h.add(a,b);
    printBlock(t,"add (empty + non-empty)","Empty polynomial behaves as additive identity.",gas,"[] + [1,2]",fmtHexArr(out),fmtDecArr([1,2]));
  });

  // sub
  it("sub: empty - non-empty", async function () {
    t++;
    const a: string[] = [];
    const b=[await qInt(h,1),await qInt(h,2)];
    await touchGas(h, "sub", [a, b]);
    const gas = await estimateGas(h,"sub",[a,b]);
    const out = await h.sub(a,b);
    printBlock(t,"sub (empty - non-empty)","Subtracting from empty yields negation of the second.",gas,"[] - [1,2]",fmtHexArr(out),fmtDecArr(subNum([], [1,2])));
  });

  it("sub: different lengths", async function () {
    t++;
    const a=[await qInt(h,5),await qInt(h,3),await qInt(h,1)];
    const b=[await qInt(h,2)];
    await touchGas(h, "sub", [a, b]);
    const gas = await estimateGas(h,"sub",[a,b]);
    const out = await h.sub(a,b);
    printBlock(t,"sub (len mismatch)","Longer left polynomial retains higher terms.",gas,"[5,3,1] - [2]",fmtHexArr(out),fmtDecArr(subNum([5,3,1],[2])));
  });

  // mulScalar
  it("mulScalar: zero scalar", async function () {
    t++;
    const p=[await qInt(h,5),await qInt(h,4),await qInt(h,3)];
    const k=await qInt(h,0);
    await touchGas(h, "mulScalar", [p, k]);
    const gas = await estimateGas(h,"mulScalar",[p,k]);
    const out = await h.mulScalar(p,k);
    printBlock(t,"mulScalar (k=0)","Scaling by zero produces the zero polynomial.",gas,"[5,4,3] * 0",fmtHexArr(out),fmtDecArr(mulScalarNum([5,4,3],0)));
  });

  it("mulScalar: negative scalar", async function () {
    t++;
    const p=[await qInt(h,1),await qInt(h,2),await qInt(h,3)];
    const k=await qInt(h,-2);
    await touchGas(h, "mulScalar", [p, k]);
    const gas = await estimateGas(h,"mulScalar",[p,k]);
    const out = await h.mulScalar(p,k);
    printBlock(t,"mulScalar (k=-2)","Supports negative scaling (sign flip).",gas,"[1,2,3] * -2",fmtHexArr(out),fmtDecArr(mulScalarNum([1,2,3],-2)));
  });

  // mul
  it("mul: empty * poly", async function () {
    t++;
    const a: string[] = [];
    const b=[await qInt(h,1),await qInt(h,2)];
    await touchGas(h, "mul", [a, b]);
    const gas = await estimateGas(h,"mul",[a,b]);
    const out = await h.mul(a,b);
    printBlock(t,"mul (empty * poly)","Empty multiplicand yields zero polynomial by convention.",gas,"[] * [1,2]",fmtHexArr(out),fmtDecArr([0]));
  });

  it("mul: zero poly * poly", async function () {
    t++;
    const a=[await qInt(h,0)];
    const b=[await qInt(h,5),await qInt(h,2)];
    await touchGas(h, "mul", [a, b]);
    const gas = await estimateGas(h,"mul",[a,b]);
    const out = await h.mul(a,b);
    printBlock(t,"mul (zero * poly)","Explicit zero polynomial short-circuits to zero.",gas,"[0] * [5,2]",fmtHexArr(out),fmtDecArr([0]));
  });

  // integral
  it("integral: empty coeffs + C", async function () {
    t++;
    const a: string[] = [];
    const C = await qInt(h, 9);
    await touchGas(h, "integral", [a, C]);
    const gas = await estimateGas(h,"integral",[a,C]);
    const out = await h.integral(a, C);
    printBlock(t,"integral (empty + C)","Integrating empty returns just the constant term C.",gas,"[] + C=9",fmtHexArr(out),fmtDecArr([9]));
  });

  it("integral: constant only", async function () {
    t++;
    const a=[await qInt(h,7)];
    const C = await qInt(h, 0);
    await touchGas(h, "integral", [a, C]);
    const gas = await estimateGas(h,"integral",[a,C]);
    const out = await h.integral(a, C);
    printBlock(t,"integral (const)","Integral of a constant c is C + c·x.",gas,"[7] + C=0",fmtHexArr(out),fmtDecArr(integralNum([7],0)));
  });

  // degree
  it("degree: empty -> 0", async function () {
    t++;
    const a: string[] = [];
    await touchGas(h, "degree", [a]);
    const gas = await estimateGas(h,"degree",[a]);
    const deg = await h.degree(a);
    printBlock(t,"degree (empty)","Empty is treated as degree 0 (zero polynomial).",gas,"[]",`deg=${deg}`,`deg=0`);
  });

  it("degree: all zeros -> 0", async function () {
    t++;
    const a=[await qInt(h,0),await qInt(h,0),await qInt(h,0)];
    await touchGas(h, "degree", [a]);
    const gas = await estimateGas(h,"degree",[a]);
    const deg = await h.degree(a);
    printBlock(t,"degree (all zeros)","All-zero arrays normalize to degree 0.",gas,"[0,0,0]",`deg=${deg}`,`deg=0`);
  });

  // trimTrailingZeros
  it("trimTrailingZeros: already trimmed", async function () {
    t++;
    const a=[await qInt(h,3),await qInt(h,2)];
    await touchGas(h, "trimTrailingZeros", [a]);
    const gas = await estimateGas(h,"trimTrailingZeros",[a]);
    const out = await h.trimTrailingZeros(a);
    printBlock(t,"trimTrailingZeros (already)","No change when highest term is non-zero.",gas,"[3,2]",fmtHexArr(out),fmtDecArr([3,2]));
  });

  it("trimTrailingZeros: all zeros -> [0]", async function () {
    t++;
    const a=[await qInt(h,0),await qInt(h,0),await qInt(h,0)];
    await touchGas(h, "trimTrailingZeros", [a]);
    const gas = await estimateGas(h,"trimTrailingZeros",[a]);
    const out = await h.trimTrailingZeros(a);
    printBlock(t,"trimTrailingZeros (all zeros)","Collapses to a single zero coefficient.",gas,"[0,0,0]",fmtHexArr(out),fmtDecArr([0]));
  });
});

// ===================== STRESS SUITE (values + gas) =====================
describe("Polynomial — Stress (values + gas)", function () {
  let h: Harness; let t = 0;

  before(async () => {
    h = await deployHarness();
    expect(await h.getAddress()).to.be.properAddress;
  });

  // helpers
  async function buildCoeffs(n: number): Promise<string[]> {
    const arr: string[] = [];
    for (let i = 1; i <= n; i++) arr.push(await qInt(h, i)); // asc [1..n]
    return arr;
  }
  function headTailHex(arr: string[], k = 3) {
    if (arr.length <= 2 * k) return `[${arr.join(", ")}]`;
    const head = arr.slice(0, k).join(", ");
    const tail = arr.slice(-k).join(", ");
    return `[${head}, …, ${tail}] (len=${arr.length})`;
  }
  function headTailDec(arr: number[], k = 3) {
    if (arr.length <= 2 * k) return `[${arr.join(", ")}]`;
    const head = arr.slice(0, k).join(", ");
    const tail = arr.slice(-k).join(", ");
    return `[${head}, …, ${tail}] (len=${arr.length})`;
  }

  const SIZES = [16, 32, 64];

  it("evaluateHorners — stress (n = 16/32/64)", async function () {
    for (const n of SIZES) {
      t++;
      const coeffs = await buildCoeffs(n);
      const xHex = await qFrac(h, 5, 4); // 1.25
      await touchGas(h, "evaluateHorners", [coeffs, xHex]);
      const gas = await estimateGas(h,"evaluateHorners",[coeffs,xHex]);
      const yHex = await h.evaluateHorners(coeffs, xHex);
      const yDec = evalHornerNum(Array.from({ length: n }, (_, i) => i + 1), 1.25);
      printBlock(
        t,
        `evaluateHorners (n=${n})`,
        "O(n) Horner evaluation; should scale roughly linearly with degree.",
        gas,
        `asc coeffs [1..${n}], x=1.25`,
        `y=${yHex}`,
        `y=${yDec}`
      );
    }
  });

  it("add — stress (n = 16/32/64)", async function () {
    for (const n of SIZES) {
      t++;
      const a = await buildCoeffs(n);
      const b = await buildCoeffs(n);
      await touchGas(h, "add", [a, b]);
      const gas = await estimateGas(h,"add",[a,b]);
      const outHex = await h.add(a, b);
      const outNum = addNum(
        Array.from({ length: n }, (_, i) => i + 1),
        Array.from({ length: n }, (_, i) => i + 1)
      );
      printBlock(
        t,
        `add (n=${n})`,
        "Coefficient-wise O(n) addition; linear growth with length.",
        gas,
        `a=[1..${n}], b=[1..${n}]`,
        headTailHex(outHex),
        headTailDec(outNum)
      );
    }
  });

  it("mul (convolution) — stress (n = 16/32/64)", async function () {
    for (const n of SIZES) {
      t++;
      const a = await buildCoeffs(n);
      const b = await buildCoeffs(n);
      await touchGas(h, "mul", [a, b]);
      const gas = await estimateGas(h,"mul",[a,b]);
      const outHex = await h.mul(a, b);
      const base = Array.from({ length: n }, (_, i) => i + 1);
      const outNum = mulNum(base, base); // length 2n-1
      printBlock(
        t,
        `mul (n=${n} × n=${n})`,
        "Polynomial convolution; expected ~O(n²) growth vs degree.",
        gas,
        `a=[1..${n}], b=[1..${n}]`,
        headTailHex(outHex),
        headTailDec(outNum)
      );
    }
  });
});