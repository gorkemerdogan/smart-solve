// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

type Harness = Contract & {
  qFromInt(n: bigint): Promise<string>;
  qFromFrac(num: bigint, den: bigint): Promise<string>;
  evaluateHorners(coeffs: string[], x: string): Promise<string>;
  evaluateWithDerivative(
    coeffs: string[],
    x: string
  ): Promise<{ px: string; dpx: string }>;
  derivative(coeffs: string[]): Promise<string[]>;
  add(a: string[], b: string[]): Promise<string[]>;
  sub(a: string[], b: string[]): Promise<string[]>;
  mulScalar(a: string[], k: string): Promise<string[]>;
  mul(a: string[], b: string[]): Promise<string[]>;
  integral(a: string[], C: string): Promise<string[]>;
  degree(a: string[]): Promise<bigint>;
  trimTrailingZeros(a: string[]): Promise<string[]>;
  syntheticDivide?(coeffs: string[], root: string): Promise<[string[], string]>;
  evalHornerMonic?(lowerCoeffs: string[], x: string): Promise<string>;
  getAddress(): Promise<string>;
};

// Deploy harness
async function deployHarness(): Promise<Harness> {
  const F = await ethers.getContractFactory("PolynomialHarness");
  const h = (await F.deploy()) as unknown as Harness;
  await h.waitForDeployment();
  return h;
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

/** Print pretty test output */
function printBlock(n: number, method: string, input: string, outHex: string, outDec: string) {
  const sep = "-".repeat(60);
  console.log(`\n${sep}\nTest ${n}\nMethod: ${method}\nInput: ${input}\nOutput (hex): ${outHex}\nOutput: ${outDec}`);
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
    const yHex=await h.evaluateHorners(coeffs,xHex);
    printBlock(t,"evaluateHorners","f(x)=3x^2+2x+5, x=5",`y=${yHex}`,`y=${evalHornerNum([5,2,3],5)}`);
  });

  it("evaluateWithDerivative", async function () {
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    const xHex=await qFrac(h,6,5);
    await touchGas(h,"evaluateWithDerivative",[coeffs,xHex]);
    const {px,dpx}=await h.evaluateWithDerivative(coeffs,xHex);
    const [pDec,dpDec]=evalHornerWithDerivNum([5,2,3],1.2);
    printBlock(t,"evaluateWithDerivative","f(x)=3x^2+2x+5, x=1.2",`p=${px}, p'=${dpx}`,`p=${pDec}, p'=${dpDec}`);
  });

  it("derivative", async function () {
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    await touchGas(h,"derivative",[coeffs]);
    const dHex=await h.derivative(coeffs);
    printBlock(t,"derivative","[5,2,3]",fmtHexArr(dHex),fmtDecArr(derivativeCoeffsNum([5,2,3])));
  });

  it("add", async function () {
    t++;
    const p1=[await qInt(h,2),await qInt(h,1)], p2=[await qInt(h,4),await qInt(h,3)];
    await touchGas(h,"add",[p1,p2]);
    const out=await h.add(p1,p2);
    printBlock(t,"add","(x+2)+(3x+4)",fmtHexArr(out),fmtDecArr(addNum([2,1],[4,3])));
  });

  it("sub", async function () {
    t++;
    const p1=[await qInt(h,2),await qInt(h,1)], p2=[await qInt(h,4),await qInt(h,3)];
    await touchGas(h,"sub",[p1,p2]);
    const out=await h.sub(p1,p2);
    printBlock(t,"sub","(x+2)-(3x+4)",fmtHexArr(out),fmtDecArr(subNum([2,1],[4,3])));
  });

  it("mulScalar", async function () {
    t++;
    const p=[await qInt(h,1),await qInt(h,3),await qInt(h,4)], k=await qInt(h,2);
    await touchGas(h,"mulScalar",[p,k]);
    const out=await h.mulScalar(p,k);
    printBlock(t,"mulScalar","2*(4x^2+3x+1)",fmtHexArr(out),fmtDecArr(mulScalarNum([1,3,4],2)));
  });

  it("mul (convolution)", async function () {
    t++;
    const p1=[await qInt(h,1),await qInt(h,1)],p2=[await qInt(h,1),await qInt(h,1)];
    await touchGas(h,"mul",[p1,p2]);
    const out=await h.mul(p1,p2);
    printBlock(t,"mul","(x+1)*(x+1)",fmtHexArr(out),fmtDecArr(mulNum([1,1],[1,1])));
  });

  it("integral", async function () {
    t++;
    const p=[await qInt(h,12),await qInt(h,6)],C=await qInt(h,5);
    await touchGas(h,"integral",[p,C]);
    const out=await h.integral(p,C);
    printBlock(t,"integral","∫(6x+12)dx+C",fmtHexArr(out),fmtDecArr(integralNum([12,6],5)));
  });

  it("degree", async function () {
    t++;
    const p = [await qInt(h,5), await qInt(h,2), await qInt(h,0), await qInt(h,0)];
    await touchGas(h, "degree", [p]);
    const deg = await h.degree(p);
    printBlock(t, "degree", "[5,2,0,0]", `deg=${deg}`, `deg=${degreeNum([5,2,0,0])}`);
  });

  it("trimTrailingZeros", async function () {
    t++;
    const p = [await qInt(h,5), await qInt(h,2), await qInt(h,0), await qInt(h,0)];
    await touchGas(h, "trimTrailingZeros", [p]);
    const trimmed = await h.trimTrailingZeros(p);
    printBlock(t, "trimTrailingZeros", "[5,2,0,0]", fmtHexArr(trimmed), fmtDecArr(trimTrailingZerosNum([5,2,0,0])));
  });

  it("syntheticDivide (optional)", async function () {
    if (!(h as any).syntheticDivide) return this.skip();
    t++;
    const coeffs=[await qInt(h,5),await qInt(h,2),await qInt(h,3)];
    const root=await qFrac(h,3,2);
    await touchGas(h,"syntheticDivide",[coeffs,root]);
    const [qHex,rHex]=await (h as any).syntheticDivide(coeffs,root);
    const {q:rQ,r:rR}=syntheticDivideNum([5,2,3],1.5);
    printBlock(t,"syntheticDivide","P=[5,2,3],(x-1.5)",`Q=${fmtHexArr(qHex)},R=${rHex}`,`Q=${fmtDecArr(rQ)},R=${rR}`);
  });

  it("evalHornerMonic (optional)", async function () {
    if (!(h as any).evalHornerMonic) return this.skip();
    t++;
    const lower=[await qInt(h,2),await qFrac(h,1,2),await qInt(h,4)];
    const xHex=await qInt(h,5);
    await touchGas(h,"evalHornerMonic",[lower,xHex]);
    const yHex=await (h as any).evalHornerMonic(lower,xHex);
    printBlock(t,"evalHornerMonic","x^3+4x^2+0.5x+2 at x=5",`y=${yHex}`,`y=${evalHornerMonicNum([2,0.5,4],5)}`);
  });
});