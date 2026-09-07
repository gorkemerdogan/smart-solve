// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type { BaseContract, ContractFactory, ContractTransactionResponse } from "ethers";

const QONE = "0x3fff0000000000000000000000000000";
const QTWO = "0x40000000000000000000000000000000";

type GasRow = {
  workload: string;
  linkedCalls: number;
  linkedGas: bigint;
  inlinedGas: bigint;
};

function byteLength(bytecode: string): number {
  return (bytecode.length - 2) / 2;
}

function formatRatio(numerator: bigint, denominator: bigint): string {
  const hundredths = (numerator * 100n) / denominator;
  return `${hundredths / 100n}.${(hundredths % 100n).toString().padStart(2, "0")}x`;
}

async function deploymentGas(contract: BaseContract): Promise<bigint> {
  const transaction = contract.deploymentTransaction() as ContractTransactionResponse | null;
  if (!transaction) throw new Error("Missing deployment transaction");
  const receipt = await transaction.wait();
  if (!receipt) throw new Error("Missing deployment receipt");
  return receipt.gasUsed;
}

describe("MathLib linking overhead microbenchmark", function () {
  this.timeout(120_000);

  let mathLib: BaseContract;
  let linked: BaseContract;
  let inlined: BaseContract;
  let mathLibFactory: ContractFactory;
  let linkedFactory: ContractFactory;
  let inlinedFactory: ContractFactory;
  const rows: GasRow[] = [];

  before(async function () {
    mathLibFactory = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    mathLib = await mathLibFactory.deploy();
    await mathLib.waitForDeployment();

    linkedFactory = await ethers.getContractFactory("MathLibLinkedBenchmarkHarness", {
      libraries: { MathLib: await mathLib.getAddress() },
    });
    inlinedFactory = await ethers.getContractFactory("MathLibInlinedBenchmarkHarness");

    linked = await linkedFactory.deploy();
    inlined = await inlinedFactory.deploy();
    await Promise.all([linked.waitForDeployment(), inlined.waitForDeployment()]);

    console.log("============================================================");
    console.log("MATHLIB LINKING OVERHEAD MICROBENCHMARK — TEST-ONLY");
    console.log("Execution model: direct harness estimateGas; not algorithm-level gas");
    console.log("Inputs: deterministic IEEE-754 binary128 values; optimizer/viaIR from Hardhat config");
    console.log("============================================================");
  });

  async function compare(
    workload: string,
    linkedCalls: number,
    method: "arithmeticLoop" | "horner" | "dot",
    args: readonly unknown[]
  ): Promise<void> {
    const linkedMethod = linked.getFunction(method);
    const inlinedMethod = inlined.getFunction(method);
    const [linkedResult, inlinedResult, linkedGas, inlinedGas] = await Promise.all([
      linkedMethod.staticCall(...args),
      inlinedMethod.staticCall(...args),
      linkedMethod.estimateGas(...args),
      inlinedMethod.estimateGas(...args),
    ]);

    expect(linkedResult, `${workload}: implementations returned different values`).to.equal(inlinedResult);
    rows.push({ workload, linkedCalls, linkedGas, inlinedGas });
    console.log(
      `MATHLIB_MICROBENCH | workload=${workload} | linkedCalls=${linkedCalls} | ` +
      `linkedGas=${linkedGas} | inlinedGas=${inlinedGas} | delta=${linkedGas - inlinedGas} | ` +
      `linked/inlined=${formatRatio(linkedGas, inlinedGas)}`
    );
  }

  it("compares add/sub/mul/div dependency chains", async function () {
    for (const iterations of [1, 5, 10, 20, 40]) {
      await compare(
        `arithmetic-4op-loop-${iterations}`,
        iterations * 4,
        "arithmeticLoop",
        [QONE, QTWO, QONE, BigInt(iterations)]
      );
    }
  });

  it("compares Horner-style polynomial evaluation", async function () {
    for (const coefficientCount of [3, 5, 9, 17]) {
      const coefficients = Array.from(
        { length: coefficientCount },
        (_, index) => (index % 2 === 0 ? QONE : QTWO)
      );
      await compare(
        `horner-degree-${coefficientCount - 1}`,
        2 * (coefficientCount - 1),
        "horner",
        [QTWO, coefficients]
      );
    }
  });

  it("compares dot-product accumulations", async function () {
    for (const length of [2, 4, 8, 16]) {
      const a = Array.from({ length }, (_, index) => (index % 2 === 0 ? QONE : QTWO));
      const b = Array.from({ length }, (_, index) => (index % 2 === 0 ? QTWO : QONE));
      await compare(`dot-length-${length}`, 2 * length, "dot", [a, b]);
    }
  });

  after(async function () {
    const [mathLibCode, linkedCode, inlinedCode] = await Promise.all([
      ethers.provider.getCode(await mathLib.getAddress()),
      ethers.provider.getCode(await linked.getAddress()),
      ethers.provider.getCode(await inlined.getAddress()),
    ]);
    const [mathLibDeployGas, linkedDeployGas, inlinedDeployGas] = await Promise.all([
      deploymentGas(mathLib),
      deploymentGas(linked),
      deploymentGas(inlined),
    ]);

    console.log("------------------------------------------------------------");
    console.log("MATHLIB_MICROBENCH_DEPLOYMENT");
    console.log(`MathLib creation bytes=${byteLength(mathLibFactory.bytecode)} | runtime bytes=${byteLength(mathLibCode)} | deploymentGas=${mathLibDeployGas}`);
    console.log(`Linked harness creation bytes=${byteLength(linkedFactory.bytecode)} | runtime bytes=${byteLength(linkedCode)} | deploymentGas=${linkedDeployGas}`);
    console.log(`Inlined harness creation bytes=${byteLength(inlinedFactory.bytecode)} | runtime bytes=${byteLength(inlinedCode)} | deploymentGas=${inlinedDeployGas}`);
    console.log(`Linked first-consumer deploymentGas=${mathLibDeployGas + linkedDeployGas} (MathLib + harness)`);
    console.log(`Linked incremental-consumer deploymentGas=${linkedDeployGas} (shared MathLib already deployed)`);

    const totalLinkedGas = rows.reduce((sum, row) => sum + row.linkedGas, 0n);
    const totalInlinedGas = rows.reduce((sum, row) => sum + row.inlinedGas, 0n);
    console.log(`Aggregate across ${rows.length} measured calls: linkedGas=${totalLinkedGas} | inlinedGas=${totalInlinedGas} | delta=${totalLinkedGas - totalInlinedGas} | linked/inlined=${formatRatio(totalLinkedGas, totalInlinedGas)}`);
    console.log("============================================================");
  });
});
