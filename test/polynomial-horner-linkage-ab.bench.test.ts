// SPDX-License-Identifier: MIT
import { expect } from "chai";
import { ethers } from "hardhat";
import type {
  BaseContract,
  ContractFactory,
  ContractTransactionReceipt,
  ContractTransactionResponse,
} from "ethers";

const QONE = "0x3fff0000000000000000000000000000";
const QTWO = "0x40000000000000000000000000000000";
const QFIVE_QUARTERS = "0x3fff4000000000000000000000000000";
const EIP170_RUNTIME_LIMIT = 24_576;

type Measurement = {
  degree: number;
  arithmeticCalls: number;
  directLinkedGas: bigint;
  directInlinedGas: bigint;
  routedLinkedGas: bigint;
  routedInlinedGas: bigint;
};

function byteLength(bytecode: string): number {
  return (bytecode.length - 2) / 2;
}

function formatRatio(numerator: bigint, denominator: bigint): string {
  const hundredths = (numerator * 100n) / denominator;
  return `${hundredths / 100n}.${(hundredths % 100n).toString().padStart(2, "0")}x`;
}

async function receiptOf(contract: BaseContract): Promise<ContractTransactionReceipt> {
  const transaction = contract.deploymentTransaction() as ContractTransactionResponse | null;
  if (!transaction) throw new Error("Missing deployment transaction");
  const receipt = await transaction.wait();
  if (!receipt) throw new Error("Missing deployment receipt");
  return receipt;
}

async function installSingleFacet(
  diamondCutFacet: BaseContract,
  facet: BaseContract,
  selector: string
): Promise<{ diamond: BaseContract; installationGas: bigint }> {
  const [owner] = await ethers.getSigners();
  const SmartSolve = await ethers.getContractFactory("SmartSolve");
  const diamond = await SmartSolve.deploy(owner.address, await diamondCutFacet.getAddress());
  await diamond.waitForDeployment();

  const diamondCut = await ethers.getContractAt("IDiamondCut", await diamond.getAddress());
  const transaction = await diamondCut.diamondCut(
    [{ facetAddress: await facet.getAddress(), action: 0, functionSelectors: [selector] }],
    ethers.ZeroAddress,
    "0x"
  );
  const receipt = await transaction.wait();
  if (!receipt) throw new Error("Missing Diamond installation receipt");
  return { diamond, installationGas: receipt.gasUsed };
}

describe("Polynomial Horner MathLib-linkage algorithm A/B benchmark", function () {
  this.timeout(120_000);

  let mathLib: BaseContract;
  let linkedFacet: BaseContract;
  let inlinedFacet: BaseContract;
  let productionFacet: BaseContract;
  let linkedRouted: BaseContract;
  let inlinedRouted: BaseContract;
  let mathLibFactory: ContractFactory;
  let linkedFactory: ContractFactory;
  let inlinedFactory: ContractFactory;
  let productionFactory: ContractFactory;
  let linkedInstallationGas: bigint;
  let inlinedInstallationGas: bigint;
  const measurements: Measurement[] = [];

  before(async function () {
    mathLibFactory = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    mathLib = await mathLibFactory.deploy();
    await mathLib.waitForDeployment();

    linkedFactory = await ethers.getContractFactory("PolynomialLinkedHornerBenchmarkFacet", {
      libraries: { MathLib: await mathLib.getAddress() },
    });
    inlinedFactory = await ethers.getContractFactory("PolynomialInlinedHornerBenchmarkFacet");
    productionFactory = await ethers.getContractFactory("PolynomialFacet", {
      libraries: { MathLib: await mathLib.getAddress() },
    });

    [linkedFacet, inlinedFacet, productionFacet] = await Promise.all([
      linkedFactory.deploy(),
      inlinedFactory.deploy(),
      productionFactory.deploy(),
    ]);
    await Promise.all([
      linkedFacet.waitForDeployment(),
      inlinedFacet.waitForDeployment(),
      productionFacet.waitForDeployment(),
    ]);

    const DiamondCutFacet = await ethers.getContractFactory("DiamondCutFacet");
    const diamondCutFacet = await DiamondCutFacet.deploy();
    await diamondCutFacet.waitForDeployment();
    const selector = linkedFacet.interface.getFunction("polyEvaluate")!.selector;
    expect(inlinedFacet.interface.getFunction("polyEvaluate")!.selector).to.equal(selector);

    const [linkedDiamond, inlinedDiamond] = await Promise.all([
      installSingleFacet(diamondCutFacet, linkedFacet, selector),
      installSingleFacet(diamondCutFacet, inlinedFacet, selector),
    ]);
    linkedInstallationGas = linkedDiamond.installationGas;
    inlinedInstallationGas = inlinedDiamond.installationGas;
    linkedRouted = await ethers.getContractAt(
      "PolynomialLinkedHornerBenchmarkFacet",
      await linkedDiamond.diamond.getAddress()
    );
    inlinedRouted = await ethers.getContractAt(
      "PolynomialInlinedHornerBenchmarkFacet",
      await inlinedDiamond.diamond.getAddress()
    );

    console.log("============================================================");
    console.log("POLYNOMIAL HORNER LINKAGE A/B — TEST-ONLY ALGORITHM BENCHMARK");
    console.log("Linked variant: existing Polynomial.evaluateHorners -> public linked MathLib");
    console.log("Inlined variant: equivalent Horner loop -> internal ABDK add/mul");
    console.log("Execution model: paired direct-facet and minimal-Diamond estimateGas calls");
    console.log("============================================================");
  });

  for (const degree of [2, 4, 8, 16, 32]) {
    it(`compares linked and inlined Horner evaluation at degree ${degree}`, async function () {
      const coefficients = Array.from(
        { length: degree + 1 },
        (_, index) => (index % 2 === 0 ? QONE : QTWO)
      );
      const args = [coefficients, QFIVE_QUARTERS] as const;
      const linkedDirectMethod = linkedFacet.getFunction("polyEvaluate");
      const inlinedDirectMethod = inlinedFacet.getFunction("polyEvaluate");
      const linkedRoutedMethod = linkedRouted.getFunction("polyEvaluate");
      const inlinedRoutedMethod = inlinedRouted.getFunction("polyEvaluate");

      const [
        directLinkedResult,
        directInlinedResult,
        routedLinkedResult,
        routedInlinedResult,
        directLinkedGas,
        directInlinedGas,
        routedLinkedGas,
        routedInlinedGas,
      ] = await Promise.all([
        linkedDirectMethod.staticCall(...args),
        inlinedDirectMethod.staticCall(...args),
        linkedRoutedMethod.staticCall(...args),
        inlinedRoutedMethod.staticCall(...args),
        linkedDirectMethod.estimateGas(...args),
        inlinedDirectMethod.estimateGas(...args),
        linkedRoutedMethod.estimateGas(...args),
        inlinedRoutedMethod.estimateGas(...args),
      ]);

      expect(directLinkedResult).to.equal(directInlinedResult);
      expect(routedLinkedResult).to.equal(directLinkedResult);
      expect(routedInlinedResult).to.equal(directLinkedResult);

      const arithmeticCalls = degree * 2;
      measurements.push({
        degree,
        arithmeticCalls,
        directLinkedGas,
        directInlinedGas,
        routedLinkedGas,
        routedInlinedGas,
      });

      console.log(
        `POLYNOMIAL_HORNER_AB | degree=${degree} | arithmeticCalls=${arithmeticCalls} | ` +
        `directLinkedGas=${directLinkedGas} | directInlinedGas=${directInlinedGas} | ` +
        `directDelta=${directLinkedGas - directInlinedGas} | ` +
        `directRatio=${formatRatio(directLinkedGas, directInlinedGas)} | ` +
        `routedLinkedGas=${routedLinkedGas} | routedInlinedGas=${routedInlinedGas} | ` +
        `routedDelta=${routedLinkedGas - routedInlinedGas} | ` +
        `routedRatio=${formatRatio(routedLinkedGas, routedInlinedGas)}`
      );
    });
  }

  after(async function () {
    const [mathLibCode, linkedCode, inlinedCode, productionCode] = await Promise.all([
      ethers.provider.getCode(await mathLib.getAddress()),
      ethers.provider.getCode(await linkedFacet.getAddress()),
      ethers.provider.getCode(await inlinedFacet.getAddress()),
      ethers.provider.getCode(await productionFacet.getAddress()),
    ]);
    const [mathLibReceipt, linkedReceipt, inlinedReceipt, productionReceipt] = await Promise.all([
      receiptOf(mathLib),
      receiptOf(linkedFacet),
      receiptOf(inlinedFacet),
      receiptOf(productionFacet),
    ]);

    const linkedRuntimeBytes = byteLength(linkedCode);
    const inlinedRuntimeBytes = byteLength(inlinedCode);
    expect(linkedRuntimeBytes).to.be.lessThan(EIP170_RUNTIME_LIMIT);
    expect(inlinedRuntimeBytes).to.be.lessThan(EIP170_RUNTIME_LIMIT);

    console.log("------------------------------------------------------------");
    console.log("POLYNOMIAL_HORNER_AB_DEPLOYMENT");
    console.log(`MathLib creationBytes=${byteLength(mathLibFactory.bytecode)} | runtimeBytes=${byteLength(mathLibCode)} | deploymentGas=${mathLibReceipt.gasUsed}`);
    console.log(`Linked Horner facet creationBytes=${byteLength(linkedFactory.bytecode)} | runtimeBytes=${linkedRuntimeBytes} | deploymentGas=${linkedReceipt.gasUsed} | EIP170Headroom=${EIP170_RUNTIME_LIMIT - linkedRuntimeBytes}`);
    console.log(`Inlined Horner facet creationBytes=${byteLength(inlinedFactory.bytecode)} | runtimeBytes=${inlinedRuntimeBytes} | deploymentGas=${inlinedReceipt.gasUsed} | EIP170Headroom=${EIP170_RUNTIME_LIMIT - inlinedRuntimeBytes}`);
    console.log(`Current full PolynomialFacet creationBytes=${byteLength(productionFactory.bytecode)} | runtimeBytes=${byteLength(productionCode)} | deploymentGas=${productionReceipt.gasUsed} | EIP170Headroom=${EIP170_RUNTIME_LIMIT - byteLength(productionCode)}`);
    console.log(`Linked first-consumer deploymentGas=${mathLibReceipt.gasUsed + linkedReceipt.gasUsed} (MathLib + Horner facet)`);
    console.log(`Linked incremental-consumer deploymentGas=${linkedReceipt.gasUsed} (shared MathLib already deployed)`);
    console.log(`Diamond selector installationGas: linked=${linkedInstallationGas} | inlined=${inlinedInstallationGas}`);

    const directLinkedTotal = measurements.reduce((sum, item) => sum + item.directLinkedGas, 0n);
    const directInlinedTotal = measurements.reduce((sum, item) => sum + item.directInlinedGas, 0n);
    const routedLinkedTotal = measurements.reduce((sum, item) => sum + item.routedLinkedGas, 0n);
    const routedInlinedTotal = measurements.reduce((sum, item) => sum + item.routedInlinedGas, 0n);
    console.log(`Direct aggregate: linkedGas=${directLinkedTotal} | inlinedGas=${directInlinedTotal} | delta=${directLinkedTotal - directInlinedTotal} | ratio=${formatRatio(directLinkedTotal, directInlinedTotal)}`);
    console.log(`Diamond aggregate: linkedGas=${routedLinkedTotal} | inlinedGas=${routedInlinedTotal} | delta=${routedLinkedTotal - routedInlinedTotal} | ratio=${formatRatio(routedLinkedTotal, routedInlinedTotal)}`);
    console.log("============================================================");
  });
});
