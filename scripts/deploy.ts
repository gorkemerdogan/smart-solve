import { ethers } from "hardhat";
import { ContractTransactionResponse, FunctionFragment, Interface } from "ethers";

const FacetCutAction = {
  Add: 0,
} as const;

const NUMERIC_FACETS = [
  "NumericConfigFacet",
  "RootFindingFacet",
  "IntegrationFacet",
  "DifferentiationFacet",
  "ODESolverFacet",
  "PolynomialFacet",
  "MatrixMasterFacet",
  "LinearSolversFacet",
  "SteepestDescentFacet",
] as const;

const MATHLIB_NUMERIC_FACETS = new Set<string>([
  "RootFindingFacet",
  "IntegrationFacet",
  "DifferentiationFacet",
  "ODESolverFacet",
  "PolynomialFacet",
  "MatrixMasterFacet",
  "LinearSolversFacet",
  "SteepestDescentFacet",
]);

type DeployableContract = {
  deploymentTransaction(): ContractTransactionResponse | null;
};

type FacetContract = {
  interface: Interface;
};

function selectorsOf(facet: FacetContract): string[] {
  return facet.interface.fragments
    .filter(FunctionFragment.isFragment)
    .map((fragment) => fragment.selector);
}

async function gasUsedByDeployment(contract: DeployableContract): Promise<bigint> {
  const transaction = contract.deploymentTransaction();
  if (!transaction) {
    throw new Error("Missing deployment transaction");
  }
  const receipt = await transaction.wait();
  if (!receipt) {
    throw new Error("Missing deployment receipt");
  }
  return receipt.gasUsed;
}

export type FullSmartSolveDeployment = {
  diamondAddress: string;
  mathLibAddress: string;
  facetAddresses: Record<string, string>;
  deploymentGasUsed: bigint;
};

/**
 * Deploys the production SmartSolve Diamond: its management facets and every
 * currently shipped numerical facet. MathLib is deployed once and linked into
 * each facet that has an external MathLib reference.
 */
export async function deployFullSmartSolve(): Promise<FullSmartSolveDeployment> {
  const [deployer] = await ethers.getSigners();
  const deploymentContracts: DeployableContract[] = [];

  const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
  const mathLib = await MathLib.deploy();
  await mathLib.waitForDeployment();
  deploymentContracts.push(mathLib);

  const DiamondCutFacet = await ethers.getContractFactory("DiamondCutFacet");
  const diamondCutFacet = await DiamondCutFacet.deploy();
  await diamondCutFacet.waitForDeployment();
  deploymentContracts.push(diamondCutFacet);

  const SmartSolve = await ethers.getContractFactory("SmartSolve");
  const smartSolve = await SmartSolve.deploy(
    deployer.address,
    await diamondCutFacet.getAddress()
  );
  await smartSolve.waitForDeployment();
  deploymentContracts.push(smartSolve);

  const facetNames = ["OwnershipFacet", "DiamondLoupeFacet", ...NUMERIC_FACETS] as const;
  const facetAddresses: Record<string, string> = {
    DiamondCutFacet: await diamondCutFacet.getAddress(),
  };
  const cut = [];

  for (const name of facetNames) {
    const factory = MATHLIB_NUMERIC_FACETS.has(name)
      ? await ethers.getContractFactory(name, {
          libraries: { MathLib: await mathLib.getAddress() },
        })
      : await ethers.getContractFactory(name);
    const facet = await factory.deploy();
    await facet.waitForDeployment();
    deploymentContracts.push(facet);

    facetAddresses[name] = await facet.getAddress();
    cut.push({
      facetAddress: facetAddresses[name],
      action: FacetCutAction.Add,
      functionSelectors: selectorsOf(facet),
    });
  }

  const diamondCut = await ethers.getContractAt("IDiamondCut", await smartSolve.getAddress());
  const installationReceipt = await (
    await diamondCut.diamondCut(cut, ethers.ZeroAddress, "0x")
  ).wait();
  if (!installationReceipt) {
    throw new Error("Missing Diamond installation receipt");
  }

  const deploymentGasUsed = (
    await Promise.all(deploymentContracts.map(gasUsedByDeployment))
  ).reduce((total, gasUsed) => total + gasUsed, installationReceipt.gasUsed);

  return {
    diamondAddress: await smartSolve.getAddress(),
    mathLibAddress: await mathLib.getAddress(),
    facetAddresses,
    deploymentGasUsed,
  };
}

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying full SmartSolve with account:", deployer.address);

  const deployment = await deployFullSmartSolve();
  console.log("MathLib deployed at:", deployment.mathLibAddress);
  console.log("SmartSolve (diamond) deployed at:", deployment.diamondAddress);
  console.log("Installed facets:");
  for (const [name, address] of Object.entries(deployment.facetAddresses)) {
    console.log(`  ${name}: ${address}`);
  }
  console.log("Full deployment and installation gas:", deployment.deploymentGasUsed.toString());
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
