import { artifacts, ethers, network } from "hardhat";
import { ContractTransactionResponse, FunctionFragment, Interface } from "ethers";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

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
  "NumericConfigFacet",
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

export type DeploymentGasSummary = {
  contracts: Record<string, string>;
  diamondCutInstallation: string;
  total: string;
};

export type SmartSolveDeploymentArtifact = {
  format: "smart-solve-deployment-v1";
  network: string;
  chainId: string;
  deployer: string;
  diamondOwner: string;
  diamondAddress: string;
  mathLibAddress: string;
  facetAddresses: Record<string, string>;
  facetSelectors: Record<string, string[]>;
  installedSelectorCount: number;
  deploymentGas: DeploymentGasSummary;
  compiler?: {
    solcVersion: string;
    solcLongVersion: string;
    optimizer: { enabled?: boolean; runs?: number };
    viaIR?: boolean;
    evmVersion?: string;
  };
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
  facetSelectors: Record<string, string[]>;
  installedSelectorCount: number;
  deploymentGas: DeploymentGasSummary;
  deploymentGasUsed: bigint;
};

/**
 * Deploys the production SmartSolve Diamond: its management facets and every
 * currently shipped numerical facet. MathLib is deployed once and linked into
 * each facet that has an external MathLib reference.
 */
export async function deployFullSmartSolve(): Promise<FullSmartSolveDeployment> {
  const [deployer] = await ethers.getSigners();
  const deploymentContracts: Array<{ name: string; contract: DeployableContract }> = [];

  const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
  const mathLib = await MathLib.deploy();
  await mathLib.waitForDeployment();
  deploymentContracts.push({ name: "MathLib", contract: mathLib });

  const DiamondCutFacet = await ethers.getContractFactory("DiamondCutFacet");
  const diamondCutFacet = await DiamondCutFacet.deploy();
  await diamondCutFacet.waitForDeployment();
  deploymentContracts.push({ name: "DiamondCutFacet", contract: diamondCutFacet });

  const SmartSolve = await ethers.getContractFactory("SmartSolve");
  const smartSolve = await SmartSolve.deploy(
    deployer.address,
    await diamondCutFacet.getAddress()
  );
  await smartSolve.waitForDeployment();
  deploymentContracts.push({ name: "SmartSolve", contract: smartSolve });

  const facetNames = ["OwnershipFacet", "DiamondLoupeFacet", ...NUMERIC_FACETS] as const;
  const facetAddresses: Record<string, string> = {
    DiamondCutFacet: await diamondCutFacet.getAddress(),
  };
  const facetSelectors: Record<string, string[]> = {
    DiamondCutFacet: selectorsOf(diamondCutFacet),
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
    deploymentContracts.push({ name, contract: facet });

    facetAddresses[name] = await facet.getAddress();
    facetSelectors[name] = selectorsOf(facet);
    cut.push({
      facetAddress: facetAddresses[name],
      action: FacetCutAction.Add,
      functionSelectors: facetSelectors[name],
    });
  }

  const diamondCut = await ethers.getContractAt("IDiamondCut", await smartSolve.getAddress());
  const installationReceipt = await (
    await diamondCut.diamondCut(cut, ethers.ZeroAddress, "0x")
  ).wait();
  if (!installationReceipt) {
    throw new Error("Missing Diamond installation receipt");
  }

  const contractGasEntries = await Promise.all(
    deploymentContracts.map(async ({ name, contract }) => [name, await gasUsedByDeployment(contract)] as const)
  );
  const contractGas = Object.fromEntries(contractGasEntries);
  const deploymentGasUsed = Object.values(contractGas)
    .reduce((total, gasUsed) => total + gasUsed, installationReceipt.gasUsed);

  const diamondAddress = await smartSolve.getAddress();
  const loupe = await ethers.getContractAt("IDiamondLoupe", diamondAddress);
  const installedSelectorCount = (await loupe.facets())
    .reduce((total: number, facet: { functionSelectors: string[] }) => total + facet.functionSelectors.length, 0);
  const deploymentGas: DeploymentGasSummary = {
    contracts: Object.fromEntries(
      Object.entries(contractGas).map(([name, gasUsed]) => [name, gasUsed.toString()])
    ),
    diamondCutInstallation: installationReceipt.gasUsed.toString(),
    total: deploymentGasUsed.toString(),
  };

  return {
    diamondAddress,
    mathLibAddress: await mathLib.getAddress(),
    facetAddresses,
    facetSelectors,
    installedSelectorCount,
    deploymentGas,
    deploymentGasUsed,
  };
}

async function compilerMetadata(): Promise<SmartSolveDeploymentArtifact["compiler"]> {
  const buildInfo = await artifacts.getBuildInfo("contracts/SmartSolve.sol:SmartSolve");
  if (!buildInfo) return undefined;

  const settings = buildInfo.input.settings;
  return {
    solcVersion: buildInfo.solcVersion,
    solcLongVersion: buildInfo.solcLongVersion,
    optimizer: {
      enabled: settings.optimizer.enabled,
      runs: settings.optimizer.runs,
    },
    viaIR: settings.viaIR,
    evmVersion: settings.evmVersion,
  };
}

export async function createDeploymentArtifact(
  deployment: FullSmartSolveDeployment,
): Promise<SmartSolveDeploymentArtifact> {
  const [deployer] = await ethers.getSigners();
  const chain = await ethers.provider.getNetwork();
  const ownership = await ethers.getContractAt("IERC173", deployment.diamondAddress);

  return {
    format: "smart-solve-deployment-v1",
    network: network.name,
    chainId: chain.chainId.toString(),
    deployer: deployer.address,
    diamondOwner: await ownership.owner(),
    diamondAddress: deployment.diamondAddress,
    mathLibAddress: deployment.mathLibAddress,
    facetAddresses: deployment.facetAddresses,
    facetSelectors: deployment.facetSelectors,
    installedSelectorCount: deployment.installedSelectorCount,
    deploymentGas: deployment.deploymentGas,
    compiler: await compilerMetadata(),
  };
}

export async function writeDeploymentArtifact(
  deployment: FullSmartSolveDeployment,
): Promise<{ artifact: SmartSolveDeploymentArtifact; path: string }> {
  const artifact = await createDeploymentArtifact(deployment);
  const defaultPath = join(
    "deployments",
    `smart-solve-${artifact.network}-${artifact.chainId}-${artifact.diamondAddress.toLowerCase()}.json`,
  );
  const path = process.env.DEPLOYMENT_ARTIFACT ?? defaultPath;
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  return { artifact, path };
}

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying full SmartSolve with account:", deployer.address);

  const deployment = await deployFullSmartSolve();
  const { path } = await writeDeploymentArtifact(deployment);
  console.log("MathLib deployed at:", deployment.mathLibAddress);
  console.log("SmartSolve (diamond) deployed at:", deployment.diamondAddress);
  console.log("Installed facets:");
  for (const [name, address] of Object.entries(deployment.facetAddresses)) {
    console.log(`  ${name}: ${address}`);
  }
  console.log("Installed selector count:", deployment.installedSelectorCount);
  console.log("Full deployment and installation gas:", deployment.deploymentGas.total);
  console.log("Deployment artifact:", path);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
