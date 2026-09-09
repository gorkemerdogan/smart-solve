import { artifacts, ethers } from "hardhat";
import { Interface } from "ethers";
import { readFile } from "node:fs/promises";
import type { SmartSolveDeploymentArtifact } from "./deploy";

const ZERO_BYTES16 = `0x${"00".repeat(16)}`;
const ONE = "0x3fff0000000000000000000000000000";
const TWO = "0x40000000000000000000000000000000";
const THREE = "0x40008000000000000000000000000000";

const KEY_METHODS: Record<string, string> = {
  DiamondCutFacet: "diamondCut",
  DiamondLoupeFacet: "facets",
  OwnershipFacet: "owner",
  NumericConfigFacet: "getRootFindingConfig",
  RootFindingFacet: "rootFindingBisection",
  IntegrationFacet: "integrateSimpson13WithN",
  DifferentiationFacet: "centeredDiff",
  ODESolverFacet: "eulerIter",
  PolynomialFacet: "polyEvaluate",
  MatrixMasterFacet: "createIdentityMatrix",
  LinearSolversFacet: "gaussianElimination",
  SteepestDescentFacet: "steepestDescent",
};

export type DeploymentVerificationResult = {
  diamondAddress: string;
  installedSelectorCount: number;
  rootFindingConfig: { tol: string; minTol: string; maxIter: string };
  polynomialResult: string;
};

function sameAddress(left: string, right: string): boolean {
  return left.toLowerCase() === right.toLowerCase();
}

async function facetInterface(name: string): Promise<Interface> {
  const artifact = await artifacts.readArtifact(name);
  return new Interface(artifact.abi);
}

/**
 * Read-only verification of a recorded production deployment. The artifact is
 * required so expected facet addresses and every installed selector can be
 * checked against the exact deployment rather than only current source ABIs.
 */
export async function verifyDeployment(
  deployment: SmartSolveDeploymentArtifact,
  diamondAddress = deployment.diamondAddress,
): Promise<DeploymentVerificationResult> {
  if (!sameAddress(diamondAddress, deployment.diamondAddress)) {
    throw new Error("DIAMOND_ADDRESS does not match the deployment artifact");
  }

  const chain = await ethers.provider.getNetwork();
  if (chain.chainId.toString() !== deployment.chainId) {
    throw new Error(
      `Chain ID mismatch: artifact=${deployment.chainId}, provider=${chain.chainId.toString()}`,
    );
  }

  const diamondCode = await ethers.provider.getCode(diamondAddress);
  if (diamondCode === "0x") {
    throw new Error(`No contract code at Diamond address ${diamondAddress}`);
  }

  const loupe = await ethers.getContractAt("IDiamondLoupe", diamondAddress);
  const actualFacets = await loupe.facets();
  const actualFacetAddresses = new Set(actualFacets.map((facet) => facet.facetAddress.toLowerCase()));
  const expectedFacetAddresses = new Set(
    Object.values(deployment.facetAddresses).map((address) => address.toLowerCase()),
  );
  const actualSelectorCount = actualFacets.reduce(
    (total, facet) => total + facet.functionSelectors.length,
    0,
  );

  if (actualSelectorCount !== deployment.installedSelectorCount) {
    throw new Error(
      `Installed selector count mismatch: expected=${deployment.installedSelectorCount}, actual=${actualSelectorCount}`,
    );
  }
  if (actualFacetAddresses.size !== expectedFacetAddresses.size) {
    throw new Error(
      `Installed facet count mismatch: expected=${expectedFacetAddresses.size}, actual=${actualFacetAddresses.size}`,
    );
  }

  for (const [name, expectedAddress] of Object.entries(deployment.facetAddresses)) {
    if (!actualFacetAddresses.has(expectedAddress.toLowerCase())) {
      throw new Error(`${name} is not installed at expected address ${expectedAddress}`);
    }
    if (await ethers.provider.getCode(expectedAddress) === "0x") {
      throw new Error(`${name} has no runtime code at ${expectedAddress}`);
    }
  }

  for (const [facetName, selectors] of Object.entries(deployment.facetSelectors)) {
    const expectedAddress = deployment.facetAddresses[facetName];
    if (!expectedAddress) throw new Error(`Artifact omits address for ${facetName}`);

    await Promise.all(selectors.map(async (selector) => {
      const actualAddress = await loupe.facetAddress(selector);
      if (!sameAddress(actualAddress, expectedAddress)) {
        throw new Error(
          `Selector ${selector} routes to ${actualAddress}, expected ${facetName} at ${expectedAddress}`,
        );
      }
    }));
  }

  for (const [facetName, method] of Object.entries(KEY_METHODS)) {
    const expectedAddress = deployment.facetAddresses[facetName];
    if (!expectedAddress) throw new Error(`Artifact omits required facet ${facetName}`);
    const selector = (await facetInterface(facetName)).getFunction(method)!.selector;
    const actualAddress = await loupe.facetAddress(selector);
    if (!sameAddress(actualAddress, expectedAddress)) {
      throw new Error(`${facetName}.${method} is not routed to its expected facet`);
    }
    console.log(`Verified selector | ${facetName}.${method} -> ${actualAddress}`);
  }

  const ownership = await ethers.getContractAt("IERC173", diamondAddress);
  const owner = await ownership.owner();
  if (!sameAddress(owner, deployment.diamondOwner)) {
    throw new Error(`Diamond owner mismatch: expected=${deployment.diamondOwner}, actual=${owner}`);
  }

  const numericConfig = await ethers.getContractAt("NumericConfigFacet", diamondAddress);
  const [tol, minTol, maxIter] = await numericConfig.getRootFindingConfig();
  const diffStep = await numericConfig.getDiffStep();
  if (tol === ZERO_BYTES16 || minTol === ZERO_BYTES16 || diffStep === ZERO_BYTES16 || maxIter === 0n) {
    throw new Error("Numeric configuration getters returned unset raw values instead of effective defaults");
  }
  console.log(`Effective root config | tol=${tol} | minTol=${minTol} | maxIter=${maxIter}`);
  console.log(`Effective differentiation step | h=${diffStep}`);

  const polynomial = await ethers.getContractAt("PolynomialFacet", diamondAddress);
  const polynomialResult = await polynomial.polyEvaluate([ONE, ONE], TWO);
  if (polynomialResult.toLowerCase() !== THREE) {
    throw new Error(`Diamond polynomial proxy call returned ${polynomialResult}, expected ${THREE}`);
  }
  console.log(`Verified Diamond proxy call | polyEvaluate([1, 1], 2) = ${polynomialResult}`);

  return {
    diamondAddress,
    installedSelectorCount: actualSelectorCount,
    rootFindingConfig: { tol, minTol, maxIter: maxIter.toString() },
    polynomialResult,
  };
}

async function main() {
  const artifactPath = process.env.DEPLOYMENT_ARTIFACT;
  if (!artifactPath) {
    throw new Error("Set DEPLOYMENT_ARTIFACT to the deployment JSON produced by scripts/deploy.ts");
  }

  const deployment = JSON.parse(
    await readFile(artifactPath, "utf8"),
  ) as SmartSolveDeploymentArtifact;
  const diamondAddress = process.env.DIAMOND_ADDRESS ?? deployment.diamondAddress;
  const result = await verifyDeployment(deployment, diamondAddress);
  console.log("Deployment verification passed:", JSON.stringify(result));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
