import { expect } from "chai";
import { artifacts } from "hardhat";

/** EIP-170 maximum deployed runtime bytecode size. */
const EIP170_RUNTIME_LIMIT = 24_576;

/**
 * Surface facets that are becoming expensive before they approach the hard
 * EIP-170 limit.  This is informational unless a facet-specific budget is
 * also configured below.
 */
const PROJECT_WARNING_THRESHOLD = 20_000;

/**
 * Production facets installed by the deployment helpers.  The tight budgets
 * deliberately leave reviewable headroom below EIP-170 for the largest
 * contracts while allowing normal maintenance of smaller facets.
 */
const PRODUCTION_FACETS: ReadonlyArray<{
  name: string;
  maxRuntimeBytes?: number;
}> = [
  { name: "DiamondCutFacet" },
  { name: "DiamondLoupeFacet" },
  { name: "OwnershipFacet" },
  { name: "NumericConfigFacet" },
  { name: "RootFindingFacet" },
  { name: "IntegrationFacet" },
  { name: "DifferentiationFacet" },
  { name: "ODESolverFacet" },
  { name: "PolynomialFacet" },
  // MatrixMasterFacet is currently the closest production facet to EIP-170.
  { name: "MatrixMasterFacet", maxRuntimeBytes: 23_000 },
  { name: "LinearSolversFacet", maxRuntimeBytes: 20_000 },
  { name: "SteepestDescentFacet" },
];

function runtimeByteLength(deployedBytecode: string): number {
  expect(deployedBytecode, "artifact must contain runtime bytecode").to.match(/^0x/);
  expect(
    (deployedBytecode.length - 2) % 2,
    "runtime bytecode must contain complete bytes",
  ).to.equal(0);

  return (deployedBytecode.length - 2) / 2;
}

describe("Production facet runtime bytecode budgets", function () {
  it("reports every facet size and enforces EIP-170 plus project budgets", async function () {
    for (const facet of PRODUCTION_FACETS) {
      const artifact = await artifacts.readArtifact(facet.name);
      const runtimeBytes = runtimeByteLength(artifact.deployedBytecode);
      const headroom = EIP170_RUNTIME_LIMIT - runtimeBytes;
      const projectBudget = facet.maxRuntimeBytes ?? "none";
      const report = [
        "BYTECODE_SIZE",
        `facet=${facet.name}`,
        `runtimeBytes=${runtimeBytes}`,
        `eip170Headroom=${headroom}`,
        `eip170Limit=${EIP170_RUNTIME_LIMIT}`,
        `projectBudget=${projectBudget}`,
      ].join(" | ");

      if (runtimeBytes >= PROJECT_WARNING_THRESHOLD) {
        console.warn(`BYTECODE_WARNING | ${report}`);
      } else {
        console.log(report);
      }

      expect(
        runtimeBytes,
        `${facet.name} exceeds the EIP-170 deployed runtime bytecode limit`,
      ).to.be.at.most(EIP170_RUNTIME_LIMIT);

      if (facet.maxRuntimeBytes !== undefined) {
        expect(
          runtimeBytes,
          `${facet.name} exceeds its project runtime bytecode budget`,
        ).to.be.at.most(facet.maxRuntimeBytes);
      }
    }
  });
});
