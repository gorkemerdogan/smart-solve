import { ethers } from "hardhat";
import { Contract } from "ethers";

/*
 * =============== deploy.ts ===============
 *
 * Deploys the full SmartSolve diamond system.
 *   1. Deploy DiamondCutFacet (for upgrades)
 *   2. Deploy SmartSolve (the diamond proxy)
 *   3. Deploy OwnershipFacet + DiamondLoupeFacet
 *   4. Use diamondCut to add Ownership + Loupe functions
 */

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with account:", deployer.address);

  // 1. Deploy DiamondCutFacet
  const DiamondCutFacet = await ethers.getContractFactory("DiamondCutFacet");
  const diamondCutFacet = await DiamondCutFacet.deploy();
  await diamondCutFacet.waitForDeployment();
  console.log("DiamondCutFacet deployed at:", await diamondCutFacet.getAddress());

  // 2. Deploy SmartSolve (the diamond proxy)
  const SmartSolve = await ethers.getContractFactory("SmartSolve");
  const smartSolve = await SmartSolve.deploy(await deployer.getAddress(), await diamondCutFacet.getAddress());
  await smartSolve.waitForDeployment();
  console.log("SmartSolve (diamond) deployed at:", await smartSolve.getAddress());

  // 3. Deploy OwnershipFacet
  const OwnershipFacet = await ethers.getContractFactory("OwnershipFacet");
  const ownershipFacet = await OwnershipFacet.deploy();
  await ownershipFacet.waitForDeployment();
  console.log("OwnershipFacet deployed at:", await ownershipFacet.getAddress());

  // 4. Deploy DiamondLoupeFacet
  const DiamondLoupeFacet = await ethers.getContractFactory("DiamondLoupeFacet");
  const diamondLoupeFacet = await DiamondLoupeFacet.deploy();
  await diamondLoupeFacet.waitForDeployment();
  console.log("DiamondLoupeFacet deployed at:", await diamondLoupeFacet.getAddress());

  // 5. Add Ownership + Loupe facets via diamondCut
  const diamondCut = await ethers.getContractAt("IDiamondCut", await smartSolve.getAddress());

  const cut = [
    {
      facetAddress: await ownershipFacet.getAddress(),
      action: 0, // Add
      functionSelectors: Object.keys(ownershipFacet.interface.functions).map(fn =>
        ownershipFacet.interface.getSighash(fn)
      ),
    },
    {
      facetAddress: await diamondLoupeFacet.getAddress(),
      action: 0, // Add
      functionSelectors: Object.keys(diamondLoupeFacet.interface.functions).map(fn =>
        diamondLoupeFacet.interface.getSighash(fn)
      ),
    },
  ];

  const tx = await diamondCut.diamondCut(cut, ethers.ZeroAddress, "0x");
  await tx.wait();

  console.log("SmartSolve initialized with OwnershipFacet + DiamondLoupeFacet.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});