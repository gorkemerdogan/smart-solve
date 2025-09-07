import { expect } from "chai";
import { ethers } from "hardhat";

/*
 * =============== smart-solve.test.ts ===============
 *
 * Verify deployment of SmartSolve (diamond).
 * Ensure OwnershipFacet + DiamondLoupeFacet are attached.
 * Test basic functionality: owner, facet discovery.
 */

describe("SmartSolve Diamond", function () {
  it("deploys and initializes facets", async function () {
    const [deployer, other] = await ethers.getSigners();

    // 1. Deploy DiamondCutFacet
    const DiamondCutFacetFactory = await ethers.getContractFactory("DiamondCutFacet");
    const diamondCutFacet = await DiamondCutFacetFactory.deploy();
    await diamondCutFacet.waitForDeployment();

    // 2. Deploy SmartSolve
    const SmartSolveFactory = await ethers.getContractFactory("SmartSolve");
    const smartSolve = await SmartSolveFactory.deploy(
      deployer.address,
      await diamondCutFacet.getAddress()
    );
    await smartSolve.waitForDeployment();

    // 3. Deploy OwnershipFacet
    const OwnershipFacetFactory = await ethers.getContractFactory("OwnershipFacet");
    const ownershipFacet = await OwnershipFacetFactory.deploy();
    await ownershipFacet.waitForDeployment();

    // 4. Deploy DiamondLoupeFacet
    const DiamondLoupeFacetFactory = await ethers.getContractFactory("DiamondLoupeFacet");
    const diamondLoupeFacet = await DiamondLoupeFacetFactory.deploy();
    await diamondLoupeFacet.waitForDeployment();

    // 5. Build selectors from factories (safe in ethers v6)
    // OwnershipFacet selectors
    const ownershipSelectors = Object.values(OwnershipFacetFactory.interface.fragments)
      .filter((frag) => frag.type === "function")
      .map((frag) =>
        OwnershipFacetFactory.interface.getFunction(frag.format("sighash")).selector
      );

    // LoupeFacet selectors
    const loupeSelectors = Object.values(DiamondLoupeFacetFactory.interface.fragments)
      .filter((frag) => frag.type === "function")
      .map((frag) =>
        DiamondLoupeFacetFactory.interface.getFunction(frag.format("sighash")).selector
      );

    // 6. Use diamondCut to add facets
    const diamondCut = await ethers.getContractAt("IDiamondCut", await smartSolve.getAddress());

    const cut = [
      {
        facetAddress: await ownershipFacet.getAddress(),
        action: 0, // Add
        functionSelectors: ownershipSelectors,
      },
      {
        facetAddress: await diamondLoupeFacet.getAddress(),
        action: 0, // Add
        functionSelectors: loupeSelectors,
      },
    ];

    const tx = await diamondCut.diamondCut(cut, ethers.ZeroAddress, "0x");
    await tx.wait();

    // 7. Check owner
    const ownerFacet = await ethers.getContractAt("IERC173", await smartSolve.getAddress());
    expect(await ownerFacet.owner()).to.equal(deployer.address);

    // 8. Transfer ownership
    await ownerFacet.transferOwnership(other.address);
    expect(await ownerFacet.owner()).to.equal(other.address);

    // 9. Check Loupe facet
    const loupe = await ethers.getContractAt("IDiamondLoupe", await smartSolve.getAddress());
    const facetAddresses = await loupe.facetAddresses();

    expect(facetAddresses).to.include(await ownershipFacet.getAddress());
    expect(facetAddresses).to.include(await diamondLoupeFacet.getAddress());
  });
});