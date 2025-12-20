import { expect } from "chai";
import { ethers } from "hardhat";
import { FunctionFragment } from "ethers";

/**
 * @title  SmartSolve Diamond Initialization Test
 * @notice Verifies correct deployment, initialization, and basic functionality
 *         of the SmartSolve Diamond contract and its core facets.
 *
 * @dev    This test deploys the DiamondCutFacet, the SmartSolve diamond proxy, and the
 *         OwnershipFacet and DiamondLoupeFacet, then wires them together using
 *         'diamondCut'. Function selectors are derived dynamically from facet ABIs.
 *         The test validates ownership initialization, ownership transfer, and correct
 *         facet registration via the Diamond Loupe interface.
 * 
 *         Limitations:
 *          - Covers only core Diamond mechanics; does not test application-specific facets.
 *          - Assumes compliant implementations of IERC173, IDiamondCut, and IDiamondLoupe.
 */

// Helper enum for readability
const FacetCutAction = {
  Add: 0,
  Replace: 1,
  Remove: 2,
};

describe("SmartSolve Diamond", function () {
  it("deploys and initializes facets", async function () {
    const [deployer, other] = await ethers.getSigners();

    // 1. Deploy DiamondCutFacet
    const DiamondCutFacetFactory = await ethers.getContractFactory("DiamondCutFacet");
    const diamondCutFacet = await DiamondCutFacetFactory.deploy();
    await diamondCutFacet.waitForDeployment();

    // 2. Deploy the Diamond - SmartSolve
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

    // 5. Build selectors (Simplified)
    const ownershipSelectors = OwnershipFacetFactory.interface
      .fragments
      .filter(FunctionFragment.isFragment)
      .map(frag => frag.selector);

    const loupeSelectors = DiamondLoupeFacetFactory.interface
      .fragments
      .filter(FunctionFragment.isFragment)
      .map(frag => frag.selector);

    // 6. Use diamondCut to add facets
    const diamondCut = await ethers.getContractAt("IDiamondCut", await smartSolve.getAddress());

    const cut = [
      {
        facetAddress: await ownershipFacet.getAddress(),
        action: FacetCutAction.Add, // Use enum for clarity
        functionSelectors: ownershipSelectors,
      },
      {
        facetAddress: await diamondLoupeFacet.getAddress(),
        action: FacetCutAction.Add, // Use enum for clarity
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

    // 9. Check Loupe facet (Strengthened Assertions)
    const loupe = await ethers.getContractAt("IDiamondLoupe", await smartSolve.getAddress());
    const facetAddresses = await loupe.facetAddresses();

    // 10. Check that the array contains exactly the 3 facets we expect
    expect(facetAddresses).to.have.lengthOf(3);

    const expectedFacetAddrs = [
      await diamondCutFacet.getAddress(),
      await ownershipFacet.getAddress(),
      await diamondLoupeFacet.getAddress(),
    ];

    expect(
      facetAddresses.map(a => a.toLowerCase())
    ).to.have.members(
      expectedFacetAddrs.map(a => a.toLowerCase())
    );

  });
});