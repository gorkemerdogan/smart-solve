// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LibSmartSolve } from "./libraries/LibSmartSolve.sol";
import { IDiamondCut } from "./interfaces/IDiamondCut.sol";

/**
 * @title SmartSolve
 * @notice Primary diamond (EIP-2535) proxy contract for the SmartSolve system.
 *
 * @dev
 * Implements the diamond storage and facet routing pattern using LibSmartSolve.
 * This contract contains:
 *   - Constructor: initializes ownership and installs the DiamondCutFacet
 *   - fallback(): routes function selectors to the correct facet via delegatecall
 *   - receive(): enables the diamond to accept ETH
 *
 * The diamond itself stores:
 *   - Facet registry
 *   - Selector-to-facet mapping
 *   - Ownership and supported interface data
 *
 * The actual logic is implemented in facets; SmartSolve acts strictly as the proxy.
 */
contract SmartSolve {

    /**
     * @notice Deploys the diamond and installs initial upgrade capability.
     *
     * @param _owner Address to set as the initial contract owner.
     * @param _diamondCutFacet Address of the facet providing the diamondCut function.
     *
     * @dev
     * Steps performed:
     *   1. Set contract owner in diamond storage.
     *   2. Register `diamondCut` selector under the provided facet, enabling upgrades.
     *   3. No initialization call is executed (init = address(0)).
     */
    constructor(address _owner, address _diamondCutFacet) {
        LibSmartSolve.setContractOwner(_owner);

        // Register diamondCut function so upgrades are possible
        bytes4[] memory selectors = new bytes4[](1);
        selectors[0] = IDiamondCut.diamondCut.selector;

        IDiamondCut.FacetCut[] memory cut = new IDiamondCut.FacetCut[](1);
        cut[0] = IDiamondCut.FacetCut({
            facetAddress: _diamondCutFacet,
            functionSelectors: selectors,
            action: IDiamondCut.FacetCutAction.Add
        });

        // Store facet and finalize upgrade step
        LibSmartSolve.diamondCut(cut, address(0), "");
    }

    /**
     * @notice Fallback function routing all non-existing function calls to the correct facet.
     *
     * @dev
     * Mechanism:
     *   - Looks up msg.sig in selectorToFacetAndPosition
     *   - Executes delegatecall to the corresponding facet
     *   - Returns or bubbles up any revert reason
     *
     * Reverts if no facet implements the given selector.
     */
    fallback() external payable {
        LibSmartSolve.DiamondStorage storage ds = LibSmartSolve.diamondStorage();

        // Locate facet for the function selector
        address facet = ds.selectorToFacetAndPosition[msg.sig].facetAddress;
        require(facet != address(0), "SmartSolve: Function not found");

        // Forward call to facet
        assembly {
            calldatacopy(0, 0, calldatasize())
            let result := delegatecall(gas(), facet, 0, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            switch result
                case 0 {
                    revert(0, returndatasize())
                }
                default {
                    return(0, returndatasize())
                }
        }
    }

    /**
     * @notice Accepts direct ETH transfers sent to the diamond.
     * @dev ETH may be consumed by facets implementing payable logic.
     */
    receive() external payable {}
}
