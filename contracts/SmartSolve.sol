// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LibSmartSolve } from "./libraries/LibSmartSolve.sol";
import { IDiamondCut } from "./interfaces/IDiamondCut.sol";

/*
 * =============== SmartSolve.sol ===============
 *
 * The "diamond contract" (the proxy).
 * Holds the storage (via LibSmartSolve).
 * Forwards calls to the right facet using `delegatecall`.
 *
 * Constructor:
 * - Sets the initial owner
 * - Registers the DiamondCutFacet so upgrades are possible
 *
 * Functions:
 * - fallback() → catches all calls, routes to correct facet
 * - receive() → handles plain ETH transfers
 */

contract SmartSolve {
    constructor(address _owner, address _diamondCutFacet) {
        // Set contract owner
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

        LibSmartSolve.diamondCut(cut, address(0), "");
    }

    /// @notice Routes calls to the right facet
    fallback() external payable {
        LibSmartSolve.DiamondStorage storage ds = LibSmartSolve.diamondStorage();

        // Find facet for the function selector
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

    /// @notice Allows contract to receive ETH
    receive() external payable {}
}
