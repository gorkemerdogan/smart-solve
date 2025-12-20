// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IDiamondLoupe } from "../interfaces/IDiamondLoupe.sol";
import { LibSmartSolve } from "../libraries/LibSmartSolve.sol";

/**
 * @title  DiamondLoupeFacet
 * @notice Implements the IDiamondLoupe interface (EIP-2535).
 *         Allows querying facet addresses, selectors, and mappings from selectors to facets.
 */

contract DiamondLoupeFacet is IDiamondLoupe {
    function facets() external view override returns (Facet[] memory facets_) {
        LibSmartSolve.DiamondStorage storage ds = LibSmartSolve.diamondStorage();
        uint256 numFacets = ds.facetAddresses.length;

        facets_ = new Facet[](numFacets);
        for (uint256 i; i < numFacets; i++) {
            address facetAddr = ds.facetAddresses[i];
            facets_[i].facetAddress = facetAddr;
            facets_[i].functionSelectors = ds.facetFunctionSelectors[facetAddr].selectors;
        }
    }

    function facetFunctionSelectors(address _facet)
        external
        view
        override
        returns (bytes4[] memory facetFunctionSelectors_)
    {
        facetFunctionSelectors_ = LibSmartSolve
            .diamondStorage()
            .facetFunctionSelectors[_facet]
            .selectors;
    }

    function facetAddresses() external view override returns (address[] memory facetAddresses_) {
        facetAddresses_ = LibSmartSolve.diamondStorage().facetAddresses;
    }

    function facetAddress(bytes4 _functionSelector) external view override returns (address facetAddress_) {
        facetAddress_ = LibSmartSolve
            .diamondStorage()
            .selectorToFacetAndPosition[_functionSelector]
            .facetAddress;
    }
}