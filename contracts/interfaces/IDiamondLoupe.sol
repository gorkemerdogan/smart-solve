// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
 * =============== IDiamondLoupe.sol ===============
 *
 * Defines the "loupe" API for diamonds.
 * Lets anyone *inspect* the diamond to see which facets
 * and function selectors are currently active.
 *
 * Functions:
 * 1- facets() : returns all facets + their selectors
 * 2- facetFunctionSelectors(address facet) : list of selectors for one facet
 * 3- facetAddresses() : list of all facet addresses
 * 4- facetAddress(bytes4 selector) : facet that implements this function
 */

interface IDiamondLoupe {
    struct Facet {
        address facetAddress;
        bytes4[] functionSelectors;
    }

    /// @notice Get all facets and their selectors
    function facets() external view returns (Facet[] memory facets_);

    /// @notice Get all function selectors provided by a specific facet
    function facetFunctionSelectors(address facet) external view returns (bytes4[] memory);

    /// @notice Get all facet addresses used by the diamond
    function facetAddresses() external view returns (address[] memory);

    /// @notice Get the facet that supports a given selector
    function facetAddress(bytes4 selector) external view returns (address);
}