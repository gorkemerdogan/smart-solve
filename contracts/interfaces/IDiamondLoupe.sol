// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IDiamondLoupe
 * @notice Interface defining the inspection (loupe) functions for EIP-2535 diamonds.
 *         Provides read-only access to facet addresses and their associated
 *         function selectors.
 * @dev Enables enumeration of facets, selectors within each facet, the full
 *      facet address list, and the facet responsible for a specific function
 *      selector.
 *
 *      Functions:
 *      - facets(): returns all facets and their selectors
 *      - facetFunctionSelectors(facet): returns selectors for the specified facet
 *      - facetAddresses(): returns all facet addresses
 *      - facetAddress(selector): returns the facet implementing the selector
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