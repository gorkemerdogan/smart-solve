// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IDiamondCut
 * @notice Interface defining the upgrade mechanism for EIP-2535 diamonds.
 *         Supports adding, replacing, and removing function selectors across facets.
 */
interface IDiamondCut {
    enum FacetCutAction { Add, Replace, Remove }

    struct FacetCut {
        address facetAddress;
        FacetCutAction action;
        bytes4[] functionSelectors;
    }

    /** @notice Add/replace/remove facet functions and optionally execute setup call
     *  @param _diamondCut Facet addresses & function selectors
     *  @param _init Optional init contract
     *  @param _calldata Optional init function call
     */
    function diamondCut(
        FacetCut[] calldata _diamondCut,
        address _init,
        bytes calldata _calldata
    ) external;

    event DiamondCut(FacetCut[] _diamondCut, address _init, bytes _calldata);
}