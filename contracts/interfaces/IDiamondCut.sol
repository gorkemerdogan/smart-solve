// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
 * =============== IDiamondCut.sol ===============
 *
 * Defines the "upgrade mechanism" for diamonds.
 * Lets the diamond owner add, replace, or remove functions
 * by pointing them to different facets.
 *
 * Functions:
 * - diamondCut(facetCuts, init, calldata)
 *   * facetCuts = array of changes (add/replace/remove)
 *   * init = address of contract to run setup code after upgrade
 *   * calldata = the encoded function call for `init`
 *
 * Enums & Structs:
 * - FacetCutAction : { Add, Replace, Remove }
 * - FacetCut : { facetAddress, functionSelectors, action }
 *
 * Events:
 * - DiamondCut : emitted after every upgrade
 */

interface IDiamondCut {
    enum FacetCutAction { Add, Replace, Remove }

    struct FacetCut {
        address facetAddress;
        FacetCutAction action;
        bytes4[] functionSelectors;
    }

    /// @notice Add/replace/remove facet functions and optionally execute setup call
    /// @param _diamondCut Facet addresses & function selectors
    /// @param _init Optional init contract
    /// @param _calldata Optional init function call
    function diamondCut(
        FacetCut[] calldata _diamondCut,
        address _init,
        bytes calldata _calldata
    ) external;

    event DiamondCut(FacetCut[] _diamondCut, address _init, bytes _calldata);
}