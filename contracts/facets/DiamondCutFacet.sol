// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IDiamondCut } from "../interfaces/IDiamondCut.sol";
import { LibSmartSolve } from "../libraries/LibSmartSolve.sol";

/*
 * =============== DiamondCutFacet.sol ===============
 *
 * Implements IDiamondCut.
 * Exposes the diamondCut() function so the owner can
 * add/replace/remove function selectors to facets.
 *
 * The actual logic lives in LibSmartSolve.diamondCut().
 * This facet is just the external entry point for users.
 *
 * Access control:
 * Only the contract owner (IERC173) can call diamondCut().
 */

contract DiamondCutFacet is IDiamondCut {
    /// @notice Add/replace/remove functions and optionally execute a function
    ///         with delegatecall, often used for initialization
    /// @param _diamondCut Array of facet addresses and function selectors
    /// @param _init Address of contract or facet to execute calldata on
    /// @param _calldata Calldata to execute with delegatecall
    function diamondCut(
        FacetCut[] calldata _diamondCut,
        address _init,
        bytes calldata _calldata
    ) external override {
        // Only the owner can perform upgrades
        LibSmartSolve.enforceIsContractOwner();
        // Delegate to the library's implementation
        LibSmartSolve.diamondCut(_diamondCut, _init, _calldata);
    }
}