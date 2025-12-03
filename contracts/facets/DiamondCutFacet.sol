// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IDiamondCut } from "../interfaces/IDiamondCut.sol";
import { LibSmartSolve } from "../libraries/LibSmartSolve.sol";

/**
 * @title DiamondCutFacet
 * @notice Implements IDiamondCut.
 *         Exposes the diamondCut() function so the owner can
 *         add/replace/remove function selectors to facets.
 *  
 *         Delegates all logic to LibSmartSolve.diamondCut().
 *         This facet only exposes the external entry point required by the diamond standard.
 *         Access control is restricted to the contract owner (IERC173).
 */

contract DiamondCutFacet is IDiamondCut {
    
    /** 
     * @notice Add/replace/remove functions and optionally execute a function
     *         with delegatecall, often used for initialization
     * @param _diamondCut Array of facet addresses and function selectors
     * @param _init Address of contract or facet to execute calldata on
     * @param _calldata Calldata to execute with delegatecall
     */
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