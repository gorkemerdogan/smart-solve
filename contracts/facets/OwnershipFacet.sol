// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LibSmartSolve } from "../libraries/LibSmartSolve.sol";
import { IERC173 } from "../interfaces/IERC173.sol";

/*
 * =============== OwnershipFacet.sol ===============
 *
 * Implements the IERC173 (Ownership) standard.
 * Provides external functions to get and transfer ownership.
 *
 *
 * Functions:
 * - owner() : returns current contract owner
 * - transferOwnership(newOwner) : changes the owner (only callable by current owner)
 */

contract OwnershipFacet is IERC173 {
    /// @notice Returns the address of the current owner
    function owner() external view override returns (address) {
        return LibSmartSolve.contractOwner();
    }

    /// @notice Transfers ownership to a new address
    /// @param newOwner The address of the new owner
    function transferOwnership(address newOwner) external override {
        LibSmartSolve.enforceIsContractOwner();
        
        address prevOwner = LibSmartSolve.setContractOwner(newOwner);        
        emit OwnershipTransferred(prevOwner, newOwner);
    }
}