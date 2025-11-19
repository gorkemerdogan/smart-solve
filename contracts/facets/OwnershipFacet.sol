// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LibSmartSolve } from "../libraries/LibSmartSolve.sol";
import { IERC173 } from "../interfaces/IERC173.sol";

/**
 * @title OwnershipFacet
 * @notice Implements the IERC173 ownership standard, providing external
 *         accessors for retrieving and updating the contract owner.
 * @dev
 *      - owner(): returns the current contract owner
 *      - transferOwnership(): updates the owner; restricted to the existing owner
 *        via LibSmartSolve.enforceIsContractOwner().
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