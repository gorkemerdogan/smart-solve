// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
 * =============== IERC173.sol ===============
 *
 * Defines a **standard for ownership** of contracts.
 * Lets a contract have a single "owner" who can transfer ownership.
 *
 *
 * Functions:
 * 1- owner() : returns the current owner address
 * 2- transferOwnership(newOwner) : updates ownership (only owner can call this)
 *
 * Events:
 * - OwnershipTransferred(previousOwner, newOwner) : logs ownership changes
 */

interface IERC173 {
    /// @notice Emitted when ownership changes
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /// @notice Get the address of the current owner
    function owner() external view returns (address);

    /// @notice Transfer ownership to a new address
    function transferOwnership(address newOwner) external;
}