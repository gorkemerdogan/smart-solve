// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IERC173
 * @notice ERC-173 ownership standard defining a minimal interface for
 *         single-owner contract control.
 */
interface IERC173 {
    /**
     * @notice Emitted when ownership changes.
     * @param previousOwner Address of the former owner.
     * @param newOwner Address of the new owner.
     */
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @notice Returns the address of the current contract owner.
     */
    function owner() external view returns (address);

    /**
     * @notice Updates the contract owner.
     * @param newOwner The new owner address.
     */
    function transferOwnership(address newOwner) external;
}