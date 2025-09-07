// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/* 
 * =============== IERC165.sol ===============
 *
 * Defines the ERC-165 standard interface detection mechanism.
 */

interface IERC165 {
    /// @notice Query if a contract implements an interface
    /// @param interfaceId The interface identifier (ERC-165 calculated)
    /// @return true if the contract implements interfaceId
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}