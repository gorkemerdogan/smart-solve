// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IERC165
 * @notice ERC-165 standard interface for contract interface detection.
 */
interface IERC165 {
    /**
     * @notice Indicates whether the contract supports the specified interface.
     * @param interfaceId The 4-byte ERC-165 interface identifier.
     * @return True if the interface is supported.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}