// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../../libraries/LibSmartSolve.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

contract NumericConfigFacet {
    /**
     * Set numerical tolerance (tol)
     * Only callable by the contract owner (diamond owner).
     * @param tol The tolerance value as an ABDKMathQuad (bytes16).
     * Example: 0x3E2386F26FC0948A0000000000000000 represents 1e-9
     */
    function setTol(bytes16 tol) external {
        LibSmartSolve.enforceIsContractOwner();
        LibNumericConfig.setTol(tol);
    }

    // Set maximum iterations for algorithms
    // Only callable by the contract owner
    function setMaxIter(uint256 m) external {
        LibSmartSolve.enforceIsContractOwner();
        LibNumericConfig.setMaxIter(m);
    }

    // Get current numerical tolerance (tol)
    function getTol() external view returns (bytes16) {
        return LibNumericConfig.cfg().tol;
    }

    // Get current maximum iteration count
    function getMaxIter() external view returns (uint256) {
        return LibNumericConfig.cfg().maxIter;
    }
}