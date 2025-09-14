// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LibSmartSolve } from "../libraries/LibSmartSolve.sol";
import { LibNumericConfig } from "../storagelibs/LibNumericConfig.sol";

contract NumericConfigFacet {
    /**
     * Set numerical tolerance (eps)
     * Only callable by the contract owner (diamond owner).
     * Example: eps = 1e9 to represent 1e-9 in SD59x18 fixed-point
     */
    function setEps(int256 eps) external {
        LibSmartSolve.enforceIsContractOwner();
        LibNumericConfig.setEps(eps);
    }

    // Set maximum iterations for algorithms
    // Only callable by the contract owner
    function setMaxIter(uint256 m) external {
        LibSmartSolve.enforceIsContractOwner();
        LibNumericConfig.setMaxIter(m);
    }

    // Get current numerical tolerance (eps)
    function getEps() external view returns (int256) {
        return LibNumericConfig.cfg().eps;
    }

    // Get current maximum iteration count
    function getMaxIter() external view returns (uint256) {
        return LibNumericConfig.cfg().maxIter;
    }
}