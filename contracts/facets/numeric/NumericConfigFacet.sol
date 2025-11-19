// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../../libraries/LibSmartSolve.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title NumericConfigFacet
 * @notice Facet providing read/write access to numerical configuration
 *         parameters used across all numeric libraries.
 *         Exposes tolerance (tol) and maximum iteration count (maxIter).
 *
 * @dev Writes are restricted to the diamond owner. All values are stored in
 *      LibNumericConfig and retrieved by integration, root-finding, and
 *      polynomial algorithms. The facet implements a minimal ABI surface:
 *      setTol(), setMaxIter(), getTol(), and getMaxIter().
 */
contract NumericConfigFacet {
    
    /**
     * @notice Sets the numerical tolerance (tol) for numerical algorithms.
     * @dev Only callable by the diamond owner.
     * @param tol Tolerance value encoded as IEEE-754 quadruple precision (bytes16).
     */
    function setTol(bytes16 tol) external {
        LibSmartSolve.enforceIsContractOwner();
        LibNumericConfig.setTol(tol);
    }

    /**
     * @notice Sets the maximum iteration count for numerical algorithms.
     * @dev Only callable by the diamond owner.
     * @param m Maximum allowed iteration count.
     */
    function setMaxIter(uint256 m) external {
        LibSmartSolve.enforceIsContractOwner();
        LibNumericConfig.setMaxIter(m);
    }

    /**
     * @notice Returns the current numerical tolerance (tol).
     */
    function getTol() external view returns (bytes16) {
        return LibNumericConfig.cfg().tol;
    }

    /**
     * @notice Returns the maximum iteration count.
     */
    function getMaxIter() external view returns (uint256) {
        return LibNumericConfig.cfg().maxIter;
    }
}