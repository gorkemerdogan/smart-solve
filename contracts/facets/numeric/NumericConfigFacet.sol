// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../../libraries/LibSmartSolve.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";
import { MathLib } from "../../libraries/MathLib.sol";

/**
 * @title NumericConfigFacet
 * @notice Facet providing read/write access to numerical configuration
 *         parameters used across all numeric libraries.
 *         Exposes tolerance, minimum tolerance, maximum iteration count, and
 *         the default differentiation step.
 *
 * @dev Writes are restricted to the diamond owner. All values are stored in
 *      LibNumericConfig and retrieved by integration, root-finding, and
 *      polynomial algorithms. Zero is retained as the backwards-compatible
 *      "unset" value: effective getters then return LibNumericConfig defaults.
 */
contract NumericConfigFacet {

    function _requireNonNegative(bytes16 value, string memory message) private pure {
        require(MathLib.cmp(value, MathLib.fromInt(0)) >= 0, message);
    }
    
    /**
     * @notice Sets the numerical tolerance (tol) for numerical algorithms.
     * @dev Only callable by the diamond owner.
     * @param tol Tolerance value encoded as IEEE-754 quadruple precision (bytes16).
     */
    function setTol(bytes16 tol) external {
        LibSmartSolve.enforceIsContractOwner();
        _requireNonNegative(tol, "NumericConfig: tol must be non-negative");
        LibNumericConfig.setTol(tol);
    }

    /**
     * @notice Sets the minimum numerical tolerance used as a convergence floor.
     * @dev Only callable by the diamond owner. Pass zero to clear the override.
     */
    function setMinTol(bytes16 minTol) external {
        LibSmartSolve.enforceIsContractOwner();
        _requireNonNegative(minTol, "NumericConfig: minTol must be non-negative");
        LibNumericConfig.setMinTol(minTol);
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
     * @notice Sets the default finite-difference step used when a caller passes zero.
     * @dev Only callable by the diamond owner. Pass zero to clear the override.
     */
    function setDiffStep(bytes16 h) external {
        LibSmartSolve.enforceIsContractOwner();
        _requireNonNegative(h, "NumericConfig: diffStep must be non-negative");
        LibNumericConfig.setDiffStep(h);
    }

    /**
     * @notice Returns the current numerical tolerance (tol).
     */
    function getTol() external view returns (bytes16) {
        return LibNumericConfig.getTol();
    }

    /**
     * @notice Returns the effective minimum tolerance, including the unset default.
     */
    function getMinTol() external view returns (bytes16) {
        return LibNumericConfig.getMinTol();
    }

    /**
     * @notice Returns the maximum iteration count.
     */
    function getMaxIter() external view returns (uint256) {
        return LibNumericConfig.getMaxIter();
    }

    /**
     * @notice Returns the effective default finite-difference step.
     */
    function getDiffStep() external view returns (bytes16) {
        return LibNumericConfig.getDiffStep();
    }

    /**
     * @notice Returns the effective configuration consumed by RootFindingFacet.
     * @dev The root-finding unset minimum is 1e-36 and its unset iteration
     *      limit is 200; these are intentionally more specific than shared
     *      numerical defaults.
     */
    function getRootFindingConfig() external view returns (bytes16 tol, bytes16 minTol, uint256 maxIter) {
        return LibNumericConfig.getRootFindingConfig();
    }
}
