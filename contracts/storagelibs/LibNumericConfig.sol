// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { QuadConstants as QC } from "../libraries/QuadConstants.sol";

/**
 * @title LibNumericConfig
 * @notice Stores global numeric configuration in diamond storage.
 */
library LibNumericConfig {

    uint256 private constant ROOT_FINDING_DEFAULT_MAX_ITER = 200;

    // Fixed storage slot for numeric config (unique hash key)
    bytes32 internal constant SLOT = keccak256("smart-solve.numeric.config.v1");

    /**
    * @notice Global numeric configuration values.
    * @param  tol     Numerical tolerance encoded as bytes16
    * @param  minTol  Minimum allowable tolerance encoded as bytes16
    * @param  maxIter Maximum iteration count for numeric loops
    * @param  maxIter Step size 'h' for differentiation.
    */
    struct NumericConfig {
        bytes16 tol;
        bytes16 minTol;
        uint256 maxIter;
        bytes16 diffStep;
    }

    /**
     * @notice Returns a pointer to NumericConfig in storage
     *         Uses inline assembly to assign SLOT as the storage location.
     */
    function cfg() internal pure returns (NumericConfig storage c) {
        bytes32 position = SLOT;
        assembly { c.slot := position }
    }

    // ------------------------------------------------------------
    // Setters
    // ------------------------------------------------------------

    /**
     * @notice Update tolerance in storage
     * @param tol The new tolerance value as ABDKMathQuad (bytes16)
     */
    function setTol(bytes16 tol) internal {
        cfg().tol = tol;
    }

    /**
     * @notice Update min tolerance in storage
     * @param minTol The new min tolerance value as ABDKMathQuad (bytes16)
     */
    function setMinTol(bytes16 minTol) internal {
        cfg().minTol = minTol;
    }

    /**
     * @notice Update maxIter (maximum iteration count) in storage
     * @param m The new maximum iteration count
     */
    function setMaxIter(uint256 m) internal {
        cfg().maxIter = m;
    }

    /**
     * @notice Set the step size 'h' for numerical differentiation.
     * @param _h The step size in IEEE-754 binary128 (bytes16).
     */
    function setDiffStep(bytes16 _h) internal {
        cfg().diffStep = _h;
    }

    // ------------------------------------------------------------
    // Getters
    // ------------------------------------------------------------

    /**
     * @notice Get the global default tolerance.
     *         Returns DEFAULT_TOL if uninitialized.
     */
    function getTol() internal view returns (bytes16) {
        bytes16 t = cfg().tol;
        if (MathLib.cmp(t, MathLib.fromInt(0)) == 0) {
            return QC.DEFAULT_TOL();
        }
        return t;
    }

    /**
     * @notice Get the minimum allowable tolerance (Gas Guardrail).
     *         Returns DEFAULT_MIN_TOL if uninitialized.
     */
    function getMinTol() internal view returns (bytes16) {
        bytes16 t = cfg().minTol;
        if (MathLib.cmp(t, MathLib.fromInt(0)) == 0) {
            return QC.DEFAULT_MIN_TOL();
        }
        return t;
    }

    /**
     * @notice Get the maximum number of iterations.
     *         Returns the shared default maximum iteration count if uninitialized.
     */
    function getMaxIter() internal view returns (uint256) {
        uint256 m = cfg().maxIter;
        if (m == 0) {
            return QC.DEFAULT_MAX_ITER;
        }
        return m;
    }

    /**
     * @notice Get the differentiation step size.
     *         Returns DEFAULT_DIFF_STEP if uninitialized.
     */
    function getDiffStep() internal view returns (bytes16) {
        bytes16 h = cfg().diffStep;
        if (MathLib.cmp(h, MathLib.fromInt(0)) == 0) {
            return QC.DEFAULT_DIFF_STEP();
        }
        return h;
    }

    // ------------------------------------------------------------
    // Root-finding effective configuration
    // ------------------------------------------------------------

    /**
     * @notice Returns the root-finding minimum tolerance.
     * @dev RootFinding historically supports a tighter floor than the shared
     *      numerical default. Keep that behavior explicit and centralized.
     */
    function getRootFindingMinTol() internal view returns (bytes16) {
        bytes16 t = cfg().minTol;
        if (MathLib.cmp(t, MathLib.fromInt(0)) == 0) {
            return QC.EPS_1e36();
        }
        return t;
    }

    /**
     * @notice Returns the effective root-finding configuration.
     */
    function getRootFindingConfig()
        internal
        view
        returns (bytes16 tol, bytes16 minTol, uint256 maxIter)
    {
        minTol = getRootFindingMinTol();
        tol = cfg().tol;
        if (MathLib.cmp(tol, MathLib.fromInt(0)) == 0) {
            tol = QC.DEFAULT_TOL();
        }
        if (MathLib.cmp(tol, minTol) < 0) {
            tol = minTol;
        }

        maxIter = cfg().maxIter;
        if (maxIter == 0) {
            maxIter = ROOT_FINDING_DEFAULT_MAX_ITER;
        }
    }
}
