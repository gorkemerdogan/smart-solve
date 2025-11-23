// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LibNumericConfig
 * @notice Stores global numeric configuration in diamond storage.
 * @dev Contains tolerance values (tol, minTol) encoded as bytes16 (ABDKMathQuad)
 *      and maxIter as the global iteration cap for numerical routines.
 *      Default Max Iterations: 100
 *          uint256 private constant DEFAULT_MAX_ITER = 100;
 *      Default Tolerance: 1e-18 (Standard high precision)
 *          0x3fc56d5cfaacc19b601e1b7b51944da0 (approx 1e-18 in Quad)
 *      Default Differentiation Step: 1e-8
 */
library LibNumericConfig {
    // Fixed storage slot for numeric config (unique hash key)
    bytes32 internal constant SLOT = keccak256("smart-solve.numeric.config.v1");

    /**
    * @notice Global numeric configuration values.
    * @param tol Numerical tolerance encoded as bytes16
    * @param minTol Minimum allowable tolerance encoded as bytes16
    * @param maxIter Maximum iteration count for numeric loops
    * @param maxIter Step size 'h' for differentiation.
    */
    struct NumericConfig {
        bytes16 tol;
        bytes16 minTol;
        uint256 maxIter;
        bytes16 diffStep;
    }

    /**
     * @notice Returns a pointer to NumericConfig in storage
     * @dev Uses inline assembly to assign SLOT as the storage location.
     * This is the standard EIP-2535 diamond storage pattern.
     */
    function cfg() internal pure returns (NumericConfig storage c) {
        bytes32 position = SLOT;
        assembly { c.slot := position }
    }

    // ------------------------------------------------------------
    //  Setters
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
     * @notice Set the step size `h` for numerical differentiation.
     * @param _h The step size in IEEE-754 binary128 (bytes16).
     */
    function setDiffStep(bytes16 _h) internal {
        cfg().diffStep = _h;
    }

    // ------------------------------------------------------------
    //  Getters
    // ------------------------------------------------------------

    /**
     * @notice Get the global default tolerance.
     * @dev Returns 1e-18 if uninitialized.
     */
    function getTol() internal view returns (bytes16) {
        bytes16 t = cfg().tol;
        if (MathLib.cmp(t, MathLib.fromInt(0)) == 0) {
             return MathLib.fromInt(1).div(MathLib.fromInt(1000000000000000000));
        }
        return t;
    }

    /**
     * @notice Get the minimum allowable tolerance (Gas Guardrail).
     * @dev Returns 1e-32 if uninitialized.
     */
    function getMinTol() internal view returns (bytes16) {
        bytes16 t = cfg().minTol;
        if (MathLib.cmp(t, MathLib.fromInt(0)) == 0) {
            return MathLib.fromInt(1).div(MathLib.fromUInt(10**32));
        }
        return t;
    }

    /**
     * @notice Get the maximum number of iterations.
     * @dev Returns 100 if uninitialized.
     */
    function getMaxIter() internal view returns (uint256) {
        uint256 m = cfg().maxIter;
        return m == 0 ? DEFAULT_MAX_ITER : m;
    }

    /**
     * @notice Get the differentiation step size.
     * @dev Returns 1e-8 if uninitialized.
     */
    function getDiffStep() internal view returns (bytes16) {
        bytes16 h = cfg().diffStep;
        if (MathLib.cmp(h, MathLib.fromInt(0)) == 0) {
            return MathLib.fromInt(1).div(MathLib.fromInt(100000000));
        }
        return h;
    }
}