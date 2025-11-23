// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { QuadConstants as QC } from "../libraries/QuadConstants.sol";
import { Differentiation } from "../libraries/numeric/Differentiation.sol";

/**
 * @title DifferentiationHarness
 * @notice Test harness for the Differentiation library.
 * @dev Exposes internal library functions to external callers for testing.
 * Includes mock functions (x^2, x^3, etc.) to act as integration targets.
 */
contract DifferentiationHarness {
    using MathLib for bytes16;

    // ------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------

    function qFromInt(int256 x) external pure returns (bytes16) {
        return MathLib.fromInt(x);
    }

    function qFromFrac(int256 num, int256 den) external pure returns (bytes16) {
        return MathLib.div(MathLib.fromInt(num), MathLib.fromInt(den));
    }

    /// @notice Returns a hardcoded IEEE-754 binary128 constant for PI.
    function PI() external pure returns (bytes16) {
        return QC.PI();
    }

    // ------------------------------------------------------------
    //  Differentiation Wrappers
    // ------------------------------------------------------------

    /**
     * @notice Wrapper for Differentiation.forwardDiff.
     * @param h Step size (pass 0 to use library defaults).
     */
    function forwardDiffHarness(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16) {
        return Differentiation.forwardDiff(target, selector, x, h);
    }

    /**
     * @notice Wrapper for Differentiation.backwardDiff.
     * @param h Step size (pass 0 to use library defaults).
     */
    function backwardDiffHarness(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16) {
        return Differentiation.backwardDiff(target, selector, x, h);
    }

    /**
     * @notice Wrapper for Differentiation.centeredDiff.
     * @param h Step size (pass 0 to use library defaults).
     */
    function centeredDiffHarness(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16) {
        return Differentiation.centeredDiff(target, selector, x, h);
    }

    // ------------------------------------------------------------
    //  Mock Functions (Targets for differentiation) 
    // ------------------------------------------------------------

    /**
     * @notice f(x) = x^2
     * @dev Derivative f'(x) = 2x
     */
    function f_square(bytes16 x) external pure returns (bytes16) {
        return x.mul(x);
    }

    /**
     * @notice f(x) = 3x − 2
     * @dev Derivative f'(x) = 3
     */
    function f_linear(bytes16 x) external pure returns (bytes16) {
        bytes16 three = MathLib.fromInt(3);
        bytes16 two   = MathLib.fromInt(2);
        return three.mul(x).sub(two); // 3x - 2
    }

    /**
     * @notice f(x) = 5
     * @dev Derivative f'(x) = 0
     */
    function f_constFive(bytes16) external pure returns (bytes16) {
        return MathLib.fromInt(5);
    }

    /**
     * @notice f(x) = |x|
     * @dev Derivative is -1 for x < 0, 1 for x > 0. Undefined at 0.
     */
    function f_abs(bytes16 x) external pure returns (bytes16) {
        return MathLib.abs(x);
    }

    /**
     * @notice f(x) = x^3
     * @dev Derivative f'(x) = 3x^2
     */
    function f_cube(bytes16 x) external pure returns (bytes16) {
        return x.mul(x).mul(x); // x * x * x
    }
}