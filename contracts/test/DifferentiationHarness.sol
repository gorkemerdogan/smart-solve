// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { QuadConstants as QC } from "../libraries/QuadConstants.sol";
import { Differentiation } from "../libraries/numeric/Differentiation.sol";

/**
 * @title DifferentiationHarness
 * @notice Test harness for the Differentiation library.
 *         Includes mock functions (x^2, x^3, etc.) to act as integration targets.
 */
contract DifferentiationHarness {
    using MathLib for bytes16;

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
     *         Derivative f'(x) = 2x
     */
    function f_square(bytes16 x) external pure returns (bytes16) {
        return x.mul(x);
    }

    /**
     * @notice f(x) = 3x − 2
     *         Derivative f'(x) = 3
     */
    function f_linear(bytes16 x) external pure returns (bytes16) {
        bytes16 three = MathLib.fromInt(3);
        bytes16 two   = MathLib.fromInt(2);
        return three.mul(x).sub(two); // 3x - 2
    }

    /**
     * @notice f(x) = 5
     *         Derivative f'(x) = 0
     */
    function f_constFive(bytes16) external pure returns (bytes16) {
        return MathLib.fromInt(5);
    }

    /**
     * @notice f(x) = |x|
     *         Derivative is -1 for x < 0, 1 for x > 0. Undefined at 0.
     */
    function f_abs(bytes16 x) external pure returns (bytes16) {
        return MathLib.abs(x);
    }

    /**
     * @notice f(x) = x^3
     *         Derivative f'(x) = 3x^2
     */
    function f_cube(bytes16 x) external pure returns (bytes16) {
        return x.mul(x).mul(x); // x * x * x
    }

    // ------------------------------------------------------------
    //  Numerical Helpers
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to IEEE-754 quadruple precision.
     * @param  n Signed integer.
     * @return q Quadruple-precision representation of 'n'.
     */
    function qFromInt(int256 n) external pure returns (bytes16 q) {
        q = MathLib.fromInt(n);
    }

    /**
     * @notice Converts a uint256 into its quad-precision representation.
     * @param x Unsigned integer value.
     * @return q Quad-precision number representing x.
     */
    function qFromUInt(uint256 x) external pure returns (bytes16) {
        return MathLib.fromUInt(x);
    }

    /**
     * @notice Converts a rational number num/den to quadruple precision.
     *         Reverts if 'den' equals zero.
     * @param num Signed numerator.
     * @param den Signed denominator (must be non-zero).
     * @return q  Quadruple-precision value representing num/den.
     */
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = MathLib.div(qNum, qDen);
    }

    /// @notice Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e12;

    /**
     * @notice Converts a quadruple-precision number into a scaled integer (scaled by SCALE).
     * @param x Quadruple-precision value.
     * @dev MathLib.toInt truncates toward zero. This helper is a 1e12
     *      fixed-point view and must not be used to claim binary128 accuracy.
     * @return Integer representing trunc(x * SCALE).
     */
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }

    /**
     * @notice Converts a scaled integer (scaled by SCALE) into quadruple precision.
     *         Example: scaledValue = 1234500000000 → represents 1.2345.
     * @param scaledValue Integer representing a float multiplied by SCALE.
     * @return Quadruple-precision value.
     */
    function fromFloat(int256 scaledValue) external pure returns (bytes16) {
        bytes16 qInt = MathLib.fromInt(scaledValue);
        bytes16 qScale = MathLib.fromUInt(SCALE);
        return MathLib.div(qInt, qScale);
    }
    
    /// @notice Returns a hardcoded IEEE-754 binary128 constant for PI.
    function PI() external pure returns (bytes16) {
        return QC.PI();
    }

}
