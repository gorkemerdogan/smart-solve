// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title RootFindingHarness
 * @notice Provides deterministic test functions for root-finding algorithms and
 *         helper constructors for quadruple-precision values.
 *         Includes reference functions f(x) and df(x) used in Newton/Bisection tests.
 */
contract RootFindingHarness {
    using MathLib for bytes16;

    // ------------------------------------------------------------------------
    // Test functions f(x) and df(x) for root-finding
    // ------------------------------------------------------------------------

    /**
     * @notice Computes f(x) = x² − 4.
     * @param x Input value in quadruple precision.
     * @return y Result of x² − 4.
     */
    function f_x2_minus_4(bytes16 x) external pure returns (bytes16 y) {
        bytes16 four = MathLib.fromInt(4);
        y = x.mul(x).sub(four);
    }

    /**
     * @notice Computes df(x) = 2x, the derivative of x² − 4.
     * @param x Input value in quadruple precision.
     * @return y Value of 2x.
     */
    function df_2x(bytes16 x) external pure returns (bytes16 y) {
        bytes16 two = MathLib.fromInt(2);
        y = two.mul(x);
    }

    /**
     * @notice Computes f(x) = x³ − x − 2.
     * @param x Input value in quadruple precision.
     * @return y Result of x³ − x − 2.
     */
    function f_cubic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 x2 = x.mul(x);
        bytes16 x3 = x2.mul(x);
        bytes16 two = MathLib.fromInt(2);
        y = x3.sub(x).sub(two);
    }

    /**
     * @notice Computes df(x) = 3x² − 1, the derivative of x³ − x − 2.
     * @param x Input value in quadruple precision.
     * @return y Value of 3x² − 1.
     */
    function df_cubic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 three = MathLib.fromInt(3);
        bytes16 one   = MathLib.fromInt(1);
        bytes16 x2    = x.mul(x);
        y = three.mul(x2).sub(one);
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

    /// Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e12;

    /**
     * @notice Converts a quadruple-precision number into a scaled integer (scaled by SCALE).
     * @param x Quadruple-precision value.
     * @return Integer representing x * SCALE.
     */
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }

    /**
     * @notice Compares two quadruple-precision values.
     * @dev Returns:
     *      -1 if a < b  
     *       0 if a == b  
     *      +1 if a > b
     * @param a First operand.
     * @param b Second operand.
     * @return Comparison result in {-1, 0, +1}.
     */
    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return MathLib.cmp(a, b);
    }
}