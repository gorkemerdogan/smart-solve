// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title RootFindingHarness
 * @notice Test harness providing:
 *         1) Quadruple-precision helper constructors (qFromInt, qFromFrac)
 *         2) Black-box test functions f(x) and df(x) for root-finding:
 *              - f_x2_minus_4(x) = x^2 - 4
 *              - df_2x(x)       = 2x
 *              - f_cubic(x)     = x^3 - x - 2
 *              - df_cubic(x)    = 3x^2 - 1
 *
 * @dev All numerics use IEEE-754 quadruple precision via ABDKMathQuad (bytes16).
 *      Functions are external/pure to be callable from facets in a diamond.
 */
contract RootFindingHarness {
    using MathLib for bytes16;

    // ───────────────────────────── Constants ─────────────────────────────

    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

    /// @notice 1 as quadruple
    function ONE() external pure returns (bytes16) {
        return MathLib.fromInt(1);
    }

    /// @notice 2 as quadruple
    function TWO() external pure returns (bytes16) {
        return MathLib.fromInt(2);
    }

    /// @notice 3 as quadruple
    function THREE() external pure returns (bytes16) {
        return MathLib.fromInt(3);
    }

    /// @notice 4 as quadruple
    function FOUR() external pure returns (bytes16) {
        return MathLib.fromInt(4);
    }

    // ───────────────────── Quadruple helper constructors ─────────────────────

    /**
     * @notice Convert a signed integer to ABDK quadruple precision.
     * @param n Signed integer
     * @return q IEEE-754 quadruple (bytes16) representing `n`
     */
    function qFromInt(int256 n) external pure returns (bytes16 q) {
        q = MathLib.fromInt(n);
    }

    /**
     * @notice Convert a rational number num/den to ABDK quadruple precision.
     * @dev Reverts if den == 0.
     * @param num Signed numerator
     * @param den Signed denominator (must be non-zero)
     * @return q IEEE-754 quadruple (bytes16) representing num/den
     */
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = qNum.div(qDen);
    }

    // ─────────────────────── Test functions for f/df ───────────────────────

    /**
     * @notice f(x) = x^2 - 4
     * @param x Input (bytes16)
     * @return y f(x) as quadruple
     */
    function f_x2_minus_4(bytes16 x) external pure returns (bytes16 y) {
        bytes16 four = MathLib.fromInt(4);
        y = x.mul(x).sub(four);
    }

    /**
     * @notice df(x) = 2x (derivative of x^2 - 4)
     * @param x Input (bytes16)
     * @return y 2x as quadruple
     */
    function df_2x(bytes16 x) external pure returns (bytes16 y) {
        bytes16 two = MathLib.fromInt(2);
        y = two.mul(x);
    }

    /**
     * @notice f(x) = x^3 - x - 2
     * @param x Input (bytes16)
     * @return y f(x) as quadruple
     */
    function f_cubic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 x2 = x.mul(x);
        bytes16 x3 = x2.mul(x);
        bytes16 two = MathLib.fromInt(2);
        y = x3.sub(x).sub(two);
    }

    /**
     * @notice df(x) = 3x^2 - 1 (derivative of x^3 - x - 2)
     * @param x Input (bytes16)
     * @return y df(x) as quadruple
     */
    function df_cubic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 three = MathLib.fromInt(3);
        bytes16 one   = MathLib.fromInt(1);
        bytes16 x2    = x.mul(x);
        y = three.mul(x2).sub(one);
    }

    // ──────────────────────────── Optional utils ────────────────────────────

    /**
     * @notice Returns 0 in quadruple precision.
     */
    function ZERO() external pure returns (bytes16) {
        return QZERO;
    }

    /**
     * @notice Compare a and b: returns -1 if a<b, 0 if equal, +1 if a>b.
     * @param a First operand
     * @param b Second operand
     */
    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return MathLib.cmp(a, b);
    }
}