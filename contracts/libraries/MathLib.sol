// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad } from "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title MathLib
 * @notice Thin wrapper around ABDKMathQuad with PUBLIC functions.
 *         Purpose: avoid inlining ABDKMathQuad into every trigonometric library.
 *
 *         Using public functions ensures all trig libraries CALL this library
 *         instead of embedding the heavy ABDK code => bytecode drops massively.
 */
library MathLib {

    // ------------------------------------------------------------
    // Basic Operations
    // ------------------------------------------------------------

    /**
     * @notice Adds two IEEE-754 quadruple-precision numbers.
     * @param a First operand (bytes16)
     * @param b Second operand (bytes16)
     * @return bytes16 Result of a + b
     */
    function add(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.add(a, b);
    }

    /**
     * @notice Subtracts two IEEE-754 quadruple-precision numbers.
     * @param a Minuend
     * @param b Subtrahend
     * @return bytes16 a − b
     */
    function sub(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.sub(a, b);
    }

    /**
     * @notice Multiplies two quadruple-precision numbers.
     * @param a First operand
     * @param b Second operand
     * @return bytes16 a x b
     */
    function mul(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.mul(a, b);
    }

    /**
     * @notice Divides two quadruple-precision numbers.
     *         Reverts if 'b == 0'.
     * @param a Numerator
     * @param b Denominator
     * @return bytes16 a / b
     */
    function div(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.div(a, b);
    }

    /**
     * @notice Computes sqrt(a) in IEEE-754 quad.
     *         Returns NaN if a < 0.
     * @param a Input number
     * @return bytes16 √a
     */
    function sqrt(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.sqrt(a);
    }

    /**
     * @notice Returns the negation of a quadruple-precision number.
     * @param a Input value
     * @return bytes16 −a
     */
    function neg(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.neg(a);
    }

    /**
     * @notice Returns |a| in quad precision.
     * @param a Input value
     * @return bytes16 Absolute value |a|
     */
    function abs(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.abs(a);
    }

    // ------------------------------------------------------------
    // Floor
    // ------------------------------------------------------------

    /**
     * @notice Computes floor(x) in IEEE-754 quadruple precision.
     * @dev ABDKMathQuad has no native floor(), so this uses truncation rules:
     *       - If x ≥ 0: floor(x) = trunc(x)
     *       - If x < 0 and x is integer: floor(x) = x
     *       - If x < 0 and fractional: floor(x) = trunc(x) − 1
     *
     * @param x Quad-precision input value
     * @return floorX Quad-precision floor(x)
     */
    function floorQuad(bytes16 x) public pure returns (bytes16 floorX) {
        // Case 1: x >= 0
        // Truncation and floor are equivalent for non-negative
        // numbers because both move toward zero.
        if (cmp(x, fromInt(0)) >= 0) {
            int256 ti = toInt(x);      // remove fractional part toward zero
            return fromInt(ti);        // floor(x) = trunc(x)
        }

        // Case 2: x < 0
        // Truncation (toward zero) moves upward for negatives.
        //
        // Examples:
        //   x = -2.3 → trunc = -2 → floor = -3
        //   x = -4.0 → trunc = -4 → floor = -4
        int256 ti2 = toInt(x);         // truncate toward zero
        bytes16 b  = fromInt(ti2);     // convert to quad

        // If trunc(x) equals x exactly, x is already an integer.
        if (cmp(b, x) == 0) { return b; }

        // Negative non-integer case: floor = trunc(x) - 1
        return sub(b, fromInt(1));
    }

    /**
     * @notice Computes floor(x) and returns it as int256.
     * @dev Same logic as floorQuad(), but outputs a native integer.
     *      Safe because ABDKMathQuad.toInt() already bounds-checks.
     *
     * @param x Quad-precision input value
     * @return floorI Signed integer floor(x)
     */
    function floorInt(bytes16 x) public pure returns (int256 floorI) {
        // Case 1: x >= 0
        // trunc(x) moves toward zero → equivalent to floor(x)
        if (cmp(x, fromInt(0)) >= 0) { return toInt(x); }// floor = trunc

        // Case 2: x < 0
        // trunc still moves upward; adjust for non-integers.
        int256 ti = toInt(x);         // truncation
        bytes16 t = fromInt(ti);      // convert back to quad

        // If exact integer, return as-is.
        if (cmp(t, x) == 0) { return ti; }

        // Negative non-integer: subtract 1
        return ti - 1;
    }

    // ------------------------------------------------------------
    // Comparison
    // ------------------------------------------------------------

    /**
     * @notice Compares two quad numbers.
     *         Returns:
     *            -1 if a < b
     *             0 if a == b
     *             1 if a > b
     *
     * @param a First operand
     * @param b Second operand
     * @return int256 Comparison result
     */
    function cmp(bytes16 a, bytes16 b) public pure returns (int256) {
        return ABDKMathQuad.cmp(a, b);
    }

    /**
     * @notice Returns true if the number is NaN.
     * @param a Quad number
     * @return bool True if a is NaN
     */
    function isNaN(bytes16 a) public pure returns (bool) {
        return ABDKMathQuad.isNaN(a);
    }

    /**
    * @notice Returns true if 'x' is exactly zero in IEEE-754 quad format.
    *         Uses cmp(x, 0) to correctly treat +0 and -0 as zero.
    *
    * @param x  Quadruple-precision value (bytes16)
    * @return   True if x == 0, false otherwise
    */
    function isZero(bytes16 x) internal pure returns (bool) {
        // QZERO is cheaper to inline than reading a constant
        bytes16 zero = bytes16(0);
        return cmp(x, zero) == 0;
    }

    /**
     * @notice Compares two floating-point numbers for "approximate equality"
     *         using a combined absolute and relative tolerance.
     * @dev Uses a hybrid absolute + relative tolerance:
     *      diff = |a − b|
     *      tolerance = max(absTol, relTol * max(|a|, |b|))
     *      nearlyEqual ⇔ diff ≤ tolerance
     * @param a First value
     * @param b Second value
     * @param absTol Absolute tolerance (useful near zero)
     * @param relTol Relative tolerance (scales with magnitude)
     */
    function nearlyEqual(bytes16 a, bytes16 b, bytes16 absTol, bytes16 relTol) internal pure returns (bool) {
        // diff = |a - b|
        bytes16 diff = abs(sub(a, b));

        // |a| and |b|
        bytes16 absA = abs(a);
        bytes16 absB = abs(b);

        // largest = max(|a|, |b|) using cmp
        bytes16 largest = cmp(absA, absB) >= 0 ? absA : absB;

        // scaled = largest * relTol
        bytes16 scaled = mul(largest, relTol);

        // tolerance = max(absTol, scaled) using cmp
        bytes16 tolerance = cmp(absTol, scaled) >= 0 ? absTol : scaled;

        // nearlyEqual ⇔ |a - b| <= tolerance
        return cmp(diff, tolerance) <= 0;
    }

    /**
     * @notice Restricts a value x to stay within a lower bound (lo) and upper bound (hi).
     *         Reverts if lo > hi.
     * @param x The value to clamp (bytes16)
     * @param lo The lower bound (bytes16)
     * @param hi The upper bound (bytes16)
     * @return The clamped value (bytes16)
     */
    function clamp(bytes16 x, bytes16 lo, bytes16 hi) internal pure returns (bytes16) {
        // Ensure lo <= hi
        require(ABDKMathQuad.cmp(lo, hi) <= 0, "Scalar: lo must be <= hi");

        // If x < lo → return lo
        if (ABDKMathQuad.cmp(x, lo) < 0) { return lo; }

        // If x > hi → return hi
        if (ABDKMathQuad.cmp(x, hi) > 0) { return hi; }

        // Otherwise x is inside [lo, hi]
        return x;
    }

    // ------------------------------------------------------------
    // Conversions
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to IEEE-754 quad.
     * @param v Integer value
     * @return bytes16 Quad representation
     */
    function fromInt(int256 v) public pure returns (bytes16) {
        return ABDKMathQuad.fromInt(v);
    }

    /**
     * @notice Converts an unsigned integer to IEEE-754 quad.
     * @param v Unsigned integer
     * @return bytes16 Quad representation
     */
    function fromUInt(uint256 v) public pure returns (bytes16) {
        return ABDKMathQuad.fromUInt(v);
    }

    /**
     * @notice Converts a quad number to a signed integer via truncation.
     * @dev Reverts if out of int256 range.
     * @param v Quad-precision number
     * @return int256 Truncated integer representation
     */
    function toInt(bytes16 v) public pure returns (int256) {
        return ABDKMathQuad.toInt(v);
    }
}