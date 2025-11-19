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

    /*──────────────────────────────────────────────────────────
        BASIC OPS
    ───────────────────────────────────────────────────────────*/

    function add(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.add(a, b);
    }

    function sub(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.sub(a, b);
    }

    function mul(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.mul(a, b);
    }

    function div(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.div(a, b);
    }

    function sqrt(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.sqrt(a);
    }

    function neg(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.neg(a);
    }

    function abs(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.abs(a);
    }

    /*──────────────────────────────────────────────────────────
        FLOOR
    ───────────────────────────────────────────────────────────*/

    function floorQuad(bytes16 x) public pure returns (bytes16) {
        // If x >= 0: floor = trunc
        if (cmp(x, fromInt(0)) >= 0) {
            int256 ti = toInt(x);          // truncate
            return fromInt(ti);
        }

        // x < 0
        int256 ti2 = toInt(x);             // truncate toward zero
        bytes16 b = fromInt(ti2);

        // If x is already integer, floor(x) == x
        if (cmp(b, x) == 0) {
            return b;
        }

        // else: floor(x) = trunc(x) - 1
        return sub(b, fromInt(1));
    }

    function floorInt(bytes16 x) public pure returns (int256) {
        if (cmp(x, fromInt(0)) >= 0) {
            return toInt(x); // trunc = floor
        }

        int256 ti = toInt(x); // trunc
        bytes16 t = fromInt(ti);

        if (cmp(t, x) == 0) {
            return ti; // already integer
        }

        return ti - 1; // subtract 1 for negative non-integers
    }

    /*──────────────────────────────────────────────────────────
        COMPARE
    ───────────────────────────────────────────────────────────*/

    function cmp(bytes16 a, bytes16 b) public pure returns (int256) {
        return ABDKMathQuad.cmp(a, b);
    }

    function isNaN(bytes16 a) public pure returns (bool) {
        return ABDKMathQuad.isNaN(a);
    }

    /**
     * @notice Compares two floating-point numbers for "approximate equality"
     * using a combined absolute and relative tolerance.
     * @param a First value
     * @param b Second value
     * @param absTol The absolute tolerance (good for comparisons near zero)
     * @param relTol The relative tolerance (scales with magnitude of a and b)
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
     * @dev Reverts if lo > hi.
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

    /*──────────────────────────────────────────────────────────
        CONVERSIONS
    ───────────────────────────────────────────────────────────*/

    function fromInt(int256 v) public pure returns (bytes16) {
        return ABDKMathQuad.fromInt(v);
    }

    function fromUInt(uint256 v) public pure returns (bytes16) {
        return ABDKMathQuad.fromUInt(v);
    }

    function toInt(bytes16 v) public pure returns (int256) {
        return ABDKMathQuad.toInt(v);
    }
}