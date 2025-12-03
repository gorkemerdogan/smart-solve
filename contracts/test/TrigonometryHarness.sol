// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import "../libraries/Trigonometry/Trigonometry.sol";

/**
 * @title TrigonometryHarness
 * @notice Testing harness providing:
 *         - Common quadruple-precision constants
 *         - Conversions between quad ↔ int and quad ↔ scaled floats
 *         - NaN and comparison utilities
 *         - Direct passthrough calls to the Trigonometry library
 *
 */
contract TrigonometryHarness {
    using MathLib for bytes16;

    // ---------------------------------------------------------
    // QUAD CONSTANTS
    // ---------------------------------------------------------

    bytes16 public constant QPI        = 0x4000921FB54442D18469898CC51701B8;
    bytes16 public constant QHALF_PI   = 0x3FFF921FB54442D18469898CC51701B8;
    bytes16 public constant QQUARTER_PI= 0x3FFE921FB54442D18469898CC51701B8;
    bytes16 public constant QTWO_PI    = 0x4001921FB54442D18469898CC51701B8;

    bytes16 public constant QNAN       = 0x7fff8000000000000000000000000000;

    // ------------------------------------------------------------
    // QUAD <-> INT Conversions
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to quadruple precision.
     * @param x Signed integer input.
     * @return Quadruple-precision value representing x.
     */
    function fromDouble(int256 x) external pure returns (bytes16) {
        return MathLib.fromInt(x);
    }

    /**
     * @notice Converts an unsigned integer to quadruple precision.
     * @param x Unsigned integer input.
     * @return Quadruple-precision value representing x.
     */
    function fromUInt(uint256 x) external pure returns (bytes16) {
        return MathLib.fromUInt(x);
    }

    /**
     * @notice Converts a quadruple-precision value to a signed integer.
     *         Truncates toward zero, per ABDKMathQuad.toInt().
     * @param x Quadruple-precision input.
     * @return Signed integer result.
     */
    function toDouble(bytes16 x) external pure returns (int256) {
        return MathLib.toInt(x);
    }

    // ------------------------------------------------------------
    // QUAD <-> FLOAT (scaled) (JS testing compatible)
    // ------------------------------------------------------------

    /// @notice Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e12;

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

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Computes |x| in quadruple precision.
     * @param x Input value.
     * @return Absolute value of x.
     */
    function abs(bytes16 x) external pure returns (bytes16) {
        return MathLib.abs(x);
    }

    /**
     * @notice Computes the negation of x.
     * @param x Input value.
     * @return -x in quadruple precision.
     */
    function neg(bytes16 x) external pure returns (bytes16) {
        return MathLib.neg(x);
    }

    /**
     * @notice Compares two quadruple-precision values.
     *         Returns:
     *             -1 if a < b  
     *              0 if a == b  
     *             +1 if a > b
     * @param a First operand.
     * @param b Second operand.
     * @return Comparison result.
     */
    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return MathLib.cmp(a, b);
    }

    /**
     * @notice Returns true if x is IEEE-754 NaN.
     * @param x Quadruple-precision value.
     * @return True if x is NaN.
     */
    function isNaN(bytes16 x) external pure returns (bool) {
        return MathLib.isNaN(x);
    }

    /**
     * @notice Computes a + b in quadruple precision.
     * @param a First operand.
     * @param b Second operand.
     * @return Sum of a and b.
     */
    function add(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return MathLib.add(a, b);
    } 

    /**
     * @notice Computes a × b in quadruple precision.
     * @param a First operand.
     * @param b Second operand.
     * @return Product of a and b.
     */
    function mul(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return MathLib.mul(a, b);
    }

    // ------------------------------------------------------------
    // Trigonometry Functions
    // ------------------------------------------------------------

    /**
     * @notice Computes sin(x) using high-precision trigonometric reduction.
     * @param x Quadruple-precision angle.
     * @return sin(x).
     */
    function sin(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.sin(x);
    }

    /**
     * @notice Computes cos(x).
     * @param x Quadruple-precision angle.
     * @return cos(x).
     */
    function cos(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.cos(x);
    }

    /**
     * @notice Computes tan(x).
     * @param x Quadruple-precision angle.
     * @return tan(x) or NaN if undefined.
     */
    function tan(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.tan(x);
    }

    /**
     * @notice Computes cot(x).
     * @param x Quadruple-precision angle.
     * @return cot(x) or NaN if undefined.
     */
    function cot(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.cot(x);
    }

    /**
     * @notice Computes arcsin(x).
     * @param x Quadruple-precision input in [-1, 1].
     * @return arcsin(x).
     */
    function asin(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.asin(x);
    }

    /**
     * @notice Computes arccos(x).
     * @param x Quadruple-precision input in [-1, 1].
     * @return arccos(x).
     */
    function acos(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.acos(x);
    }

    /**
     * @notice Computes arctan(x).
     * @param x Quadruple-precision value.
     * @return arctan(x).
     */
    function atan(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.atan(x);
    }
}