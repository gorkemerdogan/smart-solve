// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title QuadConstants
 * @notice Shared high-precision constants and tiny helpers for IEEE-754
 *         binary128 (quadruple precision) using ABDKMathQuad.
 */
library QuadConstants {
    using ABDKMathQuad for bytes16;

    // -------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------

    /**
     * @notice Constructs a quadruple-precision rational number num/den.
     *         Convenience wrapper for ABDKMathQuad.fromInt(num).div(fromInt(den)).
     *
     * @param num Numerator (signed integer)
     * @param den Denominator (signed integer), must be non-zero
     * @return bytes16 Quadruple-precision value num / den
     */
    function fromFrac(int256 num, int256 den) internal pure returns (bytes16) {
        bytes16 n = ABDKMathQuad.fromInt(num);
        bytes16 d = ABDKMathQuad.fromInt(den);
        return n.div(d);
    }

    /**
     * @notice Returns 0.5 as a quadruple-precision number.
     * @return bytes16 1/2 in quad format
     */
    function HALF() internal pure returns (bytes16) {
        return fromFrac(1, 2);
    }

    // -------------------------------------------------------------
    // π
    // -------------------------------------------------------------

    /**
     * @notice Returns π (pi) in IEEE-754 quadruple precision.
     *         Encoded as a canonical 128-bit constant approximating:
     *         3.1415926535897932384626433832795
     *         Provides 31 decimal digits, sufficient for quad-precision math.
     * @return bytes16 π
     */
    function PI() internal pure returns (bytes16) {
        return 0x4000921FB54442D18469898CC51701B8;
    }

    /**
     * @notice Returns π/4 in quadruple precision.
     * @return bytes16 π/4
     */
    function QUARTER_PI() internal pure returns (bytes16) {
        return 0x3FFE921FB54442D18469898CC51701B8;
    }

    /**
     * @notice Returns π/2 in quadruple precision.
     * @return bytes16 π/2
     */
    function HALF_PI() internal pure returns (bytes16) {
        return 0x3FFF921FB54442D18469898CC51701B8;
    }

    /**
     * @notice Returns 2π in quadruple precision.
     * @return bytes16 2π
     */
    function TWO_PI() internal pure returns (bytes16) {
        return 0x4001921FB54442D18469898CC51701B8;
    }

    // -------------------------------------------------------------
    // Tolerences
    // -------------------------------------------------------------

    /**
     * @notice Returns 1e-6 as a quadruple-precision constant.
     *         Useful as a "small-angle" threshold in trigonometric approximations.
     * @return bytes16 1 x 10^-6
     */
    function EPS_1e6() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000);
    }

    /**
     * @notice Returns 1e-9 as a quadruple-precision constant.
     *         Good for mid-range error bounds.
     * @return bytes16 1 x 10^-9
     */
    function EPS_1e9() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000);
    }

    /**
     * @notice Returns 1e-12 as a quadruple-precision constant.
     *         Default tolerance used in SmartSolve numeric routines.
     * @return bytes16 1 x 10^-12
     */
    function EPS_1e12() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000_000);
    }

    /**
     * @notice Returns 1e-15 as a quadruple-precision constant.
     *         Recommended minimum tolerance ('minTol') for root-finding algorithms.
     *
     * @return bytes16 1 x 10^-15
     */
    function EPS_1e15() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000_000_000);
    }

    /**
     * @notice Returns 1e-18 as a quadruple-precision constant.
     *         Mainly useful for testing clamping logic (too strict tolerances).
     *
     * @return bytes16 1 x 10^-18
     */
    function EPS_1e18() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000_000_000_000);
    }

    /**
     * @notice Returns 1e-30 as a quadruple-precision constant.
     *         Very small epsilon, stored as a precomputed hex literal for accuracy.
     *
     * @return bytes16 1 x 10^-30
     */
    function EPS_1e30() internal pure returns (bytes16) {
        return 0x3cd203af9ee756159b00000000000000;
    }

    // -------------------------------------------------------------
    // Default Config
    // -------------------------------------------------------------

    /**
     * @notice Returns the default numerical tolerance for SmartSolve.
     *         Alias for EPS_1e12(), used by root-finding and integration routines.
     *
     * @return bytes16 1 x 10^-12
     */
    function DEFAULT_TOL() internal pure returns (bytes16) {
        return EPS_1e12();
    }

    /**
     * @notice Returns the default minimum tolerance for SmartSolve.
     *         Alias for EPS_1e15(), used as a floor when clamping tolerance.
     *
     * @return bytes16 1 x 10^-15
     */
    function DEFAULT_MIN_TOL() internal pure returns (bytes16) {
        return EPS_1e15();
    }
}