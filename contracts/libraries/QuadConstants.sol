// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title QuadConstants
 * @notice Shared high-precision constants and tiny helpers for IEEE-754
 *         binary128 (quadruple precision) using ABDKMathQuad.
 *
 *         Design notes:
 *         - Everything is returned as bytes16 (ABDK's quad type).
 *         - Implemented as `internal pure` functions, not `constant` values,
 *           because Solidity does not allow calling library functions
 *           (e.g. fromInt, div) in constant initializers.
 *         - This centralizes π, tolerances, and small integers so that
 *           Integration/RootFinding/Trigonometric libraries all use the
 *           *same* definitions.
 */
library QuadConstants {
    using ABDKMathQuad for bytes16;

    // -------------------------------------------------------------
    // HELPERS
    // -------------------------------------------------------------

    /**
     * @dev Rational helper: num/den in quad.
     *      Avoids repeating the fromInt+div pattern everywhere.
     */
    function fromFrac(int256 num, int256 den) internal pure returns (bytes16) {
        bytes16 n = ABDKMathQuad.fromInt(num);
        bytes16 d = ABDKMathQuad.fromInt(den);
        return n.div(d);
    }

    // -------------------------------------------------------------
    // BASIC NUMERIC CONSTANTS
    // -------------------------------------------------------------

    /// @return 0.0 as quad
    function ZERO() internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(0);
    }

    /// @return 1.0 as quad
    function ONE() internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(1);
    }

    /// @return -1.0 as quad
    function MONE() internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(-1);
    }

    /// @return 2.0 as quad
    function TWO() internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(2);
    }

    /// @return 3.0 as quad
    function THREE() internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(3);
    }

    /// @return 4.0 as quad
    function FOUR() internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(4);
    }

    /// @return 0.5 as quad
    function HALF() internal pure returns (bytes16) {
        // 1 / 2
        return fromFrac(1, 2);
    }

    // -------------------------------------------------------------
    // π AND FRIENDS
    // -------------------------------------------------------------

    /**
     * @notice π ≈ 3.141592653589793
     * @dev Represented as 3141592653589793 / 10^15.
     */
    function PI() internal pure returns (bytes16) {
        return fromFrac(3141592653589793, 1000000000000000); // 3.141592653589793
    }

    /// @return π/2
    function HALF_PI() internal pure returns (bytes16) {
        return PI().div(TWO());
    }

    /// @return 2π
    function TWO_PI() internal pure returns (bytes16) {
        return PI().mul(TWO());
    }

    /// @return π/4
    function QUARTER_PI() internal pure returns (bytes16) {
        return PI().div(ABDKMathQuad.fromInt(4));
    }

    // -------------------------------------------------------------
    // TOLERANCES
    // -------------------------------------------------------------

    /// @return 1e-6 as quad
    function EPS_1e6() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000);
    }

    /// @return 1e-12 as quad (good default tolerance)
    function EPS_1e12() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000_000);
    }

    /// @return 1e-15 as quad (default minTol in your numeric config)
    function EPS_1e15() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000_000_000);
    }

    /// @return 1e-18 as quad (for “too strict, must clamp up” tests)
    function EPS_1e18() internal pure returns (bytes16) {
        return fromFrac(1, 1_000_000_000_000_000_000);
    }

    // -------------------------------------------------------------
    // DEFAULT NUMERIC CONFIG
    // -------------------------------------------------------------

    /// @return default tolerance (1e-12)
    function DEFAULT_TOL() internal pure returns (bytes16) {
        return EPS_1e12();
    }

    /// @return default minimum tolerance (1e-15)
    function DEFAULT_MIN_TOL() internal pure returns (bytes16) {
        return EPS_1e15();
    }
}