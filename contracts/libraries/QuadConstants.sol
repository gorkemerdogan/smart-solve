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

    /// @return 0.5 as quad
    function HALF() internal pure returns (bytes16) {
        // 1 / 2
        return fromFrac(1, 2);
    }

    // -------------------------------------------------------------
    // π AND FRIENDS
    // -------------------------------------------------------------

    /**
     * @notice π ≈ 3.1415926535897932384626433832795
     * @dev Represented as 31415926535897932384626433832795 / 10^31.
     * This provides 31 digits of precision, suitable for quad-precision.
     */
    function PI() internal pure returns (bytes16) {
        return fromFrac(
            31415926535897932384626433832795, // 31 digits
            10000000000000000000000000000000  // 10^31
        );
    }

    /// @return π/2
    function HALF_PI() internal pure returns (bytes16) {
        return PI().div(ABDKMathQuad.fromInt(2));
    }

    /// @return 2π
    function TWO_PI() internal pure returns (bytes16) {
        return PI().mul(ABDKMathQuad.fromInt(2));
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