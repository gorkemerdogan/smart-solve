// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title Scalar
 * @notice Provides helper functions for scalar (single number) operations
 * compatible with the ABDKMathQuad (bytes16) backend.
 */
library Scalar {
    /**
     * @notice Returns the minimum of two ABDKQuad values.
     * @param a First value (bytes16)
     * @param b Second value (bytes16)
     * @return The smaller of a or b
     */
    function min(bytes16 a, bytes16 b) internal pure returns (bytes16) {
        // Use ABDKMathQuad.cmp: returns -1 if a < b
        return ABDKMathQuad.cmp(a, b) < 0 ? a : b;
    }

    /**
     * @notice Returns the maximum of two ABDKQuad values.
     * @param a First value (bytes16)
     * @param b Second value (bytes16)
     * @return The larger of a or b
     */
    function max(bytes16 a, bytes16 b) internal pure returns (bytes16) {
        // Use ABDKMathQuad.cmp: returns 1 if a > b
        return ABDKMathQuad.cmp(a, b) > 0 ? a : b;
    }

    /**
     * @notice Restricts a value x to stay within a lower bound (lo) and upper bound (hi).
     * @dev If x is smaller than lo → return lo.
     * @dev If x is bigger than hi → return hi.
     * @dev Otherwise, return x unchanged.
     * @param x The value to clamp (bytes16)
     * @param lo The lower bound (bytes16)
     * @param hi The upper bound (bytes16)
     * @return The clamped value (bytes16)
     */
    function clamp(bytes16 x, bytes16 lo, bytes16 hi) internal pure returns (bytes16) {
        // Check if x < lo
        if (ABDKMathQuad.cmp(x, lo) < 0) {
            return lo;
        }
        // Check if x > hi
        if (ABDKMathQuad.cmp(x, hi) > 0) {
            return hi;
        }
        // x is between lo and hi
        return x;
    }
}
