// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
* Scalar Utility Library
* Provides helper functions for scalar (single number) operations
* used across numerical methods, compatible with both fixed-point
* and floating-point backends.
*/
library Scalar {
    function min(int256 a, int256 b) internal pure returns (int256) { return a < b ? a : b; }
    function max(int256 a, int256 b) internal pure returns (int256) { return a > b ? a : b; }

    // Restricts a value x to stay within a lower bound (lo) and upper bound (hi).
    // If x is smaller than lo → return lo.
    // If x is bigger than hi → return hi.
    // Otherwise, return x unchanged.
    // Example: clamp(x, lo, hi) = “Make sure x doesn’t go below lo or above hi.”
    function clamp(int256 x, int256 lo, int256 hi) internal pure returns (int256) {
        return x < lo ? lo : (x > hi ? hi : x);
    }

    // Absolute function to return absolute value of a
    function abs(int256 a) internal pure returns (int256) {
        // Standard non-fixed-point absolute value
        return a >= 0 ? a : -a; 
    }
}