// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * AbdkQuad
 * Tiny facade for ABDKMathQuad (IEEE-754 quadruple precision, bytes16).
 * Conversions to/from int for convenience. Avoid heavy string conversions on-chain.
 */
library AbdkQuad {
    using ABDKMathQuad for bytes16;

    function fromInt(int256 x) internal pure returns (bytes16) {
        return ABDKMathQuad.fromInt(x);
    }

    function toInt(bytes16 x) internal pure returns (int256) {
        return ABDKMathQuad.toInt(x);
    }

    function add(bytes16 a, bytes16 b) internal pure returns (bytes16) { return a.add(b); }
    function sub(bytes16 a, bytes16 b) internal pure returns (bytes16) { return a.sub(b); }
    function mul(bytes16 a, bytes16 b) internal pure returns (bytes16) { return a.mul(b); }
    function div(bytes16 a, bytes16 b) internal pure returns (bytes16) { return a.div(b); }

    // Optional helpers for comparisons
    function abs(bytes16 a) internal pure returns (bytes16) {
        return a.abs();
    }

    // Compares two fixed-point numbers for "approximate equality"
    // @param a First value
    // @param b Second value
    // @param tolerance Allowed difference (e.g., 1e9 for 1e-9 precision)
    function nearlyEqual(bytes16 a, bytes16 b, bytes16 eps) internal pure returns (bool) {
        bytes16 diff = abs(a.sub(b));
        return ABDKMathQuad.cmp(diff, eps) <= 0;
    }
}