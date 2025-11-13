// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";
import { Scalar } from "./numeric/Scalar.sol";

/**
 * Helper
 * Tiny facade for additional functions
 * Uses ABDKMathQuad (IEEE-754 quadruple precision, bytes16).
 */
library Helper {
    using ABDKMathQuad for bytes16;

    /**
     * @notice Compares two floating-point numbers for "approximate equality"
     * using a combined absolute and relative tolerance.
     * @param a First value
     * @param b Second value
     * @param absTol The absolute tolerance (good for comparisons near zero)
     * @param relTol The relative tolerance (scales with magnitude of a and b)
     */
    function nearlyEqual(bytes16 a, bytes16 b, bytes16 absTol, bytes16 relTol) internal pure returns (bool) {
        bytes16 diff = a.sub(b).abs();

        // Standard check for most cases
        bytes16 largest = Scalar.max(a.abs(), b.abs());
        bytes16 tolerance = Scalar.max(absTol, largest.mul(relTol));

        return ABDKMathQuad.cmp(diff, tolerance) <= 0;
    }
}