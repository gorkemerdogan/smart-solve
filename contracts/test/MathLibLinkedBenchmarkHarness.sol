// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title MathLibLinkedBenchmarkHarness
 * @notice Test-only microbenchmark for the production-style linked MathLib path.
 * @dev This contract is intentionally isolated from production numerical code.
 */
contract MathLibLinkedBenchmarkHarness {
    function arithmeticLoop(bytes16 initial, bytes16 a, bytes16 b, uint256 iterations)
        external
        pure
        returns (bytes16 value)
    {
        value = initial;
        for (uint256 i = 0; i < iterations; ++i) {
            value = MathLib.add(value, a);
            value = MathLib.sub(value, b);
            value = MathLib.mul(value, b);
            value = MathLib.div(value, a);
        }
    }

    function horner(bytes16 x, bytes16[] calldata coefficients)
        external
        pure
        returns (bytes16 value)
    {
        require(coefficients.length > 0, "MathLibBenchmark: empty polynomial");
        value = coefficients[0];
        for (uint256 i = 1; i < coefficients.length; ++i) {
            value = MathLib.add(MathLib.mul(value, x), coefficients[i]);
        }
    }

    function dot(bytes16[] calldata a, bytes16[] calldata b)
        external
        pure
        returns (bytes16 value)
    {
        require(a.length == b.length, "MathLibBenchmark: vector length mismatch");
        for (uint256 i = 0; i < a.length; ++i) {
            value = MathLib.add(value, MathLib.mul(a[i], b[i]));
        }
    }
}
