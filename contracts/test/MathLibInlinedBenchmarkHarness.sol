// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad } from "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title MathLibInlinedBenchmarkHarness
 * @notice Test-only microbenchmark using ABDK internal arithmetic directly.
 * @dev This contract is intentionally isolated from production numerical code.
 */
contract MathLibInlinedBenchmarkHarness {
    function arithmeticLoop(bytes16 initial, bytes16 a, bytes16 b, uint256 iterations)
        external
        pure
        returns (bytes16 value)
    {
        value = initial;
        for (uint256 i = 0; i < iterations; ++i) {
            value = ABDKMathQuad.add(value, a);
            value = ABDKMathQuad.sub(value, b);
            value = ABDKMathQuad.mul(value, b);
            value = ABDKMathQuad.div(value, a);
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
            value = ABDKMathQuad.add(ABDKMathQuad.mul(value, x), coefficients[i]);
        }
    }

    function dot(bytes16[] calldata a, bytes16[] calldata b)
        external
        pure
        returns (bytes16 value)
    {
        require(a.length == b.length, "MathLibBenchmark: vector length mismatch");
        for (uint256 i = 0; i < a.length; ++i) {
            value = ABDKMathQuad.add(value, ABDKMathQuad.mul(a[i], b[i]));
        }
    }
}
