// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { AbdkQuad as Quad } from "./AbdkQuad.sol";

/**
 * @title Polynomial
 * @notice ABDKMathQuad (bytes16) polynomial utilities.
 * coeffs[i] corresponds to coefficient of x^i. All coeffs and x are bytes16.
 */
library Polynomial {
    using Quad for bytes16;

    /**
     * @notice Evaluate a polynomial at point x using Horner’s method
     * @param coeffs Array of coefficients, where coeffs[i] is the coefficient of x^i
     * Example: for f(x) = 3x^2 + 2x + 5, coeffs = [Quad.fromInt(5), Quad.fromInt(2), Quad.fromInt(3)]
     * @param x The input value (bytes16) at which to evaluate the polynomial
     * @return y The computed value f(x) (bytes16)
     *
     * Horner’s method rewrites the polynomial:
     * a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
     * into a nested form:
     * (((a_n * x + a_{n-1}) * x + a_{n-2}) * x + ...) * x + a_0
     */
    function evaluateHorners(bytes16[] memory coeffs, bytes16 x) internal pure returns (bytes16 y) {
        // Handle empty array case
        if (coeffs.length == 0) {
            return Quad.fromInt(0); // Return 0 in ABDK bytes16 format
        }

        // Initialize result 'y' with the highest-degree coefficient
        y = coeffs[coeffs.length - 1];

        // Loop down from the second highest degree (n-1) to the constant term (0)
        for (uint256 i = coeffs.length - 1; i > 0; i--) {
            // y = (y * x) + coeffs[i-1]
            y = y.mul(x).add(coeffs[i - 1]);
        }
    }

    /**
     * @notice Compute the derivative polynomial coefficients
     * @param coeffs Array of coefficients of the original polynomial
     * @return d Array of coefficients of the derivative polynomial
     *
     * Rule: derivative of (a_i * x^i) = (i * a_i) * x^(i-1)
     * - Constant term disappears (so length reduces by 1).
     * - Each coefficient is multiplied by its power index i.
     *
     * Example:
     * f(x) = 5 + 2x + 3x^2  -> coeffs = [Q.fromInt(5), Q.fromInt(2), Q.fromInt(3)]
     * f’(x) = 2 + 6x        -> derivative = [Q.fromInt(2), Q.fromInt(6)]
     */
    function derivative(bytes16[] memory coeffs) internal pure returns (bytes16[] memory d) {
        if (coeffs.length <= 1) {
            d = new bytes16[](1);
            d[0] = Quad.fromInt(0); // Derivative of a constant is 0
            return d;
        }

        d = new bytes16[](coeffs.length - 1);
        for (uint256 i = 1; i < coeffs.length; i++) {
            // Cast the uint256 index 'i' to int256
            int256 i_int = int256(i);
            
            // Convert the integer 'i' to ABDK bytes16 format
            bytes16 i_quad = Quad.fromInt(i_int);
            
            // d[i-1] = coeffs[i] * i
            d[i - 1] = coeffs[i].mul(i_quad);
        }
    }
}
