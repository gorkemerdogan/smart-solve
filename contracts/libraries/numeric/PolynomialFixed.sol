// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { FixedPoint as FP } from "./FixedPoint.sol";

/**
 * PolynomialFixed
 * Fixed-point (SD59x18) polynomial utilities.
 * coeffs[i] corresponds to coefficient of x^i. All coeffs and x are SD59x18.
 */
library PolynomialFixed {
    using FP for int256;

    /**
     * Evaluate a polynomial at point x using Horner’s method
     * @param coeffs Array of coefficients, where coeffs[i] is the coefficient of x^i
     *        Example: for f(x) = 3x^2 + 2x + 5, coeffs = [5, 2, 3]
     * @param x The input value at which to evaluate the polynomial
     * @return y The computed value f(x)
     *
     * Horner’s method rewrites the polynomial:
     *   a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
     * into a nested form:
     *   (((a_n * x + a_{n-1}) * x + a_{n-2}) * x + ...) * x + a_0
     *
     * This reduces the number of multiplications (more gas-efficient).
     */
    function evaluateHorners(int256[] memory coeffs, int256 x) internal pure returns (int256 y) {
        // Handle empty array case
        if (coeffs.length == 0) {
            return 0;
        }

        // Initialize result 'y' with the highest-degree coefficient
        y = coeffs[coeffs.length - 1];

        // Loop down from the second highest degree (n-1) to the constant term (0)
        for (uint256 i = coeffs.length - 1; i > 0; i--) {
            // Multiply current result by x and add next coefficient
            y = y.mul(x).add(coeffs[i - 1]);
        }
    }

    /**
     * Compute the derivative polynomial coefficients
     * @param coeffs Array of coefficients of the original polynomial
     * @return d Array of coefficients of the derivative polynomial
     *
     * Rule: derivative of (a_i * x^i) = (i * a_i) * x^(i-1)
     * - Constant term disappears (so length reduces by 1).
     * - Each coefficient is multiplied by its power index i.
     *
     * Example:
     *   f(x) = 5 + 2x + 3x^2  -> coeffs = [5, 2, 3]
     *   f’(x) = 2 + 6x        -> derivative = [2, 6]
     */
    function derivative(int256[] memory coeffs) internal pure returns (int256[] memory d) {
        if (coeffs.length <= 1) {
            d = new int256[](1);
            d[0] = 0; // Derivative of a constant is 0, which is represented by [0]
            return d;
        }

        d = new int256[](coeffs.length - 1);
        for (uint256 i = 1; i < coeffs.length; i++) {
            // Cast the uint256 index 'i' to int256
            int256 i_int = int256(i);
            
            // Scale the integer 'i' to SD59x18 format
            int256 i_fp = FP.fromInt(i_int);
            
            // Perform fixed-point multiplication: d[i-1] = coeffs[i] * i
            d[i - 1] = coeffs[i].mul(i_fp);
        }
    }
}