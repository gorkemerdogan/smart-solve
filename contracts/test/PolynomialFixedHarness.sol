// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { PolynomialFixed } from "../libraries/numeric/PolynomialFixed.sol";

/**
 * PolynomialFixedHarness
 * Minimal harness to expose internal library functions of PolynomialFixed for testing.
 * The library functions are `internal`, so we wrap them in `public` functions.
 */
contract PolynomialFixedHarness {
    /**
     * Evaluate a polynomial at point x via Horner’s method (library passthrough).
     * @param coeffs Array where coeffs[i] is the coefficient of x^i, in SD59x18.
     * @param x The input point (SD59x18).
     * @return y The value f(x) (SD59x18).
     */
    function evaluateHorners(int256[] memory coeffs, int256 x)
        public
        pure
        returns (int256 y)
    {
        return PolynomialFixed.evaluateHorners(coeffs, x);
    }

    /**
     * Compute the derivative polynomial coefficients (library passthrough).
     * @param coeffs Original polynomial coefficients (SD59x18).
     * @return d Derivative coefficients (SD59x18), where d[i] corresponds to x^i.
     */
    function derivative(int256[] memory coeffs)
        public
        pure
        returns (int256[] memory d)
    {
        return PolynomialFixed.derivative(coeffs);
    }
}