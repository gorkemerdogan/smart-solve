// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Polynomial } from "../libraries/numeric/Polynomial.sol";

/**
 * @title PolynomialHarness
 * @notice Minimal harness to expose internal library functions of PolynomialQuad for testing.
 * The library functions are `internal`, so we wrap them in `public` functions.
 */
contract PolynomialHarness {
    /**
     * @notice Evaluate a polynomial at point x via Horner’s method (library passthrough).
     * @param coeffs Array where coeffs[i] is the coefficient of x^i, in ABDKMathQuad (bytes16).
     * @param x The input point (bytes16).
     * @return y The value f(x) (bytes16).
     */
    function evaluateHorners(bytes16[] memory coeffs, bytes16 x)
        public
        pure
        returns (bytes16 y)
    {
        return Polynomial.evaluateHorners(coeffs, x);
    }

    /**
     * @notice Compute the derivative polynomial coefficients (library passthrough).
     * @param coeffs Original polynomial coefficients (bytes16).
     * @return d Derivative coefficients (bytes16), where d[i] corresponds to x^i.
     */
    function derivative(bytes16[] memory coeffs)
        public
        pure
        returns (bytes16[] memory d)
    {
        return Polynomial.derivative(coeffs);
    }
}
