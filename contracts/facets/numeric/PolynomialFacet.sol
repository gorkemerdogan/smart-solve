// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Polynomial } from "../../libraries/numeric/Polynomial.sol";

/**
 * @title  PolynomialFacet
 * @notice External Diamond facet that exposes Polynomial (ABDK quad) utilities.
 *         All inputs/outputs use IEEE-754 quadruple precision as bytes16.
 *         Library functions require memory arrays, therefore calldata arrays are copied to memory once per call.
 */
contract PolynomialFacet {

    // ------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------

    /**
     * @notice Evaluate a polynomial at point x using Horner’s method
     * @dev    Horner’s method rewrites the polynomial:
     *         a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
     *         into a nested form:
     *         (((a_n * x + a_{n-1}) * x + a_{n-2}) * x + ...) * x + a_0
     * @param  coeffs Array of coefficients, where coeffs[i] is the coefficient of x^i
     *                Example: for f(x) = 3x^2 + 2x + 5, coeffs = [MathLib.fromInt(5), MathLib.fromInt(2), MathLib.fromInt(3)]
     * @param  x The input value (bytes16) at which to evaluate the polynomial
     * @return y The computed value f(x) (bytes16)
     *
     */
    function polyEvaluate(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 y) {
        return Polynomial.evaluateHorners(_toMemory(coeffs), x);
    }

    /**
     * @notice Computes both p(x) and p’(x) in a single pass using an extended Horner method.
     *         Iterates once through all coefficients, producing both function and derivative values.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @param  x      Evaluation point encoded as bytes16.
     * @return px     Value of the polynomial p(x).
     * @return dpx    Value of the derivative p’(x).
     */
    function polyEvaluateWithDerivative(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 px, bytes16 dpx) {
        return Polynomial.evaluateWithDerivative(_toMemory(coeffs), x);
    }

    /**
     * @notice Evaluate a monic polynomial at x using Horner’s method.
     *         The leading coefficient is implicitly 1, so the input excludes it.
     *         p(x) = x^n + coeffs[n-1] * x^(n-1) + ... + coeffs[1] * x + coeffs[0]
     *
     * @param  coeffs  Lower coefficients in ascending order: [a0, a1, ..., a_{n-1}]
     * @param  x       Evaluation point (bytes16, ABDK quad)
     * @return y       p(x) as bytes16 (ABDK quad)
     *
     * @custom:gas iterates once over coeffs, O(n), no storage.
     */
    function polyEvalHornerMonic(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 y) {
        return Polynomial.evalHornerMonic(_toMemory(coeffs), x);
    }

    /**
     * @notice Compute the derivative polynomial coefficients
     * @dev    Rule: derivative of (a_i * x^i) = (i * a_i) * x^(i-1)
     *          - Constant term disappears (so length reduces by 1).
     *          - Each coefficient is multiplied by its power index i.
     *
     *          Example:
     *           f(x) = 5 + 2x + 3x^2  -> coeffs = [Q.fromInt(5), Q.fromInt(2), Q.fromInt(3)]
     *           f’(x) = 2 + 6x        -> derivative = [Q.fromInt(2), Q.fromInt(6)]
     * @param  coeffs Array of coefficients of the original polynomial
     * @return d      Array of coefficients of the derivative polynomial
     */
    function polyDerivative(bytes16[] calldata coeffs) external pure returns (bytes16[] memory d) {
        return Polynomial.derivative(_toMemory(coeffs));
    }

    /**
     * @notice Computes the coefficients of the indefinite integral of a polynomial.
     *         Produces q(x) such that q'(x) = p(x) and q(0) = C.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @param  C      Constant of integration encoded as bytes16.
     * @return out    Coefficients of the integral polynomial.
     */
    function polyIntegral(bytes16[] calldata coeffs, bytes16 C) external pure returns (bytes16[] memory out) {
        return Polynomial.integral(_toMemory(coeffs), C);
    }

    // ------------------------------------------------------------
    // Arithmetic Operations
    // ------------------------------------------------------------

    /**
     * @notice Computes coefficient-wise polynomial addition.
     *         Missing terms are treated as zero. Returns a zero polynomial when both inputs are empty.
     * @param  coeffs_a Coefficients of the first polynomial.
     * @param  coeffs_b Coefficients of the second polynomial.
     * @return out      Coefficient array representing coeffs_a + coeffs_b.
     */
    function polyAdd(bytes16[] calldata coeffs_a, bytes16[] calldata coeffs_b) external pure returns (bytes16[] memory out) {
        return Polynomial.add(_toMemory(coeffs_a), _toMemory(coeffs_b));
    }

    /**
     * @notice Computes coefficient-wise polynomial subtraction.
     *         Missing terms are treated as zero. When coeffs_a is empty, returns -coeffs_b.
     * @param  coeffs_a Minuend polynomial coefficients.
     * @param  coeffs_b Subtrahend polynomial coefficients.
     * @return out      Resulting coefficients representing coeffs_a - coeffs_b.
     */
    function polySub(bytes16[] calldata coeffs_a, bytes16[] calldata coeffs_b) external pure returns (bytes16[] memory out) {
        return Polynomial.sub(_toMemory(coeffs_a), _toMemory(coeffs_b));
    }

    /**
     * @notice Multiplies each coefficient of a polynomial by a scalar.
     *         Returns a zero polynomial when the input is empty or the scalar is zero.
     * @param  coeffs Polynomial coefficients.
     * @param  k      Scalar multiplier encoded as bytes16.
     * @return out    Coefficients after scalar multiplication.
     */
    function polyMulScalar(bytes16[] calldata coeffs, bytes16 k) external pure returns (bytes16[] memory out) {
        return Polynomial.mulScalar(_toMemory(coeffs), k);
    }

    /**
     * @notice Computes polynomial multiplication via convolution.
     *         Returns a zero polynomial if either polynomial is zero. Complexity is O(n*m).
     * @param  coeffs_a First polynomial coefficients.
     * @param  coeffs_b Second polynomial coefficients.
     * @return out      Resulting coefficients representing the convolution product.
     */
    function polyMul(bytes16[] calldata coeffs_a, bytes16[] calldata coeffs_b) external pure returns (bytes16[] memory out) {
        return Polynomial.mul(_toMemory(coeffs_a), _toMemory(coeffs_b));
    }

    /**
     * @notice Performs synthetic division of a polynomial by (x - root).
     *         Given P(x) = a0 + a1 x + ... + an x^n, returns Q(x) and remainder r such that:
     *         P(x) = (x - root) * Q(x) + r. Inputs must be in ascending powers.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @param  root   The value r for division by (x - r), encoded as bytes16.
     * @return q      Quotient coefficients in ascending powers.
     * @return r      Remainder term encoded as bytes16.
     */
    function polySyntheticDivide(bytes16[] calldata coeffs, bytes16 root) external pure returns (bytes16[] memory q, bytes16 r) {
        return Polynomial.syntheticDivide(_toMemory(coeffs), root);
    }

    // ------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------

    /// @notice Degree of the polynomial (highest i with non-zero coeff). Zero poly -> 0.
    function polyDegree(bytes16[] calldata coeffs) external pure returns (uint256) {
        return Polynomial.degree(_toMemory(coeffs));
    }

    /// @notice Trim trailing zeros to canonical length.
    function polyTrim(bytes16[] calldata coeffs) external pure returns (bytes16[] memory out) {
        return Polynomial.trimTrailingZeros(_toMemory(coeffs));
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    // Copy coeffs calldata bytes16[] into memory (library expects memory).
    function _toMemory(bytes16[] calldata coeffs) internal pure returns (bytes16[] memory m) {
        m = new bytes16[](coeffs.length);
        for (uint256 i = 0; i < coeffs.length; ++i) m[i] = coeffs[i];
    }
}