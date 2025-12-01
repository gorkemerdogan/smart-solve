// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Polynomial } from "../libraries/numeric/Polynomial.sol";
import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title PolynomialHarness
 * @notice Exposes internal functions of the Polynomial library for testing and verification.
 * @dev All methods are direct passthroughs to library implementations. Additional helpers
 *      are provided for constructing quad-precision constants and producing debug strings.
 */
contract PolynomialHarness {
    using MathLib for bytes16;

    // ---------------------------------------------------------
    // Polynomial library
    // ---------------------------------------------------------

    /**
     * @notice Evaluates a polynomial using Horner's method.
     * @param coeffs Coefficient array in ascending powers: coeffs[i] = aᵢ.
     * @param x Evaluation point.
     * @return y Polynomial value p(x) in quad precision.
     */
    function evaluateHorners(bytes16[] memory coeffs, bytes16 x) public pure returns (bytes16) {
        return Polynomial.evaluateHorners(coeffs, x);
    }

    /**
     * @notice Computes both p(x) and p′(x) using extended Horner’s method.
     * @param coeffs Coefficient array in ascending powers.
     * @param x Evaluation point.
     * @return px Value of p(x).
     * @return dpx Value of p′(x).
     */
    function evaluateWithDerivative(bytes16[] memory coeffs, bytes16 x) public pure returns (bytes16 px, bytes16 dpx) {
        return Polynomial.evaluateWithDerivative(coeffs, x);
    }

    /**
     * @notice Evaluates a monic polynomial of the form:
     *         p(x) = xⁿ + aₙ₋₁ xⁿ⁻¹ + ... + a₁ x + a₀.
     * @param lowerCoeffs Coefficients [a₀, a₁, ..., aₙ₋₁].
     * @param x Evaluation point.
     * @return y Value of p(x).
     */
    function evalHornerMonic(bytes16[] memory lowerCoeffs, bytes16 x) public pure returns (bytes16) {
        return Polynomial.evalHornerMonic(lowerCoeffs, x);
    }

    /**
     * @notice Computes p(x) + q(x).
     * @param coeffs_a First polynomial coefficients.
     * @param coeffs_b Second polynomial coefficients.
     * @return out Coefficient array representing the sum.
     */
    function add(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) public pure returns (bytes16[] memory) {
        return Polynomial.add(coeffs_a, coeffs_b);
    }

    /**
     * @notice Computes p(x) − q(x).
     * @param coeffs_a First polynomial coefficients.
     * @param coeffs_b Second polynomial coefficients.
     * @return out Coefficient array representing the difference.
     */
    function sub(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) public pure returns (bytes16[] memory) {
        return Polynomial.sub(coeffs_a, coeffs_b);
    }

    /**
     * @notice Computes scalar multiplication k · p(x).
     * @param coeffs Polynomial coefficients.
     * @param k Scalar multiplier.
     * @return out Coefficients of the resulting polynomial.
     */
    function mulScalar(bytes16[] memory coeffs, bytes16 k) public pure returns (bytes16[] memory) {
        return Polynomial.mulScalar(coeffs, k);
    }

    /**
     * @notice Computes polynomial multiplication p(x) · q(x).
     * @param coeffs_a First polynomial coefficients.
     * @param coeffs_b Second polynomial coefficients.
     * @return out Coefficient array representing the product.
     */
    function mul(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) public pure returns (bytes16[] memory) {
        return Polynomial.mul(coeffs_a, coeffs_b);
    }

    /**
     * @notice Performs synthetic division by (x − root).
     * @param coeffs Polynomial coefficients in ascending powers.
     * @param root Root value r, represented in quad precision.
     * @return q Quotient coefficients.
     * @return r Remainder p(r).
     */
    function syntheticDivide(bytes16[] memory coeffs, bytes16 root) public pure returns (bytes16[] memory q, bytes16 r) {
        return Polynomial.syntheticDivide(coeffs, root);
    }

    // ---------------------------------------------------------
    // Calculus
    // ---------------------------------------------------------

    /**
     * @notice Computes the derivative polynomial coefficients.
     * @param coeffs Polynomial coefficients in ascending powers.
     * @return d Coefficient array of derivative p′(x).
     */
    function derivative(bytes16[] memory coeffs) public pure returns (bytes16[] memory) {
        return Polynomial.derivative(coeffs);
    }

    /**
     * @notice Computes the indefinite integral of a polynomial.
     * @param coeffs Polynomial coefficients p(x).
     * @param C Integration constant.
     * @return out Coefficients of ∫p(x)dx + C.
     */
    function integral(bytes16[] memory coeffs, bytes16 C) public pure returns (bytes16[] memory) {
        return Polynomial.integral(coeffs, C);
    }

    // ---------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------

    /**
     * @notice Computes the polynomial degree (highest non-zero coefficient index).
     * @param coeffs Polynomial coefficients.
     * @return deg Degree of the polynomial. Zero polynomial returns 0.
     */
    function degree(bytes16[] memory coeffs) public pure returns (uint256) {
        return Polynomial.degree(coeffs);
    }

    /**
     * @notice Removes trailing zero coefficients and returns the minimal-length canonical form.
     * @param coeffs Polynomial coefficients.
     * @return out Trimmed coefficient array.
     */
    function trimTrailingZeros(bytes16[] memory coeffs) public pure returns (bytes16[] memory) {
        return Polynomial.trimTrailingZeros(coeffs);
    }

    // ---------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------

    /**
     * @notice Converts an integer to quad precision.
     * @param x Integer value.
     * @return q Quad-precision representation.
     */
    function qFromInt(int256 x) public pure returns (bytes16) {
        return MathLib.fromInt(x);
    }

    /**
     * @notice Converts a rational number num/den into quad precision.
     * @dev Reverts if den = 0.
     * @param num Signed numerator.
     * @param den Signed denominator.
     * @return q Quad-precision representation of num/den.
     */
    function qFromFrac(int256 num, int256 den) public pure returns (bytes16) {
        require(den != 0, "den=0");
        return MathLib.fromInt(num).div(MathLib.fromInt(den));
    }

    /**
     * @notice Returns a hexadecimal string representation of a quad value.
     * @dev Intended for logging and debugging during unit tests.
     * @param x Quad-precision number.
     * @return str Hexadecimal string "0x…".
     */
    function qToString(bytes16 x) public pure returns (string memory) {
        bytes memory b = abi.encodePacked(x);
        bytes memory hexChars = "0123456789abcdef";
        bytes memory str = new bytes(2 + b.length * 2);
        str[0] = "0";
        str[1] = "x";
        for (uint i = 0; i < b.length; i++) {
            str[2 + i * 2] = hexChars[uint8(b[i] >> 4)];
            str[3 + i * 2] = hexChars[uint8(b[i] & 0x0f)];
        }
        return string(str);
    }
}