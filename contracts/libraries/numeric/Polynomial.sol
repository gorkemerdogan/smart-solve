// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title Polynomial
 * @notice ABDKMathQuad (bytes16) polynomial utilities.
 * coeffs[i] corresponds to coefficient of x^i. All coeffs and x are bytes16.
 */
library Polynomial {
    using ABDKMathQuad for bytes16;
    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

    /**
     * @notice Evaluate a polynomial at point x using Horner’s method
     * @param coeffs Array of coefficients, where coeffs[i] is the coefficient of x^i
     * Example: for f(x) = 3x^2 + 2x + 5, coeffs = [ABDKMathQuad.fromInt(5), ABDKMathQuad.fromInt(2), ABDKMathQuad.fromInt(3)]
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
            return ABDKMathQuad.fromInt(0); // Return 0 in ABDK bytes16 format
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
            d[0] = ABDKMathQuad.fromInt(0); // Derivative of a constant is 0
            return d;
        }

        d = new bytes16[](coeffs.length - 1);
        for (uint256 i = 1; i < coeffs.length; i++) {
            // Cast the uint256 index 'i' to int256
            int256 i_int = int256(i);
            
            // Convert the integer 'i' to ABDK bytes16 format
            bytes16 i_quad = ABDKMathQuad.fromInt(i_int);
            
            // d[i-1] = coeffs[i] * i
            d[i - 1] = coeffs[i].mul(i_quad);
        }
    }


    /**
    * @notice Polynomial addition: out[i] = a[i] + b[i]
    * @param a coeffs of first polynomial (bytes16[])
    * @param b coeffs of second polynomial (bytes16[])
    */
    function add(bytes16[] memory a, bytes16[] memory b) internal pure returns (bytes16[] memory out) {
        if (a.length == 0 && b.length == 0) return _zeroPoly();
        if (a.length == 0) return b;
        if (b.length == 0) return a;

        uint256 n = a.length > b.length ? a.length : b.length;
        out = new bytes16[](n);
        for (uint256 i = 0; i < n; i++) {
            bytes16 ai = i < a.length ? a[i] : ABDKMathQuad.fromInt(0);
            bytes16 bi = i < b.length ? b[i] : ABDKMathQuad.fromInt(0);
            out[i] = ai.add(bi);
        }
        // return trimTrailingZeros(out); // enable if you want canonical form
    }

    /**
    * @notice Polynomial subtraction: out[i] = a[i] - b[i]
    */
    function sub(bytes16[] memory a, bytes16[] memory b) internal pure returns (bytes16[] memory out) {
        if (a.length == 0 && b.length == 0) return _zeroPoly();
        if (a.length == 0) {
            // 0 - b
            out = new bytes16[](b.length);
            for (uint256 i = 0; i < b.length; i++) out[i] = ABDKMathQuad.fromInt(0).sub(b[i]);
            return out;
        }
        if (b.length == 0) return a;

        uint256 n = a.length > b.length ? a.length : b.length;
        out = new bytes16[](n);
        for (uint256 i = 0; i < n; i++) {
            bytes16 ai = i < a.length ? a[i] : ABDKMathQuad.fromInt(0);
            bytes16 bi = i < b.length ? b[i] : ABDKMathQuad.fromInt(0);
            out[i] = ai.sub(bi);
        }
        // return trimTrailingZeros(out);
    }

    /**
    * @notice Scalar multiplication: out[i] = a[i] * k
    */
    function mulScalar(bytes16[] memory a, bytes16 k) internal pure returns (bytes16[] memory out) {
        if (a.length == 0) return _zeroPoly();
        // If k == 0 => zero polynomial
        if (_isZero(k)) return _zeroPoly();

        out = new bytes16[](a.length);
        for (uint256 i = 0; i < a.length; i++) {
            out[i] = a[i].mul(k);
        }
        // return trimTrailingZeros(out);
    }

    /**
    * @notice Polynomial convolution: (a * b)
    * @dev If either is zero, returns [0]
    */
    function mul(bytes16[] memory a, bytes16[] memory b) internal pure returns (bytes16[] memory out) {
        if (a.length == 0 || b.length == 0) return _zeroPoly();
        // quick zero checks (optional)
        if (a.length == 1 && _isZero(a[0])) return _zeroPoly();
        if (b.length == 1 && _isZero(b[0])) return _zeroPoly();

        out = new bytes16[](a.length + b.length - 1);
        for (uint256 i = 0; i < a.length; i++) {
            for (uint256 j = 0; j < b.length; j++) {
                // out[i+j] += a[i]*b[j]
                bytes16 term = a[i].mul(b[j]);
                out[i + j] = out[i + j].add(term);
            }
        }
        // return trimTrailingZeros(out);
    }

    // --- calculus ---

    /**
    * @notice Indefinite integral with constant C: q'(x)=p(x), q(0)=C
    * @param a polynomial coefficients a[i] for x^i
    * @param C constant term for the integral (bytes16)
    * @return out coefficients for q(x)
    */
    function integral(bytes16[] memory a, bytes16 C) internal pure returns (bytes16[] memory out) {
        if (a.length == 0) {
            out = new bytes16[](1);
            out[0] = C;
            return out;
        }
        out = new bytes16[](a.length + 1);
        out[0] = C;
        for (uint256 i = 0; i < a.length; i++) {
            bytes16 denom = ABDKMathQuad.fromInt(int256(i + 1));
            out[i + 1] = a[i].div(denom);
        }
    }

    /**
    * @notice Extended Horner: compute p(x) and p’(x) in one pass.
    * @return px = p(x), dpx = p’(x)
    */
    function evaluateWithDerivative(bytes16[] memory a, bytes16 x)
        internal
        pure
        returns (bytes16 px, bytes16 dpx)
    {
        px = ABDKMathQuad.fromInt(0);
        dpx = ABDKMathQuad.fromInt(0);
        if (a.length == 0) return (px, dpx);

        for (uint256 i = a.length; i > 0; i--) {
            dpx = dpx.mul(x).add(px);
            px  = px.mul(x).add(a[i - 1]);
        }
    }

    // --- Helpers ---

    function _zeroPoly() private pure returns (bytes16[] memory z) {
        z = new bytes16[](1);
        z[0] = ABDKMathQuad.fromInt(0);
    }

    function _isZero(bytes16 c) private pure returns (bool) {
        // For IEEE-754 quad, exact zero check is ok
        return c == QZERO;
    }

    /**
    * @notice Returns degree (highest i with non-zero coeff); zero poly -> 0
    */
    function degree(bytes16[] memory a) internal pure returns (uint256) {
        if (a.length == 0) return 0;
        for (uint256 i = a.length; i > 0; i--) {
            if (!_isZero(a[i - 1])) return i - 1;
        }
        return 0;
    }

    /**
    * @notice Optionally trim trailing zeros to canonical length.
    */
    function trimTrailingZeros(bytes16[] memory a) internal pure returns (bytes16[] memory out) {
        if (a.length == 0) return _zeroPoly();
        uint256 deg = degree(a);
        out = new bytes16[](deg + 1);
        for (uint256 i = 0; i <= deg; i++) out[i] = a[i];
    }
}
