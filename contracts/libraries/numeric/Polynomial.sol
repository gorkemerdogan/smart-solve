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
            return QZERO; // Return 0 in ABDK bytes16 format
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
    * @notice Evaluate a monic polynomial at x using Horner’s method.
    *         The leading coefficient is implicitly 1, so the input excludes it.
    *         p(x) = x^n + coeffs[n-1] * x^(n-1) + ... + coeffs[1] * x + coeffs[0]
    *
    * @param coeffs  Lower coefficients in ascending order: [a0, a1, ..., a_{n-1}]
    * @param x       Evaluation point (bytes16, ABDK quad)
    * @return y      p(x) as bytes16 (ABDK quad)
    *
    * Gas note: iterates once over coeffs, O(n), no storage.
    */
    function evalHornerMonic(bytes16[] memory coeffs, bytes16 x) internal pure returns (bytes16 y) {
        // Degree-1 monic: p(x) = x
        if (coeffs.length == 0) {
            return x;
        }

        // Horner with implicit leading 1:
        // acc_0 = 1
        // acc_{k+1} = acc_k * x + a_{n-1-k}, ending at a0
        bytes16 acc = ABDKMathQuad.fromInt(1); // leading coefficient
        for (uint256 i = coeffs.length; i > 0; i--) {
            acc = acc.mul(x).add(coeffs[i - 1]);
        }
        return acc;
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
            d[0] = QZERO; // Derivative of a constant is 0
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
    * @notice Polynomial addition: out[i] = coeffs_a[i] + coeffs_b[i]
    * @param coeffs_a coeffs of first polynomial (bytes16[])
    * @param coeffs_b coeffs of second polynomial (bytes16[])
    */
    function add(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) internal pure returns (bytes16[] memory out) {
        if (coeffs_a.length == 0 && coeffs_b.length == 0) return _zeroPoly();
        if (coeffs_a.length == 0) return coeffs_b;
        if (coeffs_b.length == 0) return coeffs_a;

        uint256 n = coeffs_a.length > coeffs_b.length ? coeffs_a.length : coeffs_b.length;
        out = new bytes16[](n);
        for (uint256 i = 0; i < n; i++) {
            bytes16 ai = i < coeffs_a.length ? coeffs_a[i] : QZERO;
            bytes16 bi = i < coeffs_b.length ? coeffs_b[i] : QZERO;
            out[i] = ai.add(bi);
        }
        return out;
    }

    /**
    * @notice Polynomial subtraction: out[i] = coeffs_a[i] - coeffs_b[i]
    */
    function sub(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) internal pure returns (bytes16[] memory out) {
        if (coeffs_a.length == 0 && coeffs_b.length == 0) return _zeroPoly();
        if (coeffs_a.length == 0) {
            // 0 - b
            out = new bytes16[](coeffs_b.length);
            for (uint256 i = 0; i < coeffs_b.length; i++) out[i] = QZERO.sub(coeffs_b[i]);
            return out;
        }
        if (coeffs_b.length == 0) return coeffs_a;

        uint256 n = coeffs_a.length > coeffs_b.length ? coeffs_a.length : coeffs_b.length;
        out = new bytes16[](n);
        for (uint256 i = 0; i < n; i++) {
            bytes16 ai = i < coeffs_a.length ? coeffs_a[i] : QZERO;
            bytes16 bi = i < coeffs_b.length ? coeffs_b[i] : QZERO;
            out[i] = ai.sub(bi);
        }
        return out;
    }

    /**
    * @notice Scalar multiplication: out[i] = coeffs[i] * k
    */
    function mulScalar(bytes16[] memory coeffs, bytes16 k) internal pure returns (bytes16[] memory out) {
        if (coeffs.length == 0) return _zeroPoly();
        // If k == 0 => zero polynomial
        if (_isZero(k)) return _zeroPoly();

        out = new bytes16[](coeffs.length);
        for (uint256 i = 0; i < coeffs.length; i++) {
            out[i] = coeffs[i].mul(k);
        }
        return out;
    }

    /**
    * @notice Polynomial convolution: (a * b)
    * @dev If either is zero, returns [0]
    */
    function mul(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) internal pure returns (bytes16[] memory out) {
        if (coeffs_a.length == 0 || coeffs_b.length == 0) return _zeroPoly();
        // quick zero checks (optional)
        if (coeffs_a.length == 1 && _isZero(coeffs_a[0])) return _zeroPoly();
        if (coeffs_b.length == 1 && _isZero(coeffs_b[0])) return _zeroPoly();

        out = new bytes16[](coeffs_a.length + coeffs_b.length - 1);
        for (uint256 i = 0; i < coeffs_a.length; i++) {
            for (uint256 j = 0; j < coeffs_b.length; j++) {
                bytes16 term = coeffs_a[i].mul(coeffs_b[j]);
                out[i + j] = out[i + j].add(term);
            }
        }
        return out;
    }

    // --- calculus ---

    /**
    * @notice Indefinite integral with constant C: q'(x)=p(x), q(0)=C
    * @param coeffs polynomial coefficients coeffs[i] for x^i
    * @param C constant term for the integral (bytes16)
    * @return out coefficients for q(x)
    */
    function integral(bytes16[] memory coeffs, bytes16 C) internal pure returns (bytes16[] memory out) {
        if (coeffs.length == 0) {
            out = new bytes16[](1);
            out[0] = C;
            return out;
        }
        out = new bytes16[](coeffs.length + 1);
        out[0] = C;
        for (uint256 i = 0; i < coeffs.length; i++) {
            bytes16 denom = ABDKMathQuad.fromInt(int256(i + 1));
            out[i + 1] = coeffs[i].div(denom);
        }
    }

    /**
    * @notice Extended Horner: compute p(x) and p’(x) in one pass.
    * @return px = p(x), dpx = p’(x)
    */
    function evaluateWithDerivative(bytes16[] memory coeffs, bytes16 x)
        internal
        pure
        returns (bytes16 px, bytes16 dpx)
    {
        px = QZERO;
        dpx = QZERO;
        if (coeffs.length == 0) return (px, dpx);

        for (uint256 i = coeffs.length; i > 0; i--) {
            dpx = dpx.mul(x).add(px);
            px  = px.mul(x).add(coeffs[i - 1]);
        }
    }

    /**
    * @notice Synthetic division by (x - root) for polynomials with asc coeffs.
    *         Given P(x) = a0 + a1*x + ... + an*x^n, returns Q(x) and remainder R such that:
    *         P(x) = (x - root) * Q(x) + R
    *
    * @param coeffs  Polynomial coefficients in ascending powers: [a0, a1, ..., an]
    * @param root    The root r in (x - r) as bytes16 (ABDK quad)
    * @return q      Quotient coefficients in ascending powers, length = max(1, n-1)
    * @return r      Remainder as bytes16 (ABDK quad)
    */
    function syntheticDivide(bytes16[] memory coeffs, bytes16 root) internal pure returns (bytes16[] memory q, bytes16 r) {
        uint256 n = coeffs.length;
        if (n == 0) {
            // P(x) = 0 -> Q = [0], R = 0
            q = new bytes16[](1);
            q[0] = ABDKMathQuad.fromInt(0);
            r   = ABDKMathQuad.fromInt(0);
            return (q, r);
        }
        if (n == 1) {
            // Degree 0: P(x) = a0 -> Q = [0], R = a0
            q = new bytes16[](1);
            q[0] = ABDKMathQuad.fromInt(0);
            r    = coeffs[0];
            return (q, r);
        }

        // Degree n: build quotient of degree n-1 (length n-1)
        // Let q[n-2] = a_n (top coeff)
        // For i = n-2 .. 0: q[i-1] = a_{i+1} + root * q[i]
        q = new bytes16[](n - 1);
        q[n - 2] = coeffs[n - 1]; // highest coeff copied

        for (uint256 i = n - 1; i > 1; i--) {
            // i runs: n-1, n-2, ... , 2
            // write q[i-2] = a_{i-1} + root * q[i-1]
            q[i - 2] = coeffs[i - 1].add(root.mul(q[i - 1]));
        }

        // Remainder: r = a0 + root * q[0]
        r = coeffs[0].add(root.mul(q[0]));
        return (q, r);
    }

    // --- Helpers ---

    function _zeroPoly() private pure returns (bytes16[] memory z) {
        z = new bytes16[](1);
        z[0] = QZERO;
    }

    function _isZero(bytes16 c) private pure returns (bool) {
        return c == QZERO;
    }

    /**
    * @notice Returns degree (highest i with non-zero coeff); zero poly -> 0
    */
    function degree(bytes16[] memory coeffs) internal pure returns (uint256) {
        if (coeffs.length == 0) return 0;
        for (uint256 i = coeffs.length; i > 0; i--) {
            if (!_isZero(coeffs[i - 1])) return i - 1;
        }
        return 0;
    }

    /**
    * @notice Optionally trim trailing zeros to canonical length.
    */
    function trimTrailingZeros(bytes16[] memory coeffs) internal pure returns (bytes16[] memory out) {
        if (coeffs.length == 0) return _zeroPoly();
        uint256 deg = degree(coeffs);
        out = new bytes16[](deg + 1);
        for (uint256 i = 0; i <= deg; i++) out[i] = coeffs[i];
    }
}
