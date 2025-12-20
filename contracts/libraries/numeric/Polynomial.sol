// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";

/**
 * @title Polynomial
 * @notice ABDKMathQuad (bytes16) polynomial utilities.
 *         coeffs[i] corresponds to coefficient of x^i. All coeffs and x are bytes16.
 */
library Polynomial {
    using MathLib for bytes16;
    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

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
     * @param  coeffs  Lower coefficients in ascending order: [a0, a1, ..., a_{n-1}]
     * @param  x       Evaluation point (bytes16, ABDK quad)
     * @return y       p(x) as bytes16 (ABDK quad)
     *
     * @custom:gas iterates once over coeffs, O(n), no storage.
     */
    function eval'onic(bytes16[] memory coeffs, bytes16 x) internal pure returns (bytes16 y) {
        // Degree-1 monic: p(x) = x
        if (coeffs.length == 0) {
            return MathLib.fromInt(1);
        }

        // Horner with implicit leading 1:
        // acc_0 = 1
        // acc_{k+1} = acc_k * x + a_{n-1-k}, ending at a0
        bytes16 acc = MathLib.fromInt(1); // leading coefficient
        for (uint256 i = coeffs.length; i > 0; i--) {
            acc = acc.mul(x).add(coeffs[i - 1]);
        }
        return acc;
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
            bytes16 i_quad = MathLib.fromInt(i_int);
            
            // d[i-1] = coeffs[i] * i
            d[i - 1] = coeffs[i].mul(i_quad);
        }
    }


    /**
     * @notice Computes coefficient-wise polynomial addition.
     *         Missing terms are treated as zero. Returns a zero polynomial when both inputs are empty.
     * @param  coeffs_a Coefficients of the first polynomial.
     * @param  coeffs_b Coefficients of the second polynomial.
     * @return out      Coefficient array representing coeffs_a + coeffs_b.
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
     * @notice Computes coefficient-wise polynomial subtraction.
     *         Missing terms are treated as zero. When coeffs_a is empty, returns -coeffs_b.
     * @param  coeffs_a Minuend polynomial coefficients.
     * @param  coeffs_b Subtrahend polynomial coefficients.
     * @return out      Resulting coefficients representing coeffs_a - coeffs_b.
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
     * @notice Multiplies each coefficient of a polynomial by a scalar.
     *         Returns a zero polynomial when the input is empty or the scalar is zero.
     * @param  coeffs Polynomial coefficients.
     * @param  k      Scalar multiplier encoded as bytes16.
     * @return out    Coefficients after scalar multiplication.
     */
    function mulScalar(bytes16[] memory coeffs, bytes16 k) internal pure returns (bytes16[] memory out) {
        if (coeffs.length == 0) return _zeroPoly();
        // If k == 0 => zero polynomial
        if (MathLib.isZero(k)) return _zeroPoly();

        out = new bytes16[](coeffs.length);
        for (uint256 i = 0; i < coeffs.length; i++) {
            out[i] = coeffs[i].mul(k);
        }
        return out;
    }

    /**
     * @notice Computes polynomial multiplication via convolution.
     *         Returns a zero polynomial if either polynomial is zero. Complexity is O(n*m).
     * @param  coeffs_a First polynomial coefficients.
     * @param  coeffs_b Second polynomial coefficients.
     * @return out      Resulting coefficients representing the convolution product.
     */
    function mul(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) internal pure returns (bytes16[] memory out) {
        if (coeffs_a.length == 0 || coeffs_b.length == 0) return _zeroPoly();
        // quick zero checks (optional)
        if (coeffs_a.length == 1 && MathLib.isZero(coeffs_a[0])) return _zeroPoly();
        if (coeffs_b.length == 1 && MathLib.isZero(coeffs_b[0])) return _zeroPoly();

        out = new bytes16[](coeffs_a.length + coeffs_b.length - 1);
        for (uint256 i = 0; i < coeffs_a.length; i++) {
            for (uint256 j = 0; j < coeffs_b.length; j++) {
                bytes16 term = coeffs_a[i].mul(coeffs_b[j]);
                out[i + j] = out[i + j].add(term);
            }
        }
        return out;
    }

    /**
     * @notice Computes the coefficients of the indefinite integral of a polynomial.
     *         Produces q(x) such that q'(x) = p(x) and q(0) = C.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @param  C      Constant of integration encoded as bytes16.
     * @return out    Coefficients of the integral polynomial.
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
            bytes16 denom = MathLib.fromInt(int256(i + 1));
            out[i + 1] = coeffs[i].div(denom);
        }
    }

    /**
     * @notice Computes both p(x) and p’(x) in a single pass using an extended Horner method.
     *         Iterates once through all coefficients, producing both function and derivative values.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @param  x      Evaluation point encoded as bytes16.
     * @return px     Value of the polynomial p(x).
     * @return dpx    Value of the derivative p’(x).
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
     * @notice Performs synthetic division of a polynomial by (x - root).
     *         Given P(x) = a0 + a1 x + ... + an x^n, returns Q(x) and remainder r such that:
     *         P(x) = (x - root) * Q(x) + r. Inputs must be in ascending powers.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @param  root   The value r for division by (x - r), encoded as bytes16.
     * @return q      Quotient coefficients in ascending powers.
     * @return r      Remainder term encoded as bytes16.
     */
    function syntheticDivide(bytes16[] memory coeffs, bytes16 root) internal pure returns (bytes16[] memory q, bytes16 r) {
        uint256 n = coeffs.length;
        if (n == 0) {
            // P(x) = 0 -> Q = [0], R = 0
            q = new bytes16[](1);
            q[0] = MathLib.fromInt(0);
            r   = MathLib.fromInt(0);
            return (q, r);
        }
        if (n == 1) {
            // Degree 0: P(x) = a0 -> Q = [0], R = a0
            q = new bytes16[](1);
            q[0] = MathLib.fromInt(0);
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

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Constructs a zero polynomial. Allocates a single-term polynomial representing 0.
     * @return z A polynomial with one coefficient equal to zero.
     */
    function _zeroPoly() private pure returns (bytes16[] memory z) {
        z = new bytes16[](1);
        z[0] = QZERO;
    }

    /**
     * @notice Computes the degree of a polynomial.
     *         Returns the highest index with a nonzero coefficient. Zero polynomial returns 0.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @return Degree of the polynomial.
     */
    function degree(bytes16[] memory coeffs) internal pure returns (uint256) {
        if (coeffs.length == 0) return 0;
        for (uint256 i = coeffs.length; i > 0; i--) {
            if (!MathLib.isZero(coeffs[i - 1])) return i - 1;
        }
        return 0;
    }

    /**
     * @notice Removes trailing zeros from the coefficient array.
     *         Produces a canonical representation with the minimal degree equivalent polynomial.
     * @param  coeffs Polynomial coefficients in ascending powers.
     * @return out    Trimmed coefficient array.
     */
    function trimTrailingZeros(bytes16[] memory coeffs) internal pure returns (bytes16[] memory out) {
        if (coeffs.length == 0) return _zeroPoly();
        uint256 deg = degree(coeffs);
        out = new bytes16[](deg + 1);
        for (uint256 i = 0; i <= deg; i++) out[i] = coeffs[i];
    }
}