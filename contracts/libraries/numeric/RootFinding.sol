// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title RootFinding
 * @notice High-precision numerical root-finding algorithms implemented in quadruple precision (bytes16).
 * @dev This library provides pure functional implementations of:
 *      - Bisection method (guaranteed convergence for sign-changing intervals)
 *      - Newton–Raphson method (requires derivative)
 *      - Secant method (derivative-free, uses two initial points)
 *
 * All arithmetic is performed using ABDKMathQuad to achieve IEEE-754 binary128 precision.
 * The library is stateless and suitable for pure/view use in smart contracts or Diamond facets.
 */
library RootFinding {
    using ABDKMathQuad for bytes16;

    bytes16 private constant ZERO     = 0x00000000000000000000000000000000; // 0.0
    bytes16 private constant ONE = 0x3FFF0000000000000000000000000000; // 1.0
    bytes16 private constant TWO      = 0x40000000000000000000000000000000; // 2.0
    bytes16 private constant MIN_TOL  = 0x3DAB7349B82000000000000000000000; // ≈1e-15

    // ------------------------------------------------------------------------
    // INTERNAL HELPER
    // ------------------------------------------------------------------------

    /**
     * @notice Evaluates an external function f(x) implemented in another contract.
     * @dev The external target must expose `f(bytes16) → bytes16`.
     *
     * @param target   Address of the target contract exposing f(x)
     * @param selector Function selector for f(bytes16)
     * @param x        Evaluation point in ABDK quad precision
     * @return fx      Evaluated function value f(x)
     *
     * Reverts if the call fails or returns a payload not exactly 16 bytes.
     */
    function _eval(address target, bytes4 selector, bytes16 x)
        private
        view
        returns (bytes16 fx)
    {
        (bool ok, bytes memory ret) = target.staticcall(abi.encodeWithSelector(selector, x));
        require(ok && ret.length == 16, "f(x) call failed");
        fx = abi.decode(ret, (bytes16));
    }

    /**
     * @notice Returns true if `a` is strictly less than `b`.
     * @dev Uses ABDKMathQuad.cmp(a, b) < 0 for quadruple-precision comparison.
     * @param a The first operand (bytes16, IEEE-754 quadruple precision)
     * @param b The second operand (bytes16, IEEE-754 quadruple precision)
     * @return True if a < b, otherwise false
     */
    function _lt(bytes16 a, bytes16 b) private pure returns (bool) { 
        return ABDKMathQuad.cmp(a, b) < 0;
    }


    /**
     * @notice Returns true if `a` is less than or equal to `b`.
     * @dev Uses ABDKMathQuad.cmp(a, b) ≤ 0 for quadruple-precision comparison.
     * @param a The first operand (bytes16, IEEE-754 quadruple precision)
     * @param b The second operand (bytes16, IEEE-754 quadruple precision)
     * @return True if a <= b, otherwise false
     */
    function _lte(bytes16 a, bytes16 b) private pure returns (bool) {
        return ABDKMathQuad.cmp(a, b) <= 0;
    }

    /**
     * @notice Returns the absolute value of `a`.
     * @dev Uses ABDKMathQuad.abs(a) for quadruple-precision absolute value.
     * @param a The operand (bytes16, IEEE-754 quadruple precision)
     * @return Absolute value of `a` as bytes16
     */
    function _abs(bytes16 a) private pure returns (bytes16) {
        return ABDKMathQuad.abs(a);
    }

    // ------------------------------------------------------------------------
    // BISECTION METHOD
    // ------------------------------------------------------------------------

    /**
     * @notice Computes a root of f(x) within [a, b] using the **Bisection method**.
     * @dev Guaranteed convergence if f(a) and f(b) have opposite signs and f is continuous.
     *
     * Iteratively halves the interval until either:
     *  - |f(mid)| ≤ tol, or  
     *  - (b − a)/2 ≤ tol.
     *
     * @param target   Address of contract exposing f(bytes16) → bytes16.
     * @param fSelector Function selector for f.
     * @param a        Left endpoint of the interval.
     * @param b        Right endpoint of the interval.
     * @param tol      Tolerance for convergence (minimum ≈1e-15 enforced).
     * @param maxIter  Maximum iteration limit.
     * @return root      Final root approximation midpoint.
     * @return iterations Number of iterations executed.
     * @return converged  True if convergence criteria satisfied.
     * @return fAtRoot    Value f(root) at termination.
     */
    function bisection(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b,
        bytes16 tol,
        uint256 maxIter
    )
        internal
        view
        returns (bytes16 root, uint256 iterations, bool converged, bytes16 fAtRoot)
    {
        bytes16 atTol = tol.cmp(MIN_TOL) < 0 ? MIN_TOL : tol;
        bytes16 fa = _eval(target, fSelector, a);
        bytes16 fb = _eval(target, fSelector, b);
        require(_lt(fa.mul(fb), ZERO), "No sign change");

        bytes16 left = a;
        bytes16 right = b;
        bytes16 mid;
        bytes16 fm;
        uint256 k = 0;

        while (k < maxIter) {
            mid = (left.add(right)).div(TWO);
            fm = _eval(target, fSelector, mid);

            // stopping criteria: |f(m)| <= tol  OR  (right - left)/2 <= tol
            if (_lte(_abs(fm), atTol) || _lte((right.sub(left)).div(TWO), atTol)) {
                return (mid, k + 1, true, fm);
            }

            // choose the subinterval with sign change: f(a)*f(m) < 0
            if (_lt(fa.mul(fm), ZERO)) {
                right = mid;
                fb = fm;
            } else {
                left = mid;
                fa = fm;
            }
            unchecked { ++k; }
        }
        return (mid, k, false, fm);
    }

    // ------------------------------------------------------------------------
    // NEWTON–RAPHSON METHOD
    // ------------------------------------------------------------------------

    /**
     * @notice Approximates a root of f(x) using the **Newton–Raphson iterative method**.
     * @dev Requires both f(x) and its derivative f′(x).
     *      Converges quadratically near a simple root with non-zero derivative.
     *
     * Iteration formula:
     *      xₖ₊₁ = xₖ − f(xₖ) / f′(xₖ)
     *
     * Terminates when |f(xₖ₊₁)| ≤ tol or |xₖ₊₁ − xₖ| ≤ tol.
     *
     * @param target     Address exposing f(bytes16) → bytes16.
     * @param fSelector  Function selector for f(x).
     * @param dfTarget   Address exposing f′(bytes16) → bytes16.
     * @param dfSelector Function selector for f′(x).
     * @param x0         Initial guess (bytes16).
     * @param tol        Tolerance for convergence (minimum ≈ 1e-15 enforced).
     * @param maxIter    Maximum iteration limit.
     * @return root       Final root approximation.
     * @return iterations Iteration count.
     * @return converged  True if tolerance satisfied.
     * @return fAtRoot    f(root) value at termination.
     *
     * @custom:reverts Zero derivative encountered (df = 0).
     */
    function newton(
        address target,
        bytes4 fSelector,
        address dfTarget,
        bytes4 dfSelector,
        bytes16 x0,
        bytes16 tol,
        uint256 maxIter
    )
        internal
        view
        returns (bytes16 root, uint256 iterations, bool converged, bytes16 fAtRoot)
    {
        bytes16 atTol = tol.cmp(MIN_TOL) < 0 ? MIN_TOL : tol;
        bytes16 x = x0;
        bytes16 fx = _eval(target, fSelector, x);
        if (_lte(_abs(fx), atTol)) {
            return (x, 0, true, fx);
        }

        uint256 k = 0;
        while (k < maxIter) {
            bytes16 dfx = _eval(dfTarget, dfSelector, x);
            require(!dfx.eq(ZERO), "Zero derivative");

            bytes16 xNext = x.sub(fx.div(dfx));
            bytes16 fxNext = _eval(target, fSelector, xNext);

            if (
                _lte(_abs(fxNext), atTol) ||
                _lte(_abs(xNext.sub(x)), atTol)
            ) {
                return (xNext, k + 1, true, fxNext);
            }

            x = xNext;
            fx = fxNext;
            unchecked { ++k; }
        }
        return (x, k, false, fx);
    }

    // ------------------------------------------------------------------------
    // SECANT METHOD
    // ------------------------------------------------------------------------

    /**
     * @notice Approximates a root of f(x) using the **Secant method** (derivative-free).
     * @dev Uses two initial guesses (x₀, x₁) and approximates the derivative by a secant line.
     *      Converges super-linearly under smoothness assumptions.
     *
     * Update rule:
     *      xₖ₊₁ = xₖ − f(xₖ) · (xₖ − xₖ₋₁) / (f(xₖ) − f(xₖ₋₁))
     *
     * @param target   Address exposing f(bytes16) → bytes16.
     * @param fSelector Function selector for f.
     * @param x0       First initial point (bytes16).
     * @param x1       Second initial point (bytes16).
     * @param tol      Tolerance for convergence (minimum ≈ 1e-15 enforced).
     * @param maxIter  Maximum iteration limit.
     * @return root       Final root approximation.
     * @return iterations Iteration count.
     * @return converged  True if tolerance satisfied.
     * @return fAtRoot    f(root) value at termination.
     *
     * @custom:reverts Zero slope encountered (f(xₖ) − f(xₖ₋₁) = 0).
     */
    function secant(
        address target,
        bytes4 fSelector,
        bytes16 x0,
        bytes16 x1,
        bytes16 tol,
        uint256 maxIter
    )
        internal
        view
        returns (bytes16 root, uint256 iterations, bool converged, bytes16 fAtRoot)
    {
        bytes16 atTol = tol.cmp(MIN_TOL) < 0 ? MIN_TOL : tol;
        bytes16 xPrev = x0;
        bytes16 x = x1;
        bytes16 fPrev = _eval(target, fSelector, xPrev);
        bytes16 fx = _eval(target, fSelector, x);

        if (_lte(_abs(fx), atTol)) {
            return (x, 0, true, fx);
        }
        uint256 k = 0;
        while (k < maxIter) {
            bytes16 denom = fx.sub(fPrev);
            require(!denom.eq(ZERO), "Zero slope");

            bytes16 xNext = x.sub(fx.mul(x.sub(xPrev)).div(denom));
            bytes16 fxNext = _eval(target, fSelector, xNext);

            if (
                _lte(_abs(fxNext), atTol) ||
                _lte(_abs(xNext.sub(x)), atTol)
            ) {
                return (xNext, k + 1, true, fxNext);
            }

            xPrev = x;
            fPrev = fx;
            x = xNext;
            fx = fxNext;
            unchecked { ++k; }
        }
        return (x, k, false, fx);
    }
}