// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";
import { QuadConstants } from "../QuadConstants.sol";

/**
 * @title RootFinding
 * @notice High-precision numerical root-finding algorithms (Bisection, Newton, Secant)
 *         implemented with IEEE-754 binary128 (bytes16) via ABDKMathQuad (from MathLib).
 */
library RootFinding {
    using MathLib for bytes16;

    bytes16 private constant QZERO = 0x00000000000000000000000000000000;

    // ------------------------------------------------------------
    // Result & internal working-state structs
    // ------------------------------------------------------------

    /**
     * @notice Result container for any root-finding algorithm.
     *         Encapsulates the final approximation, iteration count, convergence flag, and f(root).
     * @param root       Final root approximation (bytes16)
     * @param iterations Number of iterations performed
     * @param converged  True if tolerance satisfied
     * @param fAtRoot    f(root) at termination
     */
    struct RootResult {
        bytes16 root;
        uint256 iterations;
        bool converged;
        bytes16 fAtRoot;
    }

    /**
     * @notice Internal transient state used by the Newton–Raphson method.
     * @param  x Current iterate
     * @param  fx Current value f(x)
     * @param  atTol Effective tolerance applied during iteration
     */
    struct NewtonState {
        bytes16 x;     // current iterate
        bytes16 fx;    // f(x)
        bytes16 atTol; // active tolerance (max(requested, MIN))
    }

    /**
     * @notice Internal state for the bisection algorithm.
     * @param left Current left endpoint
     * @param right Current right endpoint
     * @param fa f(left)
     * @param fb f(right)
     * @param mid Midpoint of current interval
     * @param fm f(mid)
     * @param atTol Effective tolerance applied during iteration
     */
    struct BisectionState {
        bytes16 left;
        bytes16 right;
        bytes16 fa;
        bytes16 fb;
        bytes16 mid;
        bytes16 fm;
        bytes16 atTol;
    }

    /**
     * @notice Internal transient state for the secant method.
     * @param  xPrev Previous iterate
     * @param  x Current iterate
     * @param  fPrev f(xPrev)
     * @param  fx f(x)
     * @param  atTol Effective tolerance applied during iteration
     */
    struct SecantState {
        bytes16 xPrev;
        bytes16 x;
        bytes16 fPrev;
        bytes16 fx;
        bytes16 atTol;
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Evaluates the integrand or function via staticcall.
     *         Calls 'target.selector(x)' and decodes a bytes16 return value.
     *         Reverts on failure or insufficient return data.
     * @param  target Address exposing f(bytes16) -> bytes16
     * @param  sel    Function selector for f
     * @param  x      Input point
     * @return y      Result of f(x) encoded as bytes16
     */
    function _eval(address target, bytes4 sel, bytes16 x) internal view returns (bytes16 y) {
        (bool ok, bytes memory data) = target.staticcall(abi.encodeWithSelector(sel, x));
        require(ok && data.length >= 32, "RootFinding: eval failed");
        assembly {
            y := mload(add(data, 32))
        }
    }

    /**
     * @notice Produces the effective tolerance used by a root-finding routine.
     * @dev Enforces the following resolution order:
     *      - Non-zero requestedTol is used but clamped to ≥ minTol.
     *      - Otherwise uses configured tolerance from storage.
     *      - If unset, uses the library default tolerance.
     *      - Result is always ≥ configured or default minimum tolerance.
     * @param  requestedTol Tolerance supplied by the caller.
     * @return tol Validated and clamped tolerance, guaranteed > 0.
     */
    function _clampTol(bytes16 requestedTol) internal view returns (bytes16 tol) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();

        // Load minimum tolerance from storage or fallback default
        bytes16 minTol = cfg.minTol;
        if (minTol == bytes16(0)) {
            minTol = QuadConstants.EPS_1e36();
        }

        // Case 1: Caller provided a specific, non-zero requestedTol
        if (!MathLib.isZero(requestedTol)) {
            // Clamp upward to ensure tolerance >= minTol
            if (MathLib.cmp(requestedTol, minTol) < 0) {
                return minTol;
            }
            return requestedTol;
        }

        // Case 2: No caller-supplied tolerance → load system config
        tol = cfg.tol;

        // If system-configured tolerance is unset → fallback default
        if (MathLib.isZero(tol)) {
            tol = QuadConstants.DEFAULT_TOL();
        }

        // Enforce minimum tolerance
        if (MathLib.cmp(tol, minTol) < 0) {
            tol = minTol;
        }
        return tol;
    }

    // ------------------------------------------------------------
    // Bisection
    // ------------------------------------------------------------

    /**
     * @notice Computes a root using the bisection method on [a, b].
     *         Requires f(a) and f(b) to have opposite signs. Endpoints are normalized
     *         to ensure left ≤ right. Convergence is triggered when either:
     *           - |f(mid)| ≤ tol, or
     *           - interval width / 2 ≤ tol.
     * @param  target     Contract exposing f(bytes16) -> bytes16
     * @param  fSelector  Selector for f(bytes16)
     * @param  a          First endpoint of the interval
     * @param  b          Second endpoint of the interval
     * @param  tol        Requested tolerance (clamped to configured minimum)
     * @param  maxIter    Maximum number of iterations
     * @return RootResult Struct containing root approximation and metadata
     */
    function bisection(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b,
        bytes16 tol,
        uint256 maxIter
    ) internal view returns (RootResult memory) {
        BisectionState memory s;

        s.atTol = _clampTol(tol);
        // normalize [left, right]
        bool isALess = MathLib.cmp(a, b) < 0;
        s.left  = isALess ? a : b;
        s.right = isALess ? b : a;

        s.fa = _eval(target, fSelector, s.left);
        s.fb = _eval(target, fSelector, s.right);

        if (MathLib.cmp(s.fa, MathLib.fromInt(0)) == 0) {
            return RootResult(s.left, 0, true, s.fa);
        }

        if (MathLib.cmp(s.fb, MathLib.fromInt(0)) == 0) {
            return RootResult(s.right, 0, true, s.fb);
        }

        // require sign change
        require(MathLib.cmp(s.fa.mul(s.fb), QZERO) < 0, "No sign change");

        uint256 k = 0;
        while (k < maxIter) {
            s.mid = s.left.add(s.right).div(MathLib.fromInt(2));
            s.fm  = _eval(target, fSelector, s.mid);

            if (MathLib.cmp(MathLib.abs(s.fm), s.atTol) <= 0 ||
                MathLib.cmp(
                    s.right.sub(s.left).div(MathLib.fromInt(2)),
                    s.atTol
                ) <= 0
            ) {
                return RootResult(s.mid, k + 1, true, s.fm);
            }

            if (MathLib.cmp(s.fa.mul(s.fm), QZERO) < 0) {
                s.right = s.mid;
                s.fb = s.fm;
            } else {
                s.left = s.mid;
                s.fa = s.fm;
            }
            unchecked { ++k; }
        }
        return RootResult(s.mid, k, false, s.fm);
    }

    // ------------------------------------------------------------
    // Newton–Raphson
    // ------------------------------------------------------------

    /**
     * @notice Computes a root using the Newton–Raphson method with analytic derivative.
     *         Requires non-zero derivative at each iterate. Convergence is based on:
     *           - |f(xNext)| ≤ tol, or
     *           - |xNext − x| ≤ tol.
     * @param  target     Contract exposing f(bytes16) -> bytes16
     * @param  fSelector  Selector for f(bytes16)
     * @param  dfTarget   Contract exposing f'(bytes16)
     * @param  dfSelector Selector for f'(bytes16)
     * @param  x0         Initial guess
     * @param  tol        Requested tolerance (clamped to configured minimum)
     * @param  maxIter    Maximum number of iterations
     * @return RootResult Struct with the final iterate, iteration count, and convergence flag
     */
    function newton(
        address target,
        bytes4 fSelector,
        address dfTarget,
        bytes4 dfSelector,
        bytes16 x0,
        bytes16 tol,
        uint256 maxIter
    ) internal view returns (RootResult memory) {
        NewtonState memory s;
        s.atTol = _clampTol(tol);
        s.x = x0;
        s.fx = _eval(target, fSelector, s.x);

        if (MathLib.cmp(MathLib.abs(s.fx), s.atTol) <= 0) {
            return RootResult(s.x, 0, true, s.fx);
        }

        uint256 k = 0;
        while (k < maxIter) {
            bytes16 dfx = _eval(dfTarget, dfSelector, s.x);
            require(MathLib.cmp(dfx, MathLib.fromInt(0)) != 0, "Zero derivative");

            bytes16 xNext = s.x.sub(s.fx.div(dfx));
            bytes16 fxNext = _eval(target, fSelector, xNext);

            bytes16 errF = MathLib.abs(fxNext);          // |f(xNext)|
            bytes16 errX = MathLib.abs(xNext.sub(s.x));  // |xNext – x|

            if (MathLib.cmp(errF, s.atTol) <= 0 || MathLib.cmp(errX, s.atTol) <= 0) {
                return RootResult(xNext, k + 1, true, fxNext);
            }

            s.x = xNext;
            s.fx = fxNext;
            unchecked { ++k; }
        }
        return RootResult(s.x, k, false, s.fx);
    }

    // ------------------------------------------------------------
    // Secant
    // ------------------------------------------------------------

    /**
     * @notice Computes a root using the secant method (derivative-free).
     *         Uses two initial values and updates via the secant update formula.
     *         Reverts if consecutive function values yield zero slope. Converges when:
     *           - |f(xNext)| ≤ tol, or
     *           - |xNext − x| ≤ tol.
     * @param  target     Contract exposing f(bytes16) -> bytes16
     * @param  fSelector  Selector for f(bytes16)
     * @param  x0         First initial point
     * @param  x1         Second initial point
     * @param  tol        Requested tolerance (clamped to configured minimum)
     * @param  maxIter    Maximum number of iterations
     * @return RootResult Struct containing the resulting approximation
     */
    function secant(
        address target,
        bytes4 fSelector,
        bytes16 x0,
        bytes16 x1,
        bytes16 tol,
        uint256 maxIter
    ) internal view returns (RootResult memory) {
        SecantState memory s;
        s.atTol  = _clampTol(tol);
        s.xPrev  = x0;
        s.x      = x1;
        s.fPrev  = _eval(target, fSelector, s.xPrev);
        s.fx     = _eval(target, fSelector, s.x);

        if (MathLib.cmp(MathLib.abs(s.fx), s.atTol) <= 0) {
            return RootResult(s.x, 0, true, s.fx);
        }

        uint256 k = 0;
        while (k < maxIter) {
            bytes16 denom = s.fx.sub(s.fPrev);
            require(MathLib.cmp(denom, MathLib.fromInt(0)) != 0, "Zero slope");

            bytes16 xNext = s.x.sub( s.fx.mul( s.x.sub(s.xPrev) ).div(denom) );
            bytes16 fxNext = _eval(target, fSelector, xNext);

            if (MathLib.cmp(MathLib.abs(fxNext), s.atTol) <= 0 ||
                MathLib.cmp(MathLib.abs(xNext.sub(s.x)), s.atTol) <= 0) {
                return RootResult(xNext, k + 1, true, fxNext);
            }

            s.xPrev = s.x;
            s.fPrev = s.fx;
            s.x = xNext;
            s.fx = fxNext;
            unchecked { ++k; }
        }
        return RootResult(s.x, k, false, s.fx);
    }
}