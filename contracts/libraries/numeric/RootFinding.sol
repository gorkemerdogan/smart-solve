// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";
import { QuadConstants } from "./QuadConstants.sol";

/**
 * @title RootFinding
 * @notice High-precision numerical root-finding algorithms (Bisection, Newton, Secant)
 *         implemented with IEEE-754 binary128 (bytes16) via ABDKMathQuad.
 * @dev Stateless, pure/parametric library. Facets (or other contracts) should inject
 *      tolerance (tol) and maxIter from configuration (e.g., LibNumericConfig).
 */
library RootFinding {
    using ABDKMathQuad for bytes16;

    // ---- constants (quad literals) ----
    bytes16 private constant QZERO = 0x00000000000000000000000000000000; // 0.0

    // ================================================================
    // Result & internal working-state structs
    // ================================================================

    /**
     * @notice Result container for any root-finding algorithm.
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

    /// @notice Transient state for Newton–Raphson.
    struct NewtonState {
        bytes16 x;     // current iterate
        bytes16 fx;    // f(x)
        bytes16 atTol; // active tolerance (max(requested, MIN))
    }

    /// @notice Working state for Bisection.
    struct BisectionState {
        bytes16 left;
        bytes16 right;
        bytes16 fa;
        bytes16 fb;
        bytes16 mid;
        bytes16 fm;
        bytes16 atTol;
    }

    /// @notice Transient state for Secant.
    struct SecantState {
        bytes16 xPrev;
        bytes16 x;
        bytes16 fPrev;
        bytes16 fx;
        bytes16 atTol;
    }

    // ================================================================
    // Internal helpers
    // ================================================================

    /// @dev f(bytes16) -> bytes16 by staticcall into `target` with selector `sel`.
    function _eval(address target, bytes4 sel, bytes16 x) private view returns (bytes16 y) {
        (bool ok, bytes memory data) = target.staticcall(abi.encodeWithSelector(sel, x));
        require(ok && data.length >= 32, "RootFinding: eval failed");
        assembly {
            y := mload(add(data, 32))
        }
    }

    /// @dev a < b  (quad compare)
    function _lt(bytes16 a, bytes16 b) private pure returns (bool) {
        return ABDKMathQuad.cmp(a, b) < 0;
    }

    /// @dev a <= b (quad compare)
    function _lte(bytes16 a, bytes16 b) private pure returns (bool) {
        return ABDKMathQuad.cmp(a, b) <= 0;
    }

    /// @dev |a|
    function _abs(bytes16 a) private pure returns (bytes16) {
        return ABDKMathQuad.abs(a);
    }

    /// @dev max(a, b)
    function _max(bytes16 a, bytes16 b) private pure returns (bytes16) {
        return _lt(a, b) ? b : a;
    }

    /// @dev true if a == 0 (handles -0 too via cmp)
    function _isZero(bytes16 a) private pure returns (bool) {
        return ABDKMathQuad.cmp(a, QZERO) == 0;
    }

    /// @dev clamp tolerance from storage
    function _clampTol(bytes16 tol) private view returns (bytes16) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        bytes16 minTol = cfg.minTol;

        // Fefault fallback if not set on storage
        if (minTol == bytes16(0)) {
            minTol = minTol = QuadConstants.EPS_1e15();
        }

        return ABDKMathQuad.cmp(tol, minTol) < 0 ? minTol : tol;
    }

    // ================================================================
    // Bisection (internal)
    // ================================================================

    /**
     * @notice Root by Bisection over [a,b] with f(a)*f(b)<0.
     * @param target   Contract exposing f(bytes16)->bytes16
     * @param fSelector Selector of f
     * @param a        Left endpoint
     * @param b        Right endpoint
     * @param tol      Requested tolerance (will be clamped to >= 1e-15)
     * @param maxIter  Iteration cap
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
        s.left  = _lt(a, b) ? a : b;
        s.right = _lt(a, b) ? b : a;

        s.fa = _eval(target, fSelector, s.left);
        s.fb = _eval(target, fSelector, s.right);

        if (_isZero(s.fa)) return RootResult(s.left, 0, true, s.fa);
        if (_isZero(s.fb)) return RootResult(s.right, 0, true, s.fb);

        // require sign change
        require(_lt(s.fa.mul(s.fb), QZERO), "No sign change");

        uint256 k = 0;
        while (k < maxIter) {
            s.mid = s.left.add(s.right).div(ABDKMathQuad.fromInt(2));
            s.fm  = _eval(target, fSelector, s.mid);

            if (_lte(_abs(s.fm), s.atTol) || _lte(s.right.sub(s.left).div(ABDKMathQuad.fromInt(2)), s.atTol)) {
                return RootResult(s.mid, k + 1, true, s.fm);
            }

            if (_lt(s.fa.mul(s.fm), QZERO)) {
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

    // ================================================================
    // Newton–Raphson (internal)
    // ================================================================

    /**
     * @notice Root by Newton–Raphson with analytic derivative.
     * @param target     Contract exposing f(bytes16)->bytes16
     * @param fSelector  Selector of f
     * @param dfTarget   Contract exposing f'(bytes16)->bytes16
     * @param dfSelector Selector of f'
     * @param x0         Initial guess
     * @param tol        Requested tolerance (clamped to >= 1e-15)
     * @param maxIter    Iteration cap
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

        if (_lte(_abs(s.fx), s.atTol)) return RootResult(s.x, 0, true, s.fx);

        uint256 k = 0;
        while (k < maxIter) {
            bytes16 dfx = _eval(dfTarget, dfSelector, s.x);
            require(!_isZero(dfx), "Zero derivative");
            bytes16 xNext = s.x.sub(s.fx.div(dfx));
            bytes16 fxNext = _eval(target, fSelector, xNext);

            if (_lte(_abs(fxNext), s.atTol) || _lte(_abs(xNext.sub(s.x)), s.atTol)) {
                return RootResult(xNext, k + 1, true, fxNext);
            }
            s.x = xNext;
            s.fx = fxNext;
            unchecked { ++k; }
        }
        return RootResult(s.x, k, false, s.fx);
    }

    // ================================================================
    // Secant (internal)
    // ================================================================

    /**
     * @notice Root by Secant (derivative-free) using two initial points.
     * @param target   Contract exposing f(bytes16)->bytes16
     * @param fSelector Selector of f
     * @param x0       First start
     * @param x1       Second start
     * @param tol      Requested tolerance (clamped to >= 1e-15)
     * @param maxIter  Iteration cap
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

        if (_lte(_abs(s.fx), s.atTol)) return RootResult(s.x, 0, true, s.fx);

        uint256 k = 0;
        while (k < maxIter) {
            bytes16 denom = s.fx.sub(s.fPrev);
            require(!_isZero(denom), "Zero slope");
            bytes16 xNext = s.x.sub( s.fx.mul( s.x.sub(s.xPrev) ).div(denom) );
            bytes16 fxNext = _eval(target, fSelector, xNext);

            if (_lte(_abs(fxNext), s.atTol) || _lte(_abs(xNext.sub(s.x)), s.atTol)) {
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