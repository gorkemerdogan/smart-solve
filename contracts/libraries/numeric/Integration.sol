// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";

/**
 * @title Integration Library
 * @notice High-precision numerical integration on uniform grids using
 *         trapezoidal rule, Simpson's 1/3 rule, and Simpson's 3/8 rule.
 *
 *         The integrand f is provided as an external view function
 *         f(bytes16) -> bytes16, addressed by (target, selector).
 */
library Integration {
    using MathLib for bytes16;

    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Evaluates the integrand at a given point by staticcalling the target contract.
     *         Performs a staticcall to 'target' using selector 'sel' with argument 'x'.
     *         Reverts if the call fails or returns insufficient data.
     *
     * @param  target Address of the contract exposing f(bytes16) -> bytes16
     * @param  sel Function selector for the integrand
     * @param  x Input point where the integrand is evaluated
     * @return y Result of f(x) as bytes16
     */
    function _eval(address target, bytes4 sel, bytes16 x) private view returns (bytes16 y) {
        (bool ok, bytes memory data) = target.staticcall(abi.encodeWithSelector(sel, x));
        require(ok && data.length >= 32, "Integration: eval failed");
        assembly {
            y := mload(add(data, 32))
        }
    }

    /**
     * @notice Validates shared integration constraints for all numerical schemes.
     *         Ensures:
     *          - target address is nonzero
     *          - n > 0
     *          - b >= a (based on MathLib.cmp)
     *          Used by all integration routines prior to computation.
     *
     * @param target Address of the contract exposing the integrand
     * @param a Lower bound
     * @param b Upper bound
     * @param n Number of subintervals
     */
    function _validateCommon(address target, bytes16 a, bytes16 b, uint256 n) private pure {
        require(target != address(0), "Integration: target is zero");
        require(n > 0, "Integration: n must be > 0");
        require(MathLib.cmp(b, a) >= 0, "Integration: upper bound b must be >= a");
    }

    // ------------------------------------------------------------
    // Trapezoidal rule
    // ------------------------------------------------------------

    /**
     * @notice Computes the composite trapezoidal rule on a uniform grid.
     *         Requires n > 0 and b >= a. Performs O(n) staticcalls to the integrand.
     *         All computation is done in memory; no storage access occurs.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @param  n Number of subintervals (must be > 0)
     * @return I Approximate integral value encoded as bytes16
     */
    function trapezoidal(address target, bytes4 fSelector, bytes16 a, bytes16 b, uint256 n) internal view returns (bytes16 I) {
        _validateCommon(target, a, b, n);

        // h = (b - a) / n
        bytes16 h = b.sub(a).div(MathLib.fromUInt(n));
        if (MathLib.cmp(h, QZERO) == 0) {
            return QZERO; // a == b
        }

        bytes16 fa = _eval(target, fSelector, a);
        bytes16 fb = _eval(target, fSelector, b);

        // sum = 0.5*(f(a) + f(b)) = (f(a) + f(b))/2
        bytes16 sum = fa.add(fb).div(MathLib.fromInt(2));

        // i = 1..n-1: sum += f(a + i*h)
        for (uint256 i = 1; i < n; i++) {
            bytes16 xi = a.add(h.mul(MathLib.fromUInt(i)));
            sum = sum.add(_eval(target, fSelector, xi));
        }

        // I = h * sum
        return h.mul(sum);
    }

    // ------------------------------------------------------------
    // Simpson's 1/3 rule
    // ------------------------------------------------------------

    /**
     * @notice Computes the composite Simpson’s 1/3 rule on a uniform grid.
     *         Requires n > 0, n even, and b >= a. Performs O(n) staticcalls to the integrand.
     *         All computation occurs in memory; no storage is modified.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @param  n Number of subintervals (must be even and > 0)
     * @return I Approximate integral value encoded as bytes16
     */
    function simpson13(address target, bytes4 fSelector, bytes16 a, bytes16 b, uint256 n) internal view returns (bytes16 I) {
        _validateCommon(target, a, b, n);
        require(n % 2 == 0, "Integration: Simpson 1/3 requires even n");

        bytes16 h = b.sub(a).div(MathLib.fromUInt(n));
        if (MathLib.cmp(h, QZERO) == 0) {
            return QZERO; // a == b
        }

        bytes16 sumOdd = QZERO;  // i = 1,3,5,...,n-1
        bytes16 sumEven = QZERO; // i = 2,4,6,...,n-2

        for (uint256 i = 1; i < n; i++) {
            bytes16 xi = a.add(h.mul(MathLib.fromUInt(i)));
            bytes16 fx = _eval(target, fSelector, xi);
            if (i % 2 == 1) {
                sumOdd = sumOdd.add(fx);
            } else {
                sumEven = sumEven.add(fx);
            }
        }

        bytes16 fa = _eval(target, fSelector, a);
        bytes16 fb = _eval(target, fSelector, b);

        // I = (h/3) * (f(a) + 4*sumOdd + 2*sumEven + f(b))
        bytes16 inner = fa
            .add(sumOdd.mul(MathLib.fromInt(4)))
            .add(sumEven.mul(MathLib.fromInt(2)))
            .add(fb);

        return h.div(MathLib.fromInt(3)).mul(inner);
    }

    // ------------------------------------------------------------
    // Simpson's 3/8 rule
    // ------------------------------------------------------------

    /**
     * @notice Computes the composite Simpson’s 3/8 rule on a uniform grid.
     *         Requires n > 0, n divisible by 3, and b >= a. Performs O(n) staticcalls
     *         to the integrand. All arithmetic is performed in memory.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @param  n Number of subintervals (must be > 0 and divisible by 3)
     * @return I Approximate integral value encoded as bytes16
     */
    function simpson38(address target, bytes4 fSelector, bytes16 a, bytes16 b, uint256 n) internal view returns (bytes16 I) {
        _validateCommon(target, a, b, n);
        require(n % 3 == 0, "Integration: Simpson 3/8 requires n % 3 == 0");

        bytes16 h = b.sub(a).div(MathLib.fromUInt(n));
        if (MathLib.cmp(h, QZERO) == 0) {
            return QZERO; // a == b
        }

        bytes16 sum3 = QZERO;     // i multiple of 3: 3,6,...,n-3
        bytes16 sumNot3 = QZERO;  // i not multiple of 3: 1,2,4,5,...,n-1 (excluding multiples of 3)

        for (uint256 i = 1; i < n; i++) {
            bytes16 xi = a.add(h.mul(MathLib.fromUInt(i)));
            bytes16 fx = _eval(target, fSelector, xi);
            if (i % 3 == 0) {
                sum3 = sum3.add(fx);
            } else {
                sumNot3 = sumNot3.add(fx);
            }
        }

        bytes16 fa = _eval(target, fSelector, a);
        bytes16 fb = _eval(target, fSelector, b);

        // I = (3h/8) * (f(a) + 3*sumNot3 + 2*sum3 + f(b))
        bytes16 inner = fa
            .add(sumNot3.mul(MathLib.fromInt(3)))
            .add(sum3.mul(MathLib.fromInt(2)))
            .add(fb);

        bytes16 factor = h
            .mul(MathLib.fromInt(3))
            .div(MathLib.fromInt(8));

        return factor.mul(inner);
    }
}