// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Trigonometry } from "../libraries/Trigonometry/Trigonometry.sol";
import { QuadConstants as QC } from "../libraries/QuadConstants.sol";
import { MathLib } from "../libraries/MathLib.sol";
import { Integration } from "../libraries/numeric/Integration.sol";

/**
 * @title IntegrationHarness
 * @notice Test harness exposing the Integration library for Hardhat tests.
 *         Provides helper constructors for quad values and simple integrand
 *         functions used during verification of numeric accuracy.
 */
contract IntegrationHarness {
    using MathLib for bytes16;

    // ------------------------------------------------------------
    //  Helpers: constants / conversions
    // ------------------------------------------------------------

    /**
     * @notice Converts an int256 into its quad-precision representation.
     * @param x Signed integer value.
     * @return q Quad-precision number representing x.
     */
    function qFromInt(int256 x) external pure returns (bytes16) {
        return MathLib.fromInt(x);
    }

    /**
     * @notice Converts a uint256 into its quad-precision representation.
     * @param x Unsigned integer value.
     * @return q Quad-precision number representing x.
     */
    function qFromUInt(uint256 x) external pure returns (bytes16) {
        return MathLib.fromUInt(x);
    }

    /**
     * @notice Produces a quad-precision fraction num/den.
     * @param num Signed numerator.
     * @param den Signed denominator.
     * @return q Quad-precision representation of num/den.
     */
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16) {
        bytes16 n = MathLib.fromInt(num);
        bytes16 d = MathLib.fromInt(den);
        return n.div(d);
    }

    /**
     * @notice Performs quad-precision comparison.
     * @param a First quad-precision value.
     * @param b Second quad-precision value.
     * @return result Three-way comparison result:
     *         -1 if a < b, 0 if equal, +1 if a > b.
     */
    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return MathLib.cmp(a, b);
    }

    /**
     * @notice Computes the absolute difference |a − b|.
     * @param a First quad-precision value.
     * @param b Second quad-precision value.
     * @return diff Quad-precision absolute difference.
     */
    function absDiff(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return a.sub(b).abs();
    }


    // ------------------------------------------------------------
    //  Integrand functions f(x)
    // ------------------------------------------------------------

    /**
     * @notice Integrand f(x) = 0.
     * @return v Quad-precision value equal to 0.
     */
    function f_zero(bytes16 /*x*/) external pure returns (bytes16) {
        return MathLib.fromInt(0);
    }

    /**
     * @notice Integrand f(x) = 1.
     * @return v Quad-precision value equal to 1.
     */
    function f_one(bytes16 /*x*/) external pure returns (bytes16) {
        return MathLib.fromInt(1);
    }

    /**
     * @notice Integrand f(x) = x.
     * @param x Input value.
     * @return v Quad-precision output equal to x.
     */
    function f_linear(bytes16 x) external pure returns (bytes16) {
        return x;
    }

    /**
     * @notice Integrand f(x) = x².
     * @param x Input value.
     * @return v Quad-precision square of x.
     */
    function f_square(bytes16 x) external pure returns (bytes16) {
        return x.mul(x);
    }

    /**
     * @notice Integrand f(x) = x³.
     * @param x Input value.
     * @return v Quad-precision cube of x.
     */
    function f_cube(bytes16 x) external pure returns (bytes16) {
        return x.mul(x).mul(x);
    }

    /**
     * @notice Integrand that always reverts.
     * @dev Used to verify that Integration library short-circuits when a == b
     *      and avoids unnecessary function calls.
     */
    function f_revert(bytes16 /*x*/) external pure returns (bytes16) {
        revert("f_revert called");
    }

    /**
     * @notice Integrand f(x) = 5.
     * @return v Quad-precision constant value 5.
     */
    function f_const5(bytes16 /*x*/) external pure returns (bytes16) {
        return MathLib.fromInt(5);
    }

    /**
     * @notice Integrand f(x) = sin(x).
     * @param x Input value.
     * @return v Quad-precision sine of x.
     */
    function f_sin(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.sin(x);
    }

    /**
     * @notice Integrand f(x) = 1/x.
     * @dev Domain excludes x = 0.
     * @param x Input value.
     * @return v Quad-precision value 1/x.
     */
    function f_inv(bytes16 x) external pure returns (bytes16) {
        return MathLib.fromInt(1).div(x);
    }


    /**
     * @notice Integrand f(x) = 1e−30 · x.
     * @param x Input value.
     * @return v Scaled quad-precision result.
     */
    function f_tiny(bytes16 x) external pure returns (bytes16) {
        bytes16 scale = MathLib.fromUInt(1).div(MathLib.fromUInt(10**30));
        return x.mul(scale);
    }

    /**
     * @notice Integrand f(x) = 1e20 · x.
     * @param x Input value.
     * @return v Scaled quad-precision result.
     */
    function f_large(bytes16 x) external pure returns (bytes16) {
        bytes16 scale = MathLib.fromUInt(10**20);
        return x.mul(scale);
    }

    /**
     * @notice Piecewise integrand:
     *         - [0,1] → 1
     *         - (1,2] → 3
     * @dev Reverts outside [0,2].
     * @param x Input value.
     * @return v Quad-precision piecewise-defined value.
     */
    function f_piecewise(bytes16 x) external pure returns (bytes16) {
        bytes16 one = MathLib.fromInt(1);
        bytes16 two = MathLib.fromInt(2);

        if (MathLib.cmp(x, one) <= 0) {
            return MathLib.fromInt(1);
        } else if (MathLib.cmp(x, two) <= 0) {
            return MathLib.fromInt(3);
        } else {
            revert("piecewise domain");
        }
    }

    /**
     * @notice Returns π in quad precision.
     * @return p Quad-precision constant π.
     */
    function PI() external pure returns (bytes16) {
        return QC.PI();
    }

    // ------------------------------------------------------------
    //  Wrappers around Integration library
    // ------------------------------------------------------------

    /**
     * @notice Computes the integral using the composite trapezoidal rule.
     * @param target Contract address providing f(bytes16) → bytes16.
     * @param sel Function selector of f in the target contract.
     * @param a Lower integration bound.
     * @param b Upper integration bound.
     * @param n Number of subintervals (must be > 0).
     * @return I Quad-precision integral approximation.
     */
    function trapezoidal (
        address target,
        bytes4 sel,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16) {
        return Integration.trapezoidal(target, sel, a, b, n);
    }

    /**
     * @notice Computes the integral using composite Simpson's 1/3 rule.
     * @param target Contract address providing f(bytes16) → bytes16.
     * @param sel Function selector of f in the target contract.
     * @param a Lower integration bound.
     * @param b Upper integration bound.
     * @param n Number of subintervals (must be even).
     * @return I Quad-precision integral approximation.
     */
    function simpson13(
        address target,
        bytes4 sel,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16) {
        return Integration.simpson13(target, sel, a, b, n);
    }

    /**
     * @notice Computes the integral using composite Simpson's 3/8 rule.
     * @param target Contract address providing f(bytes16) → bytes16.
     * @param sel Function selector of f in the target contract.
     * @param a Lower integration bound.
     * @param b Upper integration bound.
     * @param n Number of subintervals (must be divisible by 3).
     * @return I Quad-precision integral approximation.
     */
    function simpson38(
        address target,
        bytes4 sel,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16) {
        return Integration.simpson38(target, sel, a, b, n);
    }
}