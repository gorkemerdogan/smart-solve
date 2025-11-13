// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad as Q } from "abdk-libraries-solidity/ABDKMathQuad.sol";
import { Trigonometric } from "../libraries/Trigonometric.sol";
import { QuadConstants as QC } from "../libraries/QuadConstants.sol";
import { Integration } from "../libraries/numeric/Integration.sol";

/**
 * @title IntegrationHarness
 * @notice Thin wrapper around the Integration library to make it testable
 *         from Hardhat tests. Also exposes some helper utilities and
 *         simple integrand functions.
 */
contract IntegrationHarness {
    using Q for bytes16;

    // ---------- Helpers: constants / conversions ----------

    function ZERO() external pure returns (bytes16) {
        return Q.fromInt(0);
    }

    function ONE() external pure returns (bytes16) {
        return Q.fromInt(1);
    }

    function TWO() external pure returns (bytes16) {
        return Q.fromInt(2);
    }

    function qFromInt(int256 x) external pure returns (bytes16) {
        return Q.fromInt(x);
    }

    function qFromUInt(uint256 x) external pure returns (bytes16) {
        return Q.fromUInt(x);
    }

    function qFromFrac(int256 num, int256 den) external pure returns (bytes16) {
        bytes16 n = Q.fromInt(num);
        bytes16 d = Q.fromInt(den);
        return n.div(d);
    }

    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return Q.cmp(a, b);
    }

    /// @notice |a - b| in quad
    function absDiff(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return a.sub(b).abs();
    }

    // ---------- Integrand functions f(x) ----------

    /// f(x) = 0
    function f_zero(bytes16 /*x*/) external pure returns (bytes16) {
        return Q.fromInt(0);
    }

    /// f(x) = 1
    function f_one(bytes16 /*x*/) external pure returns (bytes16) {
        return Q.fromInt(1);
    }

    /// f(x) = x
    function f_x(bytes16 x) external pure returns (bytes16) {
        return x;
    }

    /// f(x) = x^2
    function f_x2(bytes16 x) external pure returns (bytes16) {
        return x.mul(x);
    }

    /// f(x) = x^3
    function f_x3(bytes16 x) external pure returns (bytes16) {
        return x.mul(x).mul(x);
    }

    /// Always reverts – to test that a==b short-circuits before calling f
    function f_revert(bytes16 /*x*/) external pure returns (bytes16) {
        revert("f_revert called");
    }

        /// f(x) = 5
    function f_const5(bytes16 /*x*/) external pure returns (bytes16) {
        return Q.fromInt(5);
    }

    /// f(x) = sin(x)
    function f_sin(bytes16 x) external pure returns (bytes16) {
        return Trigonometric.sin(x);
    }

    /// f(x) = 1/x
    function f_inv(bytes16 x) external pure returns (bytes16) {
        return Q.fromInt(1).div(x);
    }

    /// f(x) = 1e-30 * x
    function f_tiny(bytes16 x) external pure returns (bytes16) {
        bytes16 scale = Q.fromUInt(1).div(Q.fromUInt(10**30));
        return x.mul(scale);
    }

    /// f(x) = 1e20 * x
    function f_large(bytes16 x) external pure returns (bytes16) {
        bytes16 scale = Q.fromUInt(10**20);
        return x.mul(scale);
    }

    /// f(x) = piecewise:
    /// [0,1]: f=1
    /// [1,2]: f=3
    function f_piecewise(bytes16 x) external pure returns (bytes16) {
        bytes16 one = Q.fromInt(1);
        bytes16 two = Q.fromInt(2);

        if (Q.cmp(x, one) <= 0) {
            return Q.fromInt(1);
        } else if (Q.cmp(x, two) <= 0) {
            return Q.fromInt(3);
        } else {
            revert("piecewise domain");
        }
    }

    /// π constant
    function PI() external pure returns (bytes16) {
        return QC.PI;
    }

    // ---------- Wrappers around Integration library ----------

    function trap(
        address target,
        bytes4 sel,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16) {
        return Integration.trapezoidal(target, sel, a, b, n);
    }

    function simpson13(
        address target,
        bytes4 sel,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16) {
        return Integration.simpson13(target, sel, a, b, n);
    }

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