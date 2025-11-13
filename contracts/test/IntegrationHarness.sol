// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad as Q } from "abdk-libraries-solidity/ABDKMathQuad.sol";
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
        if (Q.cmp(a, b) >= 0) {
            return a.sub(b);
        } else {
            return b.sub(a);
        }
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