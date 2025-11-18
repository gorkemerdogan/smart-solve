// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad } from "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title MathLib
 * @notice Thin wrapper around ABDKMathQuad with PUBLIC functions.
 *         Purpose: avoid inlining ABDKMathQuad into every trigonometric library.
 *
 *         Using public functions ensures all trig libraries CALL this library
 *         instead of embedding the heavy ABDK code => bytecode drops massively.
 */
library MathLib {

    /*──────────────────────────────────────────────────────────
        BASIC OPS
    ───────────────────────────────────────────────────────────*/

    function add(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.add(a, b);
    }

    function sub(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.sub(a, b);
    }

    function mul(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.mul(a, b);
    }

    function div(bytes16 a, bytes16 b) public pure returns (bytes16) {
        return ABDKMathQuad.div(a, b);
    }

    function sqrt(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.sqrt(a);
    }

    function neg(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.neg(a);
    }

    function abs(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.abs(a);
    }

    /*──────────────────────────────────────────────────────────
        FLOOR
    ───────────────────────────────────────────────────────────*/

    function floorQuad(bytes16 x) public pure returns (bytes16) {
        // If x >= 0: floor = trunc
        if (cmp(x, fromInt(0)) >= 0) {
            int256 ti = toInt(x);          // truncate
            return fromInt(ti);
        }

        // x < 0
        int256 ti2 = toInt(x);             // truncate toward zero
        bytes16 b = fromInt(ti2);

        // If x is already integer, floor(x) == x
        if (cmp(b, x) == 0) {
            return b;
        }

        // else: floor(x) = trunc(x) - 1
        return sub(b, fromInt(1));
    }

    function floorInt(bytes16 x) public pure returns (int256) {
        if (cmp(x, fromInt(0)) >= 0) {
            return toInt(x); // trunc = floor
        }

        int256 ti = toInt(x); // trunc
        bytes16 t = fromInt(ti);

        if (cmp(t, x) == 0) {
            return ti; // already integer
        }

        return ti - 1; // subtract 1 for negative non-integers
    }

    /*──────────────────────────────────────────────────────────
        COMPARE
    ───────────────────────────────────────────────────────────*/

    function cmp(bytes16 a, bytes16 b) public pure returns (int256) {
        return ABDKMathQuad.cmp(a, b);
    }

    function isNaN(bytes16 a) public pure returns (bool) {
        return ABDKMathQuad.isNaN(a);
    }

    /*──────────────────────────────────────────────────────────
        CONVERSIONS
    ───────────────────────────────────────────────────────────*/

    function fromInt(int256 v) public pure returns (bytes16) {
        return ABDKMathQuad.fromInt(v);
    }

    function fromUInt(uint256 v) public pure returns (bytes16) {
        return ABDKMathQuad.fromUInt(v);
    }

    function toInt(bytes16 v) public pure returns (int256) {
        return ABDKMathQuad.toInt(v);
    }
}