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

    function neg(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.neg(a);
    }

    function abs(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.abs(a);
    }

    function sqrt(bytes16 a) public pure returns (bytes16) {
        return ABDKMathQuad.sqrt(a);
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