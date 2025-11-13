// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad as Q } from "abdk-libraries-solidity/ABDKMathQuad.sol";
import { Trigonometric } from "../libraries/Trigonometric.sol";
import { QuadConstants as QC } from "../libraries/QuadConstants.sol";
/**
 * @title TrigonometricHarness
 * @notice Thin test wrapper exposing all Trigonometric library functions
 *         and a set of helper utilities for quad-precision comparisons.
 *
 *         This contract is *stateless*; all computations are pure/view.
 */
contract TrigonometricHarness {
    using Q for bytes16;

    // -------------------------------------------------------------
    // Quad helpers (constants & conversions)
    // -------------------------------------------------------------
    function qFromInt(int256 x) external pure returns (bytes16) { return Q.fromInt(x); }

    function qFromFrac(int256 num, int256 den) external pure returns (bytes16) {
        bytes16 n = Q.fromInt(num);
        bytes16 d = Q.fromInt(den);
        return n.div(d);
    }

    function qAdd(bytes16 a, bytes16 b) external pure returns (bytes16) { return a.add(b); }
    function qSub(bytes16 a, bytes16 b) external pure returns (bytes16) { return a.sub(b); }
    function qMul(bytes16 a, bytes16 b) external pure returns (bytes16) { return a.mul(b); }
    function qDiv(bytes16 a, bytes16 b) external pure returns (bytes16) { return a.div(b); }
    function cmp(bytes16 a, bytes16 b) external pure returns (int256) { return Q.cmp(a, b); }

    /// @notice |a - b| in quad
    function absDiff(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return a.sub(b).abs();
    }

    // -------------------------------------------------------------
    // QuadConstants library forwarding functions
    // -------------------------------------------------------------
    function PI() external pure returns (bytes16) { return QC.PI(); }
    function HALF_PI() external pure returns (bytes16) { return QC.HALF_PI(); }
    function TWO_PI() external pure returns (bytes16) { return QC.TWO_PI(); }
    function DEFAULT_TOL() external pure returns (bytes16) { return QC.DEFAULT_TOL(); }

    // -------------------------------------------------------------
    // Trigonometric library forwarding functions
    // -------------------------------------------------------------
    function sin(bytes16 x) external pure returns (bytes16) { return Trigonometric.sin(x); }
    function cos(bytes16 x) external pure returns (bytes16) { return Trigonometric.cos(x); }
    function tan(bytes16 x) external pure returns (bytes16) { return Trigonometric.tan(x); }
    function cot(bytes16 x) external pure returns (bytes16) { return Trigonometric.cot(x); }
    function asin(bytes16 x) external pure returns (bytes16) { return Trigonometric.asin(x); }
    function acos(bytes16 x) external pure returns (bytes16) { return Trigonometric.acos(x); }
    function atan(bytes16 x) external pure returns (bytes16) { return Trigonometric.atan(x); }
    function acot(bytes16 x) external pure returns (bytes16) { return Trigonometric.acot(x); }
}