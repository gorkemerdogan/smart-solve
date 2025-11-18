// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import "../libraries/Trigonometry/Trigonometry.sol";
import "../libraries/Trigonometry/TrigonometryHelpers.sol";

/**
 * @title TrigonometryHarness
 * @notice Single test helper for Hardhat. No other helper files required.
 *
 * Provides:
 *  - Quad constants (pi, pi/2, pi/4, 2pi)
 *  - Quad <-> int and Quad <-> scaled-float conversions
 *  - NaN handling
 *  - Utility ops (abs, neg, cmp)
 *  - Direct passthrough to Trigonometry.sol functions
 * 
 * EVERYTHING tests need lives inside this one contract.
 */
contract TrigonometryHarness {
    using MathLib for bytes16;

    /*──────────────────────────────────────────────────────────
        QUAD CONSTANTS
    ──────────────────────────────────────────────────────────*/
    bytes16 public constant QPI        = 0x4000921FB54442D18469898CC51701B8;
    bytes16 public constant QHALF_PI   = 0x3FFF921FB54442D18469898CC51701B8;
    bytes16 public constant QQUARTER_PI= 0x3FFE921FB54442D18469898CC51701B8;
    bytes16 public constant QTWO_PI    = 0x4001921FB54442D18469898CC51701B8;

    bytes16 public constant QZERO      = 0x00000000000000000000000000000000;
    bytes16 public constant QONE       = 0x3fff0000000000000000000000000000;
    bytes16 public constant QNAN       = 0x7fff8000000000000000000000000000;

    /*──────────────────────────────────────────────────────────
        BASIC QUAD <-> INT CONVERSIONS
    ──────────────────────────────────────────────────────────*/

    function fromDouble(int256 x) external pure returns (bytes16) {
        return MathLib.fromInt(x);
    }

    function fromUInt(uint256 x) external pure returns (bytes16) {
        return MathLib.fromUInt(x);
    }

    function toDouble(bytes16 x) external pure returns (int256) {
        return MathLib.toInt(x);
    }

    /*──────────────────────────────────────────────────────────
        QUAD <-> FLOAT (scaled)  (JS testing compatible)
    ──────────────────────────────────────────────────────────*/

    uint256 public constant SCALE = 1e12;

    /// @notice Convert JS float x into quad using SCALE
    /// Example: x = 1.2345 → scaled = 1234500000000 → quad = scaled / 1e12
    function fromFloat(int256 scaledValue) external pure returns (bytes16) {
        bytes16 qInt = MathLib.fromInt(scaledValue);
        bytes16 qScale = MathLib.fromUInt(SCALE);
        return MathLib.div(qInt, qScale);
    }

    /// @notice Convert quad → JS float (scaled by SCALE)
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }

    /*──────────────────────────────────────────────────────────
        UTILITY HELPERS
    ──────────────────────────────────────────────────────────*/
    function abs(bytes16 x) external pure returns (bytes16) {
        return MathLib.abs(x);
    }

    function neg(bytes16 x) external pure returns (bytes16) {
        return MathLib.neg(x);
    }

    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return MathLib.cmp(a, b);
    }

    function isNaN(bytes16 x) external pure returns (bool) {
        return MathLib.isNaN(x);
    }

    function add(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return MathLib.add(a, b);
    } 

    function mul(bytes16 a, bytes16 b) external pure returns (bytes16) {
        return MathLib.mul(a, b);
    }

    /*──────────────────────────────────────────────────────────
        TRIGONOMETRY EXPOSED FUNCTIONS
    ──────────────────────────────────────────────────────────*/

    function sin(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.sin(x);
    }

    function cos(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.cos(x);
    }

    function tan(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.tan(x);
    }

    function cot(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.cot(x);
    }

    function asin(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.asin(x);
    }

    function acos(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.acos(x);
    }

    function atan(bytes16 x) external pure returns (bytes16) {
        return Trigonometry.atan(x);
    }
}