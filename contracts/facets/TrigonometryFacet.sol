// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../libraries/Trigonometry/Trigonometry.sol";

/**
 * @title TrigonometryFacet
 * @notice Facet exposing high-precision trigonometric functions that operate on
 *         IEEE-754 quadruple-precision (bytes16) values.
 *         All functions are pure passthroughs: sin, cos, tan, cot, asin, acos, and atan.
 */
contract TrigonometryFacet {
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