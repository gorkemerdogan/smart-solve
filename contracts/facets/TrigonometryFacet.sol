// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../libraries/Trigonometry/Trigonometry.sol";

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