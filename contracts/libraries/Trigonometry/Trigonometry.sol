// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { TrigonometrySinCos as TSC } from "./TrigonometrySinCos.sol";
import { TrigonometryTanCot as TT } from "./TrigonometryTanCot.sol";
import { TrigonometryArc as TA } from "./TrigonometryArc.sol";

/**
 * @title Trigonometry
 */

library Trigonometry {

    function sin(bytes16 x) internal pure returns (bytes16) {
        return TSC.sin(x);
    }

    function cos(bytes16 x) internal pure returns (bytes16) {
        return TSC.cos(x);
    }

    function tan(bytes16 x) internal pure returns (bytes16) {
        return TT.tan(x);
    }

    function cot(bytes16 x) internal pure returns (bytes16) {
        return TT.cot(x);
    }

    function asin(bytes16 x) internal pure returns (bytes16) {
        return TA.asin(x);
    }

    function acos(bytes16 x) internal pure returns (bytes16) {
        return TA.acos(x);
    }

    function atan(bytes16 x) internal pure returns (bytes16) {
        return TA.atan(x);
    }
}