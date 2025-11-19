// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { TrigonometrySinCos as TSC } from "./TrigonometrySinCos.sol";

library TrigonometryTanCot {

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;
    bytes16 internal constant QNAN  = 0x7fff8000000000000000000000000000;
    bytes16 internal constant TINY  = 0x00010000000000000000000000000000; // 2^-112

    // ===============================================================
    // === tan(x)
    // ===============================================================
    /**
     * @dev High-precision tan(x).
     *
     * tan(x) = sin(x) / cos(x)
     *
     * Domain:
     *   undefined when cos(x) = 0 → return NaN
     *
     * Precision: ~1e-34 (quad)
     */
    function tan(bytes16 x) internal pure returns (bytes16) {
        bytes16 s = TSC.sin(x);
        bytes16 c = TSC.cos(x);

        if (MathLib.isNaN(s) || MathLib.isNaN(c)) return QNAN;

        // tan undefined when cos ≈ 0
        if (MathLib.cmp(MathLib.abs(c), TINY) < 0)
            return QNAN;

        return MathLib.div(s, c);
    }

    // ===============================================================
    // === cot(x)
    // ===============================================================
    /**
     * @dev High-precision cot(x).
     *
     * cot(x) = cos(x) / sin(x)
     *
     * Domain:
     *   undefined where sin(x) = 0 → return NaN
     *
     * Precision: ~1e-34 (quad)
     */
    function cot(bytes16 x) internal pure returns (bytes16) {
        bytes16 s = TSC.sin(x);
        bytes16 c = TSC.cos(x);

        if (MathLib.isNaN(s) || MathLib.isNaN(c)) return QNAN;

        // cot undefined when sin ≈ 0
        if (MathLib.cmp(MathLib.abs(s), TINY) < 0)
            return QNAN;

        return MathLib.div(c, s);
    }
}