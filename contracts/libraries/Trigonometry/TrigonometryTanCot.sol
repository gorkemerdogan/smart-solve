// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { TrigonometrySinCos as TSC } from "./TrigonometrySinCos.sol";

/**
 * @title  TrigonometryTanCot
 * @notice High-precision tangent and cotangent in IEEE-754 binary128 (bytes16).
 *         Uses sine/cosine from TrigonometrySinCos and applies domain checks
 *         for singularities where results are undefined.
 */
library TrigonometryTanCot {

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;
    bytes16 internal constant QNAN  = 0x7fff8000000000000000000000000000;
    bytes16 internal constant TINY  = 0x00010000000000000000000000000000; // 2^-112

    // ------------------------------------------------------------
    // tan(x)
    // ------------------------------------------------------------
    /**
     * @notice Computes tan(x) in binary128 precision.
     * @dev Implementation:
     *        tan(x) = sin(x) / cos(x).
     *
     *      Domain notes:
     *        - Returns QNAN when cos(x) ≈ 0 (undefined).
     *        - Propagates QNAN from sin or cos if present.
     *
     *      Precision is limited primarily by the underlying sin/cos
     *      approximations and is typically ~1e-34.
     *
     * @param x Input angle (bytes16)
     * @return bytes16 High-precision tan(x), or QNAN if undefined
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

    // ------------------------------------------------------------
    // cot(x)
    // ------------------------------------------------------------
    /**
     * @notice Computes cot(x) in binary128 precision.
     * @dev Implementation:
     *        cot(x) = cos(x) / sin(x).
     *
     *      Domain notes:
     *        - Returns QNAN when sin(x) ≈ 0 (undefined).
     *        - Propagates QNAN from sin or cos if present.
     *
     *      Precision matches underlying sin/cos evaluation (~1e-34).
     *
     * @param x Input angle (bytes16)
     * @return bytes16 High-precision cot(x), or QNAN if undefined
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