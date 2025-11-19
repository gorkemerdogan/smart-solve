// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";

library TrigonometryHelpers {

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;

    /**
    * @dev IEEE-754 compliant angle reduction.
    *
    * Returns:
    *  - xr ∈ [-π/4, +π/4]
    *  - mask bitfield:
    *        bit0: swap (0 → use sin_poly, 1 → use cos_poly)
    *        bit1: sin sign  (1 → negative)
    *        bit2: cos sign  (1 → negative)
    */
    function reduceAngle(bytes16 x) internal pure returns (bytes16 xr, uint8 mask) {
        bytes16 halfpi  = QC.HALF_PI();
        bytes16 twopi   = QC.TWO_PI();

        // 1) mod 2π
        bytes16 t = MathLib.div(x, twopi);
        int256 k = MathLib.toInt(t);
        bytes16 kq = MathLib.fromInt(k);
        bytes16 xm = MathLib.sub(x, MathLib.mul(kq, twopi));

        // 2) mod π/2
        bytes16 t2 = MathLib.div(xm, halfpi);
        int256 k2 = MathLib.toInt(t2);
        bytes16 k2q = MathLib.fromInt(k2);
        xr = MathLib.sub(xm, MathLib.mul(k2q, halfpi));

        // 3) quadrant
        uint8 q = uint8(uint256(k2 & 3));

        uint8 swap   = (q == 1 || q == 3) ? 1 : 0; // bit0
        uint8 sinNeg = (q == 2 || q == 3) ? 1 : 0; // bit1
        uint8 cosNeg = (q == 1 || q == 2) ? 1 : 0; // bit2

        mask = (swap)
            | (sinNeg << 1)
            | (cosNeg << 2);

        return (xr, mask);
    }
}