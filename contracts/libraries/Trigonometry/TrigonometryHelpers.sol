// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { TrigonometricConstants as CS } from "./TrigonometricConstants.sol";

library TrigonometryHelpers {

    /*──────────────────────────────────────────────────────────
        HELPERS
    ──────────────────────────────────────────────────────────*/

    // ===============================================================
    // === REDUCE ANGLE FUNCTION
    // ===============================================================
    /**
     * @dev Hybrid range reduction.
     * Returns:
     * - xr: reduced angle in [-π/2, π/2]
     * - quadrant: integer in [0..3] for sin/cos symmetry
     */
    function reduceAngle(bytes16 x) internal pure returns (bytes16 xr, uint8 quadrant) {
        bytes16 ax = MathLib.abs(x);
        bytes16 halfpi = QC.HALF_PI();
        bytes16 twopi  = QC.TWO_PI();

        // Fast Path - |x| < 2^64
        if (MathLib.cmp(ax,CS.TWO_POW_64()) <= 0) {
            bytes16 t = MathLib.div(x, halfpi);
            int256 qi = MathLib.toInt(t); // TRUNCATE toward zero
            bytes16 qi_f = MathLib.fromInt(qi);

            // quadrant = qi mod 4
            // We use a bitwise AND for (qi % 4)
            // Note: In Solidity, (-1 & 3) is 3, (-2 & 3) is 2.
            quadrant = uint8(uint256(qi & 3));
            xr = MathLib.sub(x, MathLib.mul(qi_f, halfpi));
            return (xr, quadrant);
        }

        // Slow Path - PAYNE–HANEK
        // Reduce mod 2π
        bytes16 t1 = MathLib.div(x, twopi);
        int256 k1 = MathLib.floorInt(t1);
        bytes16 k1f = MathLib.fromInt(k1);
        bytes16 xm = MathLib.sub(x, MathLib.mul(k1f, twopi));

        // Reduce xm mod (π/2)
        bytes16 t2 = MathLib.div(xm, halfpi);
        int256 k2 = MathLib.floorInt(t2);
        bytes16 k2f = MathLib.fromInt(k2);

        xr = MathLib.sub(xm, MathLib.mul(k2f, halfpi));

        // quadrant = k2 mod 4
        int256 qm = k2 % 4;
        if (qm < 0) qm += 4; // normalize to [0..3]
        quadrant = uint8(uint256(qm));
    }

    // ===============================================================
    // === NaN HELPERS
    // ===============================================================
    /**
     * @dev Check if x is NaN in IEEE-754 quad format.
     * NaN is defined as: exponent all 1s AND mantissa != 0.
     */
    function isNaN(bytes16 x) internal pure returns (bool) {
        return MathLib.isNaN(x);
    }
}