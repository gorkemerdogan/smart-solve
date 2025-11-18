// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad } from "abdk-libraries-solidity/ABDKMathQuad.sol";
import { TrigonometricConstants as CS } from "./TrigonometricConstants.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";

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
        bytes16 ax = ABDKMathQuad.abs(x);
        bytes16 halfpi = QC.HALF_PI();
        bytes16 twopi  = QC.TWO_PI();

        // Fast Path - |x| < 2^64
        if (ABDKMathQuad.cmp(ax,CS.TWO_POW_64()) <= 0) {
            bytes16 t = ABDKMathQuad.div(x, halfpi);
            int256 qi = ABDKMathQuad.toInt(t); // TRUNCATE toward zero
            bytes16 qi_f = ABDKMathQuad.fromInt(qi);

            // quadrant = qi mod 4
            // We use a bitwise AND for (qi % 4)
            // Note: In Solidity, (-1 & 3) is 3, (-2 & 3) is 2.
            quadrant = uint8(uint256(qi & 3));
            xr = ABDKMathQuad.sub(x, ABDKMathQuad.mul(qi_f, halfpi));
            return (xr, quadrant);
        }

        // Slow Path - PAYNE–HANEK
        // Reduce mod 2π
        bytes16 t1 = ABDKMathQuad.div(x, twopi);
        int256 k1 = ABDKMathQuad.toInt(t1); // trunc towards zero
        bytes16 k1f = ABDKMathQuad.fromInt(k1);
        bytes16 xm = ABDKMathQuad.sub(x, ABDKMathQuad.mul(k1f, twopi));

        // Reduce xm mod (π/2)
        bytes16 t2 = ABDKMathQuad.div(xm, halfpi);
        int256 k2 = ABDKMathQuad.toInt(t2);
        bytes16 k2f = ABDKMathQuad.fromInt(k2);

        xr = ABDKMathQuad.sub(xm, ABDKMathQuad.mul(k2f, halfpi));

        // quadrant = k2 mod 4
        quadrant = uint8(uint256(k2 & 3));
    }

    // ===============================================================
    // === NaN HELPERS
    // ===============================================================
    /**
     * @dev Check if x is NaN in IEEE-754 quad format.
     * NaN is defined as: exponent all 1s AND mantissa != 0.
     */
    function isNaN(bytes16 x) internal pure returns (bool) {
        return ABDKMathQuad.isNaN(x);
    }
}