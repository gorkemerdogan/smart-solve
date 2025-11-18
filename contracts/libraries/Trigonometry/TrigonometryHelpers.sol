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
     * @dev Angle reduction function.
     * Reduces the input 'x' to the interval [-π/2, π/2] (returned as xr)
     * and determines which quadrant the angle belongs to.
     *
     * Returns:
     * - xr: Reduced angle in [-π/2, π/2].
     * - quadrant: Integer [0..3] (0: 0-90, 1: 90-180, 2: 180-270, 3: 270-360)
     */
    function reduceAngle(bytes16 x) internal pure returns (bytes16 xr, uint8 quadrant) {
        bytes16 halfpi = QC.HALF_PI();

        // Step 1: Compute x / (π/2).
        // This determines how many 90-degree slices are contained in x.
        bytes16 t = MathLib.div(x, halfpi);

        // Step 2: Get the integer part (Truncate towards zero).
        // Example: For 1.6 radians (~91 degrees), t ≈ 1.01, so qi becomes 1.
        int256 qi = MathLib.toInt(t);

        // Step 3: Convert this integer back to Quad format for calculation.
        bytes16 qi_f = MathLib.fromInt(qi);

        // Step 4: Calculate the quadrant.
        // Bitwise AND (& 3) is equivalent to (qi % 4) but handles negative numbers correctly.
        // Example: qi = 1  -> q = 1
        // Example: qi = -1 -> (-1 & 3) = 3 (Correct mathematical wrapping)
        quadrant = uint8(uint256(qi & 3));

        // Step 5: Calculate the remaining angle (xr).
        // xr = x - (qi * π/2)
        xr = MathLib.sub(x, MathLib.mul(qi_f, halfpi));

        return (xr, quadrant);
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