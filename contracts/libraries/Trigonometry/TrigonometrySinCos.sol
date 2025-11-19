// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";

/**
 * @title TrigonometrySinCos
 * @notice High-precision sine and cosine evaluation in IEEE-754 binary128 (bytes16).
 * @dev Implements quadrant-aware angle reduction followed by degree-limited
 *      core polynomials on the interval [-π/4, +π/4]. Provides sin(x) and cos(x)
 *      used by higher-level trigonometry modules.
 */
library TrigonometrySinCos {

    // ===============================================================
    // === HELPERS
    // ===============================================================

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;

    /**
     * @notice Reduces an angle into the core range [-π/4, +π/4] and encodes
     *         quadrant information in a mask.
     * @dev Reduction occurs in two stages:
     *        (1) Modulo 2π → principal domain
     *        (2) Modulo π/2 → core subrange
     *
     *      The returned bitmask contains:
     *        bit0: swap flag (0 → use sin polynomial, 1 → use cos polynomial)
     *        bit1: sine sign bit  (1 → negative)
     *        bit2: cosine sign bit (1 → negative)
     *
     * @param x Input angle (bytes16)
     * @return xr Reduced angle in [-π/4, +π/4]
     * @return mask Encoded quadrant and swap information
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

    /**
     * @notice Decodes the original quadrant from the bitmask produced by reduceAngle.
     * @dev Mapping:
     *        mask = 0 → Q0
     *        mask = 5 → Q1
     *        mask = 6 → Q2
     *        mask = 3 → Q3
     *      Returns 0 as a fallback for invalid masks.
     *
     * @param mask Encoded quadrant information
     * @return uint8 Quadrant index in {0,1,2,3}
     */
    function _quadrant(uint8 mask) private pure returns (uint8) {
        if (mask == 0) return 0;
        if (mask == 5) return 1;
        if (mask == 6) return 2;
        if (mask == 3) return 3;
        // Should not happen; default to Q0
        return 0;
    }
    
    /**
     * @notice Evaluates the core sine polynomial on the reduced domain.
     * @dev Uses a truncated Taylor expansion expressed in Horner form:
     *
     *        sin(x) = x + x * z * P(z),   z = x²
     *
     *      where P(z) is a degree-5 polynomial matching the odd terms
     *      of the sine series up to x¹³.
     *
     * @param x Angle in core interval [-π/4, +π/4]
     * @return bytes16 Approximated sin(x)
     */
    function _sin_poly(bytes16 x) internal pure returns (bytes16) {
        bytes16 z = MathLib.mul(x, x); // z = x^2

        // Coefficients for P(z): c1..c6  (for x^3..x^13 terms)
        // P(z) = c1 + c2*z + c3*z^2 + c4*z^3 + c5*z^4 + c6*z^5
        //
        // c1 = -1/3!
        // c2 =  1/5!
        // c3 = -1/7!
        // c4 =  1/9!
        // c5 = -1/11!
        // c6 =  1/13!

        bytes16 c1 = MathLib.neg(MathLib.div(MathLib.fromUInt(1), MathLib.fromUInt(6)));         // 3! = 6
        bytes16 c2 = MathLib.div(MathLib.fromUInt(1),             MathLib.fromUInt(120));        // 5! = 120
        bytes16 c3 = MathLib.neg(MathLib.div(MathLib.fromUInt(1), MathLib.fromUInt(5040)));      // 7! = 5040
        bytes16 c4 = MathLib.div(MathLib.fromUInt(1),             MathLib.fromUInt(362880));     // 9! = 362880
        bytes16 c5 = MathLib.neg(MathLib.div(MathLib.fromUInt(1), MathLib.fromUInt(39916800)));  // 11! = 39916800
        bytes16 c6 = MathLib.div(MathLib.fromUInt(1),             MathLib.fromUInt(6227020800)); // 13! = 6227020800

        // Horner: y = c6; y = c5 + z*y; ...; y = c1 + z*y;
        bytes16 y = c6;
        y = MathLib.add(c5, MathLib.mul(z, y));
        y = MathLib.add(c4, MathLib.mul(z, y));
        y = MathLib.add(c3, MathLib.mul(z, y));
        y = MathLib.add(c2, MathLib.mul(z, y));
        y = MathLib.add(c1, MathLib.mul(z, y));

        // sin(x) ≈ x + x*z*y
        bytes16 xz = MathLib.mul(x, z);
        bytes16 corr = MathLib.mul(xz, y);
        return MathLib.add(x, corr);
    }

    /**
     * @notice Evaluates the core cosine polynomial on the reduced domain.
     * @dev Uses a truncated Taylor expansion expressed in Horner form:
     *
     *        cos(x) = 1 + z * Q(z),   z = x²
     *
     *      where Q(z) is a degree-5 polynomial matching the even cosine
     *      terms up to x¹².
     *
     * @param x Angle in core interval [-π/4, +π/4]
     * @return bytes16 Approximated cos(x)
     */
    function _cos_poly(bytes16 x) internal pure returns (bytes16) {
        bytes16 z = MathLib.mul(x, x); // z = x^2

        // Q(z) = d1 + d2*z + d3*z^2 + d4*z^3 + d5*z^4 + d6*z^5
        //
        // d1 = -1/2!
        // d2 =  1/4!
        // d3 = -1/6!
        // d4 =  1/8!
        // d5 = -1/10!
        // d6 =  1/12!

        bytes16 d1 = MathLib.neg(MathLib.div(MathLib.fromUInt(1), MathLib.fromUInt(2)));        // 2! = 2
        bytes16 d2 = MathLib.div(MathLib.fromUInt(1),             MathLib.fromUInt(24));        // 4! = 24
        bytes16 d3 = MathLib.neg(MathLib.div(MathLib.fromUInt(1), MathLib.fromUInt(720)));      // 6! = 720
        bytes16 d4 = MathLib.div(MathLib.fromUInt(1),             MathLib.fromUInt(40320));     // 8! = 40320
        bytes16 d5 = MathLib.neg(MathLib.div(MathLib.fromUInt(1), MathLib.fromUInt(3628800)));  // 10! = 3628800
        bytes16 d6 = MathLib.div(MathLib.fromUInt(1),             MathLib.fromUInt(479001600)); // 12! = 479001600

        // Horner: q = d6; q = d5 + z*q; ...; q = d1 + z*q;
        bytes16 q = d6;
        q = MathLib.add(d5, MathLib.mul(z, q));
        q = MathLib.add(d4, MathLib.mul(z, q));
        q = MathLib.add(d3, MathLib.mul(z, q));
        q = MathLib.add(d2, MathLib.mul(z, q));
        q = MathLib.add(d1, MathLib.mul(z, q));

        // cos(x) ≈ 1 + z*q
        bytes16 zq = MathLib.mul(z, q);
        return MathLib.add(MathLib.fromUInt(1), zq);
    }

    // ===============================================================
    // === sin(x)
    // ===============================================================
    /**
     * @notice Computes sin(x) in binary128 precision.
     * @dev Procedure:
     *        (1) reduceAngle → core domain and mask
     *        (2) handle exact multiples of π/2
     *        (3) optionally map to complementary angle if |xr| > π/4
     *        (4) select sin or cos polynomial via mask bit0
     *        (5) apply quadrant sign adjustment (bit1)
     *
     * @param x Input angle (bytes16)
     * @return bytes16 High-precision sin(x)
     */
    function sin(bytes16 x) internal pure returns (bytes16) {
        // 1) Range reduction → xr in [-π/2, π/2], mask holds swap/sign info
        (bytes16 xr, uint8 mask) = reduceAngle(x);

        // 2) Special case: exact multiples of π/2
        //    xr == 0 => x = k * (π/2)
        if (MathLib.isZero(xr)) {
            uint8 q = _quadrant(mask);
            if (q == 1) return MathLib.fromUInt(1);                // +π/2, 5π/2, ...
            if (q == 3) return MathLib.neg(MathLib.fromUInt(1));   // -π/2, 3π/2, ...
            return QZERO; // 0, π, 2π, ...
        }

        bool swap   = (mask & 1) != 0; // bit0
        bool sinNeg = (mask & 2) != 0; // bit1

        // 3) Core mapping: |xr| > π/4 → use complementary angle
        //    xr' = sign(xr) * (π/2 - |xr|)
        bytes16 axr = MathLib.abs(xr);
        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            bytes16 newxr = MathLib.sub(QC.HALF_PI(), axr);
            if (MathLib.cmp(xr, QZERO) < 0) {
                newxr = MathLib.neg(newxr);
            }
            xr = newxr;
            swap = !swap; // toggle: sin ↔ cos
        }

        // 4) Evaluate appropriate polynomial in core domain [-π/4, +π/4]
        bytes16 y = swap ? _cos_poly(xr) : _sin_poly(xr);

        // 5) Apply final sign from quadrant
        if (sinNeg) {
            y = MathLib.neg(y);
        }

        return y;
    }

    // ===============================================================
    // === cos(x)
    // ===============================================================
    /**
     * @notice Computes cos(x) in binary128 precision.
     * @dev Procedure mirrors sin(x):
     *        (1) reduceAngle → core domain and mask
     *        (2) handle exact multiples of π/2
     *        (3) complementary mapping when |xr| > π/4
     *        (4) evaluate appropriate polynomial via swap bit
     *        (5) apply cosine sign from mask bit2
     *
     * @param x Input angle (bytes16)
     * @return bytes16 High-precision cos(x)
     */
    function cos(bytes16 x) internal pure returns (bytes16) {
        // 1) Range reduction
        (bytes16 xr, uint8 mask) = reduceAngle(x);

        // 2) Special case: exact multiples of π/2
        if (MathLib.isZero(xr)) {
            uint8 q = _quadrant(mask);
            if (q == 0) return MathLib.fromUInt(1);               // 0, 2π, ...
            if (q == 2) return MathLib.neg(MathLib.fromUInt(1));  // π, 3π, ...
            return QZERO; // ±π/2, ...
        }

        bool swap   = (mask & 1) != 0; // bit0
        bool cosNeg = (mask & 4) != 0; // bit2

        // 3) Core mapping to [-π/4, +π/4]
        bytes16 axr = MathLib.abs(xr);
        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            bytes16 newxr = MathLib.sub(QC.HALF_PI(), axr);
            if (MathLib.cmp(xr, QZERO) < 0) {
                newxr = MathLib.neg(newxr);
            }
            xr = newxr;
            swap = !swap; // cos ↔ sin
        }

        // 4) Evaluate polynomial
        bytes16 y = swap ? _sin_poly(xr) : _cos_poly(xr);

        // 5) Quadrant sign
        if (cosNeg) {
            y = MathLib.neg(y);
        }

        return y;
    }
}