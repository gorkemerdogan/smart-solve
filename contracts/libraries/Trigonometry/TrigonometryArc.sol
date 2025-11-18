// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { TrigonometricConstants as CS } from "./TrigonometricConstants.sol";
import { TrigonometryHelpers as TH } from "./TrigonometryHelpers.sol";

library TrigonometryArc {

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;
    bytes16 internal constant QNAN  = 0x7fff8000000000000000000000000000;

    // ===============================================================
    // === asin(x)
    // ===============================================================
    /**
     * @dev High-precision asin(x)
     * Domain: x ∈ [-1, 1]
     * Regions:
     *   |x| ≤ 0.5 → polynomial
     *   |x| > 0.5 → asin(x) = ±π/2 ∓ 2*asin(r), r = sqrt((1−x)/2)
     * Fully quad-precision (1e-34).
     */
    function asin(bytes16 x) internal pure returns (bytes16) {

        // Domain error
        bytes16 one = MathLib.fromUInt(1);
        bytes16 ax  = MathLib.abs(x);
        if (MathLib.cmp(ax, one) > 0) {
            return QNAN;
        }

        // -------- Region 1: |x| ≤ 0.5 ----------
        bytes16 half = MathLib.div(one, MathLib.fromUInt(2));   // 0.5
        if (MathLib.cmp(ax, half) <= 0) {
            bytes16 x2 = MathLib.mul(x, x);

            // Horner (degree-15)
            bytes16 p = CS.ASIN_C15();
            p = MathLib.add(CS.ASIN_C13(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C11(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C9(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C7(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C5(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C3(),  MathLib.mul(x2, p));

            return MathLib.add(x, MathLib.mul(x, p));
        }

        // -------- Region 2: |x| > 0.5 ----------
        //
        // Correct identity:
        //
        // For x ≥ 0:
        //   asin(x) =  π/2 − 2*asin(r)
        //
        // For x < 0:
        //   asin(x) = −π/2 + 2*asin(r)
        //
        // where r = sqrt((1 − x)/2)
        //
        bytes16 oneMinus = MathLib.sub(one, x);        // (1 − x)
        bytes16 halfTimes = MathLib.mul(half, oneMinus);
        bytes16 r = MathLib.sqrt(halfTimes);

        // asin(r) using Region-1 always
        bytes16 r2 = MathLib.mul(r, r);

        bytes16 p2 = CS.ASIN_C15();
        p2 = MathLib.add(CS.ASIN_C13(), MathLib.mul(r2, p2));
        p2 = MathLib.add(CS.ASIN_C11(), MathLib.mul(r2, p2));
        p2 = MathLib.add(CS.ASIN_C9(),  MathLib.mul(r2, p2));
        p2 = MathLib.add(CS.ASIN_C7(),  MathLib.mul(r2, p2));
        p2 = MathLib.add(CS.ASIN_C5(),  MathLib.mul(r2, p2));
        p2 = MathLib.add(CS.ASIN_C3(),  MathLib.mul(r2, p2));

        bytes16 inner = MathLib.add(r, MathLib.mul(r, p2)); // asin(r)
        bytes16 twoInner = MathLib.mul(MathLib.fromUInt(2), inner);

        if (MathLib.cmp(x, QZERO) >= 0) {
            // x ≥ 0
            return MathLib.sub(QC.HALF_PI(), twoInner);
        } else {
            // x < 0
            return MathLib.add(MathLib.neg(QC.HALF_PI()), twoInner);
        }
    }



    // ===============================================================
    // === acos(x)
    // ===============================================================
    /**
     * @dev High-precision acos(x)
     * Regions:
     *   |x| ≤ 0.5 → π/2 − ( x + xP(x²) )
     *   |x| > 0.5 → 2 * asin( sqrt((1−x)/2) )
     */
    function acos(bytes16 x) internal pure returns (bytes16) {

        bytes16 one = MathLib.fromUInt(1);
        bytes16 ax  = MathLib.abs(x);

        if (MathLib.cmp(ax, one) > 0) {
            return QNAN;
        }

        bytes16 half = MathLib.div(one, MathLib.fromUInt(2)); // 0.5

        // -------- Region 1: |x| ≤ 0.5 ----------
        if (MathLib.cmp(ax, half) <= 0) {

            bytes16 x2 = MathLib.mul(x, x);

            bytes16 y = CS.ACOS_C15();
            y = MathLib.add(CS.ACOS_C13(), MathLib.mul(x2, y));
            y = MathLib.add(CS.ACOS_C11(), MathLib.mul(x2, y));
            y = MathLib.add(CS.ACOS_C9(),  MathLib.mul(x2, y));
            y = MathLib.add(CS.ACOS_C7(),  MathLib.mul(x2, y));
            y = MathLib.add(CS.ACOS_C5(),  MathLib.mul(x2, y));
            y = MathLib.add(CS.ACOS_C3(),  MathLib.mul(x2, y));

            bytes16 inner = MathLib.add(x, MathLib.mul(x, y));
            return MathLib.sub(QC.HALF_PI(), inner);
        }

        // -------- Region 2: |x| > 0.5 ----------
        //
        // Correct stable identity:
        //
        // acos(x) = 2 * asin( sqrt((1 − x)/2) )
        //
        bytes16 oneMinus = MathLib.sub(one, x);
        bytes16 halfTimes = MathLib.mul(half, oneMinus);
        bytes16 r = MathLib.sqrt(halfTimes);

        bytes16 innerAsin = asin(r);

        return MathLib.mul(MathLib.fromUInt(2), innerAsin);
    }


    // ===============================================================
    // === atan(x)
    // ===============================================================
    /**
     * @dev High-precision atan(x)
     * Region 0: |x| > 2^112  → ±π/2
     * Region 1: |x| ≤ 1      → polynomial
     * Region 2: |x| > 1      → atan(x) = ±π/2 ∓ atan(1/x)
     */
    function atan(bytes16 x) internal pure returns (bytes16) {

        if (TH.isNaN(x)) return QNAN;

        bytes16 one = MathLib.fromUInt(1);
        bytes16 ax  = MathLib.abs(x);

        // Huge values → ±π/2
        bytes16 huge = 0x403f0000000000000000000000000000; // 2^112
        if (MathLib.cmp(ax, huge) > 0) {
            return (MathLib.cmp(x, QZERO) > 0)
                ? QC.HALF_PI()
                : MathLib.neg(QC.HALF_PI());
        }

        // -------- Region 1: |x| ≤ 1 ----------
        if (MathLib.cmp(ax, one) <= 0) {

            bytes16 x2 = MathLib.mul(x, x);

            // Horner degree-17
            bytes16 p = CS.ATAN_C17();
            p = MathLib.add(CS.ATAN_C15(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ATAN_C13(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ATAN_C11(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ATAN_C9(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ATAN_C7(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ATAN_C5(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ATAN_C3(),  MathLib.mul(x2, p));

            return MathLib.add(x, MathLib.mul(x, p));
        }

        // -------- Region 2: |x| > 1 ----------
        // atan(x) = ±π/2 ∓ atan(1/x)
        bytes16 t = MathLib.div(one, x);     // |t| < 1
        bytes16 t2 = MathLib.mul(t, t);

        bytes16 z = CS.ATAN_C17();
        z = MathLib.add(CS.ATAN_C15(), MathLib.mul(t2, z));
        z = MathLib.add(CS.ATAN_C13(), MathLib.mul(t2, z));
        z = MathLib.add(CS.ATAN_C11(), MathLib.mul(t2, z));
        z = MathLib.add(CS.ATAN_C9(),  MathLib.mul(t2, z));
        z = MathLib.add(CS.ATAN_C7(),  MathLib.mul(t2, z));
        z = MathLib.add(CS.ATAN_C5(),  MathLib.mul(t2, z));
        z = MathLib.add(CS.ATAN_C3(),  MathLib.mul(t2, z));

        bytes16 poly = MathLib.add(t, MathLib.mul(t, z));

        if (MathLib.cmp(x, QZERO) > 0) {
            // x > 1
            return MathLib.sub(QC.HALF_PI(), poly);
        } else {
            // x < −1
            return MathLib.add(MathLib.neg(QC.HALF_PI()), poly);
        }
    }
}