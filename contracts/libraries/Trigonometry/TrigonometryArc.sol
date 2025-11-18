// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { TrigonometricConstants as CS } from "./TrigonometricConstants.sol";
import { TrigonometryHelpers as TH } from "./TrigonometryHelpers.sol";
import { TrigonometrySinCos } from "./TrigonometrySinCos.sol";

/**
 * @title TrigonometryArc
 * @notice High–precision inverse trig functions (asin, acos, atan) in quad.
 *
 * Strategy:
 *  - asin(x):
 *      * Domain check |x| <= 1
 *      * |x| small: return x
 *      * |x| <= 0.5: odd polynomial (ASIN_Cn) → y0
 *      * |x|  > 0.5: half–angle identity → y0
 *      * 3x Newton iteration on f(y)=sin(y)−x using TrigonometrySinCos
 *  - acos(x) = π/2 − asin(x)
 *  - atan(x) = asin( x / sqrt(1 + x²) )
 */
library TrigonometryArc {
    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;
    bytes16 internal constant QNAN  = 0x7fff8000000000000000000000000000;

    /*──────────────────────────────────────────────────────────
        asin(x)
    ──────────────────────────────────────────────────────────*/

    /**
     * @dev High-precision asin(x).
     * Domain: x ∈ [-1, 1]
     *
     * - |x| very small  → asin(x) ≈ x
     * - |x| ≤ 0.5       → polynomial approximation + Newton refine
     * - |x|  > 0.5      → half-angle reduction + polynomial + Newton refine
     */
    function asin(bytes16 x) internal pure returns (bytes16) {
        // Domain check
        bytes16 one = MathLib.fromUInt(1);
        bytes16 ax  = MathLib.abs(x);
        if (MathLib.cmp(ax, one) > 0) {
            return QNAN;
        }

        // Tiny x: asin(x) ≈ x
        // Same threshold as sin small-angle shortcut.
        bytes16 tiny = 0x3F8A39EF35793C767300000000000000; // ~1e-6
        if (MathLib.cmp(ax, tiny) < 0) {
            return x;
        }

        bytes16 half = MathLib.div(one, MathLib.fromUInt(2)); // 0.5
        bytes16 y0; // initial approximation

        // ─────────────────────────────────────────
        // Region 1: |x| ≤ 0.5  → polynomial in x
        // asin(x) ≈ x + x·P(x²)
        // ─────────────────────────────────────────
        if (MathLib.cmp(ax, half) <= 0) {
            bytes16 x2 = MathLib.mul(x, x);

            bytes16 p = CS.ASIN_C15();
            p = MathLib.add(CS.ASIN_C13(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C11(), MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C9(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C7(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C5(),  MathLib.mul(x2, p));
            p = MathLib.add(CS.ASIN_C3(),  MathLib.mul(x2, p));

            y0 = MathLib.add(x, MathLib.mul(x, p));
        } else {
            // ─────────────────────────────────────
            // Region 2: |x| > 0.5
            //
            // Use stable identity on |x|:
            //    asin(u)  for u = |x|
            //    asin(u) = π/2 − 2·asin(r),
            //    r = sqrt((1 − u)/2)  ∈ [0, 0.5]
            // Then restore sign.
            // ─────────────────────────────────────
            bool neg = (MathLib.cmp(x, QZERO) < 0);
            bytes16 u = neg ? MathLib.neg(x) : x;

            bytes16 oneMinus = MathLib.sub(one, u);         // 1 − u
            bytes16 halfTimes = MathLib.mul(half, oneMinus); // (1 − u)/2
            bytes16 r = MathLib.sqrt(halfTimes);             // 0 ≤ r ≤ 0.5

            // asin(r) with same Region-1 polynomial
            bytes16 r2 = MathLib.mul(r, r);

            bytes16 p2 = CS.ASIN_C15();
            p2 = MathLib.add(CS.ASIN_C13(), MathLib.mul(r2, p2));
            p2 = MathLib.add(CS.ASIN_C11(), MathLib.mul(r2, p2));
            p2 = MathLib.add(CS.ASIN_C9(),  MathLib.mul(r2, p2));
            p2 = MathLib.add(CS.ASIN_C7(),  MathLib.mul(r2, p2));
            p2 = MathLib.add(CS.ASIN_C5(),  MathLib.mul(r2, p2));
            p2 = MathLib.add(CS.ASIN_C3(),  MathLib.mul(r2, p2));

            bytes16 asin_r = MathLib.add(r, MathLib.mul(r, p2)); // asin(r)
            bytes16 two    = MathLib.fromUInt(2);
            bytes16 twoAr  = MathLib.mul(two, asin_r);

            bytes16 halfPi = QC.HALF_PI();
            bytes16 approx = MathLib.sub(halfPi, twoAr);   // asin(u) for u ≥ 0

            if (neg) approx = MathLib.neg(approx);
            y0 = approx;
        }

        // ─────────────────────────────────────────
        // Newton refinement:
        //   Solve f(y) = sin(y) − x = 0
        //   y_{n+1} = y_n − f(y_n) / f'(y_n)
        //   f'(y)   = cos(y)
        // ─────────────────────────────────────────
        bytes16 y = y0;
        for (uint8 i = 0; i < 3; ++i) {
            bytes16 sy = TrigonometrySinCos.sin(y);
            bytes16 cy = TrigonometrySinCos.cos(y);
            bytes16 f  = MathLib.sub(sy, x);

            // If cos is ~0 (shouldn't happen in principal branch), just break
            if (MathLib.cmp(MathLib.abs(cy), tiny) <= 0) {
                break;
            }

            bytes16 delta = MathLib.div(f, cy);
            y = MathLib.sub(y, delta);
        }

        return y;
    }

    /*──────────────────────────────────────────────────────────
        acos(x)
    ──────────────────────────────────────────────────────────*/

    /**
     * @dev High-precision acos(x).
     * Uses identity:
     *   acos(x) = π/2 − asin(x)
     * Domain: x ∈ [-1, 1]
     */
    function acos(bytes16 x) internal pure returns (bytes16) {
        bytes16 one = MathLib.fromUInt(1);
        bytes16 ax  = MathLib.abs(x);

        if (MathLib.cmp(ax, one) > 0) {
            return QNAN;
        }

        bytes16 a = asin(x);
        return MathLib.sub(QC.HALF_PI(), a);
    }

    /*──────────────────────────────────────────────────────────
        atan(x)
    ──────────────────────────────────────────────────────────*/
    /**
    * @dev High-precision atan(x) in quadruple precision.
    *
    * Initial approximation:
    *   atan(x) ≈ asin( x / sqrt(1 + x²) )
    *
    * Then refined using two Newton iterations on:
    *   f(t)   = tan(t) - x
    *   f'(t)  = 1 + tan(t)²
    *   t_next = t - f(t) / f'(t)
    *
    * This approach maintains accuracy up to ~1e-34 and ensures
    * consistency with the primary trigonometric identities.
    */
    function atan(bytes16 x) internal pure returns (bytes16) {
        if (TH.isNaN(x)) return QNAN;

        bytes16 one = MathLib.fromUInt(1);
        bytes16 ax  = MathLib.abs(x);

        // Fast path for very large values: atan(x) → ±π/2
        bytes16 huge = 0x403f0000000000000000000000000000; // 2^112
        if (MathLib.cmp(ax, huge) > 0) {
            return (MathLib.cmp(x, QZERO) > 0)
                ? QC.HALF_PI()
                : MathLib.neg(QC.HALF_PI());
        }

        // (1) Initial estimate using asin(u)
        //     where u = x / sqrt(1 + x²), ensuring |u| < 1.
        bytes16 x2    = MathLib.mul(x, x);
        bytes16 denom = MathLib.add(one, x2);      // 1 + x²
        bytes16 root  = MathLib.sqrt(denom);       // sqrt(1 + x²)
        bytes16 u     = MathLib.div(x, root);      // normalized input

        bytes16 y = asin(u); // first-order approximation of atan(x)

        // (2) Newton-Raphson refinement on f(t) = tan(t) - x
        //     t_next = t - (tan(t) - x) / (1 + tan²(t))
        bytes16 tiny = 0x3F8A39EF35793C767300000000000000; // ~1e-6
        for (uint8 i = 0; i < 2; ++i) {
            bytes16 sy = TrigonometrySinCos.sin(y);
            bytes16 cy = TrigonometrySinCos.cos(y);

            // Skip iteration if cos(y) ≈ 0 (near singularity)
            if (MathLib.cmp(MathLib.abs(cy), tiny) <= 0) {
                break;
            }

            // Compute tan(y)
            bytes16 tanY  = MathLib.div(sy, cy);
            bytes16 tanY2 = MathLib.mul(tanY, tanY);
            bytes16 sec2  = MathLib.add(one, tanY2);   // 1 + tan²(y)

            // Update step: delta = (tan(y) - x) / (1 + tan²(y))
            bytes16 f     = MathLib.sub(tanY, x);
            bytes16 delta = MathLib.div(f, sec2);
            y = MathLib.sub(y, delta);
        }

        return y;
    }
}