// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { TrigonometrySinCos as TSC } from "./TrigonometrySinCos.sol";

/**
 * @title TrigonometryArc
 * @notice High-precision inverse trigonometric functions (asin, acos, atan) in
 *         IEEE-754 binary128 (bytes16) arithmetic.
 * @dev Implements domain checks, stable polynomial approximations, half-angle
 *      reductions, and Newton refinements. All operations are performed using
 *      MathLib and TrigonometrySinCos.
 */
library TrigonometryArc {
    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;
    bytes16 internal constant QNAN  = 0x7fff8000000000000000000000000000;

    /*──────────────────────────────────────────────────────────
        asin(x)
    ──────────────────────────────────────────────────────────*/

    /**
     * @notice Computes asin(x) in binary128 precision.
     * @dev Domain: x ∈ [-1, 1].
     *
     *      Uses a region-dependent strategy:
     *         - |x| < tiny threshold: return x
     *         - |x| ≤ 0.5: odd polynomial in x² + Newton refinement
     *         - |x|  > 0.5: stable half-angle reduction followed by the same polynomial path
     *
     *      Final approximation is refined with Newton iterations:
     *         f(y)  = sin(y) − x
     *         f'(y) = cos(y)
     *
     * @param x Input value (bytes16)
     * @return bytes16 asin(x) in radians, or QNAN for inputs outside the domain
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

        bytes16 C3  = MathLib.div(MathLib.fromInt(1),   MathLib.fromInt(6));
        bytes16 C5  = MathLib.div(MathLib.fromInt(3),   MathLib.fromInt(40));
        bytes16 C7  = MathLib.div(MathLib.fromInt(5),   MathLib.fromInt(112));
        bytes16 C9  = MathLib.div(MathLib.fromInt(35),  MathLib.fromInt(1152));
        bytes16 C11 = MathLib.div(MathLib.fromInt(63),  MathLib.fromInt(2816));
        bytes16 C13 = MathLib.div(MathLib.fromInt(231), MathLib.fromInt(13312));
        bytes16 C15 = MathLib.div(MathLib.fromInt(143), MathLib.fromInt(10240));

        bytes16 half = MathLib.div(one, MathLib.fromUInt(2)); // 0.5
        bytes16 y0; // initial approximation

        // ─────────────────────────────────────────
        // Region 1: |x| ≤ 0.5  → polynomial in x
        // asin(x) ≈ x + x·P(x²)
        // ─────────────────────────────────────────
        if (MathLib.cmp(ax, half) <= 0) {
            bytes16 x2 = MathLib.mul(x, x);
            bytes16 p = C15;

            p = MathLib.add(C13, MathLib.mul(x2, p));
            p = MathLib.add(C11, MathLib.mul(x2, p));
            p = MathLib.add(C9,  MathLib.mul(x2, p));
            p = MathLib.add(C7,  MathLib.mul(x2, p));
            p = MathLib.add(C5,  MathLib.mul(x2, p));
            p = MathLib.add(C3,  MathLib.mul(x2, p));

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

            bytes16 p2 = C15;
            p2 = MathLib.add(C13, MathLib.mul(r2, p2));
            p2 = MathLib.add(C11, MathLib.mul(r2, p2));
            p2 = MathLib.add(C9,  MathLib.mul(r2, p2));
            p2 = MathLib.add(C7,  MathLib.mul(r2, p2));
            p2 = MathLib.add(C5,  MathLib.mul(r2, p2));
            p2 = MathLib.add(C3,  MathLib.mul(r2, p2));

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
            bytes16 sy = TSC.sin(y);
            bytes16 cy = TSC.cos(y);
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
     * @notice Computes acos(x) in binary128 precision.
     * @dev Domain: x ∈ [-1, 1].
     *
     *      Implemented via the identity:
     *          acos(x) = π/2 − asin(x)
     *
     *      Returns QNAN if |x| > 1.
     *
     * @param x Input value (bytes16)
     * @return bytes16 acos(x) in radians
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
     * @notice Computes atan(x) in binary128 precision.
     * @dev Strategy:
     *       (1) Initial approximation via:
     *             atan(x) ≈ asin( x / sqrt(1 + x²) )
     *
     *       (2) Two Newton refinements on:
     *             f(t)  = tan(t) − x
     *             f'(t) = 1 + tan²(t)
     *
     *       For |x| → ∞, returns ±π/2.
     *
     * @param x Input value (bytes16)
     * @return bytes16 atan(x) in radians, or QNAN if x is NaN
     */
    function atan(bytes16 x) internal pure returns (bytes16) {
        if (MathLib.isNaN(x)) return QNAN;

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
            bytes16 sy = TSC.sin(y);
            bytes16 cy = TSC.cos(y);

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