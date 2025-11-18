// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { TrigonometricConstants as CS } from "./TrigonometricConstants.sol";
import { TrigonometryHelpers as TH } from "./TrigonometryHelpers.sol";

library TrigonometrySinCos {

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;

    // ===============================================================
    // === POLY EVAL HELPERS (Gas-optimized, unrolled Horner)
    // ===============================================================
    
    /**
     * @dev _sin_poly(x) - Evaluates the degree-19 minimax polynomial for sin(x).
     * @param x Angle in the core domain [-π/4, π/4].
     * @return sin(x) approximation.
     */
    function _sin_poly(bytes16 x) internal pure returns (bytes16) {
        bytes16 x2 = MathLib.mul(x, x);

        // Horner chain (degree 19)
        // P(x^2) = C19 + x^2*(C17 + x^2*(...))
        bytes16 y = CS.SIN_C19();
        y = MathLib.add(CS.SIN_C17(), MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C15(), MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C13(), MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C11(), MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C9(),  MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C7(),  MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C5(),  MathLib.mul(x2, y));
        y = MathLib.add(CS.SIN_C3(),  MathLib.mul(x2, y));

        // sin(x) ≈ x + x*P(x^2)
        return MathLib.add(x, MathLib.mul(x, y));
    }

    // ===============================================================
    // === sin(x)
    // ===============================================================
    /**
     * @dev High-precision sin(x).
     * Error < 1e-34.
     */
    function sin(bytes16 x) internal pure returns (bytes16) {

        bytes16 ax = MathLib.abs(x);

        // |x| < ~1e-6 → sin(x) ≈ x
        if (MathLib.cmp(ax, 0x3F8A39EF35793C767300000000000000) < 0) {
            return x;
        }

        (bytes16 xr, uint8 quadrant) = TH.reduceAngle(x);

        // -------------------------------------------------------
        // SPECIAL CASE: xr == 0
        // xr = 0 occurs at x = k * (π/2)
        //
        // k mod 4:
        //   0 → sin(0)    = 0
        //   1 → sin(π/2)  = +1
        //   2 → sin(π)    = 0
        //   3 → sin(3π/2) = -1
        // -------------------------------------------------------
        if (MathLib.cmp(xr, QZERO) == 0) {
            if (quadrant == 1) return MathLib.fromUInt(1);                           // +π/2
            if (quadrant == 3) return MathLib.neg(MathLib.fromUInt(1));         // -π/2
            return QZERO; // 0 or π
        }

        // Core mapping to [-π/4, π/4]
        bytes16 axr = MathLib.abs(xr);
        bytes16 coreX;
        bool useCosPoly;

        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            // axr in (π/4, π/2]
            // sin(axr) = cos(π/2 - axr)
            coreX = MathLib.sub(QC.HALF_PI(), axr);   // coreX ∈ [0, π/4)
            useCosPoly = true;                        // sin(axr) uses cos(coreX)
        } else {
            // axr ∈ [0, π/4]
            coreX = axr;
            useCosPoly = false;                       // sin(axr) uses sin(coreX)
        }

        // Evaluate polynomial
        bytes16 result = useCosPoly ? _cos_poly(coreX) : _sin_poly(coreX);

        // Restore sign due to original xr sign (sin is odd)
        if (MathLib.cmp(xr, QZERO) < 0) {
            result = MathLib.neg(result);
        }

        // -------------------------------------------------------
        // QUADRANT SIGN (sin)
        // quadrant meaning:
        //   0 → sin(+)
        //   1 → sin(+)
        //   2 → sin(-)
        //   3 → sin(-)
        // -------------------------------------------------------
        if (quadrant == 2 || quadrant == 3) {
            result = MathLib.neg(result);
        }

        return result;
    }

    /**
     * @dev _cos_poly(x) - Evaluates the degree-18 minimax polynomial for cos(x).
     * @param x Angle in the core domain [-π/4, π/4].
     * @return cos(x) approximation.
     */
    function _cos_poly(bytes16 x) internal pure returns (bytes16) {
        bytes16 x2 = MathLib.mul(x, x);

        // Horner chain (degree 18)
        // P(x^2) = C18 + x^2*(C16 + x^2*(...))
        bytes16 y = CS.COS_C18();
        y = MathLib.add(CS.COS_C16(), MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C14(), MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C12(), MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C10(), MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C8(),  MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C6(),  MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C4(),  MathLib.mul(x2, y));
        y = MathLib.add(CS.COS_C2(),  MathLib.mul(x2, y));
        
        // cos(x) ≈ C0 + x^2*P(x^2)
        // Note: The original had the C0 term inside the mul/add chain,
        // which is incorrect for an even polynomial.
        // It should be: C0 + x^2 * (C2 + x^2 * (C4 + ...))
        // Your unrolled Horner was correct: y = C0 + mul(x2, y)
        y = MathLib.add(CS.COS_C0(),  MathLib.mul(x2, y));

        return y;
    }

    // ===============================================================
    // === cos(x)
    // ===============================================================
    /**
     * @dev High-precision cos(x).
     * Error < 1e-34.
     */
    function cos(bytes16 x) internal pure returns (bytes16) {
        (bytes16 xr, uint8 quadrant) = TH.reduceAngle(x);

        // -------------------------------------------------------
        // SPECIAL CASE: xr == 0
        // xr = 0 occurs at x = k * (π/2)
        //
        // k mod 4:
        //   0 → cos(0)    = +1
        //   1 → cos(π/2)  =  0
        //   2 → cos(π)    = -1
        //   3 → cos(3π/2) =  0
        // -------------------------------------------------------
        if (MathLib.cmp(xr, QZERO) == 0) {
            if (quadrant == 0) return MathLib.fromUInt(1);               // +1
            if (quadrant == 2) return MathLib.neg(MathLib.fromUInt(1)); // −1
            return QZERO;                                 // 0 at ±π/2
        }


        // Core mapping to [-π/4, π/4]
        bytes16 axr = MathLib.abs(xr);
        bytes16 coreX;
        bool useCosPoly;

        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            // axr in (π/4, π/2]
            // cos(axr) = sin(π/2 - axr)
            coreX = MathLib.sub(QC.HALF_PI(), axr);
            useCosPoly = false;                       // cos(axr) uses sin(coreX)
        } else {
            // axr ∈ [0, π/4]
            coreX = axr;
            useCosPoly = true;                        // cos(axr) uses cos(coreX)
        }

        // Evaluate polynomial
        bytes16 result = useCosPoly ? _cos_poly(coreX) : _sin_poly(coreX);

        // cos(xr) is EVEN → no sign from xr itself

        // -------------------------------------------------------
        // QUADRANT SIGN FOR COS
        //   0 → +
        //   1 → -
        //   2 → -
        //   3 → +
        // -------------------------------------------------------
        if (quadrant == 1 || quadrant == 2) {
            result = MathLib.neg(result);
        }

        return result;
    }
}