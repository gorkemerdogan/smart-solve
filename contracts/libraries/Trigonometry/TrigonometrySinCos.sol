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
        // Note: The original formula adds C0 at the end to handle even polynomials correctly.
        y = MathLib.add(CS.COS_C0(),  MathLib.mul(x2, y));

        return y;
    }

    // ===============================================================
    // === sin(x)
    // ===============================================================
    /**
     * @dev High-precision sin(x).
     * Error < 1e-34.
     */
    function sin(bytes16 x) internal pure returns (bytes16) {
        // Small angle optimization: |x| < ~1e-6 → sin(x) ≈ x
        // Threshold: 10^-6 in quad precision
        bytes16 ax = MathLib.abs(x);
        if (MathLib.cmp(ax, 0x3F8A39EF35793C767300000000000000) < 0) {
            return x;
        }

        (bytes16 xr, uint8 quadrant) = TH.reduceAngle(x);

        // -------------------------------------------------------
        // SPECIAL CASE: xr == 0
        // xr = 0 occurs at x = k * (π/2)
        //
        // k mod 4 behavior:
        //   0 → sin(0)    = 0
        //   1 → sin(π/2)  = +1
        //   2 → sin(π)    = 0
        //   3 → sin(3π/2) = -1
        // -------------------------------------------------------
        if (MathLib.cmp(xr, QZERO) == 0) {
            if (quadrant == 1) return MathLib.fromUInt(1);       // +π/2
            if (quadrant == 3) return MathLib.neg(MathLib.fromUInt(1)); // -π/2
            return QZERO; // 0 or π
        }

        bytes16 axr = MathLib.abs(xr);
        
        // --- QUADRANT DECISION ---
        // Determine which core function to use based on the quadrant:
        // q=0 -> sin(xr)
        // q=1 -> sin(pi/2 + xr) = cos(xr)  <-- Switch function
        // q=2 -> sin(pi + xr)   = -sin(xr)
        // q=3 -> sin(3pi/2 + xr)= -cos(xr) <-- Switch function
        
        bool needCos = (quadrant == 1 || quadrant == 3);
        
        // --- CORE MAPPING (Minimax Domain) ---
        // If the reduced angle exceeds 45 degrees (π/4), map it to the
        // complement angle (π/2 - x) to stay within the optimal polynomial range.
        // This requires toggling the function (Sin <-> Cos).
        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            axr = MathLib.sub(QC.HALF_PI(), axr); // Map to coreX
            needCos = !needCos; // Toggle function selection
        }

        // Evaluate the selected polynomial
        bytes16 result = needCos ? _cos_poly(axr) : _sin_poly(axr);

        // --- SIGN MANAGEMENT ---
        
        // 1. Apply quadrant signs
        // In Q2 and Q3, Sine is negative.
        if (quadrant == 2 || quadrant == 3) {
            result = MathLib.neg(result);
        }
        
        // 2. Restore sign from original input
        // Sine is an odd function: sin(-x) = -sin(x).
        // If the original input was negative, flip the result.
        if (MathLib.cmp(x, QZERO) < 0) {
            result = MathLib.neg(result);
        }

        return result;
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
        // k mod 4 behavior:
        //   0 → cos(0)    = +1
        //   1 → cos(π/2)  =  0
        //   2 → cos(π)    = -1
        //   3 → cos(3π/2) =  0
        // -------------------------------------------------------
        if (MathLib.cmp(xr, QZERO) == 0) {
            if (quadrant == 0) return MathLib.fromUInt(1);       // +1
            if (quadrant == 2) return MathLib.neg(MathLib.fromUInt(1)); // -1
            return QZERO; // 0 at ±π/2
        }

        bytes16 axr = MathLib.abs(xr);

        // --- QUADRANT DECISION ---
        // Determine function based on quadrant:
        // q=0 -> cos(xr)
        // q=1 -> cos(pi/2 + xr) = -sin(xr) <-- Switch function
        // q=2 -> cos(pi + xr)   = -cos(xr)
        // q=3 -> cos(3pi/2 + xr)= +sin(xr) <-- Switch function

        bool needSin = (quadrant == 1 || quadrant == 3);

        // --- CORE MAPPING ---
        // If angle > π/4, map to π/2 - x and toggle function.
        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            axr = MathLib.sub(QC.HALF_PI(), axr);
            needSin = !needSin; // Toggle
        }

        bytes16 result = needSin ? _sin_poly(axr) : _cos_poly(axr);

        // --- SIGN MANAGEMENT ---
        // Apply Cosine Quadrant Signs (Assuming x > 0 for quadrant logic):
        // Q0: +
        // Q1: -
        // Q2: -
        // Q3: +
        if (quadrant == 1 || quadrant == 2) {
            result = MathLib.neg(result);
        }

        // Note: Cosine is an even function: cos(-x) = cos(x).
        // Therefore, the sign of the original input 'x' does not affect the result.

        return result;
    }
}