// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { TrigonometricConstants as CS } from "./TrigonometricConstants.sol";
import { TrigonometryHelpers as TH } from "./TrigonometryHelpers.sol";

library TrigonometrySinCos {

    bytes16 internal constant QZERO = 0x00000000000000000000000000000000;

    // ===============================================================
    // === POLY EVAL HELPERS
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
        // Small angle optimization: |x| < ~1e-6
        bytes16 ax = MathLib.abs(x);
        if (MathLib.cmp(ax, 0x3F8A39EF35793C767300000000000000) < 0) {
            return x;
        }

        (bytes16 xr, uint8 quadrant) = TH.reduceAngle(x);

        // Special cases: 0, pi/2, pi, 3pi/2
        if (MathLib.cmp(xr, QZERO) == 0) {
            if (quadrant == 1) return MathLib.fromUInt(1);
            if (quadrant == 3) return MathLib.neg(MathLib.fromUInt(1));
            return QZERO;
        }

        bytes16 axr = MathLib.abs(xr);
        
        // --- QUADRANT DECISION ---
        // q=0 -> sin(xr)
        // q=1 -> sin(pi/2 + xr) = cos(xr)  (Switch function)
        // q=2 -> sin(pi + xr)   = -sin(xr)
        // q=3 -> sin(3pi/2 + xr)= -cos(xr) (Switch function)
        bool needCos = (quadrant == 1 || quadrant == 3);
        
        // --- CORE MAPPING ---
        // If angle > 45 degrees, map to (90 - angle) and switch function again.
        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            axr = MathLib.sub(QC.HALF_PI(), axr);
            needCos = !needCos; // Toggle
        }

        // Evaluate the selected polynomial
        bytes16 result = needCos ? _cos_poly(axr) : _sin_poly(axr);

        // --- SIGN MANAGEMENT ---
        // Apply quadrant signs
        // In Q2 and Q3, Sine is negative.
        if (quadrant == 2 || quadrant == 3) {
            result = MathLib.neg(result);
        }

        // Restore sign from original input
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

        // Special cases
        if (MathLib.cmp(xr, QZERO) == 0) {
            if (quadrant == 0) return MathLib.fromUInt(1);
            if (quadrant == 2) return MathLib.neg(MathLib.fromUInt(1));
            return QZERO;
        }

        bytes16 axr = MathLib.abs(xr);

        // --- QUADRANT DECISION ---
        // q=0 -> cos(xr)
        // q=1 -> cos(pi/2 + xr) = -sin(xr) (Switch function)
        // q=2 -> cos(pi + xr)   = -cos(xr)
        // q=3 -> cos(3pi/2 + xr)= +sin(xr) (Switch function)
        bool needSin = (quadrant == 1 || quadrant == 3);

        // --- CORE MAPPING ---
        if (MathLib.cmp(axr, QC.QUARTER_PI()) > 0) {
            axr = MathLib.sub(QC.HALF_PI(), axr);
            needSin = !needSin; // Toggle
        }

        bytes16 result = needSin ? _sin_poly(axr) : _cos_poly(axr);

        // --- SIGN MANAGEMENT ---
        if (quadrant == 1 || quadrant == 2) {
            result = MathLib.neg(result);
        }

        return result;
    }
}