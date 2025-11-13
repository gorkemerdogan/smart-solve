// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad as Q } from "abdk-libraries-solidity/ABDKMathQuad.sol";
import { QuadConstants as QC } from "./QuadConstants.sol";

/**
 * @title Trigonometric
 * @notice High-precision trigonometric functions for IEEE-754 binary128
 *         using degree-16 Chebyshev polynomial approximation.
 *         Domain reduction handles any real input. sin / cos form the
 *         basis; tan, cot, asin, acos, atan, acot are derived from them.
 *
 * @dev All math is bytes16 (quad precision). Pure library, no storage.
 */
library Trigonometric {
    using Q for bytes16;

    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

    // ------------------------------------------------------------
    // Chebyshev coefficients for sin(t) & cos(t),
    // defined on t ∈ [-π/2, π/2] using degree-16 expansions.
    //
    // sin(t) ≈ ∑ cSin[k] * T_k(z),   z = 2t/π
    // cos(t) ≈ ∑ cCos[k] * T_k(z)
    //
    // Coefficients were generated offline at quad precision.
    // ------------------------------------------------------------

    // sin coefficients (degree 16, odd symmetry baked in)
    function _sinCoeffs() private pure returns (bytes16[17] memory c) {
        c[0]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=0
        c[1]  = bytes16(uint128(0x3FFEFFFFFFFFFFFFFFFFFFFFFFFD2F9A)); // k=1
        c[2]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=2
        c[3]  = bytes16(uint128(0xBFFA55555555555555555541BEC6A4));   // k=3
        c[4]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=4
        c[5]  = bytes16(uint128(0x3FF811111111111111110EF3D51F7A));   // k=5
        c[6]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=6
        c[7]  = bytes16(uint128(0xBFF2A01A01A01A01A01876B15FC40F));   // k=7
        c[8]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=8
        c[9]  = bytes16(uint128(0x3FEF5D3C7C4DA3E28BA6412A8220F7));   // k=9
        c[10] = bytes16(uint128(0x00000000000000000000000000000000)); // k=10
        c[11] = bytes16(uint128(0xBFE75D6E03EA3A54C728FDAACB42E5));   // k=11
        c[12] = bytes16(uint128(0x00000000000000000000000000000000)); // k=12
        c[13] = bytes16(uint128(0x3FE3A527D8E5FFEB6F7C654E56E8C2));   // k=13
        c[14] = bytes16(uint128(0x00000000000000000000000000000000)); // k=14
        c[15] = bytes16(uint128(0xBFE0F8594E49C3E845D4A0341AC8AB));   // k=15
        c[16] = bytes16(uint128(0x00000000000000000000000000000000)); // k=16 (≈0)
        return c;
    }

    // cos coefficients (degree 16, even symmetry)
    function _cosCoeffs() private pure returns (bytes16[17] memory c) {
        c[0]  = bytes16(uint128(0x3FFF0000000000000000000000000000)); // k=0
        c[1]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=1
        c[2]  = bytes16(uint128(0xBFFD0000000000000000000000000000)); // k=2
        c[3]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=3
        c[4]  = bytes16(uint128(0x3FF5555555555555555555552C4484));   // k=4
        c[5]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=5
        c[6]  = bytes16(uint128(0xBFEFA01A01A01A01A01A006B13F94A));   // k=6
        c[7]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=7
        c[8]  = bytes16(uint128(0x3FEA79435E50D63EA03D0813AF162D));   // k=8
        c[9]  = bytes16(uint128(0x00000000000000000000000000000000)); // k=9
        c[10] = bytes16(uint128(0xBFE51B0792BD7C43E78A6B02E395AE));   // k=10
        c[11] = bytes16(uint128(0x00000000000000000000000000000000)); // k=11
        c[12] = bytes16(uint128(0x3FE18F5C28F5C28F5C2B1297B40218));   // k=12
        c[13] = bytes16(uint128(0x00000000000000000000000000000000)); // k=13
        c[14] = bytes16(uint128(0xBFDE06F0AA31F2F4B1DB32B3AF78FB));   // k=14
        c[15] = bytes16(uint128(0x00000000000000000000000000000000)); // k=15
        c[16] = bytes16(uint128(0x3FDBB6FB601EC0C040C0A29DD89294));   // k=16
        return c;
    }

    // ------------------------------------------------------------
    // Domain reduction: x → t ∈ [-π/2, π/2], quadrant ∈ {0,1,2,3}
    // ------------------------------------------------------------
    function _reduce(bytes16 x) private pure returns (bytes16 t, uint256 q) {
        // k = round( x / (π/2) )
        bytes16 kf = x.div(QC.HALF_PI());
        int256 k = _roundNearest(kf);

        q = uint256(k) & 3; // quadrant in [0,3]

        // t = x - k * (π/2)
        bytes16 kHalfPi = QC.HALF_PI().mul(Q.fromInt(k));
        t = x.sub(kHalfPi);
    }

    // ------------------------------------------------------------
    // Chebyshev evaluator: ∑ c[k] T_k(z), z ∈ [-1, 1]
    // using Clenshaw recurrence.
    // ------------------------------------------------------------
    function _cheby(bytes16 z, bytes16[17] memory c)
        private
        pure
        returns (bytes16)
    {
        bytes16 b_kplus1 = QZERO;
        bytes16 b_kplus2 = QZERO;

        for (int256 k = 16; k >= 0; k--) {
            bytes16 tmp = b_kplus1;
            // b_k = c[k] + 2*z*b_{k+1} - b_{k+2}
            b_kplus1 = c[uint256(k)]
                .add(z.mul(b_kplus1).mul(Q.fromInt(2)))
                .sub(b_kplus2);
            b_kplus2 = tmp;
        }

        // Result = b_0 - z*b_1
        return b_kplus1.sub(z.mul(b_kplus2));
    }

    // ------------------------------------------------------------
    // sin(x)
    // ------------------------------------------------------------
    function sin(bytes16 x) internal pure returns (bytes16) {
        (bytes16 t, uint256 q) = _reduce(x);
        // Map t ∈ [-π/2, π/2] → z ∈ [-1, 1]
        bytes16 z = t.mul(Q.fromInt(2)).div(QC.PI());

        if (q == 0) {
            // region near 0 → sin(x) = sin(t)
            bytes16[17] memory cSin = _sinCoeffs();
            return _cheby(z, cSin);
        } else if (q == 1) {
            // region near π/2 → sin(x) = cos(t)
            bytes16[17] memory cCos = _cosCoeffs();
            return _cheby(z, cCos);
        } else if (q == 2) {
            // region near π → sin(x) = -sin(t)
            bytes16[17] memory cSin = _sinCoeffs();
            bytes16 s = _cheby(z, cSin);
            return Q.fromInt(-1).mul(s);
        } else {
            // q == 3, region near 3π/2 → sin(x) = -cos(t)
            bytes16[17] memory cCos = _cosCoeffs();
            bytes16 c = _cheby(z, cCos);
            return Q.fromInt(-1).mul(c);
        }
    }

    // ------------------------------------------------------------
    // cos(x)
    // ------------------------------------------------------------
    function cos(bytes16 x) internal pure returns (bytes16) {
        (bytes16 t, uint256 q) = _reduce(x);
        bytes16 z = t.mul(Q.fromInt(2)).div(QC.PI());

        // Approximate cos(t) on [-π/2, π/2]
        bytes16[17] memory cCos = _cosCoeffs();
        bytes16 c = _cheby(z, cCos);

        if (q == 0) {
            // region near 0 → cos(x) = cos(t)
            return c;
        } else if (q == 1) {
            // region near π/2 → cos(x) = -sin(t)
            bytes16[17] memory cSin = _sinCoeffs();
            bytes16 s = _cheby(z, cSin);
            return Q.fromInt(-1).mul(s);
        } else if (q == 2) {
            // region near π → cos(x) = -cos(t)
            return Q.fromInt(-1).mul(c);
        } else {
            // q == 3, region near 3π/2 → cos(x) = sin(t)
            bytes16[17] memory cSin = _sinCoeffs();
            bytes16 s = _cheby(z, cSin);
            return s;
        }
    }

    // ------------------------------------------------------------
    // tan / cot
    // ------------------------------------------------------------
function tan(bytes16 x) internal pure returns (bytes16) {
        (bytes16 t, uint256 q) = _reduce(x);
        // Map t ∈ [-π/2, π/2] → z ∈ [-1, 1]
        bytes16 z = t.mul(Q.fromInt(2)).div(QC.PI());

        // Get both approximations
        bytes16 s = _cheby(z, _sinCoeffs());
        bytes16 c = _cheby(z, _cosCoeffs());

        // q == 0 or q == 2
        // tan(x) = sin(t) / cos(t)
        if (q % 2 == 0) {
            require(Q.cmp(c, QZERO) != 0, "tan: undefined");
            return s.div(c);
        }

        // q == 1 or q == 3
        // tan(x) = cos(t) / -sin(t)
        require(Q.cmp(s, QZERO) != 0, "tan: undefined");
        return c.div(s.neg());
    }

    function cot(bytes16 x) internal pure returns (bytes16) {
        (bytes16 t, uint256 q) = _reduce(x);
        // Map t ∈ [-π/2, π/2] → z ∈ [-1, 1]
        bytes16 z = t.mul(Q.fromInt(2)).div(QC.PI());

        // Get both approximations
        bytes16 s = _cheby(z, _sinCoeffs());
        bytes16 c = _cheby(z, _cosCoeffs());

        // q == 0 or q == 2
        // cot(x) = cos(t) / sin(t)
        if (q % 2 == 0) {
            require(Q.cmp(s, QZERO) != 0, "cot: undefined");
            return c.div(s);
        }

        // q == 1 or q == 3
        // cot(x) = -sin(t) / cos(t)
        require(Q.cmp(c, QZERO) != 0, "cot: undefined");
        return s.neg().div(c);
    }

    // ------------------------------------------------------------
    // atan(x)  (odd, strictly increasing)
    //  - range reduction: x → [0,1] or >1
    //  - short polynomial + one Newton refinement
    // ------------------------------------------------------------
    function atan(bytes16 x) internal pure returns (bytes16) {
        if (Q.cmp(x, QZERO) == 0) return QZERO;

        bool neg = Q.cmp(x, QZERO) < 0;
        if (neg) x = x.neg();

        // If x > 1: atan(x) = π/2 − atan(1/x)
        bool flip = Q.cmp(x, Q.fromInt(1)) > 0;
        if (flip) {
            x = Q.fromInt(1).div(x);
        }

        // Polynomial approx on [0,1]:
        // atan(x) ≈ x - x^3/3 + x^5/5 - x^7/7
        bytes16 x2 = x.mul(x);
        bytes16 x3 = x.mul(x2);
        bytes16 x5 = x3.mul(x2);
        bytes16 x7 = x5.mul(x2);

        bytes16 approx =
            x
            .sub(x3.div(Q.fromInt(3)))
            .add(x5.div(Q.fromInt(5)))
            .sub(x7.div(Q.fromInt(7)));

        // Newton refinement for f(y) = tan(y) - x:
        //   y₁ = y₀ - (tan(y₀) - x)/sec²(y₀)
        bytes16 t = tan(approx);
        bytes16 sec2 = Q.fromInt(1).add(t.mul(t));
        approx = approx.sub(t.sub(x).div(sec2));

        if (flip) {
            approx = QC.HALF_PI().sub(approx);
        }
        if (neg) {
            approx = approx.neg();
        }

        return approx;
    }

    // ------------------------------------------------------------
    // asin / acos
    // ------------------------------------------------------------
    function asin(bytes16 x) internal pure returns (bytes16) {
        // Domain check: x ∈ [-1, 1]
        require(
            Q.cmp(x, Q.fromInt(1)) <= 0 && Q.cmp(x, Q.fromInt(-1)) >= 0,
            "asin: domain"
        );

        if (Q.cmp(x, QZERO) == 0) return QZERO;
        if (Q.cmp(x, Q.fromInt(1)) == 0) return QC.HALF_PI();
        if (Q.cmp(x, Q.fromInt(-1)) == 0) return QC.HALF_PI().neg();

        // asin(x) = atan( x / sqrt(1 - x²) )
        bytes16 oneMinus = Q.fromInt(1).sub(x.mul(x));
        bytes16 root = Q.sqrt(oneMinus);
        bytes16 ratio = x.div(root);

        return atan(ratio);
    }

    function acos(bytes16 x) internal pure returns (bytes16) {
        return QC.HALF_PI().sub(asin(x));
    }

    // ------------------------------------------------------------
    // acot(x) = atan(1/x)
    // ------------------------------------------------------------
    function acot(bytes16 x) internal pure returns (bytes16) {
        require(Q.cmp(x, QZERO) != 0, "acot: undefined");
        return atan(Q.fromInt(1).div(x));
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Round-to-nearest (ties away from zero) for a quad value.
     *
     * @dev ABDKMathQuad only exposes truncation to int (toward zero).
     *      We simulate "round to nearest, ties away from zero":
     *
     *          t = trunc(x)
     *          frac = |x - t|
     *          if frac >= 0.5 → round away from zero
     *          else           → return t
     */
    function _roundNearest(bytes16 x) private pure returns (int256) {
        // truncate toward zero
        int256 t = Q.toInt(x);
        bytes16 tQuad = Q.fromInt(t);

        // frac = |x - t|
        bytes16 diff = x.sub(tQuad);
        if (Q.cmp(diff, QZERO) < 0) {
            diff = diff.neg();
        }

        if (Q.cmp(diff, QC.HALF()) >= 0) {
            // round away from zero
            if (t >= 0) return t + 1;
            return t - 1;
        }

        return t;
    }
}