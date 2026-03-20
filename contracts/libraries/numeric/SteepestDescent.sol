// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { IObjectiveFunction } from "../../interfaces/IObjectiveFunction.sol";

/**
 * @title SteepestDescent
 * @notice General multivariable steepest descent solver using ABDK quad precision (bytes16).
 *         Follows the textbook-style steepest descent algorithm with step-halving
 *         and quadratic interpolation for step selection.
 *
 *         The objective function is supplied externally via IObjectiveFunction.
 */
library SteepestDescent {

    // ------------------------------------------------------------
    // Status Codes
    // ------------------------------------------------------------

    uint8 internal constant STATUS_SUCCESS               = 0;
    uint8 internal constant STATUS_ZERO_GRADIENT         = 1;
    uint8 internal constant STATUS_NO_LIKELY_IMPROVEMENT = 2;
    uint8 internal constant STATUS_MAX_ITER_EXCEEDED     = 3;

    struct Result {
        bytes16[] x;      // Final iterate
        bytes16   gx;     // Final objective value
        uint256   iters;  // Number of completed iterations
        uint8     status; // One of the STATUS_* constants
    }

    // ------------------------------------------------------------
    // Main Solver
    // ------------------------------------------------------------

    /**
     * @notice Approximate a local minimizer of g(x) using steepest descent.
     * @dev    The implementation follows the structure:
     *         - Evaluate g1 = g(x), z = grad(x), z0 = ||z||_2
     *         - Stop on zero gradient
     *         - Normalize z into a unit vector
     *         - Start with alpha3 = 1 and halve until improvement is found
     *         - Compute alpha2 = alpha3 / 2
     *         - Build quadratic interpolation terms h1, h2, h3
     *         - Compute alpha0 and compare g0 with g3
     *         - Update x using the better step
     *         - Stop when |g_new - g_old| < tol
     *
     * @param objective Objective contract implementing IObjectiveFunction
     * @param x0        Initial point in R^n
     * @param maxIter   Maximum number of iterations
     * @param tol       Tolerance used in stopping conditions
     * @return r        Solver result
     */
    function solve(IObjectiveFunction objective, bytes16[] memory x0, uint256 maxIter, bytes16 tol) internal view returns (Result memory r) {
        require(address(objective) != address(0), "SteepestDescent: objective is zero address");
        require(x0.length > 0, "SteepestDescent: empty initial point");
        require(maxIter > 0, "SteepestDescent: maxIter must be > 0");

        bytes16 zero = MathLib.fromUInt(0);
        bytes16 tolEff = MathLib.cmp(tol, zero) == 0 ? QC.DEFAULT_TOL() : tol;
        require(MathLib.cmp(tolEff, zero) > 0, "SteepestDescent: effective tol must be > 0");

        bytes16[] memory x = _copyVec(x0);

        // Scratch vector reused for all trial points g(x - alpha z)
        bytes16[] memory trialX = new bytes16[](x.length);

        bytes16 g1;
        bytes16 g2;
        bytes16 g3;
        bytes16 g0;
        bytes16 gSelected;

        bytes16[] memory z;
        bytes16 z0;

        bytes16 alpha;
        bytes16 alpha0;
        bytes16 alpha2;
        bytes16 alpha3;

        bytes16 h1;
        bytes16 h2;
        bytes16 h3;

        bytes16 one     = MathLib.fromUInt(1);
        bytes16 two     = MathLib.fromUInt(2);
        bytes16 half    = MathLib.div(one, two);
        bytes16 tolHalf = MathLib.mul(tolEff, half);

        for (uint256 k = 1; k <= maxIter; ++k) {
            // g1 = g(x), z = grad g(x), z0 = ||z||_2
            g1 = objective.g(x);
            z = objective.grad(x);
            require(z.length == x.length, "SteepestDescent: gradient dimension mismatch");

            z0 = _norm2(z);

            // zero gradient => possible local minimum
            if (MathLib.cmp(z0, zero) == 0) {
                r.x = x;
                r.gx = g1;
                r.iters = k - 1;
                r.status = STATUS_ZERO_GRADIENT;
                return r;
            }

            // Normalize gradient to unit vector, alpha3 = 1
            _normalizeInPlace(z, z0);
            alpha3 = one;

            _xMinusAlphaZInPlace(x, alpha3, z, trialX);
            g3 = objective.g(trialX);

            // alpha3 / 2 until improvement is found
            while (MathLib.cmp(g3, g1) >= 0) {
                alpha3 = MathLib.mul(alpha3, half);

                _xMinusAlphaZInPlace(x, alpha3, z, trialX);
                g3 = objective.g(trialX);

                if (MathLib.cmp(alpha3, tolHalf) < 0) {
                    r.x = x;
                    r.gx = g1;
                    r.iters = k - 1;
                    r.status = STATUS_NO_LIKELY_IMPROVEMENT;
                    return r;
                }
            }

            // alpha2 = alpha3 / 2, g2 = g(x - alpha2 z)
            alpha2 = MathLib.mul(alpha3, half);
            _xMinusAlphaZInPlace(x, alpha2, z, trialX);
            g2 = objective.g(trialX);

            // Interpolation terms
            h1 = MathLib.div(MathLib.sub(g2, g1), alpha2);
            h2 = MathLib.div(MathLib.sub(g3, g2), MathLib.sub(alpha3, alpha2));
            h3 = MathLib.div(MathLib.sub(h2, h1), alpha3);

            // alpha0 = 0.5 * (alpha2 - h1 / h3)
            // Fallback when h3 = 0 to avoid division by zero.
            if (MathLib.cmp(h3, zero) == 0) {
                alpha0 = alpha3;
            } else {
                alpha0 = MathLib.mul(
                    half,
                    MathLib.sub(alpha2, MathLib.div(h1, h3))
                );
            }

            _xMinusAlphaZInPlace(x, alpha0, z, trialX);
            g0 = objective.g(trialX);

            // Choose better of alpha0 and alpha3
            if (MathLib.cmp(g0, g3) < 0) {
                alpha = alpha0;
                gSelected = g0;
            } else {
                alpha = alpha3;
                gSelected = g3;
            }

            // x = x - alpha z
            _subScaledInPlace(x, alpha, z);

            // Stop if improvement is below tolerance
            if (MathLib.cmp(MathLib.abs(MathLib.sub(gSelected, g1)), tolEff) < 0) {
                r.x = x;
                r.gx = gSelected;
                r.iters = k;
                r.status = STATUS_SUCCESS;
                return r;
            }
        }

        // Stop if aximum iterations reached
        r.x = x;
        r.gx = objective.g(x);
        r.iters = maxIter;
        r.status = STATUS_MAX_ITER_EXCEEDED;
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice     Deep-copy a vector.
     * @param  src Source vector
     * @return dst Deep copy of src
     */
    function _copyVec(bytes16[] memory src) internal pure returns (bytes16[] memory dst) {
        dst = new bytes16[](src.length);
        for (uint256 i = 0; i < src.length; ++i) {
            dst[i] = src[i];
        }
    }

    /**
     * @notice      Compute the Euclidean norm ||v||_2 = sqrt(sum_i v_i^2).
     * @param  v    Input vector
     * @return norm Euclidean norm of v
     */
    function _norm2(bytes16[] memory v) internal pure returns (bytes16 norm) {
        bytes16 sum = MathLib.fromUInt(0);
        for (uint256 i = 0; i < v.length; ++i) {
            sum = MathLib.add(sum, MathLib.mul(v[i], v[i]));
        }
        norm = MathLib.sqrt(sum);
    }

    /**
     * @notice     Normalize a vector in place by dividing each component by the given norm.
     * @param v    Vector to normalize
     * @param norm Euclidean norm of v, must be nonzero
     */
    function _normalizeInPlace(bytes16[] memory v, bytes16 norm) internal pure {
        require(MathLib.cmp(norm, MathLib.fromUInt(0)) != 0, "SteepestDescent: division by zero");
        for (uint256 i = 0; i < v.length; ++i) {
            v[i] = MathLib.div(v[i], norm);
        }
    }

    /**
     * @notice      Compute out = x - alpha * z using a preallocated scratch vector.
     * @param x     Current point
     * @param alpha Step size
     * @param z     Direction vector
     * @param out   Preallocated output vector
     */
    function _xMinusAlphaZInPlace(bytes16[] memory x, bytes16 alpha, bytes16[] memory z, bytes16[] memory out) internal pure {
        require(x.length == z.length, "SteepestDescent: vector dimension mismatch");
        require(out.length == x.length, "SteepestDescent: output dimension mismatch");

        for (uint256 i = 0; i < x.length; ++i) {
            out[i] = MathLib.sub(x[i], MathLib.mul(alpha, z[i]));
        }
    }

    /**
     * @notice      Update x in place as x = x - alpha * z.
     * @param x     Current point, overwritten with the updated iterate
     * @param alpha Step size
     * @param z     Direction vector
     */
    function _subScaledInPlace(bytes16[] memory x, bytes16 alpha, bytes16[] memory z) internal pure {
        require(x.length == z.length, "SteepestDescent: vector dimension mismatch");

        for (uint256 i = 0; i < x.length; ++i) {
            x[i] = MathLib.sub(x[i], MathLib.mul(alpha, z[i]));
        }
    }
}