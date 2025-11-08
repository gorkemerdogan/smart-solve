// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RootFinding } from "../../libraries/numeric/RootFinding.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title RootFindingFacet
 * @notice Diamond facet exposing on-chain root-finding algorithms in quadruple precision.
 * @dev Wraps RootFinding library and retrieves tolerance (eps) and iteration limit (maxIter)
 *      from global numeric configuration via LibNumericConfig.
 *
 *      This facet follows the same pattern as PolynomialFacet:
 *        - Thin wrapper layer, only ABI exposure.
 *        - Pure computation in RootFinding library.
 *        - View-only operations (no state changes).
 *
 *      Academic Reference:
 *      - Implements canonical numerical root-finding algorithms (Bisection, Newton–Raphson, Secant)
 *        as defined in:
 *          * R. L. Burden and J. D. Faires, *Numerical Analysis*, 10th Ed., 2016.
 *          * Chapra & Canale, *Numerical Methods for Engineers*, 8th Ed., 2021.
 */
contract RootFindingFacet {
    using RootFinding for *;

    // ------------------------------------------------------------------------
    // STRUCTS
    // ------------------------------------------------------------------------

    /**
     * @notice Result container for any root-finding algorithm.
     * @dev Used as return type for all facet methods.
     * @param root Approximated root value in IEEE-754 quadruple precision (bytes16).
     * @param iterations Number of iterations performed before termination.
     * @param converged Boolean indicating convergence within global tolerance.
     * @param fAtRoot Function value f(x) evaluated at the final root approximation.
     */
    struct RootResult {
        bytes16 root;
        uint256 iterations;
        bool converged;
        bytes16 fAtRoot;
    }

    // ------------------------------------------------------------------------
    // BISSECTION METHOD
    // ------------------------------------------------------------------------

    /**
     * @notice Finds a root of f(x) within [a, b] using the Bisection method.
     * @dev Guaranteed convergence if f is continuous and f(a)·f(b) < 0.
     *
     * The algorithm repeatedly halves the interval [a, b] until:
     *  - The function value |f(mid)| ≤ eps, or
     *  - The half-interval width (b − a)/2 ≤ eps.
     *
     * @param target Address exposing f(bytes16) → bytes16.
     * @param fSelector Function selector for f(bytes16).
     * @param a Left interval endpoint (ABDK bytes16).
     * @param b Right interval endpoint (ABDK bytes16).
     * @return result Struct containing:
     *         - root: final mid-point approximation,
     *         - iterations: iteration count,
     *         - converged: true if tolerance satisfied,
     *         - fAtRoot: value f(root).
     */
    function bisection(address target, bytes4 fSelector, bytes16 a, bytes16 b) external view returns (RootResult memory result) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        (bytes16 root, uint256 iters, bool conv, bytes16 fval) =
            RootFinding.bisection(target, fSelector, a, b, cfg.eps, cfg.maxIter);
        result = RootResult(root, iters, conv, fval);
    }

    // ------------------------------------------------------------------------
    // NEWTON–RAPHSON METHOD
    // ------------------------------------------------------------------------

    /**
     * @notice Approximates a root of f(x) using the Newton–Raphson iterative method.
     * @dev Requires both f(x) and f′(x). Uses quadratic convergence near the root
     *      when derivative is well-conditioned.
     *
     * Update rule:
     *      xₖ₊₁ = xₖ − f(xₖ) / f′(xₖ)
     *
     * Termination conditions:
     *      |f(xₖ₊₁)| ≤ eps  or  |xₖ₊₁ − xₖ| ≤ eps
     *
     * @param target Address exposing f(bytes16) → bytes16.
     * @param fSelector Function selector for f(bytes16).
     * @param dfTarget Address exposing f′(bytes16) → bytes16.
     * @param dfSelector Function selector for f′(bytes16).
     * @param x0 Initial guess for the root (ABDK bytes16).
     * @return result Struct with:
     *         - root: final root approximation,
     *         - iterations: number of iterations performed,
     *         - converged: true if tolerance satisfied,
     *         - fAtRoot: final f(x) value.
     */
    function newton(address target, bytes4 fSelector, address dfTarget, bytes4 dfSelector, bytes16 x0) external view returns (RootResult memory result) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        (bytes16 root, uint256 iters, bool conv, bytes16 fval) =
            RootFinding.newton(target, fSelector, dfTarget, dfSelector, x0, cfg.eps, cfg.maxIter);
        result = RootResult(root, iters, conv, fval);
    }

    // ------------------------------------------------------------------------
    // SECANT METHOD
    // ------------------------------------------------------------------------

    /**
     * @notice Approximates a root of f(x) using the Secant method.
     * @dev Two-point derivative-free method. Uses successive secant lines to
     *      approximate the slope of f(x). Converges superlinearly under mild conditions.
     *
     * Update rule:
     *      xₖ₊₁ = xₖ − f(xₖ) * (xₖ − xₖ₋₁) / (f(xₖ) − f(xₖ₋₁))
     *
     * @param target Address exposing f(bytes16) → bytes16.
     * @param fSelector Function selector for f(bytes16).
     * @param x0 First starting point (ABDK bytes16).
     * @param x1 Second starting point (ABDK bytes16).
     * @return result Struct with:
     *         - root: final root approximation,
     *         - iterations: number of iterations performed,
     *         - converged: true if tolerance satisfied,
     *         - fAtRoot: f(root) at convergence.
     */
    function secant(address target, bytes4 fSelector, bytes16 x0, bytes16 x1) external view returns (RootResult memory result) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        (bytes16 root, uint256 iters, bool conv, bytes16 fval) =
            RootFinding.secant(target, fSelector, x0, x1, cfg.eps, cfg.maxIter);
        result = RootResult(root, iters, conv, fval);
    }
}