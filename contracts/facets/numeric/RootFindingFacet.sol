// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RootFinding } from "../../libraries/numeric/RootFinding.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";
import { QuadConstants } from "../../libraries/QuadConstants.sol";
import { MathLib } from "../../libraries/MathLib.sol";

/**
 * @title  RootFindingFacet
 * @notice Diamond facet exposing high-precision root-finding (bytes16 ABDK).
 *         Pulls tol/maxIter from LibNumericConfig and forwards
 *         to the pure RootFinding library. Returns RootFinding.RootResult.
 */
contract RootFindingFacet {
    using RootFinding for *;

    /// Loads configuration values, applying fallback defaults if unset.
    function _readCfg() internal view returns (bytes16 tol, uint256 maxIter) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();

        bytes16 minTol = cfg.minTol;
        tol = cfg.tol;
        maxIter = cfg.maxIter;

        if (minTol == bytes16(0)) {
            minTol = QuadConstants.EPS_1e15();
        }

        if (tol == bytes16(0)) {
            tol = QuadConstants.EPS_1e12();
        }

        tol = (MathLib.cmp(tol, minTol) < 0) ? minTol : tol; // tol must be at least equal to minTol

        if (maxIter == 0) {
            maxIter = 200;
        }
    }

    /**
     * @notice Computes a root using the bisection method on [a, b].
     *         Requires f(a) and f(b) to have opposite signs. Endpoints are normalized
     *         to ensure left ≤ right. Convergence is triggered when either:
     *           - |f(mid)| ≤ tol, or
     *           - interval width / 2 ≤ tol.
     * @param  target     Contract exposing f(bytes16) -> bytes16
     * @param  fSelector  Selector for f(bytes16)
     * @param  a          First endpoint of the interval
     * @param  b          Second endpoint of the interval
     * @return RootResult Struct containing root approximation and metadata
     */
    function rootFindingBisection(address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b
    ) external view returns (RootFinding.RootResult memory result) {
        (bytes16 tol, uint256 maxIter) = _readCfg();
        return RootFinding.bisection(target, fSelector, a, b, tol, maxIter);
    }

    /**
     * @notice Computes a root using the Newton–Raphson method with analytic derivative.
     *         Requires non-zero derivative at each iterate. Convergence is based on:
     *           - |f(xNext)| ≤ tol, or
     *           - |xNext − x| ≤ tol.
     * @param  target     Contract exposing f(bytes16) -> bytes16
     * @param  fSelector  Selector for f(bytes16)
     * @param  dfTarget   Contract exposing f'(bytes16)
     * @param  dfSelector Selector for f'(bytes16)
     * @param  x0         Initial guess
     * @return RootResult Struct with the final iterate, iteration count, and convergence flag
     */
    function rootFindingNewton(
        address target,
        bytes4 fSelector,
        address dfTarget,
        bytes4 dfSelector,
        bytes16 x0
    ) external view returns (RootFinding.RootResult memory result) {
        (bytes16 tol, uint256 maxIter) = _readCfg();
        return RootFinding.newton(target, fSelector, dfTarget, dfSelector, x0, tol, maxIter);
    }

    /**
     * @notice Computes a root using the secant method (derivative-free).
     *         Uses two initial values and updates via the secant update formula.
     *         Reverts if consecutive function values yield zero slope. Converges when:
     *           - |f(xNext)| ≤ tol, or
     *           - |xNext − x| ≤ tol.
     * @param  target     Contract exposing f(bytes16) -> bytes16
     * @param  fSelector  Selector for f(bytes16)
     * @param  x0         First initial point
     * @param  x1         Second initial point
     * @return RootResult Struct containing the resulting approximation
     */
    function rootFindingSecant(
        address target,
        bytes4 fSelector,
        bytes16 x0,
        bytes16 x1
    ) external view returns (RootFinding.RootResult memory result) {
        (bytes16 tol, uint256 maxIter) = _readCfg();
        return RootFinding.secant(target, fSelector, x0, x1, tol, maxIter);
    }
}