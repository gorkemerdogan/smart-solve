// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "abdk-libraries-solidity/ABDKMathQuad.sol";

import { RootFinding } from "../../libraries/numeric/RootFinding.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title RootFindingFacet
 * @notice Diamond facet exposing high-precision root-finding (bytes16 ABDK).
 * @dev Thin ABI layer: pulls tol/maxIter from LibNumericConfig and forwards
 *     to the pure RootFinding library. Returns RootFinding.RootResult.
 */
contract RootFindingFacet {
    using RootFinding for *;

    /// @dev Read config; if missing/zero, apply safe academic defaults.
    function _readCfg() internal view returns (bytes16 tol, uint256 maxIter) {
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();

        bytes16 minTol = cfg.minTol;
        tol = cfg.tol;
        maxIter = cfg.maxIter;

        if (minTol == bytes16(0)) {
            minTol = 0x3FC063E11D9235650000000000000000; // 1e-15
        }

        if (tol == bytes16(0)) {
            tol = 0x3FD3C1F07C1F07C1F07C1F07C1F07C20; // 1e-12
        }

        tol = (ABDKMathQuad.cmp(tol, minTol) < 0) ? minTol : tol; // tol must be at least equal to minTol

        if (maxIter == 0) {
            maxIter = 200;
        }
    }

    /**
     * @notice Bisection on [a,b] for target f(x).
     * @param target Address exposing f(bytes16) -> bytes16
     * @param fSelector Function selector for f
     * @param a Left endpoint
     * @param b Right endpoint
     */
    function bisection(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b
    ) external view returns (RootFinding.RootResult memory result) {
        (bytes16 tol, uint256 maxIter) = _readCfg();
        return RootFinding.bisection(target, fSelector, a, b, tol, maxIter);
    }

    /**
     * @notice Newton–Raphson with analytic derivative.
     * @param target Address exposing f(bytes16) -> bytes16
     * @param fSelector Function selector for f
     * @param dfTarget Address exposing f'(bytes16) -> bytes16
     * @param dfSelector Function selector for f'
     * @param x0 Initial guess
     */
    function newton(
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
     * @notice Secant method (derivative-free) with two initial points.
     * @param target Address exposing f(bytes16) -> bytes16
     * @param fSelector Function selector for f
     * @param x0 First starting point
     * @param x1 Second starting point
     */
    function secant(
        address target,
        bytes4 fSelector,
        bytes16 x0,
        bytes16 x1
    ) external view returns (RootFinding.RootResult memory result) {
        (bytes16 tol, uint256 maxIter) = _readCfg();
        return RootFinding.secant(target, fSelector, x0, x1, tol, maxIter);
    }
}