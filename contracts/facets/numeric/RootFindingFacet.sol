// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { RootFinding } from "../../libraries/numeric/RootFinding.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title RootFindingFacet
 * @notice Diamond facet exposing high-precision root-finding (bytes16 ABDK).
 * @dev Thin ABI layer: pulls eps/maxIter from LibNumericConfig and forwards
 *     to the pure RootFinding library. Returns RootFinding.RootResult.
 */
contract RootFindingFacet {
    using RootFinding for *;

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
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        return RootFinding.bisection(target, fSelector, a, b, cfg.eps, cfg.maxIter);
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
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        return RootFinding.newton(target, fSelector, dfTarget, dfSelector, x0, cfg.eps, cfg.maxIter);
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
        LibNumericConfig.NumericConfig storage cfg = LibNumericConfig.cfg();
        return RootFinding.secant(target, fSelector, x0, x1, cfg.eps, cfg.maxIter);
    }
}