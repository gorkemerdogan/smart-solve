// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Integration } from "../../libraries/numeric/Integration.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title  IntegrationFacet
 * @notice Diamond facet exposing high-precision numerical integration rules
 *         (Trapezoidal, Simpson 1/3, Simpson 3/8) implemented in the
 *         Integration library with ABDKMathQuad (bytes16).
 *          
 *         The integrand is provided as (target, selector): f(bytes16) -> bytes16.
 *         This facet reuses LibNumericConfig.maxIter as the default number of
 *         sub-intervals n for all rules.
 *
 *  - TODO: LibNumericConfig.tol / minTol are not used by the fixed-grid rules yet,
 *    but kept in storage for future adaptive schemes.
 *
 *         For each rule exposed two variants:
 *           1) integrateXXX(...)            → uses n from LibNumericConfig.maxIter
 *           2) integrateXXXWithN(..., n)    → uses the explicit n provided
 *
 *         The low-level Integration library is expected to work purely with in-memory
 *         variables (no storage in the library).
 */
contract IntegrationFacet {

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @return n the default grid size n taken from LibNumericConfig.
     *         If maxIter is unset (0), falls back to 102.
     *         This function guarantees n is a multiple of 6
     *         (and > 0) to be universally valid for all Simpson rules.
     */
    function _defaultN() internal view returns (uint256 n) {
        n = LibNumericConfig.getMaxIter();

        // Handle fallback if maxIter is uninitialized
        if (n == 0) {
            // Tuned for both Simpson's rules
            n = 102;
        }

        // Handle edge case where the original n was < 6.
        // Return a valid n > 0. The smallest valid n for all rules is 6.
        n = n - (n % 6);

        // Handle edge case where the original n was < 6.
        // Return a valid n > 0. The smallest valid n for all rules is 6.
        if (n == 0) {
            n = 6;
        }
    }

    // ------------------------------------------------------------
    // Trapezoidal Rule
    // ------------------------------------------------------------

    /**
     * @notice Computes the composite trapezoidal rule on a uniform grid with default N.
     *         All computation is done in memory; no storage access occurs.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @return I Approximate integral value encoded as bytes16
     */
    function integrateTrapezoidal(address target, bytes4 fSelector, bytes16 a, bytes16 b) external view returns (bytes16 I) {
        uint256 n = _defaultN();
        return Integration.trapezoidal(target, fSelector, a, b, n);
    }

    /**
     * @notice Computes the composite trapezoidal rule on a uniform grid.
     *         Requires n > 0 and b >= a. Performs O(n) staticcalls to the integrand.
     *         All computation is done in memory; no storage access occurs.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @param  n Number of subintervals (must be > 0)
     * @return I Approximate integral value encoded as bytes16
     */
    function integrateTrapezoidalWithN(address target, bytes4 fSelector, bytes16 a, bytes16 b, uint256 n) external view returns (bytes16 I) {
        return Integration.trapezoidal(target, fSelector, a, b, n);
    }

    // ------------------------------------------------------------
    // Simpson 1/3 Rule
    // ------------------------------------------------------------

    /**
     * @notice Computes the composite Simpson’s 1/3 rule on a uniform grid with default N.
     *         All computation occurs in memory; no storage is modified.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @return I Approximate integral value encoded as bytes16
     */
    function integrateSimpson13(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b
    ) external view returns (bytes16 I) {
        uint256 n = _defaultN();
        return Integration.simpson13(target, fSelector, a, b, n);
    }

    /**
     * @notice Computes the composite Simpson’s 1/3 rule on a uniform grid.
     *         Requires n > 0, n even, and b >= a. Performs O(n) staticcalls to the integrand.
     *         All computation occurs in memory; no storage is modified.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @param  n Number of subintervals (must be even and > 0)
     * @return I Approximate integral value encoded as bytes16
     */
    function integrateSimpson13WithN(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16 I) {
        return Integration.simpson13(target, fSelector, a, b, n);
    }

    // ------------------------------------------------------------
    // Simpson 3/8 Rule
    // ------------------------------------------------------------

    /**
     * @notice Computes the composite Simpson’s 3/8 rule on a uniform grid with default N.
     *         to the integrand. All arithmetic is performed in memory.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @return I Approximate integral value encoded as bytes16
     */
    function integrateSimpson38(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b
    ) external view returns (bytes16 I) {
        uint256 n = _defaultN();
        return Integration.simpson38(target, fSelector, a, b, n);
    }

    /**
     * @notice Computes the composite Simpson’s 3/8 rule on a uniform grid.
     *         Requires n > 0, n divisible by 3, and b >= a. Performs O(n) staticcalls
     *         to the integrand. All arithmetic is performed in memory.
     * @param  target Contract exposing f(bytes16) -> bytes16
     * @param  fSelector Selector of f(bytes16) in 'target'
     * @param  a Lower integration bound (bytes16)
     * @param  b Upper integration bound (bytes16)
     * @param  n Number of subintervals (must be > 0 and divisible by 3)
     * @return I Approximate integral value encoded as bytes16
     */
    function integrateSimpson38WithN(
        address target,
        bytes4 fSelector,
        bytes16 a,
        bytes16 b,
        uint256 n
    ) external view returns (bytes16 I) {
        return Integration.simpson38(target, fSelector, a, b, n);
    }

    /**
     * @notice Returns the current default grid size n used by the auto-n variants.
     */
    function getDefaultIntegrationN() external view returns (uint256 n) {
        n = _defaultN();
    }
}