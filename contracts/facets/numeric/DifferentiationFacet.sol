// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Differentiation } from "../libraries/numeric/Differentiation.sol";

/**
 * @title DifferentiationFacet
 * @notice Diamond facet exposing numerical differentiation methods (forward, backward, central diff)
 *         implemented in the Differentiation library.
 *          - Uses IEEE-754 quadruple precision (bytes16)
 *          - Evaluates target functions via staticcall
 *          - Step size resolution handled internally
 */
contract DifferentiationFacet {

    // ------------------------------------------------------------
    // Forward Difference
    // ------------------------------------------------------------

    /**
     * @notice Approximate f'(x) using forward difference
     *         (f(x + h) − f(x)) / h
     */
    function forwardDiff(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16 dfdx) {
        return Differentiation.forwardDiff(target, selector, x, h);
    }

    // ------------------------------------------------------------
    // Backward Difference
    // ------------------------------------------------------------

    /**
     * @notice Approximate f'(x) using backward difference
     *         (f(x) − f(x − h)) / h
     */
    function backwardDiff(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16 dfdx) {
        return Differentiation.backwardDiff(target, selector, x, h);
    }

    // ------------------------------------------------------------
    // Centered Difference
    // ------------------------------------------------------------

    /**
     * @notice Approximate f'(x) using centered difference
     *         (f(x + h) − f(x − h)) / (2h)
     */
    function centeredDiff(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16 dfdx) {
        return Differentiation.centeredDiff(target, selector, x, h);
    }
}