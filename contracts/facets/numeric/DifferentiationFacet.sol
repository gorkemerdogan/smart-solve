// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Differentiation } from "../libraries/numeric/Differentiation.sol";

/**
 * @title  DifferentiationFacet
 * @notice Diamond facet that exposes high-precision numerical differentiation
 *         methods (forward, backward, centered differences) as external calls.
 *
 * @dev    This facet is a thin external wrapper around the 'Differentiation' library.
 *         It forwards calls to library functions invoking the target function via 'staticcall'.
 *         All computations use IEEE-754 quadruple precision (binary128) represented as 'bytes16'.
 *          - Function pointers are represented by '(address target, bytes4 selector)'
 *            to allow differentiation of arbitrary pure/view functions.
 *
 *         Limitations:
 *          - Accuracy depends on the choice of step size 'h' and the smoothness of f(x).
 *          - Susceptible to truncation and floating-point rounding errors inherent to finite-difference methods.
 *
 * @custom:gas Gas usage is dominated by the external 'staticcall' to the target function and
 *             repeated evaluations required by the chosen difference scheme.
 */
contract DifferentiationFacet {

    // ------------------------------------------------------------
    // Forward Difference
    // ------------------------------------------------------------

    /**
     * @notice Approximates the derivative f'(x) using the Forward Difference method.
     *         Formula: f'(x) ≈ (f(x + h) - f(x)) / h
     *         Accuracy: First-order O(h).
     *
     * @param target   The address of the contract hosting f.
     * @param selector The function selector for f.
     * @param x        The point at which to differentiate.
     * @param h        The step size. Pass 0 (QZERO) to use defaults/config.
     * @return dfdx    The approximate derivative.
     */
    function forwardDiff(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16 dfdx) {
        return Differentiation.forwardDiff(target, selector, x, h);
    }

    // ------------------------------------------------------------
    // Backward Difference
    // ------------------------------------------------------------

    /**
     * @notice Approximates the derivative f'(x) using the Backward Difference method.
     *         Formula: f'(x) ≈ (f(x) - f(x - h)) / h
     *         Accuracy: First-order O(h).
     *
     * @param target   The address of the contract hosting f.
     * @param selector The function selector for f.
     * @param x        The point at which to differentiate.
     * @param h        The step size. Pass 0 (QZERO) to use defaults/config.
     * @return dfdx    The approximate derivative.
     */
    function backwardDiff(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16 dfdx) {
        return Differentiation.backwardDiff(target, selector, x, h);
    }

    // ------------------------------------------------------------
    // Centered Difference
    // ------------------------------------------------------------

    /**
     * @notice Approximates the derivative f'(x) using the Centered Difference method.
     *         Formula: f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
     *         Accuracy: Second-order O(h²).
     *
     * @param target   The address of the contract hosting f.
     * @param selector The function selector for f.
     * @param x        The point at which to differentiate.
     * @param h        The step size. Pass 0 (QZERO) to use defaults/config.
     * @return dfdx    The approximate derivative.
     */
    function centeredDiff(address target, bytes4 selector, bytes16 x, bytes16 h) external view returns (bytes16 dfdx) {
        return Differentiation.centeredDiff(target, selector, x, h);
    }
}