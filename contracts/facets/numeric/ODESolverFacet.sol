// // SPDX-License-Identifier: MIT
// pragma solidity ^0.8.20;

import { ODESolver } from "../../libraries/numeric/ODESolver.sol";

/**
 * @title  ODESolverFacet
 * @notice Diamond facet exposing single-step explicit solvers for first-order
 *         ordinary differential equations y' = f(x, y).
 *
 * @dev    This facet provides external access to the 'ODESolver' library, which implements
 *         explicit one-step integration methods (Euler, RK2 variants, RK4). The derivative
 *         function f(x, y) is evaluated via 'staticcall' using the pair
 *         '(address target, bytes4 selector)'. All computations are performed using
 *         IEEE-754 quadruple precision (binary128) represented as 'bytes16'.
 *
 *         Limitations:
 *          - Only explicit fixed-step methods are supported; adaptive step-size control
 *            and stiff solvers are not implemented.
 *          - Numerical stability and accuracy depend on the choice of step size 'h' and
 *            the smoothness of f(x, y).
 *
 * @custom:gas Gas usage scales with the number of derivative evaluations per step, increasing
 *             from Euler to higher-order Runge–Kutta methods.
 */
contract ODESolverFacet {

    // ------------------------------------------------------------
    // Euler's Method
    // ------------------------------------------------------------

    /**
     * @notice Perform one Euler step.
     * @param target Address implementing f(x,y)
     * @param selectorF Function selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     * @return yNext y(x+h)
     */
    function euler(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16 yNext) {
        yNext = ODESolver.euler(target, selectorF, x, y, h);
    }

    // ------------------------------------------------------------
    // RK2 — Midpoint Method
    // ------------------------------------------------------------

    /**
     * @notice Perform one RK2 midpoint step.
     * @param target Address implementing f(x,y)
     * @param selectorF Function selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     * @return yNext y(x+h)
     */
    function rk2Midpoint(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16 yNext) {
        yNext = ODESolver.rk2Midpoint(target, selectorF, x, y, h);
    }

    // ------------------------------------------------------------
    // RK2 - Heun's Method
    // ------------------------------------------------------------

    /**
     * @notice Perform one RK2 Heun (Improved Euler / Trapezoidal) step.
     * @param target Address implementing f(x,y)
     * @param selectorF Function selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     * @return yNext y(x+h)
     */
    function rk2Heun(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16 yNext) {
        yNext = ODESolver.rk2Heun(target, selectorF, x, y, h);
    }

    // ------------------------------------------------------------
    // Classic RK4
    // ------------------------------------------------------------

    /**
     * @notice Perform one classical RK4 step.
     * @param target Address implementing f(x,y)
     * @param selectorF Function selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     * @return yNext y(x+h)
     */
    function rk4(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16 yNext) {
        yNext = ODESolver.rk4(target, selectorF, x, y, h);
    }
}