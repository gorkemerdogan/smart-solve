// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title  IObjectiveFunction
 * @notice Interface for multivariable objective functions used by the Steepest Descent solver.
 *         The objective is a scalar function g : R^n -> R and the gradient is a vector grad g : R^n -> R^n.
 */
interface IObjectiveFunction {
    /**
     * @notice    Evaluate the scalar objective function g(x).
     * @param  x  Current point in R^n, encoded as a bytes16[] vector.
     * @return gx Objective value g(x).
     */
    function g(bytes16[] memory x) external view returns (bytes16 gx);

    /**
     * @notice   Evaluate the gradient vector grad g(x).
     * @param  x Current point in R^n, encoded as a bytes16[] vector.
     * @return gradx Gradient vector at x. Must have the same length as x.
     */
    function grad(bytes16[] memory x) external view returns (bytes16[] memory gradx);
}