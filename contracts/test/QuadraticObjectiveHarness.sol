// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IObjectiveFunction} from "../interfaces/IObjectiveFunction.sol";
import {MathLib} from "../libraries/MathLib.sol";

/**
 * @title  QuadraticObjectiveHarness
 * @notice Simple convex quadratic objective for testing the Steepest Descent solver.
 *
 *         Objective: g(x) = sum_i x_i^2
 *         Gradient: grad g(x)_i = 2 * x_i
 *
 *         The unique global minimizer is x = 0, where g(x) = 0.
 */
contract QuadraticObjectiveHarness is IObjectiveFunction {
    /**
     * @notice    Evaluate g(x) = sum_i x_i^2.
     * @param  x  Current point vector
     * @return gx Objective value
     */
    function g(bytes16[] memory x) external pure override returns (bytes16 gx) {
        gx = MathLib.fromUInt(0);

        for (uint256 i = 0; i < x.length; ++i) {
            gx = MathLib.add(gx, MathLib.mul(x[i], x[i]));
        }
    }

    /**
     * @notice       Evaluate grad g(x) = 2x.
     * @param  x     Current point vector
     * @return gradx Gradient vector
     */
    function grad(bytes16[] memory x) external pure override returns (bytes16[] memory gradx) {
        gradx = new bytes16[](x.length);
        bytes16 two = MathLib.fromUInt(2);

        for (uint256 i = 0; i < x.length; ++i) {
            gradx[i] = MathLib.mul(two, x[i]);
        }
    }
}

/**
 * @title  WeightedQuadraticObjectiveHarness
 * @notice Test objective for generic steepest descent experiments.
 *         Implements the weighted quadratic function
 *         g(x) = sum_i w_i * x_i^2
 *         with gradient
 *         grad g(x)_i = 2 * w_i * x_i
 */
contract WeightedQuadraticObjectiveHarness {
    using MathLib for bytes16;

    /**
     * @notice Evaluates the weighted quadratic objective g(x) = sum_i (i+1) * x_i^2.
     * @param  x Input vector in quad-precision.
     * @return value Objective value g(x) in quad-precision.
     */
    function g(bytes16[] memory x) external pure returns (bytes16 value) {
        value = MathLib.fromUInt(0);

        for (uint256 i = 0; i < x.length; ++i) {
            bytes16 weight = MathLib.fromUInt(i + 1);
            bytes16 xi2 = x[i].mul(x[i]);
            value = value.add(weight.mul(xi2));
        }
    }

    /**
     * @notice  Computes the gradient of the weighted quadratic objective.
     * @dev     For g(x) = sum_i (i+1) * x_i^2,
     *          grad g(x)_i = 2 * (i+1) * x_i.
     * @param x Input vector in quad-precision.
     * @return gradVec Gradient vector in quad-precision.
     */
    function grad(bytes16[] memory x) external pure returns (bytes16[] memory gradVec) {
        gradVec = new bytes16[](x.length);

        bytes16 two = MathLib.fromUInt(2);

        for (uint256 i = 0; i < x.length; ++i) {
            bytes16 weight = MathLib.fromUInt(i + 1);
            gradVec[i] = two.mul(weight).mul(x[i]);
        }
    }
}