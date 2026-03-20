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
