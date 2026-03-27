// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IObjectiveFunction} from "../interfaces/IObjectiveFunction.sol";
import {MathLib} from "../libraries/MathLib.sol";

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
    function grad(
        bytes16[] memory x
    ) external pure returns (bytes16[] memory gradVec) {
        gradVec = new bytes16[](x.length);

        bytes16 two = MathLib.fromUInt(2);

        for (uint256 i = 0; i < x.length; ++i) {
            bytes16 weight = MathLib.fromUInt(i + 1);
            gradVec[i] = two.mul(weight).mul(x[i]);
        }
    }
}

/**
 * @title SphericalObjectiveHarness
 * @notice Test objective implementing the spherical quadratic function
 *         g(x) = \sum_i x_i^2 and its gradient \nabla g(x) = 2x.
 */
contract SphericalObjectiveHarness is IObjectiveFunction {
    /**
     * @notice Evaluate the spherical quadratic objective at the given point.
     * @param  x Input vector
     * @return sum Objective value computed as \sum_i x_i^2
     */
    function g(bytes16[] memory x) external pure override returns (bytes16) {
        bytes16 sum = MathLib.fromUInt(0);
        for (uint256 i = 0; i < x.length; ++i) {
            sum = MathLib.add(sum, MathLib.mul(x[i], x[i]));
        }
        return sum;
    }

    /**
     * @notice Evaluate the gradient of the spherical quadratic objective.
     * @param  x Input vector
     * @return z Gradient vector computed as 2x
     */
    function grad(
        bytes16[] memory x
    ) external pure override returns (bytes16[] memory) {
        bytes16[] memory z = new bytes16[](x.length);
        bytes16 two = MathLib.fromUInt(2);

        for (uint256 i = 0; i < x.length; ++i) {
            z[i] = MathLib.mul(two, x[i]);
        }
        return z;
    }
}

/**
 * @title SemiWeightedQuadraticObjectiveHarness
 * @notice Test objective implementing a more softly weighted quadratic function
 *         g(x) = \sum_i w_i x_i^2 with very mild anisotropy.
 */
contract SemiWeightedQuadraticObjectiveHarness is IObjectiveFunction {
    /**
     * @notice Return the per-coordinate weight used in the objective.
     * @dev    The weights are intentionally kept close to each other so that
     *         the objective remains anisotropic, but not so stiff that the
     *         optimizer behaves like the hard weighted test case.
     * @param i Zero-based coordinate index
     * @return Weight associated with coordinate i
     */
    function _weight(uint256 i) internal pure returns (bytes16) {
        uint256[8] memory ws = [uint256(1), 1, 1, 2, 2, 2, 3, 3];
        if (i < 8) {
            return MathLib.fromUInt(ws[i]);
        }
        return MathLib.fromUInt((i / 3) + 1);
    }

    /**
     * @notice Evaluate the softly weighted quadratic objective.
     * @param x Input vector
     * @return sum Objective value computed as \sum_i w_i x_i^2
     */
    function g(bytes16[] memory x) external pure override returns (bytes16) {
        bytes16 sum = MathLib.fromUInt(0);

        for (uint256 i = 0; i < x.length; ++i) {
            bytes16 w = _weight(i);
            bytes16 term = MathLib.mul(w, MathLib.mul(x[i], x[i]));
            sum = MathLib.add(sum, term);
        }

        return sum;
    }

    /**
     * @notice Evaluate the gradient of the softly weighted quadratic objective.
     * @param x Input vector
     * @return z Gradient vector computed as 2*w_i*x_i
     */
    function grad(bytes16[] memory x) external pure override returns (bytes16[] memory z) {
        z = new bytes16[](x.length);
        bytes16 two = MathLib.fromUInt(2);

        for (uint256 i = 0; i < x.length; ++i) {
            bytes16 w = _weight(i);
            z[i] = MathLib.mul(two, MathLib.mul(w, x[i]));
        }
    }
}
