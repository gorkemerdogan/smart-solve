// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IObjectiveFunction } from "../../interfaces/IObjectiveFunction.sol";
import { SteepestDescent } from "../../libraries/numeric/SteepestDescent.sol";

/**
 * @title  SteepestDescentFacet
 * @notice Diamond facet exposing the general Steepest Descent solver.
 *         The objective function is supplied as an external contract address implementing IObjectiveFunction.
 */
contract SteepestDescentFacet {
    /**
     * @notice Approximate a local minimizer of g(x) using Steepest Descent algorithm.
     *
     * @param objective Address of the objective contract implementing IObjectiveFunction
     * @param x0data    Initial point x^(0) encoded as bytes16[]
     * @param maxIter   Maximum number of iterations
     * @param tol       Tolerance used by the textbook stopping criteria
     *
     * @return x        Final iterate
     * @return gx       Final objective value
     * @return iters    Iteration count completed
     * @return status   Solver status code:
     *                  0 = legacy success (reserved)
     *                  1 = stationary / effectively zero gradient
     *                  2 = no likely improvement
     *                  3 = maximum iterations exceeded
     */
    function steepestDescent(address objective, bytes16[] calldata x0data, uint256 maxIter, bytes16 tol)
        external view returns (bytes16[] memory x, bytes16 gx, uint256 iters, uint8 status) {
        IObjectiveFunction obj = IObjectiveFunction(objective);

        bytes16[] memory x0 = new bytes16[](x0data.length);
        
        for (uint256 i = 0; i < x0data.length; ++i) {
            x0[i] = x0data[i];
        }

        SteepestDescent.Result memory r = SteepestDescent.solve(obj, x0, maxIter, tol);
        return (r.x, r.gx, r.iters, r.status);
    }
}
