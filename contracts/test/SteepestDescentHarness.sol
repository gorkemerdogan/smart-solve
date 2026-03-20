// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { SteepestDescent } from "../libraries/numeric/SteepestDescent.sol";
import { IObjectiveFunction } from "../interfaces/IObjectiveFunction.sol";
import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title  SteepestDescentHarness
 * @notice Test harness exposing the internal SteepestDescent library.
 */
contract SteepestDescentHarness {
    /**
     * @notice Execute the steepest descent solver against a target objective contract.
     * @param  objective Address of the objective contract implementing IObjectiveFunction
     * @param  x0        Initial point vector
     * @param  maxIter   Maximum number of iterations
     * @param  tol       Stopping tolerance. If zero, library fallback logic is used.
     *
     * @return x         Final iterate
     * @return gx        Final objective value
     * @return iters     Number of completed iterations
     * @return status    Solver status code
     */
    function solve(address objective, bytes16[] memory x0, uint256 maxIter, bytes16 tol)
        external view returns (bytes16[] memory x, bytes16 gx, uint256 iters, uint8 status) {
        SteepestDescent.Result memory r = SteepestDescent.solve(IObjectiveFunction(objective), x0, maxIter, tol);
        return (r.x, r.gx, r.iters, r.status);
    }

    // ------------------------------------------------------------
    //  Numerical Helpers
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to IEEE-754 quadruple precision.
     * @param  n Signed integer.
     * @return q Quadruple-precision representation of 'n'.
     */
    function qFromInt(int256 n) external pure returns (bytes16 q) {
        q = MathLib.fromInt(n);
    }

    /**
     * @notice Converts a rational number num/den to quadruple precision.
     *         Reverts if 'den' equals zero.
     * @param num Signed numerator.
     * @param den Signed denominator (must be non-zero).
     * @return q  Quadruple-precision value representing num/den.
     */
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = MathLib.div(qNum, qDen);
    }
}