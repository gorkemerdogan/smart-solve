// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { ODESolver } from "../libraries/numeric/ODESolver.sol";

/**
 * @title ODESolverHarness
 * @notice Wrapper contract exposing ODESolver library functions for testing.
 */
contract ODESolverHarness {
    using MathLib for bytes16;

    // ------------------------------------------------------------
    // ODE Solver passthroughs
    // ------------------------------------------------------------

    function euler(address target, bytes4 selector, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16) {
        return ODESolver.euler(target, selector, x, y, h);
    }

    function rk2Midpoint(address target, bytes4 selector, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16) {
        return ODESolver.rk2Midpoint(target, selector, x, y, h);
    }

    function rk2Heun(address target, bytes4 selector, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16) {
        return ODESolver.rk2Heun(target, selector, x, y, h);
    }

    function rk4(address target, bytes4 selector, bytes16 x, bytes16 y, bytes16 h) external view returns (bytes16) {
        return ODESolver.rk4(target, selector, x, y, h);
    }

    // ------------------------------------------------------------
    //  ODE functions f(x)
    // ------------------------------------------------------------

    /**
     * @notice Constant ODE: y' = 5
     * @return v Quad-precision constant value 5.
     */
    function f_const5(bytes16 /*x*/) external pure returns (bytes16) {
        return MathLib.fromInt(5);
    }

    /** 
     * @notice Linear ODE: y' = y
     * @param  x Input value.
     * @return v Quad-precision output equal to x.
     */
    function f_linear(bytes16 x) external pure returns (bytes16) {
        return x;
    }

    /**
     * @notice Quadratic slope: y' = x^2
     * @param  x Input value.
     * @return v Quad-precision square of x.
     */
    function f_square(bytes16 x) external pure returns (bytes16) {
        return x.mul(x);
    }

    // ------------------------------------------------------------
    // Numerical helpers
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
     * @param  num Signed numerator.
     * @param  den Signed denominator (must be non-zero).
     * @return q  Quadruple-precision value representing num/den.
     */
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = MathLib.div(qNum, qDen);
    }

    /// @notice Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e12;
    
    /**
     * @notice Converts a quadruple-precision number into a scaled integer (scaled by SCALE).
     * @param  x Quadruple-precision value.
     * @return Integer representing x * SCALE.
     */
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }
}