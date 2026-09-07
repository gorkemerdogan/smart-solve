// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { ABDKMathQuad } from "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title ODESolver
 * @notice Standalone Quad-Precision ODE solvers (Euler, RK2 Midpoint, RK2 Heun, RK4)
 *         Function signature assumed for user f(x,y):
 *           function f(bytes16 x, bytes16 y) external pure returns (bytes16);
 *
 *         This library performs:
 *            target.call(abi.encodeWithSelector(selectorF, x, y))
 *         Then decodes (bytes16) from returndata.
 */
library ODESolver {
    using MathLib for bytes16;

    // ------------------------------------------------------------
    // Internal Call
    // ------------------------------------------------------------

    /**
     * @notice Call user-defined ODE function f(x,y).
     * @param target Address implementing f(x,y).
     * @param selectorF Selector for f(x,y).
     * @param x Quad-precision x
     * @param y Quad-precision y
     * @return out f(x,y) in quad precision (bytes16)
     */
    function _callF(
        address target,
        bytes4 selectorF,
        bytes16 x,
        bytes16 y
    ) internal view returns (bytes16 out) {
        (bool ok, bytes memory ret) =
            target.staticcall(abi.encodeWithSelector(selectorF, x, y));
        require(ok, "ODESolver: target call failed");
        require(ret.length == 32, "ODESolver: bad return size");
        out = abi.decode(ret, (bytes16));
    }

    // ------------------------------------------------------------
    // Euler's Method
    // ------------------------------------------------------------

    /**
     * @notice One Euler step
     *         
     *  y_{n+1} = y_n + h * f(x_n, y_n)
     * 
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     */
    function euler(
        address target,
        bytes4 selectorF,
        bytes16 x,
        bytes16 y,
        bytes16 h
    ) internal view returns (bytes16) {
        bytes16 k1 = _callF(target, selectorF, x, y);
        return y.add(h.mul(k1));
    }

    // ------------------------------------------------------------
    // RK2 — Midpoint Method
    // ------------------------------------------------------------

    /**
     * @notice One RK2 Midpoint step:
     *
     *  k1 = f(x, y)
     *  k2 = f(x + h/2, y + h*k1/2)
     *  y_{n+1} = y + h*k2
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     */
    function rk2Midpoint(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) internal view returns (bytes16) {
        bytes16 h2 = h.mul(QC.HALF());

        // k1
        bytes16 k1 = _callF(target, selectorF, x, y);

        // midpoint values
        bytes16 xMid = x.add(h2);
        bytes16 yMid = y.add(h.mul(k1).mul(QC.HALF()));

        // k2
        bytes16 k2 = _callF(target, selectorF, xMid, yMid);

        // y_{n+1}
        return y.add(h.mul(k2));
    }

    // ------------------------------------------------------------
    // RK2 - Heun's Method
    // ------------------------------------------------------------

    /**
     * @notice One RK2 Heun step (Improved Euler / trapezoid):
     *
     *  k1 = f(x,     y)
     *  k2 = f(x + h, y + h*k1)
     *
     *  y_{n+1} = y + (h/2)*(k1 + k2)
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     */
    function rk2Heun(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) internal view returns (bytes16) {
        // k1
        bytes16 k1 = _callF(target, selectorF, x, y);

        // forward-predicted point
        bytes16 xForw = x.add(h);
        bytes16 yForw = y.add(h.mul(k1));

        // k2
        bytes16 k2 = _callF(target, selectorF, xForw, yForw);

        // trapezoidal average
        bytes16 sum = k1.add(k2);
        return y.add(h.mul(sum).mul(QC.HALF()));
    }

    // ------------------------------------------------------------
    // Classic RK4
    // ------------------------------------------------------------

    /**
     * @notice One classical RK4 step:
     *
     *  k1 = f(x,           y)
     *  k2 = f(x + h/2,     y + h*k1/2)
     *  k3 = f(x + h/2,     y + h*k2/2)
     *  k4 = f(x + h,       y + h*k3)
     *
     *  y_{n+1} = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x Current x
     * @param y Current y
     * @param h Step size
     */
    function rk4(address target, bytes4 selectorF, bytes16 x, bytes16 y, bytes16 h) internal view returns (bytes16) {

        bytes16 h2 = h.mul(QC.HALF());

        /*--------------------- k1 ---------------------*/
        bytes16 k1 = _callF(target, selectorF, x, y);

        /*--------------------- k2 ---------------------*/
        bytes16 x2 = x.add(h2);
        bytes16 y2 = y.add(h.mul(k1).mul(QC.HALF()));
        bytes16 k2 = _callF(target, selectorF, x2, y2);

        /*--------------------- k3 ---------------------*/
        bytes16 y3 = y.add(h.mul(k2).mul(QC.HALF()));
        bytes16 k3 = _callF(target, selectorF, x2, y3);

        /*--------------------- k4 ---------------------*/
        bytes16 x4 = x.add(h);
        bytes16 y4 = y.add(h.mul(k3));
        bytes16 k4 = _callF(target, selectorF, x4, y4);

        /*------------------ weighted sum --------------*/
        // k1 + 2*k2 + 2*k3 + k4
        bytes16 twok2 = k2.add(k2);
        bytes16 twok3 = k3.add(k3);
        bytes16 weighted = k1.add(twok2).add(twok3).add(k4);

        return y.add(h.mul(weighted).mul(QC.ONESIXTH()));
    }

    // ------------------------------------------------------------
    // Iterative / Multi-Step Wrappers
    // ------------------------------------------------------------

    /**
     * @notice Return the fixed-grid coordinate for iteration i.
     * @dev    Deriving x_i from x0 avoids accumulating a separate rounding
     *         error from repeated coordinate updates. The i=0 fast path also
     *         preserves the exact input coordinate for the first callback.
     */
    function _gridX(bytes16 x0, bytes16 h, uint256 i) private pure returns (bytes16) {
        if (i == 0) return x0;
        // Keep the grid calculation internal: repeated external MathLib calls
        // would materially reduce the feasible step count.
        return ABDKMathQuad.add(x0, ABDKMathQuad.mul(h, ABDKMathQuad.fromUInt(i)));
    }

    /**
     * @notice Perform multiple Euler steps and return the final y value.
     *.        Repeatedly applies y <- euler(target, selectorF, x_i, y, h),
     *         where x_i = x0 + i*h.
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x0 Initial x
     * @param y0 Initial y
     * @param h Step size
     * @param steps Number of steps to execute
     * @return y Final y value after `steps` Euler iterations
     */
    function eulerIter(address target, bytes4 selectorF, bytes16 x0, bytes16 y0, bytes16 h, uint256 steps) internal view returns (bytes16 y) {
        require(steps > 0, "ODESolver: steps must be > 0");

        y = y0;

        for (uint256 i = 0; i < steps; ++i) {
            y = euler(target, selectorF, _gridX(x0, h, i), y, h);
        }
    }

    /**
     * @notice Perform multiple RK2 Midpoint steps and return the final y value.
     *         Repeatedly applies rk2Midpoint at x_i = x0 + i*h.
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x0 Initial x
     * @param y0 Initial y
     * @param h Step size
     * @param steps Number of steps to execute
     * @return y Final y value after `steps` RK2 Midpoint iterations
     */
    function rk2MidpointIter(address target, bytes4 selectorF, bytes16 x0, bytes16 y0, bytes16 h, uint256 steps) internal view returns (bytes16 y) {
        require(steps > 0, "ODESolver: steps must be > 0");

        y = y0;

        for (uint256 i = 0; i < steps; ++i) {
            y = rk2Midpoint(target, selectorF, _gridX(x0, h, i), y, h);
        }
    }

    /**
     * @notice Perform multiple RK2 Heun steps and return the final y value.
     *         Repeatedly applies rk2Heun at x_i = x0 + i*h.
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x0 Initial x
     * @param y0 Initial y
     * @param h Step size
     * @param steps Number of steps to execute
     * @return y Final y value after `steps` RK2 Heun iterations
     */
    function rk2HeunIter(address target, bytes4 selectorF, bytes16 x0, bytes16 y0, bytes16 h, uint256 steps) internal view returns (bytes16 y) {
        require(steps > 0, "ODESolver: steps must be > 0");

        y = y0;

        for (uint256 i = 0; i < steps; ++i) {
            y = rk2Heun(target, selectorF, _gridX(x0, h, i), y, h);
        }
    }

    /**
     * @notice Perform multiple RK4 steps and return the final y value.
     *         Repeatedly applies rk4 at x_i = x0 + i*h.
     *
     * @param target Address implementing f(x,y)
     * @param selectorF Selector for f(x,y)
     * @param x0 Initial x
     * @param y0 Initial y
     * @param h Step size
     * @param steps Number of steps to execute
     * @return y Final y value after `steps` RK4 iterations
     */
    function rk4Iter(address target, bytes4 selectorF, bytes16 x0, bytes16 y0, bytes16 h, uint256 steps) internal view returns (bytes16 y) {
        require(steps > 0, "ODESolver: steps must be > 0");

        y = y0;

        for (uint256 i = 0; i < steps; ++i) {
            y = rk4(target, selectorF, _gridX(x0, h, i), y, h);
        }
    }
}
