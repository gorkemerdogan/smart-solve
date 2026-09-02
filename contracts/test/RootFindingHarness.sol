// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { RootFinding } from "../libraries/numeric/RootFinding.sol";

/**
 * @title RootFindingHarness
 * @notice Provides deterministic test functions for root-finding algorithms and
 *         helper constructors for quadruple-precision values.
 *         Includes reference functions f(x) and df(x) used in Newton, Bisection,
 *         and Secant tests.
 */
contract RootFindingHarness {
    using MathLib for bytes16;

    /// Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e18;

    // ------------------------------------------------------------------------
    // Test functions f(x) and df(x) for root-finding
    // ------------------------------------------------------------------------

    /**
     * @notice Computes f(x) = x^2 - 4.
     * @param  x Input value in quadruple precision.
     * @return y Result of x^2 - 4.
     */
    function f_x2_minus_4(bytes16 x) external pure returns (bytes16 y) {
        bytes16 four = MathLib.fromInt(4);
        y = x.mul(x).sub(four);
    }

    /**
     * @notice Computes df(x) = 2x, the derivative of x^2 - 4.
     * @param  x Input value in quadruple precision.
     * @return y Value of 2x.
     */
    function df_2x(bytes16 x) external pure returns (bytes16 y) {
        bytes16 two = MathLib.fromInt(2);
        y = two.mul(x);
    }

    /**
     * @notice Computes f(x) = x^3 - x - 2.
     * @param  x Input value in quadruple precision.
     * @return y Result of x^3 - x - 2.
     */
    function f_cubic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 x2 = x.mul(x);
        bytes16 x3 = x2.mul(x);
        bytes16 two = MathLib.fromInt(2);
        y = x3.sub(x).sub(two);
    }

    /**
     * @notice Computes df(x) = 3x^2 - 1, the derivative of x^3 - x - 2.
     * @param  x Input value in quadruple precision.
     * @return y Value of 3x^2 - 1.
     */
    function df_cubic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 three = MathLib.fromInt(3);
        bytes16 one = MathLib.fromInt(1);
        bytes16 x2 = x.mul(x);
        y = three.mul(x2).sub(one);
    }

    /**
     * @notice Computes f(x) = x^4 - 10.
     * @param  x Input value in quadruple precision.
     * @return y Result of x^4 - 10.
     */
    function f_quartic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 x2 = x.mul(x);
        bytes16 x4 = x2.mul(x2);
        bytes16 ten = MathLib.fromInt(10);
        y = x4.sub(ten);
    }

    /**
     * @notice Computes df(x) = 4x^3, the derivative of x^4 - 10.
     * @param  x Input value in quadruple precision.
     * @return y Value of 4x^3.
     */
    function df_quartic(bytes16 x) external pure returns (bytes16 y) {
        bytes16 four = MathLib.fromInt(4);
        bytes16 x2 = x.mul(x);
        bytes16 x3 = x2.mul(x);
        y = four.mul(x3);
    }

    /**
     * @notice Evaluates the shifted linear function f(x) = x - 2.
     * @param  x Input value.
     * @return Function value at x.
     */
    function f_shift_small(bytes16 x) external pure returns (bytes16) {
        return MathLib.sub(x, MathLib.fromUInt(2));
    }

    /**
     * @notice Evaluates the derivative of the shifted linear function f'(x) = 1.
     * @param  x Input value.
     * @return Derivative value at x.
     */
    function df_shift_small(bytes16 x) external pure returns (bytes16) {
        x; // Silence unused parameter warning.
        return MathLib.fromUInt(1);
    }

    /**
     * @notice Evaluates the shifted linear function f(x) = x - 200.
     * @param  x Input value.
     * @return Function value at x.
     */
    function f_shift_medium(bytes16 x) external pure returns (bytes16) {
        return MathLib.sub(x, MathLib.fromUInt(200));
    }

    /**
     * @notice Evaluates the derivative of the shifted linear function f'(x) = 1.
     * @param  x Input value.
     * @return Derivative value at x.
     */
    function df_shift_medium(bytes16 x) external pure returns (bytes16) {
        x; // Silence unused parameter warning.
        return MathLib.fromUInt(1);
    }

    /**
     * @notice Evaluates the shifted linear function f(x) = x - 20000.
     * @param  x Input value.
     * @return Function value at x.
     */
    function f_shift_large(bytes16 x) external pure returns (bytes16) {
        return MathLib.sub(x, MathLib.fromUInt(20000));
    }

    /**
     * @notice Evaluates the derivative of the shifted linear function f'(x) = 1.
     * @param  x Input value.
     * @return Derivative value at x.
     */
    function df_shift_large(bytes16 x) external pure returns (bytes16) {
        x; // Silence unused parameter warning.
        return MathLib.fromUInt(1);
    }

    /// Constant non-zero residual used to exercise Newton stagnation handling.
    function f_constant_one(bytes16 x) external pure returns (bytes16) {
        x;
        return MathLib.fromUInt(1);
    }

    /// Very large derivative makes the Newton step tiny while f remains one.
    function df_very_large(bytes16 x) external pure returns (bytes16) {
        x;
        return MathLib.fromUInt(1_000_000_000_000_000_000_000_000_000_000);
    }

    // ------------------------------------------------------------
    // Numerical Helpers
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to IEEE-754 quadruple precision.
     * @param  n Signed integer.
     * @return q Quadruple-precision representation of n.
     */
    function qFromInt(int256 n) external pure returns (bytes16 q) {
        q = MathLib.fromInt(n);
    }

    /**
     * @notice Converts an unsigned integer into its quad-precision representation.
     * @param  x Unsigned integer value.
     * @return q Quad-precision number representing x.
     */
    function qFromUInt(uint256 x) external pure returns (bytes16 q) {
        q = MathLib.fromUInt(x);
    }

    /**
     * @notice Converts a rational number num/den to quadruple precision.
     *         Reverts if den equals zero.
     * @param  num Signed numerator.
     * @param  den Signed denominator.
     * @return q Quadruple-precision value representing num / den.
     */
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = MathLib.div(qNum, qDen);
    }

    /**
     * @notice Converts a quadruple-precision number into a scaled integer.
     *         The returned integer represents x * SCALE.
     * @dev MathLib.toInt truncates toward zero. This helper is a 1e18
     *      fixed-point view and does not preserve full binary128 precision.
     * @param x Quadruple-precision value.
     * @return Integer representation of trunc(x * SCALE).
     */
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }

    /**
     * @notice Compares two quadruple-precision values.
     *          Returns:
     *           -1 if a < b
     *            0 if a == b
     *           +1 if a > b
     * @param a First operand.
     * @param b Second operand.
     * @return Comparison result in {-1, 0, +1}.
     */
    function cmp(bytes16 a, bytes16 b) external pure returns (int256) {
        return MathLib.cmp(a, b);
    }

    // ------------------------------------------------------------
    // RootFinding internal helper wrappers
    // ------------------------------------------------------------

    /**
     * @notice Exposes RootFinding._eval for direct testing.
     * @dev Calls target.staticcall(selector, x) through the library helper.
     * @param target Contract exposing a function with signature f(bytes16) -> bytes16.
     * @param sel Function selector of the target function.
     * @param x Input value.
     * @return y Evaluated function value.
     */
    function evalHarness(address target, bytes4 sel, bytes16 x) external view returns (bytes16 y) {
        y = RootFinding._eval(target, sel, x);
    }

    /**
     * @notice Exposes RootFinding._clampTol for direct testing.
     * @param requestedTol Tolerance supplied by the caller.
     * @return tol Effective tolerance after config/default/min clamping.
     */
    function clampTolHarness(bytes16 requestedTol) external view returns (bytes16 tol) {
        tol = RootFinding._clampTol(requestedTol);
    }

    // ------------------------------------------------------------
    // RootFinding algorithm wrappers
    // ------------------------------------------------------------

    /**
     * @notice Exposes the RootFinding.bisection library method for testing.
     * @param  target Contract exposing f(bytes16) -> bytes16.
     * @param  fSelector Function selector of f(bytes16).
     * @param  a First endpoint.
     * @param  b Second endpoint.
     * @param  tol Requested tolerance.
     * @param  maxIter Maximum iteration count.
     * @return root Final root approximation.
     * @return iterations Number of iterations performed.
     * @return converged Whether convergence criteria were satisfied.
     * @return fAtRoot Function value at the returned root.
     */
    function rootFindingBisection(address target, bytes4 fSelector, bytes16 a, bytes16 b, bytes16 tol, uint256 maxIter)
        external view returns (bytes16 root, uint256 iterations, bool converged, bytes16 fAtRoot) {
        RootFinding.RootResult memory r = RootFinding.bisection(target, fSelector, a, b, tol, maxIter);
        return (r.root, r.iterations, r.converged, r.fAtRoot);
    }

    /**
     * @notice Exposes the RootFinding.newton library method for testing.
     * @param  target Contract exposing f(bytes16) -> bytes16.
     * @param  fSelector Function selector of f(bytes16).
     * @param  dfTarget Contract exposing f'(bytes16) -> bytes16.
     * @param  dfSelector Function selector of f'(bytes16).
     * @param  x0 Initial guess.
     * @param  tol Requested tolerance.
     * @param  maxIter Maximum iteration count.
     * @return root Final root approximation.
     * @return iterations Number of iterations performed.
     * @return converged Whether convergence criteria were satisfied.
     * @return fAtRoot Function value at the returned root.
     */
    function rootFindingNewton(address target, bytes4 fSelector, address dfTarget, bytes4 dfSelector, bytes16 x0, bytes16 tol, uint256 maxIter)
        external view returns (bytes16 root, uint256 iterations, bool converged, bytes16 fAtRoot){
        RootFinding.RootResult memory r = RootFinding.newton(target, fSelector, dfTarget, dfSelector, x0, tol, maxIter);
        return (r.root, r.iterations, r.converged, r.fAtRoot);
    }

    /**
     * @notice Exposes the RootFinding.secant library method for testing.
     * @param  target Contract exposing f(bytes16) -> bytes16.
     * @param  fSelector Function selector of f(bytes16).
     * @param  x0 First initial point.
     * @param  x1 Second initial point.
     * @param  tol Requested tolerance.
     * @param  maxIter Maximum iteration count.
     * @return root Final root approximation.
     * @return iterations Number of iterations performed.
     * @return converged Whether convergence criteria were satisfied.
     * @return fAtRoot Function value at the returned root.
     */
    function rootFindingSecant(address target, bytes4 fSelector, bytes16 x0, bytes16 x1, bytes16 tol, uint256 maxIter)
        external view returns (bytes16 root, uint256 iterations, bool converged, bytes16 fAtRoot) {
        RootFinding.RootResult memory r = RootFinding.secant(target, fSelector, x0, x1, tol, maxIter);
        return (r.root, r.iterations, r.converged, r.fAtRoot);
    }
}
