// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title Numerical Differentiation Library
 * @notice High-precision numerical differentiation using IEEE-754 binary128 (bytes16) scalars.
 * All operations are stateless and use `staticcall` to evaluate the target function `f(x)`.
 */
library Differentiation {
    using MathLib for bytes16;

    // Quadruple precision zero.
    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

    // ------------------------------------------------------------
    // Internal Helpers
    // ------------------------------------------------------------

    // @notice 
    // @dev Resolution Logic:
    //      
    //      
    //      
    // 
    // 
    /**
     * @notice Resolves the differentiation step size `h` based on precedence rules.
     *          1. If `hInput` != 0, use `hInput`.
     *          2. Else if `LibNumericConfig.getDiffStep()` != 0, use that.
     *          3. Else, use hardcoded default (1e-8).
     * @param  hInput The user-provided step size (can be 0).
     * @return step   The final resolved step size to be used in calculations.
     */
    function _getStep(bytes16 hInput) private view returns (bytes16 step) {
        // 1. User Override
        if (MathLib.cmp(hInput, QZERO) != 0) {
            return hInput;
        }

        // 2. Global Config
        bytes16 hConfig = LibNumericConfig.getDiffStep();
        if (MathLib.cmp(hConfig, QZERO) != 0) {
            return hConfig;
        }

        // 3. Hardcoded Fallback: 1e-8
        return MathLib.fromInt(1).div(MathLib.fromInt(100000000));
    }


    /**
     * @notice Evaluates the external function f(x) via staticcall.
     *         Expects the target function to have signature `function name(bytes16) external pure returns (bytes16)`.
     *
     * @param target   The address of the contract hosting the function.
     * @param selector The 4-byte function selector of f(x).
     * @param x        The point at which to evaluate f.
     * @return y       The result f(x).
     */

    function evalFunc(address target, bytes4 selector, bytes16 x) internal view returns (bytes16 y) {
        require(target != address(0), "Differentiation: target is zero address");

        bytes memory callData = abi.encodeWithSelector(selector, x);

        (bool ok, bytes memory result) = target.staticcall(callData);
        require(ok, "Differentiation: integrand call failed");
        require(result.length == 32, "Differentiation: bad return size");

        y = abi.decode(result, (bytes16));
    }

    // ------------------------------------------------------------
    // Differentiation Methods    
    // ------------------------------------------------------------

    /**
     * @notice Approximates the derivative f'(x) using the Forward Difference method.
     *         Formula: f'(x) ≈ (f(x + h) - f(x)) / h
     *         Accuracy: First-order O(h).
     *
     * @param target   The address of the contract hosting f.
     * @param selector The function selector for f.
     * @param x        The point at which to differentiate.
     * @param h        The step size. Pass 0 (QZERO) to use defaults/config.
     * @return dfdx    The approximate derivative.
     */
    function forwardDiff(address target, bytes4 selector, bytes16 x, bytes16 h) internal view returns (bytes16 dfdx) {
        bytes16 step = _getStep(h);

        bytes16 fx  = evalFunc(target, selector, x);
        bytes16 fxh = evalFunc(target, selector, x.add(step));

        // (f(x+h) - f(x)) / h
        bytes16 num = fxh.sub(fx);
        return num.div(step);
    }


    /**
    * @notice Approximates the derivative f'(x) using the Backward Difference method.
    *         Formula: f'(x) ≈ (f(x) - f(x - h)) / h
    *         Accuracy: First-order O(h).
    *
    * @param target   The address of the contract hosting f.
    * @param selector The function selector for f.
    * @param x        The point at which to differentiate.
    * @param h        The step size. Pass 0 (QZERO) to use defaults/config.
    * @return dfdx    The approximate derivative.
     */
    function backwardDiff(address target, bytes4 selector, bytes16 x, bytes16 h) internal view returns (bytes16 dfdx) {
        bytes16 step = _getStep(h);

        bytes16 fx   = evalFunc(target, selector, x);
        bytes16 fxmh = evalFunc(target, selector, x.sub(step));

        // (f(x) - f(x-h)) / h
        bytes16 num = fx.sub(fxmh);
        return num.div(step);
    }

    /**
     * @notice Approximates the derivative f'(x) using the Centered Difference method.
     *         Formula: f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
     *         Accuracy: Second-order O(h²).
     *
     * @param target   The address of the contract hosting f.
     * @param selector The function selector for f.
     * @param x        The point at which to differentiate.
     * @param h        The step size. Pass 0 (QZERO) to use defaults/config.
     * @return dfdx    The approximate derivative.
     */
    function centeredDiff(address target, bytes4 selector, bytes16 x, bytes16 h) internal view returns (bytes16 dfdx) {
        bytes16 step = _getStep(h);

        bytes16 fxp = evalFunc(target, selector, x.add(step));
        bytes16 fxm = evalFunc(target, selector, x.sub(step));

        // (f(x+h) - f(x-h)) / 2h
        bytes16 num = fxp.sub(fxm);
        bytes16 twoH = step.add(step);

        return num.div(twoH);
    }
}