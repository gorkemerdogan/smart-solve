// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Polynomial } from "../libraries/numeric/Polynomial.sol";

/**
 * @title PolynomialHarness
 * @notice Minimal harness to expose internal library functions of Polynomial for testing.
 * The library functions are `internal`, so we wrap them in `public` functions.
 * All functions are simple passthroughs to the library.
 */
contract PolynomialHarness {

    /**
     * @notice Passthrough for Polynomial.evaluateHorners
     */
    function evaluateHorners(bytes16[] memory coeffs, bytes16 x)
        public
        pure
        returns (bytes16 y)
    {
        return Polynomial.evaluateHorners(coeffs, x);
    }

    /**
     * @notice Passthrough for Polynomial.evaluateWithDerivative
     */
    function evaluateWithDerivative(bytes16[] memory a, bytes16 x)
        public
        pure
        returns (bytes16 px, bytes16 dpx)
    {
        return Polynomial.evaluateWithDerivative(a, x);
    }

    /**
     * @notice Passthrough for Polynomial.add
     */
    function add(bytes16[] memory a, bytes16[] memory b)
        public
        pure
        returns (bytes16[] memory out)
    {
        return Polynomial.add(a, b);
    }

    /**
     * @notice Passthrough for Polynomial.sub
     */
    function sub(bytes16[] memory a, bytes16[] memory b)
        public
        pure
        returns (bytes16[] memory out)
    {
        return Polynomial.sub(a, b);
    }

    /**
     * @notice Passthrough for Polynomial.mulScalar
     */
    function mulScalar(bytes16[] memory a, bytes16 k)
        public
        pure
        returns (bytes16[] memory out)
    {
        return Polynomial.mulScalar(a, k);
    }

    /**
     * @notice Passthrough for Polynomial.mul (convolution)
     */
    function mul(bytes16[] memory a, bytes16[] memory b)
        public
        pure
        returns (bytes16[] memory out)
    {
        return Polynomial.mul(a, b);
    }

    // --- CALCULUS ---

    /**
     * @notice Passthrough for Polynomial.derivative
     */
    function derivative(bytes16[] memory coeffs)
        public
        pure
        returns (bytes16[] memory d)
    {
        return Polynomial.derivative(coeffs);
    }

    /**
     * @notice Passthrough for Polynomial.integral
     */
    function integral(bytes16[] memory a, bytes16 C)
        public
        pure
        returns (bytes16[] memory out)
    {
        return Polynomial.integral(a, C);
    }

    // --- UTILS ---

    /**
     * @notice Passthrough for Polynomial.degree
     */
    function degree(bytes16[] memory a) public pure returns (uint256) {
        return Polynomial.degree(a);
    }

    /**
     * @notice Passthrough for Polynomial.trimTrailingZeros
     */
    function trimTrailingZeros(bytes16[] memory a)
        public
        pure
        returns (bytes16[] memory out)
    {
        return Polynomial.trimTrailingZeros(a);
    }
}