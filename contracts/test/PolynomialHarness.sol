// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad as Q } from "abdk-libraries-solidity/ABDKMathQuad.sol";
import { Polynomial } from "../libraries/numeric/Polynomial.sol";

/**
 * @title PolynomialHarness
 * @notice Exposes internal library functions of Polynomial for testing (ABDK quad: bytes16).
 */
contract PolynomialHarness {
    using Q for bytes16;

    // ===== Passthroughs to Polynomial library =====

    function evaluateHorners(bytes16[] memory coeffs, bytes16 x)
        public pure returns (bytes16)
    {
        return Polynomial.evaluateHorners(coeffs, x);
    }

    function evaluateWithDerivative(bytes16[] memory a, bytes16 x)
        public pure returns (bytes16 px, bytes16 dpx)
    {
        return Polynomial.evaluateWithDerivative(a, x);
    }

    function add(bytes16[] memory a, bytes16[] memory b)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.add(a, b);
    }

    function sub(bytes16[] memory a, bytes16[] memory b)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.sub(a, b);
    }

    function mulScalar(bytes16[] memory a, bytes16 k)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.mulScalar(a, k);
    }

    function mul(bytes16[] memory a, bytes16[] memory b)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.mul(a, b);
    }

    // --- Calculus ---

    function derivative(bytes16[] memory coeffs)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.derivative(coeffs);
    }

    function integral(bytes16[] memory a, bytes16 C)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.integral(a, C);
    }

    // --- Utils ---

    function degree(bytes16[] memory a) public pure returns (uint256) {
        return Polynomial.degree(a);
    }

    function trimTrailingZeros(bytes16[] memory a)
        public pure returns (bytes16[] memory)
    {
        return Polynomial.trimTrailingZeros(a);
    }

    // ===== Test helpers (no fromString/toString in ABDK) =====

    /// @notice Quad from integer (exact)
    function qFromInt(int256 x) public pure returns (bytes16) {
        return Q.fromInt(x);
    }

    /// @notice Quad from rational num/den (exact for finite decimals like 1.75=7/4)
    function qFromFrac(int256 num, int256 den) public pure returns (bytes16) {
        require(den != 0, "den=0");
        return Q.fromInt(num).div(Q.fromInt(den));
    }
}