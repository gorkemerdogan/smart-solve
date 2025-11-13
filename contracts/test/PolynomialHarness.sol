// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad as Q } from "abdk-libraries-solidity/ABDKMathQuad.sol";
import { Polynomial } from "../libraries/numeric/Polynomial.sol";

/**
 * @title PolynomialHarness
 * @notice Exposes internal library functions of Polynomial for testing (ABDK quad: bytes16).
 * @dev All functions are thin passthroughs. Helpers are provided for exact quad construction/printing.
 */
contract PolynomialHarness {
    using Q for bytes16;

    // ====== Passthroughs to Polynomial library ======

    function evaluateHorners(bytes16[] memory coeffs, bytes16 x) public pure returns (bytes16) {
        return Polynomial.evaluateHorners(coeffs, x);
    }

    function evaluateWithDerivative(bytes16[] memory coeffs, bytes16 x) public pure returns (bytes16 px, bytes16 dpx) {
        return Polynomial.evaluateWithDerivative(coeffs, x);
    }

    /// @notice Monic polynomial evaluation:
    /// p(x) = x^n + a_{n-1}x^{n-1} + ... + a1x + a0, where lowerCoeffs = [a0, a1, ..., a_{n-1}]
    function evalHornerMonic(bytes16[] memory lowerCoeffs, bytes16 x) public pure returns (bytes16) {
        return Polynomial.evalHornerMonic(lowerCoeffs, x);
    }

    function add(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) public pure returns (bytes16[] memory) {
        return Polynomial.add(coeffs_a, coeffs_b);
    }

    function sub(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) public pure returns (bytes16[] memory) {
        return Polynomial.sub(coeffs_a, coeffs_b);
    }

    function mulScalar(bytes16[] memory coeffs, bytes16 k) public pure returns (bytes16[] memory) {
        return Polynomial.mulScalar(coeffs, k);
    }

    function mul(bytes16[] memory coeffs_a, bytes16[] memory coeffs_b) public pure returns (bytes16[] memory) {
        return Polynomial.mul(coeffs_a, coeffs_b);
    }

    /// @notice Synthetic division by (x - root): returns (Q, R) where P(x) = (x - root) * Q(x) + R
    function syntheticDivide(bytes16[] memory coeffs, bytes16 root) public pure returns (bytes16[] memory q, bytes16 r) {
        return Polynomial.syntheticDivide(coeffs, root);
    }

    // ====== Calculus ======

    function derivative(bytes16[] memory coeffs) public pure returns (bytes16[] memory) {
        return Polynomial.derivative(coeffs);
    }

    function integral(bytes16[] memory coeffs, bytes16 C) public pure returns (bytes16[] memory) {
        return Polynomial.integral(coeffs, C);
    }

    // ====== Utils ======

    function degree(bytes16[] memory coeffs) public pure returns (uint256) {
        return Polynomial.degree(coeffs);
    }

    function trimTrailingZeros(bytes16[] memory coeffs) public pure returns (bytes16[] memory) {
        return Polynomial.trimTrailingZeros(coeffs);
    }

    // ====== Test Helpers ======

    /// @notice Quad from integer (exact)
    function qFromInt(int256 x) public pure returns (bytes16) {
        return Q.fromInt(x);
    }

    /// @notice Quad from rational num/den (exact for finite decimals like 7/4 = 1.75)
    function qFromFrac(int256 num, int256 den) public pure returns (bytes16) {
        require(den != 0, "den=0");
        return Q.fromInt(num).div(Q.fromInt(den));
    }

    /**
     * @notice Returns the raw hexadecimal string for a quad (e.g. "0x...")
     * @dev Useful for logging and debugging in test suites.
     */
    function qToString(bytes16 x) public pure returns (string memory) {
        bytes memory b = abi.encodePacked(x);
        bytes memory hexChars = "0123456789abcdef";
        bytes memory str = new bytes(2 + b.length * 2);
        str[0] = "0";
        str[1] = "x";
        for (uint i = 0; i < b.length; i++) {
            str[2 + i * 2] = hexChars[uint8(b[i] >> 4)];
            str[3 + i * 2] = hexChars[uint8(b[i] & 0x0f)];
        }
        return string(str);
    }
}