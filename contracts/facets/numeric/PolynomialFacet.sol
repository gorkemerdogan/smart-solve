// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Polynomial } from "../../libraries/numeric/Polynomial.sol";

/**
 * @title PolynomialFacet
 * @notice External Diamond facet that exposes Polynomial (ABDK quad) utilities.
 *         All inputs/outputs use IEEE-754 quadruple precision as bytes16.
 *
 * @dev Library functions require `memory` arrays, therefore calldata arrays are copied
 *      to memory once per call.
 */
contract PolynomialFacet {
    /*─────────────────────────── helpers ───────────────────────────*/

    /// @dev Copy a calldata bytes16[] into memory (library expects memory).
    function _toMemory(bytes16[] calldata a) internal pure returns (bytes16[] memory m) {
        m = new bytes16[](a.length);
        for (uint256 i = 0; i < a.length; ++i) m[i] = a[i];
    }

    /*────────────────────── evaluation / calculus ──────────────────*/

    /// @notice Evaluate f(x) using Horner’s method.
    function polyEvaluate(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 y) {
        return Polynomial.evaluateHorners(_toMemory(coeffs), x);
    }

    /// @notice Compute f(x) and f’(x) in a single pass (extended Horner).
    function polyEvaluateWithDerivative(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 px, bytes16 dpx) {
        return Polynomial.evaluateWithDerivative(_toMemory(coeffs), x);
    }

    /// @notice Return derivative coefficients of f.
    function polyDerivative(bytes16[] calldata coeffs) external pure returns (bytes16[] memory d) {
        return Polynomial.derivative(_toMemory(coeffs));
    }

    /// @notice Indefinite integral coefficients of f with constant C.
    function polyIntegral(bytes16[] calldata a, bytes16 C) external pure returns (bytes16[] memory out) {
        return Polynomial.integral(_toMemory(a), C);
    }

    /*──────────────────────── arithmetic ops ───────────────────────*/

    /// @notice Coefficient-wise addition: a + b.
    function polyAdd(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16[] memory out) {
        return Polynomial.add(_toMemory(a), _toMemory(b));
    }

    /// @notice Coefficient-wise subtraction: a − b.
    function polySub(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16[] memory out) {
        return Polynomial.sub(_toMemory(a), _toMemory(b));
    }

    /// @notice Convolution (polynomial multiplication): a * b.
    function polyMul(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16[] memory out) {
        return Polynomial.mul(_toMemory(a), _toMemory(b));
    }

    /// @notice Scalar multiply: k * a.
    function polyMulScalar(bytes16[] calldata a, bytes16 k) external pure returns (bytes16[] memory out) {
        return Polynomial.mulScalar(_toMemory(a), k);
    }

    /// @notice Synthetic division
    function polySyntheticDivide(bytes16[] calldata coeffs, bytes16 root) external pure returns (bytes16[] memory q, bytes16 r) {
        return Polynomial.syntheticDivide(_toMemory(coeffs), root);
    }

    /// @notice Monic Horner's Evaluation
    function polyEvalHornerMonic(bytes16[] calldata lowerCoeffs, bytes16 x) external pure returns (bytes16 y) {
        return Polynomial.evalHornerMonic(_toMemory(lowerCoeffs), x);
    }

    /*────────────────────────── utilities ──────────────────────────*/

    /// @notice Degree of the polynomial (highest i with non-zero coeff). Zero poly -> 0.
    function polyDegree(bytes16[] calldata a) external pure returns (uint256) {
        return Polynomial.degree(_toMemory(a));
    }

    /// @notice Trim trailing zeros to canonical length.
    function polyTrim(bytes16[] calldata a) external pure returns (bytes16[] memory out) {
        return Polynomial.trimTrailingZeros(_toMemory(a));
    }
}