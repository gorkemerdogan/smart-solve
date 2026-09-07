// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Polynomial } from "../libraries/numeric/Polynomial.sol";

/**
 * @title PolynomialLinkedHornerBenchmarkFacet
 * @notice Test-only facet isolating the production Horner/linked-MathLib path.
 */
contract PolynomialLinkedHornerBenchmarkFacet {
    function polyEvaluate(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 y) {
        return Polynomial.evaluateHorners(_toMemory(coeffs), x);
    }

    function _toMemory(bytes16[] calldata coeffs) private pure returns (bytes16[] memory copied) {
        copied = new bytes16[](coeffs.length);
        for (uint256 i = 0; i < coeffs.length; ++i) {
            copied[i] = coeffs[i];
        }
    }
}
