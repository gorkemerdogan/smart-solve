// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { ABDKMathQuad } from "abdk-libraries-solidity/ABDKMathQuad.sol";

/**
 * @title PolynomialInlinedHornerBenchmarkFacet
 * @notice Test-only facet isolating Horner evaluation with internal ABDK arithmetic.
 */
contract PolynomialInlinedHornerBenchmarkFacet {
    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);

    function polyEvaluate(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 y) {
        return _evaluateHorners(_toMemory(coeffs), x);
    }

    function _evaluateHorners(bytes16[] memory coeffs, bytes16 x) private pure returns (bytes16 y) {
        if (coeffs.length == 0) return QZERO;

        y = coeffs[coeffs.length - 1];
        for (uint256 i = coeffs.length - 1; i > 0; --i) {
            y = ABDKMathQuad.add(ABDKMathQuad.mul(y, x), coeffs[i - 1]);
        }
    }

    function _toMemory(bytes16[] calldata coeffs) private pure returns (bytes16[] memory copied) {
        copied = new bytes16[](coeffs.length);
        for (uint256 i = 0; i < coeffs.length; ++i) {
            copied[i] = coeffs[i];
        }
    }
}
