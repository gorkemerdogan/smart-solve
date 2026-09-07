// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title PolynomialLinkedHornerBenchmarkFacet
 * @notice Test-only facet preserving the pre-change linked-MathLib Horner path.
 */
contract PolynomialLinkedHornerBenchmarkFacet {
    using MathLib for bytes16;

    function polyEvaluate(bytes16[] calldata coeffs, bytes16 x) external pure returns (bytes16 y) {
        bytes16[] memory copied = _toMemory(coeffs);
        if (copied.length == 0) return bytes16(0);

        y = copied[copied.length - 1];
        for (uint256 i = copied.length - 1; i > 0; --i) {
            y = y.mul(x).add(copied[i - 1]);
        }
    }

    function _toMemory(bytes16[] calldata coeffs) private pure returns (bytes16[] memory copied) {
        copied = new bytes16[](coeffs.length);
        for (uint256 i = 0; i < coeffs.length; ++i) {
            copied[i] = coeffs[i];
        }
    }
}
