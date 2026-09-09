// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Test-only 1e18 fixed-point baseline for polynomial Horner evaluation.
/// @dev Multiplication truncates toward zero at each Horner step, matching Solidity
///      signed division semantics. It is intentionally not used by SmartSolve.
contract FixedPointHornerBaselineHarness {
    int256 internal constant SCALE = 1e18;

    function evaluateHorners(int256[] calldata coeffs, int256 x) external pure returns (int256 y) {
        uint256 length = coeffs.length;
        if (length == 0) return 0;

        y = coeffs[length - 1];
        for (uint256 i = length - 1; i > 0; ) {
            y = (y * x) / SCALE + coeffs[i - 1];
            unchecked {
                --i;
            }
        }
    }
}
