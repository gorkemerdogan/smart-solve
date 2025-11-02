// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LibNumericConfig
 * @notice Global numeric configuration stored in diamond storage.
 * @dev eps (tolerance) is bytes16 when using ABDKMathQuad. maxIter is a hard cap for loops.
 */
library LibNumericConfig {
    // Fixed storage slot for numeric config (unique hash key)
    bytes32 internal constant SLOT = keccak256("smart-solve.numeric.config.v1");

    /**
     * @notice Struct holding global numeric configuration values.
     * @param eps: numerical tolerance as ABDKMathQuad (bytes16)
     * @param maxIter: maximum iterations allowed for loops (e.g., 500)
     */
    struct NumericConfig {
        bytes16 eps;
        uint256 maxIter;
    }

    /**
     * @notice Returns a pointer to NumericConfig in storage
     * @dev Uses inline assembly to assign SLOT as the storage location.
     * This is the standard EIP-2535 diamond storage pattern.
     */
    function cfg() internal pure returns (NumericConfig storage c) {
        bytes32 position = SLOT;
        assembly { c.slot := position }
    }

    /**
     * @notice Update epsilon (tolerance) in storage
     * @param eps The new tolerance value as ABDKMathQuad (bytes16)
     */
    function setEps(bytes16 eps) internal {
        cfg().eps = eps;
    }

    /**
     * @notice Update maxIter (maximum iteration count) in storage
     * @param m The new maximum iteration count
     */
    function setMaxIter(uint256 m) internal {
        cfg().maxIter = m;
    }
}
