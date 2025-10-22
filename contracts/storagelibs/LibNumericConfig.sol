// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * LibNumericConfig
 * Global numeric configuration stored in diamond storage.
 * @dev eps (tolerance) is SD59x18 when using fixed-point. maxIter is a hard cap for loops.
 */
library LibNumericConfig {
    // Fixed storage slot for numeric config (unique hash key)
    bytes32 internal constant SLOT = keccak256("smart-solve.numeric.config.v1");

    // Struct holding global numeric configuration values.
    // - eps: numerical tolerance (e.g., 1e9 meaning 1e-9 in SD59x18 fixed-point)
    // - maxIter: maximum iterations allowed for loops (e.g., 500)
    struct NumericConfig {
        int256 eps;
        uint256 maxIter;
    }

    /**
     * Returns a pointer to NumericConfig in storage
     * @dev Uses inline assembly to assign SLOT as the storage location.
     * This is the standard EIP-2535 diamond storage pattern.
     */
    function cfg() internal pure returns (NumericConfig storage c) {
        bytes32 position = SLOT;
        assembly { c.slot := position }
    }

    // Update epsilon (tolerance) in storage
    function setEps(int256 eps) internal {
        cfg().eps = eps;
    }

    // Update maxIter (maximum iteration count) in storage
    function setMaxIter(uint256 m) internal {
        cfg().maxIter = m;
    }
}