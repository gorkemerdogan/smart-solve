// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LibNumericConfig
 * @notice Global numeric configuration stored in diamond storage.
 * @dev tol (tolerance) and minTol (min tolerance) is bytes16 when using ABDKMathQuad. maxIter is a hard cap for loops.
 */
library LibNumericConfig {
    // Fixed storage slot for numeric config (unique hash key)
    bytes32 internal constant SLOT = keccak256("smart-solve.numeric.config.v1");

    /**
     * @notice Struct holding global numeric configuration values.
     * @param tol: numerical tolerance as ABDKMathQuad (bytes16)
     * @param minTol: numerical minimum tolerance as ABDKMathQuad (bytes16)
     * @param maxIter: maximum iterations allowed for loops (e.g., 500)
     */
    struct NumericConfig {
        bytes16 tol;
        bytes16 minTol;
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

    /// Setters

    /**
     * @notice Update min tolerance in storage
     * @param minTol The new min tolerance value as ABDKMathQuad (bytes16)
     */
    function setMinTol(bytes16 minTol) internal {
        cfg().minTol = minTol;
    }

    /**
     * @notice Update tolerance in storage
     * @param tol The new tolerance value as ABDKMathQuad (bytes16)
     */
    function setTol(bytes16 tol) internal {
        cfg().tol = tol;
    }

    /**
     * @notice Update maxIter (maximum iteration count) in storage
     * @param m The new maximum iteration count
     */
    function setMaxIter(uint256 m) internal {
        cfg().maxIter = m;
    }

    /// Getters

    function getTol() internal view returns (bytes16) {
    return cfg().tol;
    }

    function getMinTol() internal view returns (bytes16) {
        return cfg().minTol;
    }

    function getMaxIter() internal view returns (uint256) {
        return cfg().maxIter;
    }
}