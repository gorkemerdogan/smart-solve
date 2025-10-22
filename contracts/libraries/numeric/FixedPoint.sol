// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@prb/math/contracts/PRBMathSD59x18.sol";

library FixedPoint {
    using PRBMathSD59x18 for int256;

    int256 internal constant WAD = 1e18;

    function fromInt(int256 x) internal pure returns (int256) {
        return x * WAD;
    }

    function toInt(int256 x) internal pure returns (int256) {
        return x / WAD;
    }
    
    // FIX: Use PRBMath's safe fixed-point addition and subtraction
    function add(int256 a, int256 b) internal pure returns (int256) { return a.add(b); }
    function sub(int256 a, int256 b) internal pure returns (int256) { return a.sub(b); }
    
    function mul(int256 a, int256 b) internal pure returns (int256) { return a.mul(b); }
    function div(int256 a, int256 b) internal pure returns (int256) { return a.div(b); }

    function abs(int256 a) internal pure returns (int256) {
        return a >= 0 ? a : -a;
    }

    function nearlyEqual(int256 a, int256 b, int256 eps) internal pure returns (bool) {
        // Use PRB's sub() for consistency and safety
        return abs(a.sub(b)) <= eps;
    }
}