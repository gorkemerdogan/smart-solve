// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Import the SD59x18 type wrap and unwrap functions
import { SD59x18, wrap, unwrap } from "@prb/math/src/SD59x18.sol";

/**
 * FixedPoint (SD59x18)
 * Provides simple wrappers for PRBMath operations, 
 * converting int256 inputs to the required SD59x18 type for type-safe math
 * and unwrapping the SD59x18 result back to a plain int256.
 */
library FixedPoint {
    // Attach the free function 'wrap' to the type 'int256'
    using { wrap } for int256;
    // Attach the free function 'unwrap' to the type 'SD59x18'
    using { unwrap } for SD59x18;

    int256 internal constant WAD = 1e18;

    // Functions that convert unscaled int256 inputs to SD59x18 scale
    function fromInt(int256 x) internal pure returns (int256) {
        return x * WAD;
    }

    function toInt(int256 x) internal pure returns (int256) {
        return x / WAD;
    }
    
    function fpAdd(int256 a, int256 b) internal pure returns (int256) {
        // a (int256) -> a.wrap() (SD59x18)
        // b (int256) -> b.wrap() (SD59x18)
        // .add() result (SD59x18) -> .unwrap() (int256)
        return a.wrap().add(b.wrap()).unwrap(); 
    }
    
    function fpSub(int256 a, int256 b) internal pure returns (int256) { 
        return a.wrap().sub(b.wrap()).unwrap(); 
    }
    
    function fpMul(int256 a, int256 b) internal pure returns (int256) { 
        return a.wrap().mul(b.wrap()).unwrap(); 
    }
    
    function fpDiv(int256 a, int256 b) internal pure returns (int256) { 
        return a.wrap().div(b.wrap()).unwrap(); 
    }

    // Using the native PRBMath abs() method
    function fpAbs(int256 a) internal pure returns (int256) {
        return a.wrap().abs().unwrap(); 
    }

    function nearlyEqual(int256 a, int256 b, int256 eps) internal pure returns (bool) {
        // Use the native methods for consistency and safety
        return a.wrap().sub(b.wrap()).abs().unwrap() <= eps;
    }
}