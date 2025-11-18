// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TrigonometricConstants
 * @notice All high-precision minimax coefficients, converted from
 *         `internal constant` → `internal pure returns (bytes16)`.
 */

library TrigonometricConstants {

    /*──────────────────────────────────────
        RANGE REDUCTION CONSTANT
    ──────────────────────────────────────*/

    function TWO_POW_64() internal pure returns (bytes16) {
        return 0x40300000000000000000000000000000;
    }

    /*──────────────────────────────────────
        SIN minimax coefficients
    ──────────────────────────────────────*/

    function SIN_C1()  internal pure returns (bytes16) { return 0x3fff0000000000000000000000000000; }
    function SIN_C3()  internal pure returns (bytes16) { return 0xbfffa555555555555555555555555555; }
    function SIN_C5()  internal pure returns (bytes16) { return 0x3fff1555555555555555555555555555; }
    function SIN_C7()  internal pure returns (bytes16) { return 0xbffec71c71c71c71c71c71c71c71c71c; }
    function SIN_C9()  internal pure returns (bytes16) { return 0x3ffe75d3035d3035d3035d3035d3035d; }
    function SIN_C11() internal pure returns (bytes16) { return 0xbfeda6c0960c960c960c960c960c960d; }
    function SIN_C13() internal pure returns (bytes16) { return 0x3feca7c0d0c0d0c0d0c0d0c0d0c0d0c1; }
    function SIN_C15() internal pure returns (bytes16) { return 0xbfebf03390e0e0e0e0e0e0e0e0e0e0e1; }
    function SIN_C17() internal pure returns (bytes16) { return 0x3fea30da020202020202020202020202; }
    function SIN_C19() internal pure returns (bytes16) { return 0xbfe8f2f4901010101010101010101010; }

    /*──────────────────────────────────────
        COS minimax coefficients
    ──────────────────────────────────────*/

    function COS_C0()  internal pure returns (bytes16) { return 0x3fff0000000000000000000000000000; }
    function COS_C2()  internal pure returns (bytes16) { return 0xbffe0000000000000000000000000000; }
    function COS_C4()  internal pure returns (bytes16) { return 0x3ffa5555555555555555555555555555; }
    function COS_C6()  internal pure returns (bytes16) { return 0xbff56c16c16c16c16c16c16c16c16c17; }
    function COS_C8()  internal pure returns (bytes16) { return 0x3feed00d00d00d00d00d00d00d00d00d; }
    function COS_C10() internal pure returns (bytes16) { return 0xbfe9f00e300e300e300e300e300e300f; }
    function COS_C12() internal pure returns (bytes16) { return 0x3fe51047604760476047604760476048; }
    function COS_C14() internal pure returns (bytes16) { return 0xbfe0ff807f007f007f007f007f007f01; }
    function COS_C16() internal pure returns (bytes16) { return 0x3fdbbd04008004002001000800400201; }
    function COS_C18() internal pure returns (bytes16) { return 0xbfd689e22150008a0004a00028000115; }

    /*──────────────────────────────────────
        ASIN minimax coefficients
    ──────────────────────────────────────*/

    function ASIN_C1()  internal pure returns (bytes16) { return 0x3fff0000000000000000000000000000; }
    function ASIN_C3()  internal pure returns (bytes16) { return 0x3ffe5555555555555555555555555555; }
    function ASIN_C5()  internal pure returns (bytes16) { return 0x3ffd3333333333333333333333333333; }
    function ASIN_C7()  internal pure returns (bytes16) { return 0x3ffcc925925925925925925925925926; }
    function ASIN_C9()  internal pure returns (bytes16) { return 0x3ffc3c3c3c3c3c3c3c3c3c3c3c3c3c3c; }
    function ASIN_C11() internal pure returns (bytes16) { return 0x3ffbbfbeebfbeebfbeebfbeebfbeebfc; }
    function ASIN_C13() internal pure returns (bytes16) { return 0x3ffb4f1d4f1d4f1d4f1d4f1d4f1d4f1d; }
    function ASIN_C15() internal pure returns (bytes16) { return 0x3ffae7f9e7f9e7f9e7f9e7f9e7f9e7fa; }

    /*──────────────────────────────────────
        ACOS minimax coefficients
    ──────────────────────────────────────*/

    function ACOS_C3()  internal pure returns (bytes16) { return 0xbffe5555555555555555555555555555; }
    function ACOS_C5()  internal pure returns (bytes16) { return 0xbffd3333333333333333333333333333; }
    function ACOS_C7()  internal pure returns (bytes16) { return 0xbffcc925925925925925925925925926; }
    function ACOS_C9()  internal pure returns (bytes16) { return 0xbffc3c3c3c3c3c3c3c3c3c3c3c3c3c3c; }
    function ACOS_C11() internal pure returns (bytes16) { return 0xbffbbfbeebfbeebfbeebfbeebfbeebfc; }
    function ACOS_C13() internal pure returns (bytes16) { return 0xbffb4f1d4f1d4f1d4f1d4f1d4f1d4f1d; }
    function ACOS_C15() internal pure returns (bytes16) { return 0xbffae7f9e7f9e7f9e7f9e7f9e7f9e7fa; }

    /*──────────────────────────────────────
        ATAN minimax coefficients
    ──────────────────────────────────────*/

    function ATAN_C1()  internal pure returns (bytes16) { return 0x3fff0000000000000000000000000000; }
    function ATAN_C3()  internal pure returns (bytes16) { return 0xbffe5555555555555555555555555555; }
    function ATAN_C5()  internal pure returns (bytes16) { return 0x3ffe1999999999999999999999999999; }
    function ATAN_C7()  internal pure returns (bytes16) { return 0xbffd1249249249249249249249249249; }
    function ATAN_C9()  internal pure returns (bytes16) { return 0x3ffc0c30c30c30c30c30c30c30c30c30; }
    function ATAN_C11() internal pure returns (bytes16) { return 0xbffb0afd6a52c98c7e8f5c28f5c28f5c; }
    function ATAN_C13() internal pure returns (bytes16) { return 0x3ffa0aef9db22d0e5604189374bc6a7f; }
    function ATAN_C15() internal pure returns (bytes16) { return 0xbff90a8b93fb28e8e6c1c1c1c1c1c1c2; }
    function ATAN_C17() internal pure returns (bytes16) { return 0x3ff80a7c1fcbd3d7a4ca36f5f5f5f5f6; }

}