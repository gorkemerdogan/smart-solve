// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { Integration } from "../libraries/numeric/Integration.sol";

/** @notice Test-only wrapper that exercises the unchanged production library. */
contract IntegrationNaiveBenchmarkHarness {
    function trapezoidal(address target, bytes4 selector, bytes16 a, bytes16 b, uint256 n)
        external
        view
        returns (bytes16)
    {
        return Integration.trapezoidal(target, selector, a, b, n);
    }

    function simpson13(address target, bytes4 selector, bytes16 a, bytes16 b, uint256 n)
        external
        view
        returns (bytes16)
    {
        return Integration.simpson13(target, selector, a, b, n);
    }

    function simpson38(address target, bytes4 selector, bytes16 a, bytes16 b, uint256 n)
        external
        view
        returns (bytes16)
    {
        return Integration.simpson38(target, selector, a, b, n);
    }
}

/**
 * @notice Test-only production-shaped integration variant using Neumaier
 * compensation for repeated sample accumulation. It is intentionally not
 * exposed by a production facet and does not replace Integration.
 */
contract IntegrationCompensatedBenchmarkHarness {
    using MathLib for bytes16;

    bytes16 private constant QZERO = bytes16(0);

    function trapezoidal(address target, bytes4 selector, bytes16 a, bytes16 b, uint256 n)
        external
        view
        returns (bytes16)
    {
        _validate(target, a, b, n);
        bytes16 h = b.sub(a).div(MathLib.fromUInt(n));
        if (MathLib.cmp(h, QZERO) == 0) return QZERO;

        bytes16 sum = _eval(target, selector, a)
            .add(_eval(target, selector, b))
            .div(MathLib.fromInt(2));
        bytes16 correction;
        for (uint256 i = 1; i < n; ++i) {
            bytes16 x = a.add(h.mul(MathLib.fromUInt(i)));
            (sum, correction) = _neumaierAdd(sum, correction, _eval(target, selector, x));
        }
        return h.mul(sum.add(correction));
    }

    function simpson13(address target, bytes4 selector, bytes16 a, bytes16 b, uint256 n)
        external
        view
        returns (bytes16)
    {
        _validate(target, a, b, n);
        require(n % 2 == 0, "Integration: Simpson 1/3 requires even n");
        bytes16 h = b.sub(a).div(MathLib.fromUInt(n));
        if (MathLib.cmp(h, QZERO) == 0) return QZERO;

        bytes16 sumOdd;
        bytes16 correctionOdd;
        bytes16 sumEven;
        bytes16 correctionEven;
        for (uint256 i = 1; i < n; ++i) {
            bytes16 x = a.add(h.mul(MathLib.fromUInt(i)));
            bytes16 fx = _eval(target, selector, x);
            if (i % 2 == 1) {
                (sumOdd, correctionOdd) = _neumaierAdd(sumOdd, correctionOdd, fx);
            } else {
                (sumEven, correctionEven) = _neumaierAdd(sumEven, correctionEven, fx);
            }
        }

        sumOdd = sumOdd.add(correctionOdd);
        sumEven = sumEven.add(correctionEven);
        bytes16 inner = _eval(target, selector, a)
            .add(sumOdd.mul(MathLib.fromInt(4)))
            .add(sumEven.mul(MathLib.fromInt(2)))
            .add(_eval(target, selector, b));
        return h.div(MathLib.fromInt(3)).mul(inner);
    }

    function simpson38(address target, bytes4 selector, bytes16 a, bytes16 b, uint256 n)
        external
        view
        returns (bytes16)
    {
        _validate(target, a, b, n);
        require(n % 3 == 0, "Integration: Simpson 3/8 requires n % 3 == 0");
        bytes16 h = b.sub(a).div(MathLib.fromUInt(n));
        if (MathLib.cmp(h, QZERO) == 0) return QZERO;

        bytes16 sum3;
        bytes16 correction3;
        bytes16 sumNot3;
        bytes16 correctionNot3;
        for (uint256 i = 1; i < n; ++i) {
            bytes16 x = a.add(h.mul(MathLib.fromUInt(i)));
            bytes16 fx = _eval(target, selector, x);
            if (i % 3 == 0) {
                (sum3, correction3) = _neumaierAdd(sum3, correction3, fx);
            } else {
                (sumNot3, correctionNot3) = _neumaierAdd(sumNot3, correctionNot3, fx);
            }
        }

        sum3 = sum3.add(correction3);
        sumNot3 = sumNot3.add(correctionNot3);
        bytes16 inner = _eval(target, selector, a)
            .add(sumNot3.mul(MathLib.fromInt(3)))
            .add(sum3.mul(MathLib.fromInt(2)))
            .add(_eval(target, selector, b));
        bytes16 factor = h.mul(MathLib.fromInt(3)).div(MathLib.fromInt(8));
        return factor.mul(inner);
    }

    function _neumaierAdd(bytes16 sum, bytes16 correction, bytes16 term)
        private
        pure
        returns (bytes16 nextSum, bytes16 nextCorrection)
    {
        nextSum = MathLib.add(sum, term);
        if (MathLib.cmp(MathLib.abs(sum), MathLib.abs(term)) >= 0) {
            nextCorrection = MathLib.add(correction, MathLib.add(MathLib.sub(sum, nextSum), term));
        } else {
            nextCorrection = MathLib.add(correction, MathLib.add(MathLib.sub(term, nextSum), sum));
        }
    }

    function _eval(address target, bytes4 selector, bytes16 x) private view returns (bytes16 result) {
        (bool ok, bytes memory data) = target.staticcall(abi.encodeWithSelector(selector, x));
        require(ok && data.length >= 32, "Integration: eval failed");
        assembly {
            result := mload(add(data, 32))
        }
    }

    function _validate(address target, bytes16 a, bytes16 b, uint256 n) private pure {
        require(target != address(0), "Integration: target is zero");
        require(n > 0, "Integration: n must be > 0");
        require(MathLib.cmp(b, a) >= 0, "Integration: upper bound b must be >= a");
    }
}

/** @notice Deterministic callbacks shared by both benchmark variants. */
contract IntegrationCancellationBenchmarkTarget {
    function constantFive(bytes16) external pure returns (bytes16) {
        return MathLib.fromInt(5);
    }

    /// @dev 2^64 * (x - 24) + 1; exact integral on [0,48] is 48.
    function centeredLinearModerate(bytes16 x) external pure returns (bytes16) {
        bytes16 centered = MathLib.sub(x, MathLib.fromInt(24));
        return MathLib.add(MathLib.mul(MathLib.fromInt(int256(1) << 64), centered), MathLib.fromInt(1));
    }

    /// @dev 2^106 * (x - 24) + 1; exact integral on [0,48] is 48.
    function centeredLinearStress(bytes16 x) external pure returns (bytes16) {
        bytes16 centered = MathLib.sub(x, MathLib.fromInt(24));
        return MathLib.add(MathLib.mul(MathLib.fromInt(int256(1) << 106), centered), MathLib.fromInt(1));
    }

    /// @dev 2^96 * (x - 24)^3 + 1; exact integral on [0,48] is 48.
    function centeredCubic(bytes16 x) external pure returns (bytes16) {
        bytes16 centered = MathLib.sub(x, MathLib.fromInt(24));
        bytes16 cubic = MathLib.mul(MathLib.mul(centered, centered), centered);
        return MathLib.add(MathLib.mul(MathLib.fromInt(int256(1) << 96), cubic), MathLib.fromInt(1));
    }
}
