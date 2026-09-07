// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";

/**
 * @notice Test-only accumulation variants. These deliberately use the same
 * linked MathLib arithmetic as production and are not production algorithms.
 */
contract NaiveSummationBenchmarkHarness {
    uint256 private constant MAX_TERMS = 64;

    function sum(bytes16[] calldata values) external pure returns (bytes16 result) {
        require(values.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        for (uint256 i = 0; i < values.length; ++i) {
            result = MathLib.add(result, values[i]);
        }
    }

    function dot(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16 result) {
        require(a.length == b.length, "SummationBenchmark: length mismatch");
        require(a.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        for (uint256 i = 0; i < a.length; ++i) {
            result = MathLib.add(result, MathLib.mul(a[i], b[i]));
        }
    }

    function squaredNorm(bytes16[] calldata values) external pure returns (bytes16 result) {
        require(values.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        for (uint256 i = 0; i < values.length; ++i) {
            result = MathLib.add(result, MathLib.mul(values[i], values[i]));
        }
    }

    function multiplyMatrices(
        uint256 m,
        uint256 k,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b
    ) external pure returns (bytes16[] memory result) {
        _validateMatrix(m, k, n, a.length, b.length);
        result = new bytes16[](m * n);
        for (uint256 i = 0; i < m; ++i) {
            for (uint256 j = 0; j < n; ++j) {
                bytes16 cell;
                for (uint256 t = 0; t < k; ++t) {
                    cell = MathLib.add(cell, MathLib.mul(a[i * k + t], b[t * n + j]));
                }
                result[i * n + j] = cell;
            }
        }
    }

    function _validateMatrix(uint256 m, uint256 k, uint256 n, uint256 aLength, uint256 bLength)
        private
        pure
    {
        require(m > 0 && k > 0 && n > 0, "SummationBenchmark: empty matrix");
        require(m <= 4 && k <= MAX_TERMS && n <= 4, "SummationBenchmark: matrix too large");
        require(aLength == m * k && bLength == k * n, "SummationBenchmark: matrix length mismatch");
    }
}

contract CompensatedSummationBenchmarkHarness {
    uint256 private constant MAX_TERMS = 64;

    function sum(bytes16[] calldata values) external pure returns (bytes16 result) {
        require(values.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        bytes16 correction;
        for (uint256 i = 0; i < values.length; ++i) {
            (result, correction) = _neumaierAdd(result, correction, values[i]);
        }
        result = MathLib.add(result, correction);
    }

    function dot(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16 result) {
        require(a.length == b.length, "SummationBenchmark: length mismatch");
        require(a.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        bytes16 correction;
        for (uint256 i = 0; i < a.length; ++i) {
            bytes16 term = MathLib.mul(a[i], b[i]);
            (result, correction) = _neumaierAdd(result, correction, term);
        }
        result = MathLib.add(result, correction);
    }

    function squaredNorm(bytes16[] calldata values) external pure returns (bytes16 result) {
        require(values.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        bytes16 correction;
        for (uint256 i = 0; i < values.length; ++i) {
            bytes16 term = MathLib.mul(values[i], values[i]);
            (result, correction) = _neumaierAdd(result, correction, term);
        }
        result = MathLib.add(result, correction);
    }

    function multiplyMatrices(
        uint256 m,
        uint256 k,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b
    ) external pure returns (bytes16[] memory result) {
        _validateMatrix(m, k, n, a.length, b.length);
        result = new bytes16[](m * n);
        for (uint256 i = 0; i < m; ++i) {
            for (uint256 j = 0; j < n; ++j) {
                bytes16 cell;
                bytes16 correction;
                for (uint256 t = 0; t < k; ++t) {
                    bytes16 term = MathLib.mul(a[i * k + t], b[t * n + j]);
                    (cell, correction) = _neumaierAdd(cell, correction, term);
                }
                result[i * n + j] = MathLib.add(cell, correction);
            }
        }
    }

    function _neumaierAdd(bytes16 sumValue, bytes16 correction, bytes16 term)
        private
        pure
        returns (bytes16 nextSum, bytes16 nextCorrection)
    {
        nextSum = MathLib.add(sumValue, term);
        if (MathLib.cmp(MathLib.abs(sumValue), MathLib.abs(term)) >= 0) {
            nextCorrection = MathLib.add(correction, MathLib.add(MathLib.sub(sumValue, nextSum), term));
        } else {
            nextCorrection = MathLib.add(correction, MathLib.add(MathLib.sub(term, nextSum), sumValue));
        }
    }

    function _validateMatrix(uint256 m, uint256 k, uint256 n, uint256 aLength, uint256 bLength)
        private
        pure
    {
        require(m > 0 && k > 0 && n > 0, "SummationBenchmark: empty matrix");
        require(m <= 4 && k <= MAX_TERMS && n <= 4, "SummationBenchmark: matrix too large");
        require(aLength == m * k && bLength == k * n, "SummationBenchmark: matrix length mismatch");
    }
}

contract PairwiseSummationBenchmarkHarness {
    uint256 private constant MAX_TERMS = 64;

    function sum(bytes16[] calldata values) external pure returns (bytes16) {
        require(values.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        return _pairwiseSum(values, 0, values.length);
    }

    function dot(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16) {
        require(a.length == b.length, "SummationBenchmark: length mismatch");
        require(a.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        return _pairwiseDot(a, b, 0, a.length);
    }

    function squaredNorm(bytes16[] calldata values) external pure returns (bytes16) {
        require(values.length <= MAX_TERMS, "SummationBenchmark: too many terms");
        return _pairwiseDot(values, values, 0, values.length);
    }

    function multiplyMatrices(
        uint256 m,
        uint256 k,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b
    ) external pure returns (bytes16[] memory result) {
        _validateMatrix(m, k, n, a.length, b.length);
        result = new bytes16[](m * n);
        for (uint256 i = 0; i < m; ++i) {
            for (uint256 j = 0; j < n; ++j) {
                result[i * n + j] = _pairwiseMatrixCell(a, b, i, j, k, n, 0, k);
            }
        }
    }

    function _pairwiseSum(bytes16[] calldata values, uint256 lo, uint256 hi)
        private
        pure
        returns (bytes16)
    {
        uint256 length = hi - lo;
        if (length == 0) return bytes16(0);
        if (length == 1) return values[lo];
        uint256 mid = lo + length / 2;
        return MathLib.add(_pairwiseSum(values, lo, mid), _pairwiseSum(values, mid, hi));
    }

    function _pairwiseDot(
        bytes16[] calldata a,
        bytes16[] calldata b,
        uint256 lo,
        uint256 hi
    ) private pure returns (bytes16) {
        uint256 length = hi - lo;
        if (length == 0) return bytes16(0);
        if (length == 1) return MathLib.mul(a[lo], b[lo]);
        uint256 mid = lo + length / 2;
        return MathLib.add(_pairwiseDot(a, b, lo, mid), _pairwiseDot(a, b, mid, hi));
    }

    function _pairwiseMatrixCell(
        bytes16[] calldata a,
        bytes16[] calldata b,
        uint256 row,
        uint256 column,
        uint256 k,
        uint256 n,
        uint256 lo,
        uint256 hi
    ) private pure returns (bytes16) {
        uint256 length = hi - lo;
        if (length == 1) return MathLib.mul(a[row * k + lo], b[lo * n + column]);
        uint256 mid = lo + length / 2;
        return MathLib.add(
            _pairwiseMatrixCell(a, b, row, column, k, n, lo, mid),
            _pairwiseMatrixCell(a, b, row, column, k, n, mid, hi)
        );
    }

    function _validateMatrix(uint256 m, uint256 k, uint256 n, uint256 aLength, uint256 bLength)
        private
        pure
    {
        require(m > 0 && k > 0 && n > 0, "SummationBenchmark: empty matrix");
        require(m <= 4 && k <= MAX_TERMS && n <= 4, "SummationBenchmark: matrix too large");
        require(aLength == m * k && bLength == k * n, "SummationBenchmark: matrix length mismatch");
    }
}
