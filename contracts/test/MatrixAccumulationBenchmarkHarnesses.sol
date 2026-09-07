// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { MatrixMaster } from "../libraries/numeric/MatrixMaster.sol";

/** @notice Test-only wrapper around the unchanged production accumulation path. */
contract MatrixAccumulationNaiveHarness {
    function dot(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16) {
        require(a.length == b.length && a.length > 0, "MatrixAccumulationBenchmark: vector length");
        MatrixMaster.Matrix memory av = MatrixMaster.Matrix(a.length, 1, _copy(a));
        MatrixMaster.Matrix memory bv = MatrixMaster.Matrix(b.length, 1, _copy(b));
        return MatrixMaster.dot(av, bv);
    }

    function matrixVector(uint256 m, uint256 n, bytes16[] calldata a, bytes16[] calldata x)
        external
        pure
        returns (bytes16[] memory)
    {
        _validate(m, n, a.length, x.length);
        MatrixMaster.Matrix memory matrix = MatrixMaster.Matrix(m, n, _copy(a));
        MatrixMaster.Matrix memory vector = MatrixMaster.Matrix(n, 1, _copy(x));
        return MatrixMaster.multiplyMatrixVector(matrix, vector).data;
    }

    function residualNormSquared(
        uint256 m,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata x,
        bytes16[] calldata b
    ) external pure returns (bytes16) {
        _validate(m, n, a.length, x.length);
        require(b.length == m, "MatrixAccumulationBenchmark: residual length");
        MatrixMaster.Matrix memory matrix = MatrixMaster.Matrix(m, n, _copy(a));
        MatrixMaster.Matrix memory vector = MatrixMaster.Matrix(n, 1, _copy(x));
        MatrixMaster.Matrix memory rhs = MatrixMaster.Matrix(m, 1, _copy(b));
        MatrixMaster.Matrix memory residual = MatrixMaster.subtractMatrices(
            MatrixMaster.multiplyMatrixVector(matrix, vector), rhs
        );
        return MatrixMaster.dot(residual, residual);
    }

    function _copy(bytes16[] calldata source) private pure returns (bytes16[] memory result) {
        result = new bytes16[](source.length);
        for (uint256 i = 0; i < source.length; ++i) result[i] = source[i];
    }

    function _validate(uint256 m, uint256 n, uint256 aLength, uint256 xLength) private pure {
        require(m > 0 && m <= 4 && n > 0 && n <= 64, "MatrixAccumulationBenchmark: dimensions");
        require(aLength == m * n && xLength == n, "MatrixAccumulationBenchmark: matrix length");
    }
}

contract MatrixAccumulationCompensatedHarness {
    function dot(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16) {
        require(a.length == b.length && a.length > 0, "MatrixAccumulationBenchmark: vector length");
        return _dot(_copy(a), _copy(b));
    }

    function matrixVector(uint256 m, uint256 n, bytes16[] calldata a, bytes16[] calldata x)
        external
        pure
        returns (bytes16[] memory)
    {
        _validate(m, n, a.length, x.length);
        return _matrixVector(m, n, _copy(a), _copy(x));
    }

    function residualNormSquared(
        uint256 m,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata x,
        bytes16[] calldata b
    ) external pure returns (bytes16) {
        _validate(m, n, a.length, x.length);
        require(b.length == m, "MatrixAccumulationBenchmark: residual length");
        bytes16[] memory ax = _matrixVector(m, n, _copy(a), _copy(x));
        bytes16[] memory rhs = _copy(b);
        bytes16[] memory residual = new bytes16[](m);
        for (uint256 i = 0; i < m; ++i) residual[i] = MathLib.sub(ax[i], rhs[i]);
        return _dot(residual, residual);
    }

    function _matrixVector(uint256 m, uint256 n, bytes16[] memory a, bytes16[] memory x)
        private
        pure
        returns (bytes16[] memory result)
    {
        result = new bytes16[](m);
        for (uint256 i = 0; i < m; ++i) {
            bytes16 sum;
            bytes16 correction;
            for (uint256 j = 0; j < n; ++j) {
                (sum, correction) = _neumaierAdd(sum, correction, MathLib.mul(a[i * n + j], x[j]));
            }
            result[i] = MathLib.add(sum, correction);
        }
    }

    function _dot(bytes16[] memory a, bytes16[] memory b) private pure returns (bytes16 result) {
        bytes16 correction;
        for (uint256 i = 0; i < a.length; ++i) {
            (result, correction) = _neumaierAdd(result, correction, MathLib.mul(a[i], b[i]));
        }
        return MathLib.add(result, correction);
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

    function _copy(bytes16[] calldata source) private pure returns (bytes16[] memory result) {
        result = new bytes16[](source.length);
        for (uint256 i = 0; i < source.length; ++i) result[i] = source[i];
    }

    function _validate(uint256 m, uint256 n, uint256 aLength, uint256 xLength) private pure {
        require(m > 0 && m <= 4 && n > 0 && n <= 64, "MatrixAccumulationBenchmark: dimensions");
        require(aLength == m * n && xLength == n, "MatrixAccumulationBenchmark: matrix length");
    }
}

contract MatrixAccumulationPairwiseHarness {
    function dot(bytes16[] calldata a, bytes16[] calldata b) external pure returns (bytes16) {
        require(a.length == b.length && a.length > 0, "MatrixAccumulationBenchmark: vector length");
        bytes16[] memory av = _copy(a);
        bytes16[] memory bv = _copy(b);
        return _pairwiseDot(av, bv, 0, av.length);
    }

    function matrixVector(uint256 m, uint256 n, bytes16[] calldata a, bytes16[] calldata x)
        external
        pure
        returns (bytes16[] memory)
    {
        _validate(m, n, a.length, x.length);
        return _matrixVector(m, n, _copy(a), _copy(x));
    }

    function residualNormSquared(
        uint256 m,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata x,
        bytes16[] calldata b
    ) external pure returns (bytes16) {
        _validate(m, n, a.length, x.length);
        require(b.length == m, "MatrixAccumulationBenchmark: residual length");
        bytes16[] memory ax = _matrixVector(m, n, _copy(a), _copy(x));
        bytes16[] memory rhs = _copy(b);
        bytes16[] memory residual = new bytes16[](m);
        for (uint256 i = 0; i < m; ++i) residual[i] = MathLib.sub(ax[i], rhs[i]);
        return _pairwiseDot(residual, residual, 0, residual.length);
    }

    function _matrixVector(uint256 m, uint256 n, bytes16[] memory a, bytes16[] memory x)
        private
        pure
        returns (bytes16[] memory result)
    {
        result = new bytes16[](m);
        for (uint256 i = 0; i < m; ++i) {
            result[i] = _pairwiseRow(a, x, i, n, 0, n);
        }
    }

    function _pairwiseDot(bytes16[] memory a, bytes16[] memory b, uint256 lo, uint256 hi)
        private
        pure
        returns (bytes16)
    {
        uint256 length = hi - lo;
        if (length == 1) return MathLib.mul(a[lo], b[lo]);
        uint256 mid = lo + length / 2;
        return MathLib.add(_pairwiseDot(a, b, lo, mid), _pairwiseDot(a, b, mid, hi));
    }

    function _pairwiseRow(
        bytes16[] memory a,
        bytes16[] memory x,
        uint256 row,
        uint256 n,
        uint256 lo,
        uint256 hi
    ) private pure returns (bytes16) {
        uint256 length = hi - lo;
        if (length == 1) return MathLib.mul(a[row * n + lo], x[lo]);
        uint256 mid = lo + length / 2;
        return MathLib.add(
            _pairwiseRow(a, x, row, n, lo, mid),
            _pairwiseRow(a, x, row, n, mid, hi)
        );
    }

    function _copy(bytes16[] calldata source) private pure returns (bytes16[] memory result) {
        result = new bytes16[](source.length);
        for (uint256 i = 0; i < source.length; ++i) result[i] = source[i];
    }

    function _validate(uint256 m, uint256 n, uint256 aLength, uint256 xLength) private pure {
        require(m > 0 && m <= 4 && n > 0 && n <= 64, "MatrixAccumulationBenchmark: dimensions");
        require(aLength == m * n && xLength == n, "MatrixAccumulationBenchmark: matrix length");
    }
}
