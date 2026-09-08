// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {MathLib} from "../libraries/MathLib.sol";
import {MatrixMaster} from "../libraries/numeric/MatrixMaster.sol";

/**
 * @title MatrixMasterHarness
 * @notice Wrapper contract exposing MatrixMaster library functions for testing.
 *         Matrices are passed as (rows, cols, flat data) and returned in the same form.
 *         Includes small helpers for quad construction and approximate comparison.
 */
contract MatrixMasterHarness {
    using MathLib for bytes16;

    // Short alias for library struct
    using MatrixMaster for MatrixMaster.Matrix;
    using MatrixMaster for MatrixMaster.SparseMatrix;

    // ---------------------------------------------------------
    // Helpers (Matrix)
    // ---------------------------------------------------------

    /**
     * @notice Build a MatrixMaster.Matrix from flat calldata.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dataFlat Flattened matrix data, length must be rows * cols
     */
    function _toMatrix(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat
    ) private pure returns (MatrixMaster.Matrix memory m) {
        require(rows > 0 && cols > 0, "MatrixMasterHarness: dims must be > 0");
        require(
            rows * cols == dataFlat.length,
            "MatrixMasterHarness: area mismatch"
        );

        bytes16[] memory data = new bytes16[](dataFlat.length);
        for (uint256 i = 0; i < dataFlat.length; ++i) {
            data[i] = dataFlat[i];
        }

        m = MatrixMaster.Matrix({rows: rows, cols: cols, data: data});
    }

    /**
     * @notice Return matrix components as tuple.
     */
    function _fromMatrix(
        MatrixMaster.Matrix memory m
    ) private pure returns (uint256 rows, uint256 cols, bytes16[] memory data) {
        return (m.rows, m.cols, m.data);
    }

    /**
     * @notice Elementwise approximate comparison of two matrices.
     * @param aRows Rows of A
     * @param aCols Cols of A
     * @param aData Flat data of A
     * @param bRows Rows of B
     * @param bCols Cols of B
     * @param bData Flat data of B
     * @param tol   Allowed elementwise absolute deviation
     * @return ok   True if same shape and all elements within tolerance
     */
    function matricesApproxEqual(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData,
        bytes16 tol
    ) external pure returns (bool ok) {
        if (aRows != bRows || aCols != bCols) {
            return false;
        }
        if (aData.length != bData.length) {
            return false;
        }

        for (uint256 i = 0; i < aData.length; ++i) {
            bytes16 diff = aData[i].sub(bData[i]);
            bytes16 absDiff = MathLib.abs(diff);
            if (MathLib.cmp(absDiff, tol) > 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @notice Exact comparison of two matrices (shape and data).
     */
    function matricesExactEqual(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData
    ) external pure returns (bool ok) {
        if (aRows != bRows || aCols != bCols) {
            return false;
        }
        if (aData.length != bData.length) {
            return false;
        }
        for (uint256 i = 0; i < aData.length; ++i) {
            if (aData[i] != bData[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @notice Copy calldata bytes16[] to a new memory array.
     */
    function _toMemory(
        bytes16[] calldata a
    ) internal pure returns (bytes16[] memory m) {
        m = new bytes16[](a.length);
        for (uint256 i = 0; i < a.length; ++i) {
            m[i] = a[i];
        }
    }

    /**
     * @notice Copy calldata uint256[] to a new memory array.
     */
    function _toMemory(
        uint256[] calldata a
    ) internal pure returns (uint256[] memory m) {
        m = new uint256[](a.length);
        for (uint256 i = 0; i < a.length; ++i) {
            m[i] = a[i];
        }
    }

    // ---------------------------------------------------------
    // Dense Matrix Creation wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for zeros.
     */
    function zerosHarness(
        uint256 rows,
        uint256 cols
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = MatrixMaster.createZerosMatrix(
            rows,
            cols
        );
        return _fromMatrix(m);
    }

    /**
     * @notice Wrapper for fromDiagonal.
     */
    function fromDiagonalHarness(
        bytes16[] calldata diag
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        // Copy diag into memory for the library
        bytes16[] memory d = new bytes16[](diag.length);
        for (uint256 i = 0; i < diag.length; ++i) {
            d[i] = diag[i];
        }

        MatrixMaster.Matrix memory m = MatrixMaster.createDiagonalMatrix(d);
        return _fromMatrix(m);
    }

    /**
     * @notice Wrapper for randomMatrix.
     */
    function randomMatrixHarness(
        uint256 rows,
        uint256 cols,
        bytes32 seed
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = MatrixMaster.createRandomMatrix(
            rows,
            cols,
            seed
        );
        return _fromMatrix(m);
    }

    // ---------------------------------------------------------
    // Sparse Matrix Creation wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for createZeroSparse.
     */
    function createZeroSparseHarness(
        uint256 rows,
        uint256 cols
    )
        external
        pure
        returns (
            uint256,
            uint256,
            uint256[] memory,
            uint256[] memory,
            bytes16[] memory
        )
    {
        MatrixMaster.SparseMatrix memory A = MatrixMaster.createZeroSparse(
            rows,
            cols
        );
        return (A.rows, A.cols, A.rowPtr, A.colInd, A.values);
    }

    /**
     * @notice Wrapper for createIdentitySparse.
     */
    function createIdentitySparseHarness(
        uint256 n
    )
        external
        pure
        returns (
            uint256,
            uint256,
            uint256[] memory,
            uint256[] memory,
            bytes16[] memory
        )
    {
        MatrixMaster.SparseMatrix memory A = MatrixMaster.createIdentitySparse(
            n
        );
        return (A.rows, A.cols, A.rowPtr, A.colInd, A.values);
    }

    /**
     * @notice Wrapper for createDiagonalSparse.
     */
    function createDiagonalSparseHarness(
        bytes16[] calldata diag
    )
        external
        pure
        returns (
            uint256,
            uint256,
            uint256[] memory,
            uint256[] memory,
            bytes16[] memory
        )
    {
        bytes16[] memory d = _toMemory(diag);
        MatrixMaster.SparseMatrix memory A = MatrixMaster.createDiagonalSparse(
            d
        );
        return (A.rows, A.cols, A.rowPtr, A.colInd, A.values);
    }

    /**
     * @notice Wrapper for createSparseFromTriplets.
     */
    function createSparseFromTripletsHarness(
        uint256 rows,
        uint256 cols,
        uint256[] calldata rowInd,
        uint256[] calldata colInd,
        bytes16[] calldata values
    )
        external
        pure
        returns (
            uint256,
            uint256,
            uint256[] memory,
            uint256[] memory,
            bytes16[] memory
        )
    {
        uint256[] memory r = _toMemory(rowInd);
        uint256[] memory c = _toMemory(colInd);
        bytes16[] memory v = _toMemory(values);

        MatrixMaster.SparseMatrix memory A = MatrixMaster
            .createSparseFromTriplets(rows, cols, r, c, v);
        return (A.rows, A.cols, A.rowPtr, A.colInd, A.values);
    }

    // ---------------------------------------------------------
    // Element access wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for get.
     */
    function getHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        uint256 row,
        uint256 col
    ) external pure returns (bytes16) {
        MatrixMaster.Matrix memory m = _toMatrix(rows, cols, dataFlat);
        return MatrixMaster.getElement(m, row, col);
    }

    /**
     * @notice Wrapper for set.
     *         Returns full matrix after modification.
     */
    function setHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        uint256 row,
        uint256 col,
        bytes16 val
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.setElement(m, row, col, val);
        return _fromMatrix(m);
    }

    // ---------------------------------------------------------
    // Slice & reshape wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for slice.
     */
    function sliceHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        uint256 rowStart,
        uint256 rowEnd,
        uint256 colStart,
        uint256 colEnd
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory outMat = MatrixMaster.slice(
            m,
            rowStart,
            rowEnd,
            colStart,
            colEnd
        );
        return _fromMatrix(outMat);
    }

    /**
     * @notice Wrapper for reshape.
     */
    function reshapeHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        uint256 newRows,
        uint256 newCols
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory r = MatrixMaster.reshape(
            m,
            newRows,
            newCols
        );
        return _fromMatrix(r);
    }

    // ---------------------------------------------------------
    // Transpose wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for transpose.
     */
    function transposeHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory t = MatrixMaster.transpose(a);
        return _fromMatrix(t);
    }

    // ---------------------------------------------------------
    // Elementwise arithmetic wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for add.
     */
    function addHarness(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory b = _toMatrix(bRows, bCols, bData);
        MatrixMaster.Matrix memory c = MatrixMaster.addMatrices(a, b);
        return _fromMatrix(c);
    }

    /**
     * @notice Wrapper for sub.
     */
    function subHarness(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory b = _toMatrix(bRows, bCols, bData);
        MatrixMaster.Matrix memory c = MatrixMaster.subtractMatrices(a, b);
        return _fromMatrix(c);
    }

    /**
     * @notice Wrapper for mulScalar.
     */
    function mulScalarHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        bytes16 k
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory c = MatrixMaster.multiplyScalar(a, k);
        return _fromMatrix(c);
    }

    /**
     * @notice Wrapper for divScalar.
     */
    function divScalarHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        bytes16 k
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory c = MatrixMaster.divideScalar(a, k);
        return _fromMatrix(c);
    }

    // ---------------------------------------------------------
    // Matrix multiplication wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for multiplyMatrices.
     */
    function mulMatrixHarness(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory b = _toMatrix(bRows, bCols, bData);
        MatrixMaster.Matrix memory c = MatrixMaster.multiplyMatrices(a, b);
        return _fromMatrix(c);
    }

    // ---------------------------------------------------------
    // Dense Matrix x Vector multiplication wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for multiplyMatrixVector.
     */
    function mulMatrixVectorHarness(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory b = _toMatrix(bRows, bCols, bData);
        MatrixMaster.Matrix memory c = MatrixMaster.multiplyMatrixVector(a, b);
        return _fromMatrix(c);
    }

    // ------------------------------------------------------------
    // Sparse Matrix x Vector multiplication
    // ------------------------------------------------------------

    /**
     * @notice Harness wrapper for sparse matrix × dense vector multiplication.
     */
    function mulSparseMatrixVectorHarness(
        uint256 aRows,
        uint256 aCols,
        uint256[] calldata rowPtr,
        uint256[] calldata colInd,
        bytes16[] calldata values,
        uint256 xRows,
        bytes16[] calldata xData
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.SparseMatrix memory A = MatrixMaster.SparseMatrix({
            rows: aRows,
            cols: aCols,
            rowPtr: _toMemory(rowPtr),
            colInd: _toMemory(colInd),
            values: _toMemory(values)
        });

        MatrixMaster.Matrix memory x = MatrixMaster.Matrix({
            rows: xRows,
            cols: 1,
            data: _toMemory(xData)
        });

        MatrixMaster.Matrix memory y = MatrixMaster.mulSparseMatrixVector(A, x);

        return (y.rows, y.cols, y.data);
    }

    // ---------------------------------------------------------
    // Vector dot product wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for dot.
     */
    function dotHarness(
        uint256 aRows,
        uint256 aCols,
        bytes16[] calldata aData,
        uint256 bRows,
        uint256 bCols,
        bytes16[] calldata bData
    ) external pure returns (bytes16 d) {
        MatrixMaster.Matrix memory a = _toMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory b = _toMatrix(bRows, bCols, bData);
        return MatrixMaster.dot(a, b);
    }

    // ---------------------------------------------------------
    // Determinant wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for det.
     */
    function detHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat
    ) external pure returns (bytes16) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        return MatrixMaster.det(a);
    }

    // ---------------------------------------------------------
    // Inverse wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for inverse.
     */
    function inverseHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory invA = MatrixMaster.inverse(a);
        return _fromMatrix(invA);
    }

    // ---------------------------------------------------------
    // Vector euclideanNorm (l_2) & normalization wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for euclideanNorm.
     *         Input must be vector-shaped: (n×1) or (1×n).
     * @return norm  l_2-vector norm ||v||_2
     */
    function euclideanNormHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat
    ) external pure returns (bytes16 norm) {
        MatrixMaster.Matrix memory v = _toMatrix(rows, cols, dataFlat);
        return MatrixMaster.euclideanNorm(v);
    }

    /**
     * @notice Wrapper for normalize, computing v / ||v||_2.
     *         Reverts on zero vector.
     * @return (rows, cols, data)  normalized vector with unit l_2 norm
     */
    function normalizeHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory out = MatrixMaster.normalize(v);
        return _fromMatrix(out);
    }

    // ---------------------------------------------------------
    // Vector creation wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for createVector (n × 1).
     */
    function createVectorHarness(
        bytes16[] calldata data
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = MatrixMaster.createVector(data);
        return (v.rows, v.cols, v.data);
    }

    // ---------------------------------------------------------
    // Vector randomization wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for randomVector (n × 1).
     */
    function randomVectorHarness(
        uint256 n,
        bytes32 seed
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = MatrixMaster.randomVector(n, seed);
        return _fromMatrix(v);
    }

    // ---------------------------------------------------------
    // Convergence check wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for hasConverged.
     */
    function hasConvergedHarness(
        uint256 xRows,
        uint256 xCols,
        bytes16[] calldata xData,
        uint256 yRows,
        uint256 yCols,
        bytes16[] calldata yData,
        bytes16 tol
    ) external pure returns (bool ok) {
        MatrixMaster.Matrix memory x = _toMatrix(xRows, xCols, xData);
        MatrixMaster.Matrix memory y = _toMatrix(yRows, yCols, yData);
        return MatrixMaster.hasConverged(x, y, tol);
    }

    // ---------------------------------------------------------
    // Power Iteration Wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for production-configured power iteration with status.
     */
    function powerIterationWithStatusHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        bytes32 seed,
        bytes16 tol
    )
        external
        view
        returns (
            bytes16 lambda,
            uint256 xRows,
            uint256 xCols,
            bytes16[] memory xData,
            uint256 iterCount,
            bool converged
        )
    {
        MatrixMaster.Matrix memory A = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.PowerIterationResult memory result = MatrixMaster.powerIterationWithStatus(A, seed, tol);
        return (
            result.lambda,
            result.eigenvector.rows,
            result.eigenvector.cols,
            result.eigenvector.data,
            result.iterations,
            result.converged
        );
    }

    /**
     * @notice Wrapper for bounded power iteration with convergence status.
     */
    function powerIterationWithStatusAndMaxIterHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        bytes32 seed,
        bytes16 tol,
        uint256 maxIter
    )
        external
        view
        returns (
            bytes16 lambda,
            uint256 xRows,
            uint256 xCols,
            bytes16[] memory xData,
            uint256 iterCount,
            bool converged
        )
    {
        MatrixMaster.Matrix memory A = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.PowerIterationResult memory result = MatrixMaster
            .powerIterationWithStatusAndMaxIter(A, seed, tol, maxIter);
        return (
            result.lambda,
            result.eigenvector.rows,
            result.eigenvector.cols,
            result.eigenvector.data,
            result.iterations,
            result.converged
        );
    }

    // ---------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------

    /**
     * @notice Approximate comparison of two quad values with tolerance.
     * @param a    First value
     * @param b    Second value
     * @param tol  Allowed absolute deviation |a-b| ≤ tol
     * @return ok  True if within tolerance
     */
    function qApprox(
        bytes16 a,
        bytes16 b,
        bytes16 tol
    ) public pure returns (bool ok) {
        bytes16 diff = a.sub(b);
        bytes16 absDiff = MathLib.abs(diff);
        // absDiff <= tol  <=>  cmp(absDiff, tol) <= 0
        return MathLib.cmp(absDiff, tol) <= 0;
    }

    // ------------------------------------------------------------
    //  Numerical Helpers
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to IEEE-754 quadruple precision.
     * @param  n Signed integer.
     * @return q Quadruple-precision representation of 'n'.
     */
    function qFromInt(int256 n) external pure returns (bytes16 q) {
        q = MathLib.fromInt(n);
    }

    /**
     * @notice Converts a uint256 into its quad-precision representation.
     * @param x Unsigned integer value.
     * @return q Quad-precision number representing x.
     */
    function qFromUInt(uint256 x) external pure returns (bytes16) {
        return MathLib.fromUInt(x);
    }

    /**
     * @notice Converts a rational number num/den to quadruple precision.
     *         Reverts if 'den' equals zero.
     * @param num Signed numerator.
     * @param den Signed denominator (must be non-zero).
     * @return q  Quadruple-precision value representing num/den.
     */
    function qFromFrac(
        int256 num,
        int256 den
    ) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = MathLib.div(qNum, qDen);
    }

    /// Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e12;
    
    /**
     * @notice Converts a quadruple-precision number into a scaled integer (scaled by SCALE).
     * @param x Quadruple-precision value.
     * @dev MathLib.toInt truncates toward zero. This is a 1e12 fixed-point
     *      reporting helper, not a binary128-precision conversion.
     * @return Integer representing trunc(x * SCALE).
     */
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }

    /**
     * @notice Converts a scaled integer (scaled by SCALE) into quadruple precision.
     *         Example: scaledValue = 1234500000000 → represents 1.2345.
     * @param scaledValue Integer representing a float multiplied by SCALE.
     * @return Quadruple-precision value.
     */
    function fromFloat(int256 scaledValue) external pure returns (bytes16) {
        bytes16 qInt = MathLib.fromInt(scaledValue);
        bytes16 qScale = MathLib.fromUInt(SCALE);
        return MathLib.div(qInt, qScale);
    }
}
