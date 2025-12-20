// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";
import { MatrixMaster } from "../libraries/numeric/MatrixMaster.sol";

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

        m = MatrixMaster.Matrix({ rows: rows, cols: cols, data: data });
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

    // ---------------------------------------------------------
    // Creation wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for MatrixMaster.zeros.
     */
    function zerosHarness(
        uint256 rows,
        uint256 cols
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = MatrixMaster.zeros(rows, cols);
        return _fromMatrix(m);
    }

    /**
     * @notice Wrapper for MatrixMaster.ones.
     */
    function onesHarness(
        uint256 rows,
        uint256 cols
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = MatrixMaster.ones(rows, cols);
        return _fromMatrix(m);
    }

    /**
     * @notice Wrapper for MatrixMaster.createIdentityMatrix.
     */
    function createIdentityMatrixHarness(
        uint256 n
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = MatrixMaster.createIdentityMatrix(n);
        return _fromMatrix(m);
    }

    /**
     * @notice Wrapper for MatrixMaster.fromDiagonal.
     */
    function fromDiagonalHarness(
        bytes16[] calldata diag
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        // Copy diag into memory for the library
        bytes16[] memory d = new bytes16[](diag.length);
        for (uint256 i = 0; i < diag.length; ++i) {
            d[i] = diag[i];
        }

        MatrixMaster.Matrix memory m = MatrixMaster.fromDiagonal(d);
        return _fromMatrix(m);
    }

    /**
     * @notice Wrapper for MatrixMaster.randomMatrix.
     */
    function randomMatrixHarness(
        uint256 rows,
        uint256 cols,
        bytes32 seed
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = MatrixMaster.randomMatrix(
            rows,
            cols,
            seed
        );
        return _fromMatrix(m);
    }

    // ---------------------------------------------------------
    // Element access wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for MatrixMaster.get.
     */
    function getHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        uint256 row,
        uint256 col
    ) external pure returns (bytes16) {
        MatrixMaster.Matrix memory m = _toMatrix(rows, cols, dataFlat);
        return MatrixMaster.get(m, row, col);
    }

    /**
     * @notice Wrapper for MatrixMaster.set.
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
        MatrixMaster.set(m, row, col, val);
        return _fromMatrix(m);
    }

    // ---------------------------------------------------------
    // Slice & reshape wrappers
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for MatrixMaster.slice.
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
     * @notice Wrapper for MatrixMaster.reshape.
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
     * @notice Wrapper for MatrixMaster.transpose.
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
     * @notice Wrapper for MatrixMaster.add.
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
        MatrixMaster.Matrix memory c = MatrixMaster.add(a, b);
        return _fromMatrix(c);
    }

    /**
     * @notice Wrapper for MatrixMaster.sub.
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
        MatrixMaster.Matrix memory c = MatrixMaster.sub(a, b);
        return _fromMatrix(c);
    }

    /**
     * @notice Wrapper for MatrixMaster.mulScalar.
     */
    function mulScalarHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        bytes16 k
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory c = MatrixMaster.mulScalar(a, k);
        return _fromMatrix(c);
    }

    /**
     * @notice Wrapper for MatrixMaster.divScalar.
     */
    function divScalarHarness(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata dataFlat,
        bytes16 k
    ) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory a = _toMatrix(rows, cols, dataFlat);
        MatrixMaster.Matrix memory c = MatrixMaster.divScalar(a, k);
        return _fromMatrix(c);
    }

    // ---------------------------------------------------------
    // Matrix multiplication wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for MatrixMaster.mul.
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
        MatrixMaster.Matrix memory c = MatrixMaster.mulMatrix(a, b);
        return _fromMatrix(c);
    }

    // ---------------------------------------------------------
    // Matrix-Vector multiplication wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for MatrixMaster.mulMatrixVector.
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
        MatrixMaster.Matrix memory c = MatrixMaster.mulMatrixVector(a, b);
        return _fromMatrix(c);
    }

    // ---------------------------------------------------------
    // Vector dot product wrapper
    // ---------------------------------------------------------

    /**
     * @notice Wrapper for MatrixMaster.dot.
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
     * @notice Wrapper for MatrixMaster.det.
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
     * @notice Wrapper for MatrixMaster.inverse.
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
     * @notice Wrapper for MatrixMaster.euclideanNorm.
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
     * @notice Wrapper for MatrixMaster.normalize, computing v / ||v||_2.
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
    // Vector randomization wrapper
    // ---------------------------------------------------------

    /**
    * @notice Wrapper for MatrixMaster.randomVector (n × 1).
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
     * @notice Wrapper for MatrixMaster.hasConverged.
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
    * @notice Wrapper for MatrixMaster.powerIteration.
    */
    function powerIterationHarness(
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
            bytes16[] memory xData
        )
    {
        // Build matrix
        MatrixMaster.Matrix memory A = _toMatrix(rows, cols, dataFlat);

        // Compute dominant eigenpair
        (bytes16 lam, MatrixMaster.Matrix memory x) =
            MatrixMaster.powerIteration(A, seed, tol);

        return (lam, x.rows, x.cols, x.data);
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
    function qFromFrac(int256 num, int256 den) external pure returns (bytes16 q) {
        require(den != 0, "den=0");
        bytes16 qNum = MathLib.fromInt(num);
        bytes16 qDen = MathLib.fromInt(den);
        q = MathLib.div(qNum, qDen);
    }
}