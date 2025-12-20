// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MatrixMaster } from "../../libraries/numeric/MatrixMaster.sol";

/**
 * @title MatrixMasterFacet
 * @notice External Diamond Facet exposing all MatrixMaster utilities.
 *          - All return values follow the format: (rows, cols, data[]).
 *          - For matrix inputs, calldata → memory conversion is required
 *            because MatrixMaster expects 'bytes16[] memory'.
 *          - For vectors: treat them as matrices with shape (n×1) or (1×n).
 */
contract MatrixFacet {

    // ------------------------------------------------------------
    // Creation
    // ------------------------------------------------------------

    /**
     * @notice Create a rowsxcols matrix filled with zeros.
     * @param  rows Number of rows
     * @param  cols Number of columns
     * @return m    New zero matrix
     */
    function zeros(uint256 rows, uint256 cols) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.zeros(rows, cols));
    }

    /**
     * @notice Create a rowsxcols matrix filled with ones.
     * @param  rows Number of rows
     * @param  cols Number of columns
     * @return m    New matrix with all entries = 1.0
     */
    function ones(uint256 rows, uint256 cols) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.ones(rows, cols));
    }

    /**
     * @notice Create an nxn identity matrix.
     * @param  n Matrix dimension (n > 0)
     * @return m Identity matrix with 1 on diagonal and 0 elsewhere
     */
    function createIdentityMatrix(uint256 n) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.createIdentityMatrix(n));
    }

    /**
     * @notice Create a diagonal matrix from diagonal entries.
     * @param  diag Array of diagonal entries, length = n
     * @return m    nxn matrix with diag[i] on (i,i), zeros elsewhere
     */
    function fromDiagonal(bytes16[] calldata diag) external pure returns (uint256, uint256, bytes16[] memory) {
        bytes16[] memory d = _toMemory(diag);
        return _flatten(MatrixMaster.fromDiagonal(d));
    }

    /**
     * @notice Create a rowsxcols matrix with deterministic pseudo-random values.
     *         This is NOT secure randomness. It is intended only for testing and deterministic benchmarks.
     *         Values are in [0, 1) scaled from keccak256(seed, i).
     *
     * @param  rows Number of rows
     * @param  cols Number of columns
     * @param  seed Arbitrary seed for deterministic generation
     * @return m    New matrix with pseudo-random contents
     */
    function randomMatrix(uint256 rows, uint256 cols, bytes32 seed) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.randomMatrix(rows, cols, seed));
    }

    // ------------------------------------------------------------
    // Element Access
    // ------------------------------------------------------------

    /**
     * @notice Get element at (row, col).
     * @param  rows  Number of rows of the matrix
     * @param  cols  Number of columns of the matrix
     * @param  data  Flattened matrix data (row-major)
     * @param  row   Zero-based row index
     * @param  col   Zero-based column index
     * @return value Element at (row, col)
     */
    function get(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 row, uint256 col) external pure returns (bytes16) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return MatrixMaster.get(m, row, col);
    }

    /**
     * @notice Set element at (row, col) to 'val'.
     *         Mutates the matrix in-place (in memory).
     * @param  rows Number of rows of the matrix
     * @param  cols Number of columns of the matrix
     * @param  data Flattened matrix data (row-major)
     * @param  row  Zero-based row index
     * @param  col  Zero-based column index
     * @param  val  New value to write
     * @return rowsOut Number of rows of the updated matrix
     * @return colsOut Number of columns of the updated matrix
     * @return dataOut Flattened updated matrix data
     */
    function set(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 row, uint256 col, bytes16 val)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        MatrixMaster.set(m, row, col, val);
        return _flatten(m);
    }

    // ------------------------------------------------------------
    // Slicing / Reshape
    // ------------------------------------------------------------

    /**
     * @notice Extract a contiguous submatrix [rowStart:rowEnd, colStart:colEnd).
     *         Requires 0 <= rowStart < rowEnd <= rows, same for columns.
     * @param  rows     Number of rows of the source matrix
     * @param  cols     Number of columns of the source matrix
     * @param  data     Flattened source matrix data (row-major)
     * @param  rowStart Inclusive starting row
     * @param  rowEnd   Exclusive ending row
     * @param  colStart Inclusive starting column
     * @param  colEnd   Exclusive ending column
     * @return out      New sliced matrix
     */
    function sliceMatrix(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 rowStart, uint256 rowEnd, uint256 colStart, uint256 colEnd)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(
            MatrixMaster.slice(m, rowStart, rowEnd, colStart, colEnd)
        );
    }

    /**
     * @notice Reshape matrix to newRowsxnewCols without changing data order.
     *         Requires newRows * newCols == rows * cols.
     *         Reuses the same 'data' array (no copy).
     * @param  rows      Original number of rows
     * @param  cols      Original number of columns
     * @param  data      Flattened matrix data (row-major)
     * @param  newRows   New number of rows
     * @param  newCols   New number of columns
     * @return reshaped  Reshaped matrix view
     */
    function reshape(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 newRows, uint256 newCols)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.reshape(m, newRows, newCols));
    }

    // ------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------

    /**
     * @notice Compute the transpose A^T.
     * @param  rows Number of rows of matrix A
     * @param  cols Number of columns of matrix A
     * @param  data Flattened matrix data (row-major)
     * @return t    Transposed matrix
     */
    function transpose(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.transpose(m));
    }

    // ------------------------------------------------------------
    // Elementwise Arithmetic
    // ------------------------------------------------------------

    /**
     * @notice Elementwise addition C = A + B.
     * @param  aRows Number of rows of matrix A
     * @param  aCols Number of columns of matrix A
     * @param  aData Flattened matrix A data
     * @param  bRows Number of rows of matrix B
     * @param  bCols Number of columns of matrix B
     * @param  bData Flattened matrix B data
     * @return c     Result matrix A + B
     */
    function add(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.add(A, B));
    }

    /**
     * @notice Elementwise subtraction C = A − B.
     * @param  aRows Number of rows of matrix A
     * @param  aCols Number of columns of matrix A
     * @param  aData Flattened matrix A data
     * @param  bRows Number of rows of matrix B
     * @param  bCols Number of columns of matrix B
     * @param  bData Flattened matrix B data
     * @return c     Result matrix A − B
     */
    function sub(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.sub(A, B));
    }

    /**
     * @notice Scalar multiplication C = k · A.
     * @param  rows Number of rows of the matrix
     * @param  cols Number of columns of the matrix
     * @param  data Flattened matrix data
     * @param  k    Scalar multiplier
     * @return c    Result matrix
     */
    function mulScalar(uint256 rows, uint256 cols, bytes16[] calldata data, bytes16 k) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.mulScalar(m, k));
    }

    /**
     * @notice   Scalar division C = A / k.
     * @param  rows Number of rows of the matrix
     * @param  cols Number of columns of the matrix
     * @param  data Flattened matrix data
     * @param  k    Scalar divisor
     * @return c    Result matrix
     */
    function divScalar(uint256 rows, uint256 cols, bytes16[] calldata data, bytes16 k) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.divScalar(m, k));
    }

    // ------------------------------------------------------------
    // Matrix Multiplication
    // ------------------------------------------------------------

    /**
     * @notice Matrix-matrix multiplication C = A · B.
     *         A is (mxk), B is (kxn), result is (mxn).
     *         Uses straightforward O(m·k·n) triple loop.
     * @param  aRows Number of rows of matrix A
     * @param  aCols Number of columns of matrix A
     * @param  aData Flattened matrix A data
     * @param  bRows Number of rows of matrix B
     * @param  bCols Number of columns of matrix B
     * @param  bData Flattened matrix B data
     * @return c     Product matrix A · B
     */
    function mulMatrix(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.mulMatrix(A, B));
    }

    // ------------------------------------------------------------
    // Matrix x Vector multiplication
    // ------------------------------------------------------------

    /**
     * @notice Multiply matrix A (mxn) by vector x (nx1). Result is (mx1).
     *         Much faster than full matrixxmatrix mulMatrix for power iteration.
     * @param  aRows Number of rows of matrix A
     * @param  aCols Number of columns of matrix A
     * @param  aData Flattened matrix A data
     * @param  xRows Number of rows of vector x
     * @param  xCols Number of columns of vector x
     * @param  xData Flattened vector x data
     * @return y     Result vector A · x
     */
    function mulMatrixVector(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 xRows, uint256 xCols, bytes16[] calldata xData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory x = _buildMatrix(xRows, xCols, xData);
        return _flatten(MatrixMaster.mulMatrixVector(A, x));
    }


    // ------------------------------------------------------------
    // Vector Operations (n×1 column vectors)
    // ------------------------------------------------------------

    /**
     * @notice Create a pseudo-random column vector (nx1) with entries in [0,1).
     *         Uses deterministic keccak-based generation, NOT secure randomness.
     * @param  n    Dimension of the vector (n > 0)
     * @param  seed Seed value used to generate pseudo-random entries
     * @return v    Column vector (nx1) with pseudo-random entries
     */
    function randomVector(uint256 n, bytes32 seed) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = MatrixMaster.randomVector(n, seed);
        return _flatten(v);
    }

    /**
     * @notice Compute dot product of two column vectors (nx1).
     *         Requires both vectors to be column vectors with identical lengths.
     * @param  v1Rows Number of rows of first vector
     * @param  v1Cols Number of columns of first vector
     * @param  v1Data Flattened first vector data
     * @param  v2Rows Number of rows of second vector
     * @param  v2Cols Number of columns of second vector
     * @param  v2Data Flattened second vector data
     * @return s      Scalar dot product
     */
    function dot(uint256 v1Rows, uint256 v1Cols, bytes16[] calldata v1Data, uint256 v2Rows, uint256 v2Cols, bytes16[] calldata v2Data)
        external pure returns (bytes16) {

        MatrixMaster.Matrix memory v1 = _buildMatrix(v1Rows, v1Cols, v1Data);
        MatrixMaster.Matrix memory v2 = _buildMatrix(v2Rows, v2Cols, v2Data);
        return MatrixMaster.dot(v1, v2);
    }

    /**
     * @notice Compute Euclidean 2-norm ||v||_2 of a column vector (nx1).
     * @param  rows Number of rows of the vector
     * @param  cols Number of columns of the vector
     * @param  data Flattened vector data
     * @return nrm  Euclidean 2-norm of the vector
     */
    function euclideanNorm(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (bytes16) {
        MatrixMaster.Matrix memory v = _buildMatrix(rows, cols, data);
        return MatrixMaster.euclideanNorm(v);
    }

    /**
     * @notice Normalize a column vector v into v / ||v||_2.
     * @dev    Reverts if vector has zero norm.
     * @param  rows Number of rows of the vector
     * @param  cols Number of columns of the vector
     * @param  data Flattened vector data
     * @return out  Normalized vector with unit norm
     */
    function normalize(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.normalize(v));
    }

    /**
     * @notice Check approximate convergence between two vectors via L2 tolerance.
     *         Computes ||vNew - vOld||_2 and compares to 'tol'.
     * @param  xRows Number of rows of vNew
     * @param  xCols Number of columns of vNew
     * @param  xData Flattened vNew data
     * @param  yRows Number of rows of vOld
     * @param  yCols Number of columns of vOld
     * @param  yData Flattened vOld data
     * @param  tol   Convergence tolerance
     * @return ok    True if converged
     */
    function hasConverged(uint256 xRows, uint256 xCols, bytes16[] calldata xData, uint256 yRows, uint256 yCols, bytes16[] calldata yData, bytes16 tol)
        external pure returns (bool) {

        MatrixMaster.Matrix memory vNew = _buildMatrix(xRows, xCols, xData);
        MatrixMaster.Matrix memory vOld = _buildMatrix(yRows, yCols, yData);
        return MatrixMaster.hasConverged(vNew, vOld, tol);
    }

    // ------------------------------------------------------------
    // Determinant via LU decomposition with partial pivoting
    // ------------------------------------------------------------

    /**
     * @notice Compute determinant det(A) of a square matrix using LU decomposition.
     *         Performs in-place LU on a working copy of A (Doolittle-style).
     *         Uses partial pivoting; sign of permutation affects determinant sign.
     * @param rows  Number of rows of the square matrix
     * @param cols  Number of columns of the square matrix
     * @param data  Flattened matrix data
     * @return detA Determinant of the matrix
     */
    function det(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (bytes16) {
        MatrixMaster.Matrix memory A = _buildMatrix(rows, cols, data);
        return MatrixMaster.det(A);
    }

    // ------------------------------------------------------------
    // Inversion via Gauss–Jordan elimination
    // ------------------------------------------------------------

    /**
     * @notice Compute A⁻¹ using Gauss–Jordan elimination on [A | I].
     *         Reverts if matrix is singular (no pivot above a small threshold).
     *         This is O(n^3) and intended for small/moderate n in off-chain-style usage.
     * @param  rows Number of rows of the square matrix
     * @param  cols Number of columns of the square matrix
     * @param  data Flattened matrix data
     * @return invA Inverse matrix
     */
    function inverse(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory A = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.inverse(A));
    }

    // ------------------------------------------------------------
    // Eigenvalues          
    // ------------------------------------------------------------

    /**
     * @notice Approximate the dominant eigenvalue/eigenvector of a square matrix A using power iteration.
     *
     * @dev Steps:
     *       1. Initialize x₀ using randomVector(n, seed).
     *       2. Repeat:
     *            y = A * x
     *            x_new = normalize(y)
     *            stop if hasConverged(x_new, x, tol)
     *       3. lambda ≈ x^T (A x)
     *     - A must be (nxn)
     *     - Uses cfg().maxIter from LibNumericConfig
     *     - tol must be > 0
     *
     * @param  rows   Number of rows of matrix A
     * @param  cols   Number of columns of matrix A
     * @param  data   Flattened matrix A data
     * @param  seed   Seed for deterministic initial vector
     * @param  tol    Convergence tolerance
     * @return lambda Dominant eigenvalue approximation
     * @return vecRows Number of rows of the eigenvector
     * @return vecCols Number of columns of the eigenvector
     * @return vecData Flattened eigenvector data
     */
    function powerIteration(uint256 rows, uint256 cols, bytes16[] calldata data, bytes32 seed, bytes16 tol)
        external view returns (bytes16 lambda, uint256 vecRows, uint256 vecCols, bytes16[] memory vecData) {

        MatrixMaster.Matrix memory A = _buildMatrix(rows, cols, data);
        
        // Call library
        (bytes16 l, MatrixMaster.Matrix memory v) = MatrixMaster.powerIteration(A, seed, tol);

        // Flatten results manually since return type is mixed (bytes16, Matrix)
        return (l, v.rows, v.cols, v.data);
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Copy calldata bytes16[] to a new memory array.
     */ 
    function _toMemory(bytes16[] calldata a) internal pure returns (bytes16[] memory m) {
        m = new bytes16[](a.length);
        for (uint256 i = 0; i < a.length; ++i) m[i] = a[i];
    }

    /**
     * @notice Build a MatrixMaster.Matrix from calldata.
     */ 
    function _buildMatrix(uint256 rows, uint256 cols, bytes16[] calldata data) internal pure returns (MatrixMaster.Matrix memory m) {
        require(rows * cols == data.length, "MatrixFacet: shape mismatch");
        m = MatrixMaster.Matrix({
            rows: rows,
            cols: cols,
            data: _toMemory(data)
        });
    }

    /**
     * @notice Flatten Matrix into returnable tuple.
     */ 
    function _flatten(MatrixMaster.Matrix memory m) internal pure returns (uint256, uint256, bytes16[] memory) {
        return (m.rows, m.cols, m.data);
    }
}