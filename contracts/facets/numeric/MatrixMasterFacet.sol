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
     * @param  m     Matrix to read
     * @param  row   Zero-based row index
     * @param  col   Zero-based column index
     * @return value Element m[row, col]
     */
    function get(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 row, uint256 col) external pure returns (bytes16) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return MatrixMaster.get(m, row, col);
    }

    /**
     * @notice Set element at (row, col) to 'val'.
     *         Mutates the matrix in-place (in memory).
     * @param m     Matrix to modify
     * @param row   Zero-based row index
     * @param col   Zero-based column index
     * @param val   New value to write
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
     * @param  m         Source matrix
     * @param  rowStart  Inclusive starting row
     * @param  rowEnd    Exclusive ending row
     * @param  colStart  Inclusive starting column
     * @param  colEnd    Exclusive ending column
     * @return out       New matrix with shape (rowEnd-rowStart) x (colEnd-colStart)
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
     * @param m        Input matrix
     * @param newRows  New number of rows
     * @param newCols  New number of columns
     * @return reshaped Matrix view with updated shape over same data
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
     * @notice Compute the transpose Aᵀ.
     * @param  a Input matrix A (rowsxcols)
     * @return t Transposed matrix (colsxrows)
     */
    function transpose(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.transpose(m));
    }

    // ------------------------------------------------------------
    // Elementwise Arithmetic
    // ------------------------------------------------------------

    /**
     * @notice   Elementwise addition C = A + B.
     * @param  a Left operand
     * @param  b Right operand
     * @return c Result matrix with same shape
     */
    function add(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.add(A, B));
    }

    /**
     * @notice   Elementwise subtraction C = A − B.
     * @param  a Left operand
     * @param  b Right operand
     * @return c Result matrix with same shape
     */
    function sub(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.sub(A, B));
    }

    /**
     * @notice   Scalar multiplication C = k · A.
     * @param  a Input matrix
     * @param  k Scalar multiplier
     * @return c Result matrix
     */
    function mulScalar(uint256 rows, uint256 cols, bytes16[] calldata data, bytes16 k) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.mulScalar(m, k));
    }

    /**
     * @notice   Scalar division C = A / k.
     * @param  a Input matrix
     * @param  k Scalar divisor (must be non-zero)
     * @return c Result matrix
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
     * @param  a Left operand matrix
     * @param  b Right operand matrix
     * @return c Product matrix
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
     *
     * @param  A Matrix (mxn)
     * @param  x Vector as matrix (nx1)
     * @return y = A * x (mx1 column vector)
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
     * @param  a First vector (nx1)
     * @param  b Second vector (nx1)
     * @return s Scalar dot product = Σ_i a[i] * b[i]
     */
    function dot(uint256 v1Rows, uint256 v1Cols, bytes16[] calldata v1Data, uint256 v2Rows, uint256 v2Cols, bytes16[] calldata v2Data)
        external pure returns (bytes16) {

        MatrixMaster.Matrix memory v1 = _buildMatrix(v1Rows, v1Cols, v1Data);
        MatrixMaster.Matrix memory v2 = _buildMatrix(v2Rows, v2Cols, v2Data);
        return MatrixMaster.dot(v1, v2);
    }

    /**
     * @notice Compute Euclidean 2-norm ||v||_2 of a column vector (nx1).
     * @param  v   Input vector (nx1)
     * @return nrm Vector norm = sqrt(dot(v, v))
     */
    function euclideanNorm(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (bytes16) {
        MatrixMaster.Matrix memory v = _buildMatrix(rows, cols, data);
        return MatrixMaster.euclideanNorm(v);
    }

    /**
     * @notice Normalize a column vector v into v / ||v||_2.
     * @dev    Reverts if vector has zero norm.
     * @param  v   Input vector (nx1)
     * @return out Normalized vector with unit Euclidean norm
     */
    function normalize(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.normalize(v));
    }

    /**
     * @notice Check approximate convergence between two vectors via L2 tolerance.
     *         Computes ||vNew - vOld||_2 and compares to 'tol'.
     * @param  vNew New iterate (nx1)
     * @param  vOld Previous iterate (nx1)
     * @param  tol  Convergence tolerance (positive scalar)
     * @return ok   True if ||vNew - vOld|| < tol
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
     * @param  a    Input square matrix
     * @return detA Determinant as bytes16
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
     * @param  a    Input square matrix A
     * @return invA Inverse matrix A⁻¹
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
     * @param A     Input square matrix
     * @param seed  Seed for deterministic initial vector
     * @param tol   Convergence tolerance
     *
     * @return lambda    Dominant eigenvalue approximation
     * @return x         Dominant eigenvector (nx1 unit vector)
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