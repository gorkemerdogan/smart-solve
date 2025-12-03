// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MatrixMaster } from "../../libraries/numeric/MatrixMaster.sol";

/**
 * @title MatrixMasterFacet
 * @notice External Diamond Facet exposing all MatrixMaster utilities.
 *          - All return values follow the format: (rows, cols, data[]).
 *          - For matrix inputs, calldata → memory conversion is required
 *            because MatrixMaster expects `bytes16[] memory`.
 *          - For vectors: treat them as matrices with shape (n×1) or (1×n).
 */
contract MatrixFacet {

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Copy calldata bytes16[] to a new memory array.
     */ 
    function _toMemory(bytes16[] calldata a)
        internal
        pure
        returns (bytes16[] memory m)
    {
        m = new bytes16[](a.length);
        for (uint256 i = 0; i < a.length; ++i) m[i] = a[i];
    }

    /**
     * @notice Build a MatrixMaster.Matrix from calldata.
     */ 
    function _buildMatrix(
        uint256 rows,
        uint256 cols,
        bytes16[] calldata data
    )
        internal
        pure
        returns (MatrixMaster.Matrix memory m)
    {
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
    function _flatten(MatrixMaster.Matrix memory m)
        internal
        pure
        returns (uint256, uint256, bytes16[] memory)
    {
        return (m.rows, m.cols, m.data);
    }

    // ------------------------------------------------------------
    // Creation
    // ------------------------------------------------------------

    function zeros(uint256 rows, uint256 cols) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.zeros(rows, cols));
    }

    function ones(uint256 rows, uint256 cols) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.ones(rows, cols));
    }

    function createIdentityMatrix(uint256 n) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.createIdentityMatrix(n));
    }

    function fromDiagonal(bytes16[] calldata diag) external pure returns (uint256, uint256, bytes16[] memory) {
        bytes16[] memory d = _toMemory(diag);
        return _flatten(MatrixMaster.fromDiagonal(d));
    }

    function randomMatrix(uint256 rows, uint256 cols, bytes32 seed) external pure returns (uint256, uint256, bytes16[] memory) {
        return _flatten(MatrixMaster.randomMatrix(rows, cols, seed));
    }

    // ------------------------------------------------------------
    // Element Access
    // ------------------------------------------------------------

    function get(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 row, uint256 col) external pure returns (bytes16) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return MatrixMaster.get(m, row, col);
    }

    /**
     * @notice Creates a new matrix in memory and returns it.
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

    function sliceMatrix(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 rowStart, uint256 rowEnd, uint256 colStart, uint256 colEnd)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(
            MatrixMaster.slice(m, rowStart, rowEnd, colStart, colEnd)
        );
    }

    function reshape(uint256 rows, uint256 cols, bytes16[] calldata data, uint256 newRows, uint256 newCols)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.reshape(m, newRows, newCols));
    }

    // ------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------

    function transpose(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.transpose(m));
    }

    // ------------------------------------------------------------
    // Elementwise Arithmetic
    // ------------------------------------------------------------

    function add(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.add(A, B));
    }

    function sub(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.sub(A, B));
    }

    function mulScalar(uint256 rows, uint256 cols, bytes16[] calldata data, bytes16 k) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.mulScalar(m, k));
    }

    function divScalar(uint256 rows, uint256 cols, bytes16[] calldata data, bytes16 k) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory m = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.divScalar(m, k));
    }

    // ------------------------------------------------------------
    // Matrix Multiplication
    // ------------------------------------------------------------

    function mulMatrix(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 bRows, uint256 bCols, bytes16[] calldata bData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory B = _buildMatrix(bRows, bCols, bData);
        return _flatten(MatrixMaster.mulMatrix(A, B));
    }

    // ------------------------------------------------------------
    // Determinant / Inverse
    // ------------------------------------------------------------

    function det(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (bytes16) {
        MatrixMaster.Matrix memory A = _buildMatrix(rows, cols, data);
        return MatrixMaster.det(A);
    }

    function inverse(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory A = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.inverse(A));
    }

    // ------------------------------------------------------------
    // Vector Operations (n×1 column vectors)
    // ------------------------------------------------------------

    /**
     * @notice Create a pseudo-random column vector (n×1) with entries in [0,1).
     *         Deterministic keccak-based generation. NOT secure randomness.
     */
    function randomVector(uint256 n, bytes32 seed) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = MatrixMaster.randomVector(n, seed);
        return _flatten(v);
    }

    /**
     * @notice Multiply matrix A (m×n) by vector x (n×1). Result is (m×1).
     */
    function mulMatrixVector(uint256 aRows, uint256 aCols, bytes16[] calldata aData, uint256 xRows, uint256 xCols, bytes16[] calldata xData)
        external pure returns (uint256, uint256, bytes16[] memory) {

        MatrixMaster.Matrix memory A = _buildMatrix(aRows, aCols, aData);
        MatrixMaster.Matrix memory x = _buildMatrix(xRows, xCols, xData);
        return _flatten(MatrixMaster.mulMatrixVector(A, x));
    }

    /**
     * @notice Compute dot product of two column vectors (n×1).
     *         Reverts on shape mismatch or non-vector inputs.
     */
    function dot(uint256 v1Rows, uint256 v1Cols, bytes16[] calldata v1Data, uint256 v2Rows, uint256 v2Cols, bytes16[] calldata v2Data)
        external pure returns (bytes16) {

        MatrixMaster.Matrix memory v1 = _buildMatrix(v1Rows, v1Cols, v1Data);
        MatrixMaster.Matrix memory v2 = _buildMatrix(v2Rows, v2Cols, v2Data);
        return MatrixMaster.dot(v1, v2);
    }

    /**
     * @notice Compute Euclidean 2-norm ||v||₂ of a column vector (n×1).
     */
    function euclideanNorm(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (bytes16) {
        MatrixMaster.Matrix memory v = _buildMatrix(rows, cols, data);
        return MatrixMaster.euclideanNorm(v);
    }

    /**
     * @notice Normalize a vector v into v / ||v||₂.
     *         Reverts on zero vector or non-vector shapes.
     */
    function normalize(uint256 rows, uint256 cols, bytes16[] calldata data) external pure returns (uint256, uint256, bytes16[] memory) {
        MatrixMaster.Matrix memory v = _buildMatrix(rows, cols, data);
        return _flatten(MatrixMaster.normalize(v));
    }

    /**
     * @notice Check approximate convergence between two (n×1) vectors:
     *         returns true if ||vNew - vOld||₂ < tol.
     */
    function hasConverged(uint256 xRows, uint256 xCols, bytes16[] calldata xData, uint256 yRows, uint256 yCols, bytes16[] calldata yData, bytes16 tol)
        external pure returns (bool) {

        MatrixMaster.Matrix memory vNew = _buildMatrix(xRows, xCols, xData);
        MatrixMaster.Matrix memory vOld = _buildMatrix(yRows, yCols, yData);
        return MatrixMaster.hasConverged(vNew, vOld, tol);
    }

    // ------------------------------------------------------------
    // Eigenvalues          
    // ------------------------------------------------------------

    /**
     * @notice Calculate dominant eigenvalue/vector via Power Iteration.
     */
    function powerIteration(uint256 rows, uint256 cols, bytes16[] calldata data, bytes32 seed, bytes16 tol)
        external view returns (bytes16 lambda, uint256 vecRows, uint256 vecCols, bytes16[] memory vecData) {

        MatrixMaster.Matrix memory A = _buildMatrix(rows, cols, data);
        
        // Call library
        (bytes16 l, MatrixMaster.Matrix memory v) = MatrixMaster.powerIteration(A, seed, tol);

        // Flatten results manually since return type is mixed (bytes16, Matrix)
        return (l, v.rows, v.cols, v.data);
    }
}