// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../MathLib.sol";
import { QuadConstants as QC } from "../QuadConstants.sol";
import { LibNumericConfig } from "../../storagelibs/LibNumericConfig.sol";

/**
 * @title MatrixMaster
 * @notice Dense matrix utilities using IEEE-754 binary128 (bytes16) scalars via MathLib.
 *
 * @dev
 *  - All matrices are stored in row-major order:
 *        index = row * cols + col
 *  - This library is designed for ephemeral numeric computation (off-chain
 *    simulation, benchmarking, or view calls), not for storing huge matrices
 *    permanently on-chain.
 *  - Uses strict dimension checks and reverts on mismatch.
 *  - Vectors are represented as 1xN or Nx1 matrices; no special type needed.
 */
library MatrixMaster {
    using MathLib for bytes16;

    bytes16 private constant QZERO = bytes16(0x00000000000000000000000000000000);
    bytes16 private constant QONE = bytes16(0x3fff0000000000000000000000000000);

    // ------------------------------------------------------------
    // Structs
    // ------------------------------------------------------------

    /**
    * @notice Dense matrix stored in row-major order.
    *         Represents a fully populated matrix of shape (rows × cols).
    *         
    *         Element (i, j) is located at data[i * cols + j].
    *
    * @custom:data
    *  - data.length == rows * cols
    */
    struct Matrix {
        uint256 rows;
        uint256 cols;
        bytes16[] data; // Flattened matrix entries in row-major order.
    }

    /**
    * @notice Sparse matrix stored in Compressed Sparse Row (CSR) format.
    *         Stores only non-zero elements to reduce memory usage and computation.
    *         Zero entries are implicit and are not stored.
    *
    * @custom:data
    *  - rowPtr.length == rows + 1
    *  - colInd.length == values.length
    *  - rowPtr[rows] == values.length
    *  - colInd indices within each row are sorted in ascending order
    */
    struct SparseMatrix {
        uint256 rows; /// Number of rows in the matrix.
        uint256 cols; /// Number of columns in the matrix.

        /// Row pointer array
        /// rowPtr[i] to rowPtr[i+1]-1 defines the index range of non-zero
        /// elements belonging to row i in colInd and values.
        uint256[] rowPtr;

        /// Column indices corresponding to each non-zero value
        /// colInd[k] is the column index of values[k].
        uint256[] colInd;

        /// values[k] corresponds to matrix entry (row, colInd[k]).
        bytes16[] values;
    }

    // ------------------------------------------------------------
    // Modifiers
    // ------------------------------------------------------------

    modifier validDims(uint256 r, uint256 c) {
        require(r > 0 && c > 0, "MatrixMaster: dims must be > 0");
        _;
    }

    /**
     * @notice Require that matrix is square (nxn).
     * @param  m Matrix to check
     */
    modifier isSquare(Matrix memory m) {
        require(m.rows == m.cols, "MatrixMaster: matrix must be square");
        _;
    }

    /**
     * @notice  Require that two matrices have identical shapes.
     * @param a First matrix
     * @param b Second matrix
     */
    modifier checkSameShape(Matrix memory a, Matrix memory b) {
        require(a.rows == b.rows && a.cols == b.cols, "MatrixMaster: shape mismatch");
        _;
    }

    /**
     * @notice    Basic bounds check for element access.
     * @param m   Matrix to check
     * @param row Row index to validate
     * @param col Column index to validate
     */
    modifier checkBounds(Matrix memory m, uint256 row, uint256 col) {
        require(row < m.rows && col < m.cols, "MatrixMaster: index out of bounds");
        _;
    }

    /**
     * @notice Require that 'a.cols == b.rows' for matrix multiplication.
     * @param a Left operand
     * @param b Right operand
     */
    modifier checkMulShape(Matrix memory a, Matrix memory b) {
        require(a.cols == b.rows, "MatrixMaster: multiplyMatrices dims a.cols != b.rows");
        _;
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @notice Compute row-major index into 'data' array.
     * @param  cols Number of columns in the matrix
     * @param  row  Zero-based row index
     * @param  col  Zero-based column index
     * @return idx  Linear index = row * cols + col
     */
    function _idx(uint256 cols, uint256 row, uint256 col) private pure returns (uint256 idx) {
        unchecked { return row * cols + col; }
    }

    // ------------------------------------------------------------
    // Creation
    // ------------------------------------------------------------

    /**
     * @notice Create a rowsxcols matrix filled with zeros.
     * @param  rows Number of rows
     * @param  cols Number of columns
     * @return m    New zero matrix
     */
    function createZerosMatrix(uint256 rows, uint256 cols) internal pure validDims(rows, cols) returns (Matrix memory m) {
        uint256 len = rows * cols;
        bytes16[] memory data = new bytes16[](len);
        m = Matrix({ rows: rows, cols: cols, data: data });
    }

    /**
     * @notice Create a rowsxcols matrix filled with ones.
     * @param  rows Number of rows
     * @param  cols Number of columns
     * @return m    New matrix with all entries = 1.0
     */
    function createOnesMatrix(uint256 rows, uint256 cols) internal pure validDims(rows, cols) returns (Matrix memory m) {
        uint256 len = rows * cols;
        bytes16[] memory data = new bytes16[](len);
        bytes16 one = MathLib.fromInt(1);

        for (uint256 i = 0; i < len; ++i) {
            data[i] = one;
        }

        m = Matrix({ rows: rows, cols: cols, data: data });
    }

    /**
     * @notice Create an nxn identity matrix.
     * @param  n Matrix dimension (n > 0)
     * @return m Identity matrix with 1 on diagonal and 0 elsewhere
     */
    function createIdentityMatrix(uint256 n) internal pure returns (Matrix memory m) {
        require(n > 0, "MatrixMaster: n must be > 0");
        bytes16[] memory data = new bytes16[](n * n);
        bytes16 one = MathLib.fromInt(1);

        for (uint256 i = 0; i < n; ++i) {
            data[_idx(n, i, i)] = one;
        }

        m = Matrix({ rows: n, cols: n, data: data });
    }

    /**
     * @notice Create a diagonal matrix from diagonal entries.
     * @param  diag Array of diagonal entries, length = n
     * @return m    nxn matrix with diag[i] on (i,i), zeros elsewhere
     */
    function createDiagonalMatrix(bytes16[] memory diag) internal pure returns (Matrix memory m) {
        uint256 n = diag.length;
        require(n > 0, "MatrixMaster: empty diagonal");
        bytes16[] memory data = new bytes16[](n * n);

        for (uint256 i = 0; i < n; ++i) {
            data[_idx(n, i, i)] = diag[i];
        }

        m = Matrix({ rows: n, cols: n, data: data });
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
    function createRandomMatrix(uint256 rows, uint256 cols, bytes32 seed) internal pure validDims(rows, cols) returns (Matrix memory m) {
        uint256 len = rows * cols;
        bytes16[] memory data = new bytes16[](len);
        bytes16 denom = MathLib.fromUInt(2**64);

        for (uint256 i = 0; i < len; ++i) {
            // Take top 64 bits from keccak and scale into [0,1)
            uint64 r;
            bytes32 h = keccak256(abi.encodePacked(seed, i));
            assembly {
                r := shr(192, h)
            }
            // value = r / 2^64
            bytes16 num = MathLib.fromUInt(uint256(r));
            data[i] = num.div(denom);
        }

        m = Matrix({ rows: rows, cols: cols, data: data });
    }

    // ------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------

    /**
     * @notice Get element at (row, col).
     * @param  m     Matrix to read
     * @param  row   Zero-based row index
     * @param  col   Zero-based column index
     * @return value Element m[row, col]
     */
    function getElement(Matrix memory m, uint256 row, uint256 col) internal pure checkBounds(m, row, col) returns (bytes16) {
        return m.data[_idx(m.cols, row, col)];
    }

    /**
     * @notice Set element at (row, col) to 'val'.
     *         Mutates the matrix in-place (in memory).
     * @param m     Matrix to modify
     * @param row   Zero-based row index
     * @param col   Zero-based column index
     * @param val   New value to write
     */
    function setElement(Matrix memory m, uint256 row, uint256 col, bytes16 val) internal pure checkBounds(m, row, col) {
        m.data[_idx(m.cols, row, col)] = val;
    }

    // ------------------------------------------------------------
    // Slicing & reshape
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
    function slice(Matrix memory m, uint256 rowStart, uint256 rowEnd, uint256 colStart, uint256 colEnd) internal pure returns (Matrix memory out) {
        require(rowStart < rowEnd && colStart < colEnd, "MatrixMaster: invalid slice range");
        require(rowEnd <= m.rows && colEnd <= m.cols, "MatrixMaster: slice out of bounds");

        uint256 newRows = rowEnd - rowStart;
        uint256 newCols = colEnd - colStart;
        bytes16[] memory data = new bytes16[](newRows * newCols);

        uint256 k = 0;
        for (uint256 i = rowStart; i < rowEnd; ++i) {
            for (uint256 j = colStart; j < colEnd; ++j) {
                data[k++] = m.data[_idx(m.cols, i, j)];
            }
        }

        out = Matrix({ rows: newRows, cols: newCols, data: data });
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
    function reshape(Matrix memory m, uint256 newRows, uint256 newCols) internal pure validDims(newRows, newCols) returns (Matrix memory reshaped) {
        require(
            newRows * newCols == m.rows * m.cols,
            "MatrixMaster: reshape area mismatch"
        );

        reshaped = Matrix({ rows: newRows, cols: newCols, data: m.data });
    }

    // ------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------

    /**
     * @notice Compute the transpose Aᵀ.
     * @param  a Input matrix A (rowsxcols)
     * @return t Transposed matrix (colsxrows)
     */
    function transpose(Matrix memory a) internal pure returns (Matrix memory t) {
        uint256 outRows = a.cols; // new row count
        uint256 outCols = a.rows; // new col count

        bytes16[] memory data = new bytes16[](a.rows * a.cols);

        for (uint256 i = 0; i < a.rows; ++i) {
            for (uint256 j = 0; j < a.cols; ++j) {
                data[_idx(outCols, j, i)] = a.data[_idx(a.cols, i, j)];
            }
        }

        t = Matrix({ rows: outRows, cols: outCols, data: data });
    }

    // ------------------------------------------------------------
    // Elementwise arithmetic
    // ------------------------------------------------------------

    /**
     * @notice   Elementwise addition C = A + B.
     * @param  a Left operand
     * @param  b Right operand
     * @return c Result matrix with same shape
     */
    function addMatrices(Matrix memory a, Matrix memory b) internal pure checkSameShape(a, b) returns (Matrix memory c) {
        uint256 len = a.rows * a.cols;
        bytes16[] memory data = new bytes16[](len);

        for (uint256 i = 0; i < len; ++i) {
            data[i] = a.data[i].add(b.data[i]);
        }

        c = Matrix({ rows: a.rows, cols: a.cols, data: data });
    }

    /**
     * @notice   Elementwise subtraction C = A − B.
     * @param  a Left operand
     * @param  b Right operand
     * @return c Result matrix with same shape
     */
    function subtractMatrices(Matrix memory a, Matrix memory b) internal pure checkSameShape(a, b) returns (Matrix memory c) {
        uint256 len = a.rows * a.cols;
        bytes16[] memory data = new bytes16[](len);

        for (uint256 i = 0; i < len; ++i) {
            data[i] = a.data[i].sub(b.data[i]);
        }

        c = Matrix({ rows: a.rows, cols: a.cols, data: data });
    }

    /**
     * @notice   Scalar multiplication C = k · A.
     * @param  a Input matrix
     * @param  k Scalar multiplier
     * @return c Result matrix
     */
    function multiplyScalar(Matrix memory a, bytes16 k) internal pure returns (Matrix memory c) {
        uint256 len = a.rows * a.cols;
        bytes16[] memory data = new bytes16[](len);

        for (uint256 i = 0; i < len; ++i) {
            data[i] = a.data[i].mul(k);
        }

        c = Matrix({ rows: a.rows, cols: a.cols, data: data });
    }

    /**
     * @notice   Scalar division C = A / k.
     * @param  a Input matrix
     * @param  k Scalar divisor (must be non-zero)
     * @return c Result matrix
     */
    function divideScalar(Matrix memory a, bytes16 k) internal pure returns (Matrix memory c) {
        require(MathLib.cmp(k, QZERO) != 0, "MatrixMaster: division by zero");

        uint256 len = a.rows * a.cols;
        bytes16[] memory data = new bytes16[](len);

        for (uint256 i = 0; i < len; ++i) {
            data[i] = a.data[i].div(k);
        }

        c = Matrix({ rows: a.rows, cols: a.cols, data: data });
    }

    // ------------------------------------------------------------
    // Matrix multiplication
    // ------------------------------------------------------------

    /**
     * @notice Matrix-matrix multiplication C = A · B.
     *         A is (mxk), B is (kxn), result is (mxn).
     *         Uses straightforward O(m·k·n) triple loop.
     * @param  a Left operand matrix
     * @param  b Right operand matrix
     * @return c Product matrix
     */
    function multiplyMatrices(Matrix memory a, Matrix memory b) internal pure checkMulShape(a, b) returns (Matrix memory c) {
        uint256 m = a.rows;
        uint256 k = a.cols;
        uint256 n = b.cols;

        bytes16[] memory data = new bytes16[](m * n);

        for (uint256 i = 0; i < m; ++i) {
            uint256 aRow = i * k;

            for (uint256 j = 0; j < n; ++j) {
                bytes16 acc = QZERO;
                for (uint256 t = 0; t < k; ++t) {

                    uint256 bRow = t * n;

                    bytes16 a_ik = a.data[aRow + t];
                    bytes16 b_tj = b.data[bRow + j];

                    acc = acc.add(a_ik.mul(b_tj));
                }
                data[_idx(n, i, j)] = acc;
            }
        }

        c = Matrix({ rows: m, cols: n, data: data });
    }

    // ------------------------------------------------------------
    // Matrix x Vector multiplication
    // ------------------------------------------------------------

    /**
     * @notice Multiply matrix A (mxn) by vector x (nx1). Result is (mx1).
     *
     * @param  A Matrix (mxn)
     * @param  x Vector as matrix (nx1)
     * @return y = A * x (mx1 column vector)
     */
    function multiplyMatrixVector(Matrix memory A, Matrix memory x) internal pure returns (Matrix memory y) {
        require(A.cols == x.rows, "MatrixMaster: A.cols != x.rows");
        require(x.cols == 1, "MatrixMaster: x must be column vector");

        uint256 m = A.rows;
        uint256 n = A.cols;

        bytes16[] memory out = new bytes16[](m);
        bytes16 acc;

        for (uint256 i = 0; i < m; ++i) {
            acc = QZERO;
            for (uint256 j = 0; j < n; ++j) {
                bytes16 a_ij = A.data[_idx(n, i, j)];
                bytes16 x_j = x.data[j]; // x is (n×1)
                acc = acc.add(a_ij.mul(x_j));
            }
            out[i] = acc;
        }

        y = Matrix({ rows: m, cols: 1, data: out });
    }

    // ------------------------------------------------------------
    // Vector utilities (dot, norm, normalization, random vector)
    // ------------------------------------------------------------

    /**
    * @notice Create column vector (nx1) from array
    * @param  data Vector entries
    * @return v    Column vector matrix of shape nx1
     */
    function createVector(bytes16[] memory data) internal pure returns (Matrix memory v) {
        require(data.length > 0, "MatrixMaster: empty vector");
        
        v = Matrix({rows: data.length, cols: 1, data: data});
    }

    /**
     * @notice Create a pseudo-random column vector (nx1) with entries in [0,1).
     *         Uses deterministic keccak-based generation, NOT secure randomness.
     * @param  n    Dimension of the vector (n > 0)
     * @param  seed Seed value used to generate pseudo-random entries
     * @return v    Column vector (nx1) with pseudo-random entries
     */
    function randomVector(uint256 n, bytes32 seed) internal pure validDims(n, 1) returns (Matrix memory v) {
        bytes16[] memory data = new bytes16[](n);
        bytes16 denom = MathLib.fromUInt(2**64);

        for (uint256 i = 0; i < n; ++i) {
            uint64 r;
            bytes32 h = keccak256(abi.encodePacked(seed, i));

            assembly {
                r := shr(192, h) // top 64 bits
            }

            bytes16 num = MathLib.fromUInt(uint256(r));
            data[i] = num.div(denom);
        }

        v = Matrix({ rows: n, cols: 1, data: data });
    }

    /**
     * @notice Compute dot product of two column vectors (nx1).
     *         Requires both vectors to be column vectors with identical lengths.
     * @param  a First vector (nx1)
     * @param  b Second vector (nx1)
     * @return s Scalar dot product = Σ_i a[i] * b[i]
     */
    function dot(Matrix memory a, Matrix memory b) internal pure returns (bytes16 s) {
        require(a.cols == 1 && b.cols == 1, "MatrixMaster: dot requires column vectors");
        require(a.rows == b.rows, "MatrixMaster: dot length mismatch");

        bytes16 acc = QZERO;

        for (uint256 i = 0; i < a.rows; ++i) {
            acc = acc.add(a.data[i].mul(b.data[i]));
        }

        s = acc;
    }

    /**
     * @notice Compute Euclidean 2-norm ||v||_2 of a column vector (nx1).
     * @param  v   Input vector (nx1)
     * @return nrm Vector norm = sqrt(dot(v, v))
     */
    function euclideanNorm(Matrix memory v) internal pure returns (bytes16 nrm) {
        require(v.cols == 1, "MatrixMaster: euclideanNorm requires column vector");

        bytes16 d = dot(v, v);
        nrm = MathLib.sqrt(d);
    }

    /**
     * @notice Normalize a column vector v into v / ||v||_2.
     * @dev    Reverts if vector has zero norm.
     * @param  v   Input vector (nx1)
     * @return out Normalized vector with unit Euclidean norm
     */
    function normalize(Matrix memory v) internal pure returns (Matrix memory out) {
        require(v.cols == 1, "MatrixMaster: normalize requires column vector");

        bytes16 nrm = euclideanNorm(v);
        require(MathLib.cmp(nrm, QZERO) != 0, "MatrixMaster: cannot normalize zero vector");

        out = divideScalar(v, nrm); // v / ||v||
    }

    /**
     * @notice Check approximate convergence between two vectors via L2 tolerance.
     *         Computes ||vNew - vOld||_2 and compares to 'tol'.
     * @param  vNew New iterate (nx1)
     * @param  vOld Previous iterate (nx1)
     * @param  tol  Convergence tolerance (positive scalar)
     * @return ok   True if ||vNew - vOld|| < tol
     */
    function hasConverged(Matrix memory vNew, Matrix memory vOld, bytes16 tol) internal pure checkSameShape(vNew, vOld) returns (bool ok) {
        require(vNew.cols == 1, "MatrixMaster: converged requires vectors");

        // Check Positive Direction
        Matrix memory diffPos = subtractMatrices(vNew, vOld);
        if (MathLib.cmp(euclideanNorm(diffPos), tol) < 0) {
            return true;
        }

        // Check Negative Direction (for negative eigenvalues)
        Matrix memory diffNeg = addMatrices(vNew, vOld);
        if (MathLib.cmp(euclideanNorm(diffNeg), tol) < 0) {
            return true;
        }

        return false;
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
    function det(Matrix memory a) internal pure isSquare(a) returns (bytes16 detA) {
        uint256 n = a.rows;

        // Copy data into working array for LU in-place.
        bytes16[] memory lu = new bytes16[](n * n);
        for (uint256 i = 0; i < n * n; ++i) {
            lu[i] = a.data[i];
        }

        int256 sign = 1;

        // LU decomposition with partial pivoting
        for (uint256 k = 0; k < n; ++k) {
            // Find pivot row
            uint256 pivot = k;
            bytes16 maxAbs = MathLib.abs(lu[_idx(n, k, k)]);

            for (uint256 i = k + 1; i < n; ++i) {
                bytes16 v = MathLib.abs(lu[_idx(n, i, k)]);
                if (MathLib.cmp(v, maxAbs) > 0) {
                    maxAbs = v;
                    pivot = i;
                }
            }

            // If pivot is (near) zero -> determinant ≈ 0
            if (MathLib.cmp(maxAbs, QC.EPS_1e30()) <= 0) {
                return QZERO;
            }

            // Swap rows if needed
            if (pivot != k) {
                for (uint256 j = 0; j < n; ++j) {
                    uint256 idx1 = _idx(n, k, j);
                    uint256 idx2 = _idx(n, pivot, j);
                    bytes16 tmp = lu[idx1];
                    lu[idx1] = lu[idx2];
                    lu[idx2] = tmp;
                }
                sign = -sign;
            }

            // Elimination
            bytes16 pivotVal = lu[_idx(n, k, k)];
            for (uint256 i = k + 1; i < n; ++i) {
                uint256 idx_ik = _idx(n, i, k);
                bytes16 factor = lu[idx_ik].div(pivotVal);
                lu[idx_ik] = factor;

                for (uint256 j = k + 1; j < n; ++j) {
                    uint256 idx_ij = _idx(n, i, j);
                    bytes16 subtrahend = factor.mul(lu[_idx(n, k, j)]);
                    lu[idx_ij] = lu[idx_ij].sub(subtrahend);
                }
            }
        }

        // det(A) = sign * Π diag(U) (U is upper part of lu)
        bytes16 detVal = MathLib.fromInt(1);
        for (uint256 i = 0; i < n; ++i) {
            detVal = detVal.mul(lu[_idx(n, i, i)]);
        }

        if (sign < 0) {
            detVal = detVal.neg();
        }

        detA = detVal;
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
    function inverse(Matrix memory a) internal pure isSquare(a) returns (Matrix memory invA) {
        uint256 n = a.rows;

        // Allocate augmented matrix [A | I] as n x (2n)
        uint256 augCols = 2 * n;
        bytes16[] memory aug = new bytes16[](n * augCols);
        bytes16 one = MathLib.fromInt(1);

        // Fill [A | I]
        for (uint256 i = 0; i < n; ++i) {
            for (uint256 j = 0; j < n; ++j) {
                aug[_idx(augCols, i, j)] = a.data[_idx(n, i, j)];
            }
            for (uint256 j = 0; j < n; ++j) {
                aug[_idx(augCols, i, n + j)] = (i == j) ? one : QZERO;
            }
        }

        // Gauss–Jordan
        for (uint256 k = 0; k < n; ++k) {
            // Find pivot row for column k
            uint256 pivot = k;
            bytes16 maxAbs = MathLib.abs(aug[_idx(augCols, k, k)]);

            for (uint256 i = k + 1; i < n; ++i) {
                bytes16 v = MathLib.abs(aug[_idx(augCols, i, k)]);
                if (MathLib.cmp(v, maxAbs) > 0) {
                    maxAbs = v;
                    pivot = i;
                }
            }

            // Check singularity
            require(MathLib.cmp(maxAbs, QC.EPS_1e30()) > 0, "MatrixMaster: singular matrix");

            // Swap rows if pivot != k
            if (pivot != k) {
                for (uint256 j = 0; j < augCols; ++j) {
                    uint256 idx1 = _idx(augCols, k, j);
                    uint256 idx2 = _idx(augCols, pivot, j);
                    bytes16 tmp = aug[idx1];
                    aug[idx1] = aug[idx2];
                    aug[idx2] = tmp;
                }
            }

            // Normalize pivot row
            bytes16 pivVal = aug[_idx(augCols, k, k)];
            for (uint256 j = 0; j < augCols; ++j) {
                uint256 idx = _idx(augCols, k, j);
                aug[idx] = aug[idx].div(pivVal);
            }

            // Eliminate other rows
            for (uint256 i = 0; i < n; ++i) {
                if (i == k) continue;
                bytes16 factor = aug[_idx(augCols, i, k)];
                if (MathLib.cmp(factor, QZERO) == 0) continue;

                for (uint256 j = 0; j < augCols; ++j) {
                    uint256 idx_ij = _idx(augCols, i, j);
                    bytes16 subtrahend = factor.mul(aug[_idx(augCols, k, j)]);
                    aug[idx_ij] = aug[idx_ij].sub(subtrahend);
                }
            }
        }

        // Extract inverse from right half of [I | A⁻¹]
        bytes16[] memory invData = new bytes16[](n * n);
        for (uint256 i = 0; i < n; ++i) {
            for (uint256 j = 0; j < n; ++j) {
                invData[_idx(n, i, j)] = aug[_idx(augCols, i, n + j)];
            }
        }

        invA = Matrix({ rows: n, cols: n, data: invData });
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
    function powerIteration(Matrix memory A, bytes32 seed, bytes16 tol) internal view isSquare(A) returns (bytes16 lambda, Matrix memory x) {
        uint256 n = A.rows;
        require(MathLib.cmp(tol, QZERO) > 0, "MatrixMaster: tol must be > 0");
        
        // If user passed 0, use Global Default
        if (MathLib.cmp(tol, QZERO) == 0) {
            tol = LibNumericConfig.getTol();
        }

        // Apply Safety Guardrail to ensure not goint below minTol (which would be too expensive)
        bytes16 minTol = LibNumericConfig.getMinTol();
        if (MathLib.cmp(tol, minTol) < 0) {
            tol = minTol;
        }

        uint256 maxIter = LibNumericConfig.getMaxIter();

        x = randomVector(n, seed);
        x = normalize(x);

        for (uint256 iter = 0; iter < maxIter; ++iter) {
            Matrix memory y = multiplyMatrixVector(A, x);
            
            // Avoid division by zero if matrix is singular
            if (MathLib.cmp(euclideanNorm(y), QZERO) == 0) { break; }

            Matrix memory xNew = normalize(y);

            if (hasConverged(xNew, x, tol)) {
                x = xNew;
                break;
            }

            x = xNew;
        }
        
        Matrix memory Ax = multiplyMatrixVector(A, x);
        lambda = dot(x, Ax);
    }
}