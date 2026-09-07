// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MatrixMaster } from "./MatrixMaster.sol";
import { MathLib } from "../MathLib.sol";

/**
 * @title  LinearSolvers Library
 * @notice This library provides numerical algorithms for solving linear systems
 *         and related least-squares problems using quadruple-precision arithmetic.
 *         Gradient Descent, Jacobi, Gauss–Seidel, Gaussian Elimination, and LU Decomposition
 *         implemented on top of MatrixMaster.Matrix (bytes16 / ABDK quad).
 *         All matrices use MatrixMaster.Matrix (rows, cols, data[]).
 *         Vectors are represented as column matrices (n x 1).
 */
library LinearSolvers {
    using MatrixMaster for MatrixMaster.Matrix;

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /// @notice Creates a column vector Matrix from a bytes16[] (no copy).
    function _colVec(bytes16[] memory data) internal pure returns (MatrixMaster.Matrix memory v) {
        v.rows = data.length;
        v.cols = 1;
        v.data = data;
    }

    /// @notice Ensures b is (A.rows x 1) and dimensions match for A and b.
    function _checkSystem(MatrixMaster.Matrix memory A, MatrixMaster.Matrix memory b) internal pure {
        require(b.cols == 1, "LinearSolversMM: b must be column vector");
        require(b.rows == A.rows, "LinearSolversMM: dim mismatch A vs b");
    }

    /// @notice Squared 2-norm of a column vector: ||v||^2 = dot(v, v).
    function _norm2Squared(MatrixMaster.Matrix memory v) internal pure returns (bytes16) {
        return MatrixMaster.dot(v, v);
    }

    /// @notice Access A(i,j) = A.data[i*cols + j].
    function _get(MatrixMaster.Matrix memory M, uint256 i, uint256 j) internal pure returns (bytes16) {
        require(i < M.rows && j < M.cols, "LinearSolversMM: index OOB");
        return M.data[i * M.cols + j];
    }

    /// @notice Set A(i,j) := value.
    function _set(MatrixMaster.Matrix memory M, uint256 i, uint256 j, bytes16 value) internal pure {
        require(i < M.rows && j < M.cols, "LinearSolversMM: index OOB");
        M.data[i * M.cols + j] = value;
    }

    // ------------------------------------------------------------
    // Gradient Descent (Least Squares)
    // ------------------------------------------------------------

    /**
     * @notice Solve min_x 1/2 ||A x - b||^2 via gradient descent (least squares).
     *         Convergence is guaranteed only if 0 < alpha < 2 / λ_max(A^T A).
     *
     *         This implementation:
     *          - Does NOT perform line search
     *          - Does NOT adapt alpha during iterations
     *          - Does NOT estimate spectral bounds on-chain
     *
     *         Choosing alpha too small leads to slow convergence.
     *         Choosing alpha too large leads to divergence.
     *
     * @param A       Coefficient matrix (m x n).
     * @param b       Right-hand side column vector (m x 1).
     * @param x0      Initial guess (n x 1).
     * @param alpha   Step size (bytes16).
     * @param maxIter Maximum number of iterations.
     * @param tol     Tolerance on squared gradient norm; stop when ||∇f||^2 <= tol.
     *
     * @return x      Approximate minimizer (n x 1).
     * @return iters  Number of gradient descent steps fully executed before termination.
     */
    function gradientDescentLeastSquares(
        MatrixMaster.Matrix memory A,
        MatrixMaster.Matrix memory b,
        MatrixMaster.Matrix memory x0,
        bytes16 alpha,
        uint256 maxIter,
        bytes16 tol
    ) internal pure returns (MatrixMaster.Matrix memory x, uint256 iters) {
        _checkSystem(A, b);
        
        // x0 must match the number of columns of A (n)
        // Since A is m x n, A.cols is n.
        require(x0.rows == A.cols && x0.cols == 1, "LinearSolversMM: x0 dim mismatch (n vs n)");
        
        require(maxIter > 0, "LinearSolversMM: maxIter must be > 0");
        require(MathLib.cmp(alpha, MathLib.fromUInt(0)) > 0, "LinearSolversMM: alpha <= 0");

        // x <- x0 (Correctly sized n x 1)
        x.rows = x0.rows;
        x.cols = x0.cols;
        x.data = new bytes16[](x0.data.length);
        for (uint256 i = 0; i < x0.data.length; ++i) {
            x.data[i] = x0.data[i];
        }

        uint256 m = A.rows;
        uint256 n = A.cols;
        bytes16[] memory residual = new bytes16[](m);
        bytes16[] memory gradient = new bytes16[](n);

        for (iters = 0; iters < maxIter; ++iters) {
            // Reuse fixed buffers instead of allocating Ax, residual, gradient,
            // step, and replacement-x arrays on every iteration.
            for (uint256 i = 0; i < m; ++i) {
                bytes16 sum = bytes16(0);
                uint256 rowOffset = i * n;
                for (uint256 j = 0; j < n; ++j) {
                    sum = MathLib.add(sum, MathLib.mul(A.data[rowOffset + j], x.data[j]));
                }
                residual[i] = MathLib.sub(sum, b.data[i]);
            }

            bytes16 gNorm2 = bytes16(0);
            for (uint256 j = 0; j < n; ++j) {
                bytes16 sum = bytes16(0);
                for (uint256 i = 0; i < m; ++i) {
                    sum = MathLib.add(sum, MathLib.mul(A.data[i * n + j], residual[i]));
                }
                gradient[j] = sum;
                gNorm2 = MathLib.add(gNorm2, MathLib.mul(sum, sum));
            }

            // Stopping condition: ||gradient||^2 <= tol
            if (MathLib.cmp(gNorm2, tol) <= 0) {
                return (x, iters);
            }

            // All gradient entries were computed from the same x, so updating
            // x in place here preserves the simultaneous gradient-descent step.
            for (uint256 j = 0; j < n; ++j) {
                x.data[j] = MathLib.sub(x.data[j], MathLib.mul(gradient[j], alpha));
            }
        }

        return (x, iters); // reached maxIter
    }

    // ------------------------------------------------------------
    // Jacobi Method
    // ------------------------------------------------------------

    /**
     * @notice Solve A x ≈ b using Jacobi iteration.
     *         Requires non-zero diagonal and (ideally) diagonal dominance for convergence.
     *
     * @param A        Coefficient matrix (nxn).
     * @param b        RHS column vector (nx1).
     * @param x0       Initial guess (nx1).
     * @param maxIter  Maximum iterations.
     * @param tolDiff  Tolerance on squared difference between consecutive iterates.
     *
     * @return x       Approximate solution.
     * @return iters   Number of iterations executed, where the initial guess x0
     *                 is considered iteration 0 and the first update produces x1.
     */
    function jacobi(
        MatrixMaster.Matrix memory A,
        MatrixMaster.Matrix memory b,
        MatrixMaster.Matrix memory x0,
        uint256 maxIter,
        bytes16 tolDiff
    ) internal pure returns (MatrixMaster.Matrix memory x, uint256 iters) {
        bool ignoredConverged;
        (x, iters, ignoredConverged) = jacobiWithStatus(A, b, x0, maxIter, tolDiff);
    }

    /**
     * @notice Jacobi iteration with an explicit convergence result.
     * @dev Candidate convergence requires both:
     *      ||x_k-x_{k-1}||^2 <= tolDiff, and
     *      ||A*x_k-b||^2 <= tolDiff * max(1, ||b||^2).
     *      The residual is evaluated only after the cheaper step test passes.
     * @return x Approximate solution at termination.
     * @return iters Number of completed iterations.
     * @return converged True only when both convergence criteria are satisfied.
     */
    function jacobiWithStatus(
        MatrixMaster.Matrix memory A,
        MatrixMaster.Matrix memory b,
        MatrixMaster.Matrix memory x0,
        uint256 maxIter,
        bytes16 tolDiff
    ) internal pure returns (MatrixMaster.Matrix memory x, uint256 iters, bool converged) {
        _checkSystem(A, b);
        require(x0.rows == A.cols && x0.cols == 1, "LinearSolversMM: x0 dim mismatch");
        require(maxIter > 0, "LinearSolversMM: maxIter must be > 0");

        uint256 n = A.rows;

        x.rows = x0.rows;
        x.cols = x0.cols;
        x.data = new bytes16[](x0.data.length);

        MatrixMaster.Matrix memory xOld;
        xOld.rows = x0.rows;
        xOld.cols = x0.cols;
        xOld.data = new bytes16[](x0.data.length);

        // x <- x0
        for (uint256 i = 0; i < x0.data.length; ++i) {
            x.data[i] = x0.data[i];
        }

        for (iters = 0; iters < maxIter; ++iters) {
            // xOld <- x
            for (uint256 i = 0; i < x.data.length; ++i) {
                xOld.data[i] = x.data[i];
            }

            bytes16 diff2 = bytes16(0);

            // For each i: x_i^{k+1} = (b_i - Σ_{j!=i} a_ij x_j^{k}) / a_ii
            for (uint256 i = 0; i < n; ++i) {
                bytes16 sum = b.data[i]; // start from b_i
                bytes16 aii = _get(A, i, i);
                require(MathLib.cmp(aii, MathLib.fromUInt(0)) != 0, "LinearSolversMM: zero diag Jacobi");

                for (uint256 j = 0; j < n; ++j) {
                    if (j == i) continue;
                    bytes16 aij = _get(A, i, j);
                    bytes16 xj = xOld.data[j];
                    sum = MathLib.sub(sum, MathLib.mul(aij, xj));
                }

                x.data[i] = MathLib.div(sum, aii);
                bytes16 delta = MathLib.sub(x.data[i], xOld.data[i]);
                diff2 = MathLib.add(diff2, MathLib.mul(delta, delta));
            }

            // Candidate convergence first checks ||x - xOld||^2 without
            // allocating a temporary difference vector.
            if (MathLib.cmp(diff2, tolDiff) <= 0 &&
                _residualWithinTolerance(A, b, x, tolDiff)) {
                return (x, iters + 1, true);
            }
        }

        return (x, iters, false);
    }

    // ------------------------------------------------------------
    // Gauss-Sidel Method
    // ------------------------------------------------------------

    /**
     * @notice Solve A x ≈ b using Gauss–Seidel iteration.
     *         Uses updated values within the same iteration (in-place).
     *         Requires non-zero diagonal.
     *
     * @param A        Coefficient matrix (nxn).
     * @param b        RHS column vector (nx1).
     * @param x0       Initial guess (nx1).
     * @param maxIter  Maximum iterations.
     * @param tolDiff  Tolerance on squared difference between consecutive iterates.
     *
     * @return x       Approximate solution.
     * @return iters   Number of iterations executed, where the initial guess x0
     *                 is considered iteration 0 and the first update produces x1.
     */
    function gaussSeidel(
        MatrixMaster.Matrix memory A,
        MatrixMaster.Matrix memory b,
        MatrixMaster.Matrix memory x0,
        uint256 maxIter,
        bytes16 tolDiff
    ) internal pure returns (MatrixMaster.Matrix memory x, uint256 iters) {
        bool ignoredConverged;
        (x, iters, ignoredConverged) = gaussSeidelWithStatus(A, b, x0, maxIter, tolDiff);
    }

    /**
     * @notice Gauss-Seidel iteration with an explicit convergence result.
     * @dev Uses the same combined iterate-change and scaled-residual rule as
     *      jacobiWithStatus. Residual evaluation is deferred until the step
     *      criterion is satisfied.
     * @return x Approximate solution at termination.
     * @return iters Number of completed iterations.
     * @return converged True only when both convergence criteria are satisfied.
     */
    function gaussSeidelWithStatus(
        MatrixMaster.Matrix memory A,
        MatrixMaster.Matrix memory b,
        MatrixMaster.Matrix memory x0,
        uint256 maxIter,
        bytes16 tolDiff
    ) internal pure returns (MatrixMaster.Matrix memory x, uint256 iters, bool converged) {
        _checkSystem(A, b);
        require(x0.rows == A.cols && x0.cols == 1, "LinearSolversMM: x0 dim mismatch");
        require(maxIter > 0, "LinearSolversMM: maxIter must be > 0");

        uint256 n = A.rows;

        x.rows = x0.rows;
        x.cols = x0.cols;
        x.data = new bytes16[](x0.data.length);

        MatrixMaster.Matrix memory xOld;
        xOld.rows = x0.rows;
        xOld.cols = x0.cols;
        xOld.data = new bytes16[](x0.data.length);

        // x <- x0
        for (uint256 i = 0; i < x0.data.length; ++i) {
            x.data[i] = x0.data[i];
        }

        for (iters = 0; iters < maxIter; ++iters) {
            // xOld <- x
            for (uint256 i = 0; i < x.data.length; ++i) {
                xOld.data[i] = x.data[i];
            }

            bytes16 diff2 = bytes16(0);

            for (uint256 i = 0; i < n; ++i) {
                bytes16 sum = b.data[i];
                bytes16 aii = _get(A, i, i);
                require(MathLib.cmp(aii, MathLib.fromUInt(0)) != 0, "LinearSolversMM: zero diag GS");

                for (uint256 j = 0; j < n; ++j) {
                    if (j == i) continue;
                    bytes16 aij = _get(A, i, j);
                    bytes16 xj = (j < i) ? x.data[j] : xOld.data[j];
                    sum = MathLib.sub(sum, MathLib.mul(aij, xj));
                }

                x.data[i] = MathLib.div(sum, aii);
                bytes16 delta = MathLib.sub(x.data[i], xOld.data[i]);
                diff2 = MathLib.add(diff2, MathLib.mul(delta, delta));
            }

            // Candidate convergence first checks ||x - xOld||^2 without
            // allocating a temporary difference vector.
            if (MathLib.cmp(diff2, tolDiff) <= 0 &&
                _residualWithinTolerance(A, b, x, tolDiff)) {
                return (x, iters + 1, true);
            }
        }

        return (x, iters, false);
    }

    /**
     * @dev Hybrid absolute/relative squared residual check used by stationary
     *      iterative solvers. Scaling by max(1, ||b||^2) avoids accepting a
     *      large residual merely because the iterates have stopped moving.
     */
    function _residualWithinTolerance(
        MatrixMaster.Matrix memory A,
        MatrixMaster.Matrix memory b,
        MatrixMaster.Matrix memory x,
        bytes16 tol
    ) internal pure returns (bool) {
        MatrixMaster.Matrix memory residual = MatrixMaster.subtractMatrices(
            MatrixMaster.multiplyMatrices(A, x),
            b
        );
        bytes16 residual2 = _norm2Squared(residual);
        bytes16 b2 = _norm2Squared(b);
        bytes16 one = MathLib.fromUInt(1);
        bytes16 scale2 = MathLib.cmp(b2, one) > 0 ? b2 : one;
        return MathLib.cmp(residual2, MathLib.mul(tol, scale2)) <= 0;
    }

    // ------------------------------------------------------------
    // Gaussian Elimination
    // ------------------------------------------------------------

    /**
     * @notice Solve A x = b using Gaussian elimination with partial pivoting.
     *         Operates directly on matrix buffers without forming augmented matrices.
     *
     * @param  A Coefficient matrix (n × n)
     * @param  b Right-hand side column vector (n × 1)
     * @return x Solution vector (n × 1)
     */
    function gaussianElimination(MatrixMaster.Matrix memory A, MatrixMaster.Matrix memory b) internal pure returns (MatrixMaster.Matrix memory x) {
        _checkSystem(A, b);
        uint256 n = A.rows;

        // Work copies
        MatrixMaster.Matrix memory U;
        U.rows = n;
        U.cols = n;
        U.data = new bytes16[](A.data.length);
        bytes16 matrixScale = MathLib.fromUInt(0);
        for (uint256 i = 0; i < A.data.length; ++i) {
            U.data[i] = A.data[i];
            bytes16 magnitude = MathLib.abs(A.data[i]);
            if (MathLib.cmp(magnitude, matrixScale) > 0) matrixScale = magnitude;
        }

        MatrixMaster.Matrix memory rhs;
        rhs.rows = n;
        rhs.cols = 1;
        rhs.data = new bytes16[](b.data.length);
        for (uint256 i = 0; i < b.data.length; ++i) rhs.data[i] = b.data[i];

        // Forward elimination with partial pivoting
        for (uint256 k = 0; k < n; ++k) {
            // Find pivot row
            uint256 pivotRow = k;
            bytes16 maxAbs = MathLib.abs(_get(U, k, k));
            for (uint256 i = k + 1; i < n; ++i) {
                bytes16 v = MathLib.abs(_get(U, i, k));
                if (MathLib.cmp(v, maxAbs) > 0) {
                    maxAbs = v;
                    pivotRow = i;
                }
            }
            bytes16 pivot = _get(U, pivotRow, k);
            require(
                !MatrixMaster.isUnsafePivot(maxAbs, matrixScale),
                "LinearSolversMM: unsafe pivot"
            );

            // swap rows if needed
            if (pivotRow != k) {
                _swapRows(U, k, pivotRow);
                _swapRhs(rhs, k, pivotRow);
            }

            // Eliminate
            for (uint256 i = k + 1; i < n; ++i) {
                bytes16 aik = _get(U, i, k);
                if (MathLib.cmp(aik, MathLib.fromUInt(0)) == 0) continue;
                bytes16 factor = MathLib.div(aik, pivot);
                _set(U, i, k, MathLib.fromUInt(0));

                for (uint256 j = k + 1; j < n; ++j) {
                    bytes16 uij = _get(U, i, j);
                    bytes16 ukj = _get(U, k, j);
                    _set(U, i, j, MathLib.sub(uij, MathLib.mul(factor, ukj)));
                }

                rhs.data[i] = MathLib.sub(rhs.data[i], MathLib.mul(factor, rhs.data[k]));
            }
        }

        // Back-substitution
        x.rows = n;
        x.cols = 1;
        x.data = new bytes16[](n);

        for (int256 iSigned = int256(n) - 1; iSigned >= 0; --iSigned) {
            uint256 i = uint256(iSigned);
            bytes16 sum = rhs.data[i];

            for (uint256 j = i + 1; j < n; ++j) {
                bytes16 uij = _get(U, i, j);
                sum = MathLib.sub(sum, MathLib.mul(uij, x.data[j]));
            }

            bytes16 diag = _get(U, i, i);
            require(
                !MatrixMaster.isUnsafePivot(MathLib.abs(diag), matrixScale),
                "LinearSolversMM: unsafe diag backsub"
            );
            x.data[i] = MathLib.div(sum, diag);
        }
    }

    /**
     * @notice Swap two rows of a matrix in-place.
     *         Exchanges all elements of row 'r1' with row 'r2'.
     *
     * @param M  Matrix whose rows are to be swapped
     * @param r1 Index of the first row
     * @param r2 Index of the second row
     */
    function _swapRows(MatrixMaster.Matrix memory M, uint256 r1, uint256 r2) internal pure {
        if (r1 == r2) return;
        for (uint256 j = 0; j < M.cols; ++j) {
            uint256 idx1 = r1 * M.cols + j;
            uint256 idx2 = r2 * M.cols + j;
            bytes16 tmp = M.data[idx1];
            M.data[idx1] = M.data[idx2];
            M.data[idx2] = tmp;
        }
    }

    /**
     * @notice Swap two entries of a right-hand-side column vector.
     *         Swaps elements 'rhs[i]' and 'rhs[j]' in-place.
     *         Used to keep the right-hand side vector consistent
     *         with row swaps applied to the coefficient matrix
     *         during Gaussian elimination.
     *
     * @param rhs Right-hand side column vector (n × 1)
     * @param i   Index of the first entry
     * @param j   Index of the second entry
     */
    function _swapRhs(MatrixMaster.Matrix memory rhs, uint256 i, uint256 j) internal pure {
        bytes16 tmp = rhs.data[i];
        rhs.data[i] = rhs.data[j];
        rhs.data[j] = tmp;
    }

    // ------------------------------------------------------------
    // LU Decomposition
    // ------------------------------------------------------------

    /**
     * @notice Compute LU factorization A = L*U using Doolittle's method (no pivoting).
     *         This implementation does NOT perform pivoting.
     *         Pivots unsafe relative to the original matrix scale will cause
     *         failure even if A is mathematically invertible.
     *
     * @param  A Matrix (nxn).
     * @return L Unit-lower-triangular factor (nxn).
     * @return U Upper-triangular factor (nxn).
     */
    function luDecomposition(MatrixMaster.Matrix memory A) internal pure returns (MatrixMaster.Matrix memory L, MatrixMaster.Matrix memory U) {
        require(A.rows == A.cols, "LinearSolversMM: A must be square");
        uint256 n = A.rows;

        L.rows = n;
        L.cols = n;
        L.data = new bytes16[](n * n);

        U.rows = n;
        U.cols = n;
        U.data = new bytes16[](n * n);

        bytes16 matrixScale = MathLib.fromUInt(0);

        // Initialize U = A, L = I
        for (uint256 i = 0; i < n; ++i) {
            for (uint256 j = 0; j < n; ++j) {
                bytes16 aij = _get(A, i, j);
                _set(U, i, j, aij);
                bytes16 magnitude = MathLib.abs(aij);
                if (MathLib.cmp(magnitude, matrixScale) > 0) matrixScale = magnitude;

                if (i == j) {
                    _set(L, i, j, MathLib.fromUInt(1));
                } else {
                    _set(L, i, j, MathLib.fromUInt(0));
                }
            }
        }

        for (uint256 k = 0; k < n; ++k) {
            bytes16 pivot = _get(U, k, k);
            require(
                !MatrixMaster.isUnsafePivot(MathLib.abs(pivot), matrixScale),
                "LinearSolversMM: unsafe pivot LU"
            );

            for (uint256 i = k + 1; i < n; ++i) {
                bytes16 uik = _get(U, i, k);
                if (MathLib.cmp(uik, MathLib.fromUInt(0)) == 0) continue;

                bytes16 factor = MathLib.div(uik, pivot);
                _set(L, i, k, factor);

                for (uint256 j = k; j < n; ++j) {
                    bytes16 uij = _get(U, i, j);
                    bytes16 ukj = _get(U, k, j);
                    _set(U, i, j, MathLib.sub(uij, MathLib.mul(factor, ukj)));
                }
            }
        }
    }
}
