// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LinearSolvers } from "../../libraries/numeric/LinearSolvers.sol";
import { MatrixMaster } from "../../libraries/numeric/MatrixMaster.sol";
import { MathLib } from "../../libraries/MathLib.sol";

/**
 * @title  LinearSolversFacet
 * @notice Diamond facet exposing linear system solvers and decompositions
 *         (Gradient Descent Least Squares, Jacobi Iteration, Gauss-Siedel,
 *          Gaussian Elimination, LU Decomposition are implemented with ABDK quad precision)
 *
 *         All matrices are passed as row-major bytes16 arrays.
 *         Vectors are represented as (n × 1) matrices.
 */
contract LinearSolversFacet {

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
     * @param m Number of rows of matrix A
     * @param n Number of columns of matrix A
     * @param Adata   Flattened m×n coefficient matrix A (row-major)
     * @param bdata   Flattened m×1 target vector b
     * @param x0data  Flattened n×1 initial guess vector x_0
     * @param alpha   Step size (bytes16).
     * @param maxIter Maximum number of iterations.
     * @param tol     Tolerance on squared gradient norm; stop when ||∇f||^2 <= tol.
     *
     * @return x      Approximate minimizer (n x 1).
     * @return iters  Number of gradient descent steps fully executed before termination.
     */
    function gradientDescentLeastSquares(
        uint256 m,
        uint256 n,
        bytes16[] calldata Adata,   // m × n
        bytes16[] calldata bdata,   // m × 1
        bytes16[] calldata x0data,  // n × 1
        bytes16 alpha,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (bytes16[] memory x, uint256 iters) {
        MatrixMaster.Matrix memory A =
            MatrixMaster.Matrix(m, n, _copy(Adata));
        MatrixMaster.Matrix memory b =
            MatrixMaster.Matrix(m, 1, _copy(bdata));
        MatrixMaster.Matrix memory x0 =
            MatrixMaster.Matrix(n, 1, _copy(x0data));

        MatrixMaster.Matrix memory out;
        (out, iters) =
            LinearSolvers.gradientDescentLeastSquares(A, b, x0, alpha, maxIter, tol);

        return (out.data, iters);
    }

    // ------------------------------------------------------------
    // Jacobi Iteration
    // ------------------------------------------------------------

    /**
     * @notice Solve A x ≈ b using Jacobi iteration.
     *         Requires non-zero diagonal and (ideally) diagonal dominance for convergence.
     *
     * @param n Dimension of the square coefficient matrix A (n×n)
     * @param Adata    Flattened n×n coefficient matrix A (row-major)
     * @param bdata    Flattened n×1 right-hand side vector b
     * @param x0data   Flattened n×1 initial guess vector x_0
     * @param maxIter  Maximum iterations.
     * @param tolDiff  Tolerance on squared difference between consecutive iterates.
     *
     * @return x       Approximate solution.
     * @return iters   Number of iterations executed, where the initial guess x0
     *                 is considered iteration 0 and the first update produces x1.
     */
    function jacobi(uint256 n, bytes16[] calldata Adata, bytes16[] calldata bdata, bytes16[] calldata x0data, uint256 maxIter, bytes16 tolDiff)
        external pure returns (bytes16[] memory x, uint256 iters) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix(n, n, _copy(Adata));
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix(n, 1, _copy(bdata));
        MatrixMaster.Matrix memory x0 = MatrixMaster.Matrix(n, 1, _copy(x0data));

        MatrixMaster.Matrix memory out;
        (out, iters) = LinearSolvers.jacobi(A, b, x0, maxIter, tolDiff);

        return (out.data, iters);
    }

    // ------------------------------------------------------------
    // Gauss–Seidel Iteration
    // ------------------------------------------------------------

    /**
     * @notice Solve A x ≈ b using Gauss–Seidel iteration.
     *         Uses updated values within the same iteration (in-place).
     *         Requires non-zero diagonal.
     *
     * @param n Dimension of the square coefficient matrix A (n×n)
     * @param Adata    Flattened n×n coefficient matrix A (row-major)
     * @param bdata    Flattened n×1 right-hand side vector b
     * @param x0data   Flattened n×1 initial guess vector x_0
     * @param maxIter  Maximum iterations.
     * @param tolDiff  Tolerance on squared difference between consecutive iterates.
     *
     * @return x       Approximate solution.
     * @return iters   Number of iterations executed, where the initial guess x0
     *                 is considered iteration 0 and the first update produces x1.
     */
    function gaussSeidel(uint256 n, bytes16[] calldata Adata, bytes16[] calldata bdata, bytes16[] calldata x0data, uint256 maxIter, bytes16 tolDiff)
        external pure returns (bytes16[] memory x, uint256 iters) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix(n, n, _copy(Adata));
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix(n, 1, _copy(bdata));
        MatrixMaster.Matrix memory x0 = MatrixMaster.Matrix(n, 1, _copy(x0data));

        MatrixMaster.Matrix memory out;
        (out, iters) = LinearSolvers.gaussSeidel(A, b, x0, maxIter, tolDiff);

        return (out.data, iters);
    }

    // ------------------------------------------------------------
    // Gaussian Elimination
    // ------------------------------------------------------------

    /**
     * @notice Solve A x = b using Gaussian elimination with partial pivoting.
     *         Operates directly on matrix buffers without forming augmented matrices.
     *
     * @param n Dimension of the square coefficient matrix A (n×n)
     * @param Adata Flattened n×n coefficient matrix A (row-major)
     * @param bdata Flattened n×1 right-hand side vector b
     * @return x Solution vector (n × 1)
     */
    function gaussianElimination(uint256 n, bytes16[] calldata Adata, bytes16[] calldata bdata)
        external pure returns (bytes16[] memory x) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix(n, n, _copy(Adata));
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix(n, 1, _copy(bdata));

        MatrixMaster.Matrix memory out = LinearSolvers.gaussianElimination(A, b);

        return out.data;
    }

    // ------------------------------------------------------------
    // LU Decomposition
    // ------------------------------------------------------------

    /**
     * @notice Compute LU factorization A = L*U using Doolittle's method (no pivoting).
     *         This implementation does NOT perform pivoting.
     *         Zero or near-zero pivots will cause failure even if A is invertible.
     *
     * @param n Dimension of the square input matrix A (n×n)
     * @param Adata Flattened n×n input matrix A (row-major)
     * @return L Unit-lower-triangular factor (nxn).
     * @return U Upper-triangular factor (nxn).
     */
    function luDecomposition(uint256 n, bytes16[] calldata Adata)
        external pure returns (bytes16[] memory L, bytes16[] memory U) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix(n, n, _copy(Adata));

        MatrixMaster.Matrix memory Lm;
        MatrixMaster.Matrix memory Um;
        (Lm, Um) = LinearSolvers.luDecomposition(A);

        return (Lm.data, Um.data);
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    function _copy(bytes16[] calldata src)
        internal pure returns (bytes16[] memory dst) {
        dst = new bytes16[](src.length);
        for (uint256 i = 0; i < src.length; ++i) {
            dst[i] = src[i];
        }
    }
}