// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LinearSolvers } from "../libraries/numeric/LinearSolvers.sol";
import { MatrixMaster } from "../libraries/numeric/MatrixMaster.sol";
import { MathLib } from "../libraries/MathLib.sol";

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