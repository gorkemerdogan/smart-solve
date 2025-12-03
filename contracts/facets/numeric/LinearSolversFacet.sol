// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LinearSolvers } from "../libraries/LinearSolvers.sol";

/**
 * @title LinearSolversFacet
 * @notice Diamond facet exposing linear solver utilities externally.
 * @dev
 *  - This facet is stateless; all methods are pure.
 *  - Plug into your Diamond via standard facet deployment and selector registration.
 */
contract LinearSolversFacet {
    /**
     * @notice Public gradient descent solver.
     * @param n System dimension.
     * @param A Row-major n×n coefficient matrix.
     * @param b RHS vector.
     * @param x0 Initial guess.
     * @param alpha Step size (integer).
     * @param maxIter Maximum number of iterations.
     * @param tol Squared residual tolerance.
     */
    function gradientDescent(
        uint256 n,
        int256[] calldata A,
        int256[] calldata b,
        int256[] calldata x0,
        int256 alpha,
        uint256 maxIter,
        int256 tol
    ) external pure returns (int256[] memory x, uint256 iters) {
        int256[] memory Am = new int256[](A.length);
        int256[] memory bm = new int256[](b.length);
        int256[] memory x0m = new int256[](x0.length);

        for (uint256 i = 0; i < A.length; ++i) Am[i] = A[i];
        for (uint256 i = 0; i < b.length; ++i) bm[i] = b[i];
        for (uint256 i = 0; i < x0.length; ++i) x0m[i] = x0[i];

        return LinearSolvers.gradientDescent(n, Am, bm, x0m, alpha, maxIter, tol);
    }

    /**
     * @notice Public Jacobi solver.
     */
    function jacobi(
        uint256 n,
        int256[] calldata A,
        int256[] calldata b,
        int256[] calldata x0,
        uint256 maxIter,
        int256 tol
    ) external pure returns (int256[] memory x, uint256 iters) {
        int256[] memory Am = new int256[](A.length);
        int256[] memory bm = new int256[](b.length);
        int256[] memory x0m = new int256[](x0.length);

        for (uint256 i = 0; i < A.length; ++i) Am[i] = A[i];
        for (uint256 i = 0; i < b.length; ++i) bm[i] = b[i];
        for (uint256 i = 0; i < x0.length; ++i) x0m[i] = x0[i];

        return LinearSolvers.jacobi(n, Am, bm, x0m, maxIter, tol);
    }

    /**
     * @notice Public Gauss-Seidel solver.
     */
    function gaussSeidel(uint256 n, int256[] calldata A, int256[] calldata b, int256[] calldata x0, uint256 maxIter, int256 tol)
        external pure returns (int256[] memory x, uint256 iters) {

        int256[] memory Am = new int256[](A.length);
        int256[] memory bm = new int256[](b.length);
        int256[] memory x0m = new int256[](x0.length);

        for (uint256 i = 0; i < A.length; ++i) Am[i] = A[i];
        for (uint256 i = 0; i < b.length; ++i) bm[i] = b[i];
        for (uint256 i = 0; i < x0.length; ++i) x0m[i] = x0[i];

        return LinearSolvers.gaussSeidel(n, Am, bm, x0m, maxIter, tol);
    }

    /**
     * @notice Public Gaussian elimination solver.
     */
    function gaussianElimination(uint256 n, int256[] calldata A, int256[] calldata b) external pure returns (int256[] memory x) {
        int256[] memory Am = new int256[](A.length);
        int256[] memory bm = new int256[](b.length);

        for (uint256 i = 0; i < A.length; ++i) Am[i] = A[i];
        for (uint256 i = 0; i < b.length; ++i) bm[i] = b[i];

        return LinearSolvers.gaussianElimination(n, Am, bm);
    }

    /**
     * @notice Public LU decomposition.
     */
    function luDecomposition(uint256 n, int256[] calldata A) external pure returns (int256[] memory L, int256[] memory U) {
        int256[] memory Am = new int256[](A.length);
        for (uint256 i = 0; i < A.length; ++i) Am[i] = A[i];

        return LinearSolvers.luDecomposition(n, Am);
    }
}