// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { LinearSolvers } from "../libraries/numeric/LinearSolvers.sol";
import { MatrixMaster } from "../libraries/numeric/MatrixMaster.sol";
import { MathLib } from "../libraries/MathLib.sol";

/**
 * @title LinearSolversHarness
 * @notice Thin wrapper around LinearSolvers for testing and gas profiling.
 *         All inputs use bytes16 (ABDK quad) and explicit matrix shapes.
 */
contract LinearSolversHarness {

    // ------------------------------------------------------------
    //  Gradient Descent Harness
    // ------------------------------------------------------------

    function gradientDescentLeastSquares(
        uint256 m,
        uint256 n,
        bytes16[] calldata Adata, // m×n row-major
        bytes16[] calldata bdata, // m×1
        bytes16[] calldata x0data, // n×1
        bytes16 alpha,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (bytes16[] memory x, uint256 iters) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix({rows: m, cols: n, data: _copy(Adata)});
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix({rows: m, cols: 1, data: _copy(bdata)});
        MatrixMaster.Matrix memory x0 = MatrixMaster.Matrix({rows: n, cols: 1, data: _copy(x0data)});

        MatrixMaster.Matrix memory out;
        (out, iters) =
            LinearSolvers.gradientDescentLeastSquares(A, b, x0, alpha, maxIter, tol);

        return (out.data, iters);
    }

    // ------------------------------------------------------------
    //  Jacobi Harness
    // ------------------------------------------------------------

    function jacobi(
        uint256 n,
        bytes16[] calldata Adata, // n×n
        bytes16[] calldata bdata, // n×1
        bytes16[] calldata x0data, // n×1
        uint256 maxIter,
        bytes16 tolDiff
    ) external pure returns (bytes16[] memory x, uint256 iters) {
       
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix({rows: n, cols: n, data: _copy(Adata)});
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix({rows: n, cols: 1, data: _copy(bdata)});
        MatrixMaster.Matrix memory x0 = MatrixMaster.Matrix({rows: n, cols: 1, data: _copy(x0data)});

        MatrixMaster.Matrix memory out;
        (out, iters) = LinearSolvers.jacobi(A, b, x0, maxIter, tolDiff);

        return (out.data, iters);
    }

    // ------------------------------------------------------------
    //  Gauss–Seidel Harness
    // ------------------------------------------------------------

    function gaussSeidel(
        uint256 n,
        bytes16[] calldata Adata, // n×n
        bytes16[] calldata bdata, // n×1
        bytes16[] calldata x0data, // n×1
        uint256 maxIter,
        bytes16 tolDiff
    ) external pure returns (bytes16[] memory x, uint256 iters) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix({rows: n, cols: n, data: _copy(Adata)});
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix({rows: n, cols: 1, data: _copy(bdata)});
        MatrixMaster.Matrix memory x0 = MatrixMaster.Matrix({rows: n, cols: 1, data: _copy(x0data)});

        MatrixMaster.Matrix memory out;
        (out, iters) = LinearSolvers.gaussSeidel(A, b, x0, maxIter, tolDiff);

        return (out.data, iters);
    }

    // ------------------------------------------------------------
    //  Gaussian Elimination Harness
    // ------------------------------------------------------------

    function gaussianElimination(
        uint256 n,
        bytes16[] calldata Adata, // n×n
        bytes16[] calldata bdata  // n×1
    ) external pure returns (bytes16[] memory x) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix({rows: n, cols: n, data: _copy(Adata)});
        MatrixMaster.Matrix memory b = MatrixMaster.Matrix({rows: n, cols: 1, data: _copy(bdata)});

        MatrixMaster.Matrix memory out = LinearSolvers.gaussianElimination(A, b);
        return out.data;
    }

    // ------------------------------------------------------------
    //  LU Decomposition Harness
    // ------------------------------------------------------------

    function luDecomposition(
        uint256 n,
        bytes16[] calldata Adata // n×n
    ) external pure returns (bytes16[] memory L, bytes16[] memory U) {
        
        MatrixMaster.Matrix memory A = MatrixMaster.Matrix({rows: n, cols: n, data: _copy(Adata)});

        MatrixMaster.Matrix memory Lm;
        MatrixMaster.Matrix memory Um;
        (Lm, Um) = LinearSolvers.luDecomposition(A);

        return (Lm.data, Um.data);
    }

    // ------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------

    function _copy(bytes16[] calldata src) internal pure returns (bytes16[] memory dst) {
        dst = new bytes16[](src.length);
        for (uint256 i = 0; i < src.length; ++i) {
            dst[i] = src[i];
        }
    }

    // ------------------------------------------------------------
    //  Numerical Helpers
    // ------------------------------------------------------------

    /**
     * @notice Converts a signed integer to IEEE-754 quadruple precision.
     * @param  n Signed integer.
     * @return q Quadruple-precision representation of `n`.
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
     *         Reverts if `den` equals zero.
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

    /// @notice Scaling factor used for JS-style fixed-decimal conversions.
    uint256 public constant SCALE = 1e12;

    /**
     * @notice Converts a quadruple-precision number into a scaled integer (scaled by SCALE).
     * @param x Quadruple-precision value.
     * @return Integer representing x * SCALE.
     */
    function toFloat(bytes16 x) external pure returns (int256) {
        bytes16 qScale = MathLib.fromUInt(SCALE);
        bytes16 scaled = MathLib.mul(x, qScale);
        return MathLib.toInt(scaled);
    }
}