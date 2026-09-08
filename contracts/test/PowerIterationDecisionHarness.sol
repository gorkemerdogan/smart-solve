// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MatrixMaster } from "../libraries/numeric/MatrixMaster.sol";
import { MathLib } from "../libraries/MathLib.sol";

/** @notice Focused test wrapper for power-iteration decision regressions. */
contract PowerIterationDecisionHarness {
    function initialVector(uint256 n, bytes32 seed) external pure returns (bytes16[] memory) {
        return MatrixMaster.normalize(MatrixMaster.randomVector(n, seed)).data;
    }

    function powerIteration(
        uint256 n,
        bytes16[] calldata data,
        bytes32 seed,
        bytes16 tol
    ) external view returns (bytes16 lambda, bytes16[] memory vector) {
        require(n > 0 && data.length == n * n, "PowerIterationDecisionHarness: shape");
        bytes16[] memory copied = new bytes16[](data.length);
        for (uint256 i = 0; i < data.length; ++i) copied[i] = data[i];
        MatrixMaster.Matrix memory result;
        (lambda, result) = MatrixMaster.powerIteration(
            MatrixMaster.Matrix({ rows: n, cols: n, data: copied }), seed, tol
        );
        return (lambda, result.data);
    }

    function powerIterationWithStatusAndMaxIter(
        uint256 n,
        bytes16[] calldata data,
        bytes32 seed,
        bytes16 tol,
        uint256 maxIter
    ) external view returns (bytes16 lambda, bytes16[] memory vector, uint256 iterations, bool converged) {
        require(n > 0 && data.length == n * n, "PowerIterationDecisionHarness: shape");
        bytes16[] memory copied = new bytes16[](data.length);
        for (uint256 i = 0; i < data.length; ++i) copied[i] = data[i];
        MatrixMaster.PowerIterationResult memory result = MatrixMaster.powerIterationWithStatusAndMaxIter(
            MatrixMaster.Matrix({ rows: n, cols: n, data: copied }), seed, tol, maxIter
        );
        return (result.lambda, result.eigenvector.data, result.iterations, result.converged);
    }

    /** @notice Test-only copy of the pre-hardening zero-image decision path. */
    function legacyPowerIteration(
        uint256 n,
        bytes16[] calldata data,
        bytes32 seed,
        bytes16 tol,
        uint256 maxIter
    ) external pure returns (bytes16 lambda, bytes16[] memory vector, uint256 iterations, bool converged) {
        require(n > 0 && data.length == n * n, "PowerIterationDecisionHarness: shape");
        require(maxIter > 0, "PowerIterationDecisionHarness: iterations");
        bytes16[] memory copied = new bytes16[](data.length);
        for (uint256 i = 0; i < data.length; ++i) copied[i] = data[i];
        MatrixMaster.Matrix memory matrix = MatrixMaster.Matrix({ rows: n, cols: n, data: copied });
        MatrixMaster.Matrix memory x = MatrixMaster.normalize(MatrixMaster.randomVector(n, seed));

        for (uint256 iter = 0; iter < maxIter; ++iter) {
            MatrixMaster.Matrix memory y = MatrixMaster.multiplyMatrixVector(matrix, x);
            if (MathLib.isZero(MatrixMaster.euclideanNorm(y))) break;
            MatrixMaster.Matrix memory next = MatrixMaster.normalize(y);
            iterations = iter + 1;
            if (MatrixMaster.hasConverged(next, x, tol)) {
                x = next;
                converged = true;
                break;
            }
            x = next;
        }

        MatrixMaster.Matrix memory ax = MatrixMaster.multiplyMatrixVector(matrix, x);
        return (MatrixMaster.dot(x, ax), x.data, iterations, converged);
    }
}
