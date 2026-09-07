// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { MathLib } from "../libraries/MathLib.sol";

/**
 * @notice Test-only solver-shaped A/B harness for accumulation strategies.
 * @dev The naive branches preserve the production loop order.  The stable
 *      branches change only repeated additions to Neumaier accumulation.
 *      Inputs are deliberately bounded so this cannot become an unbounded
 *      on-chain solver API.
 */
contract SolverAccumulationBenchmarkHarness {
    struct SolverResult {
        bytes16[] vector;
        uint256 iterations;
        bool converged;
        bytes16 finalMetric;
    }

    function jacobiNaive(
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b,
        bytes16[] calldata x0,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory) {
        return _stationary(n, a, b, x0, maxIter, tol, false, false);
    }

    function jacobiCompensated(
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b,
        bytes16[] calldata x0,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory) {
        return _stationary(n, a, b, x0, maxIter, tol, false, true);
    }

    function gaussSeidelNaive(
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b,
        bytes16[] calldata x0,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory) {
        return _stationary(n, a, b, x0, maxIter, tol, true, false);
    }

    function gaussSeidelCompensated(
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b,
        bytes16[] calldata x0,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory) {
        return _stationary(n, a, b, x0, maxIter, tol, true, true);
    }

    function gradientDescentNaive(
        uint256 m,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b,
        bytes16[] calldata x0,
        bytes16 alpha,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory) {
        return _gradientDescent(m, n, a, b, x0, alpha, maxIter, tol, false);
    }

    function gradientDescentCompensated(
        uint256 m,
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata b,
        bytes16[] calldata x0,
        bytes16 alpha,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory) {
        return _gradientDescent(m, n, a, b, x0, alpha, maxIter, tol, true);
    }

    /**
     * @dev Uses a supplied deterministic x0 instead of production randomVector.
     *      All subsequent update, normalization, +/- direction convergence,
     *      and Rayleigh-quotient rules match the production power-iteration shape.
     */
    function powerIterationNaive(
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata x0,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory result, bytes16 lambda) {
        return _powerIteration(n, a, x0, maxIter, tol, false);
    }

    function powerIterationCompensated(
        uint256 n,
        bytes16[] calldata a,
        bytes16[] calldata x0,
        uint256 maxIter,
        bytes16 tol
    ) external pure returns (SolverResult memory result, bytes16 lambda) {
        return _powerIteration(n, a, x0, maxIter, tol, true);
    }

    function _stationary(
        uint256 n,
        bytes16[] calldata aInput,
        bytes16[] calldata bInput,
        bytes16[] calldata x0Input,
        uint256 maxIter,
        bytes16 tol,
        bool gaussSeidel,
        bool compensated
    ) private pure returns (SolverResult memory result) {
        _validateSquare(n, aInput.length, bInput.length, x0Input.length, maxIter);
        bytes16[] memory a = _copy(aInput);
        bytes16[] memory b = _copy(bInput);
        bytes16[] memory x = _copy(x0Input);
        bytes16[] memory old = new bytes16[](n);

        for (uint256 iter = 0; iter < maxIter; ++iter) {
            for (uint256 i = 0; i < n; ++i) old[i] = x[i];
            bytes16 diffSum;
            bytes16 diffCorrection;

            for (uint256 i = 0; i < n; ++i) {
                bytes16 sum = b[i];
                bytes16 correction;
                bytes16 diagonal = a[i * n + i];
                require(MathLib.cmp(diagonal, bytes16(0)) != 0, "SolverAccumulationBenchmark: zero diagonal");

                for (uint256 j = 0; j < n; ++j) {
                    if (j == i) continue;
                    bytes16 xj = gaussSeidel && j < i ? x[j] : old[j];
                    bytes16 term = MathLib.neg(MathLib.mul(a[i * n + j], xj));
                    (sum, correction) = _add(sum, correction, term, compensated);
                }
                x[i] = MathLib.div(_finish(sum, correction, compensated), diagonal);
                bytes16 delta = MathLib.sub(x[i], old[i]);
                (diffSum, diffCorrection) = _add(
                    diffSum,
                    diffCorrection,
                    MathLib.mul(delta, delta),
                    compensated
                );
            }

            result.iterations = iter + 1;
            bytes16 diff2 = _finish(diffSum, diffCorrection, compensated);
            if (MathLib.cmp(diff2, tol) <= 0 && _residualWithinTolerance(a, b, x, n, tol, compensated)) {
                result.converged = true;
                break;
            }
        }

        result.vector = x;
        result.finalMetric = _residualNormSquared(a, b, x, n, n, compensated);
    }

    function _gradientDescent(
        uint256 m,
        uint256 n,
        bytes16[] calldata aInput,
        bytes16[] calldata bInput,
        bytes16[] calldata x0Input,
        bytes16 alpha,
        uint256 maxIter,
        bytes16 tol,
        bool compensated
    ) private pure returns (SolverResult memory result) {
        require(m > 0 && m <= 8 && n > 0 && n <= 8, "SolverAccumulationBenchmark: dimensions");
        require(aInput.length == m * n && bInput.length == m && x0Input.length == n, "SolverAccumulationBenchmark: lengths");
        require(maxIter > 0 && maxIter <= 64, "SolverAccumulationBenchmark: iterations");
        require(MathLib.cmp(alpha, bytes16(0)) > 0, "SolverAccumulationBenchmark: alpha");

        bytes16[] memory a = _copy(aInput);
        bytes16[] memory b = _copy(bInput);
        bytes16[] memory x = _copy(x0Input);
        bytes16[] memory residual = new bytes16[](m);
        bytes16[] memory gradient = new bytes16[](n);

        for (uint256 iter = 0; iter < maxIter; ++iter) {
            _matvecResidual(a, b, x, residual, m, n, compensated);
            bytes16 gradientNorm2 = _gradient(a, residual, gradient, m, n, compensated);
            result.iterations = iter;
            if (MathLib.cmp(gradientNorm2, tol) <= 0) {
                result.converged = true;
                break;
            }
            for (uint256 j = 0; j < n; ++j) {
                x[j] = MathLib.sub(x[j], MathLib.mul(gradient[j], alpha));
            }
            result.iterations = iter + 1;
        }

        _matvecResidual(a, b, x, residual, m, n, compensated);
        result.vector = x;
        result.finalMetric = _gradient(a, residual, gradient, m, n, compensated);
    }

    function _powerIteration(
        uint256 n,
        bytes16[] calldata aInput,
        bytes16[] calldata x0Input,
        uint256 maxIter,
        bytes16 tol,
        bool compensated
    ) private pure returns (SolverResult memory result, bytes16 lambda) {
        _validateSquare(n, aInput.length, 0, x0Input.length, maxIter);
        bytes16[] memory a = _copy(aInput);
        bytes16[] memory x = _normalize(_copy(x0Input), compensated);

        for (uint256 iter = 0; iter < maxIter; ++iter) {
            bytes16[] memory y = _matvec(a, x, n, n, compensated);
            if (MathLib.cmp(_dot(y, y, compensated), bytes16(0)) == 0) break;
            bytes16[] memory next = _normalize(y, compensated);
            result.iterations = iter + 1;
            if (_powerConverged(next, x, tol, compensated)) {
                x = next;
                result.converged = true;
                break;
            }
            x = next;
        }

        bytes16[] memory ax = _matvec(a, x, n, n, compensated);
        lambda = _dot(x, ax, compensated);
        result.vector = x;
        result.finalMetric = _eigenResidualSquared(ax, x, lambda, compensated);
    }

    function _matvecResidual(
        bytes16[] memory a,
        bytes16[] memory b,
        bytes16[] memory x,
        bytes16[] memory residual,
        uint256 m,
        uint256 n,
        bool compensated
    ) private pure {
        for (uint256 i = 0; i < m; ++i) {
            bytes16 sum;
            bytes16 correction;
            for (uint256 j = 0; j < n; ++j) {
                (sum, correction) = _add(sum, correction, MathLib.mul(a[i * n + j], x[j]), compensated);
            }
            residual[i] = MathLib.sub(_finish(sum, correction, compensated), b[i]);
        }
    }

    function _gradient(
        bytes16[] memory a,
        bytes16[] memory residual,
        bytes16[] memory gradient,
        uint256 m,
        uint256 n,
        bool compensated
    ) private pure returns (bytes16 norm2) {
        bytes16 normCorrection;
        for (uint256 j = 0; j < n; ++j) {
            bytes16 sum;
            bytes16 correction;
            for (uint256 i = 0; i < m; ++i) {
                (sum, correction) = _add(sum, correction, MathLib.mul(a[i * n + j], residual[i]), compensated);
            }
            gradient[j] = _finish(sum, correction, compensated);
            (norm2, normCorrection) = _add(norm2, normCorrection, MathLib.mul(gradient[j], gradient[j]), compensated);
        }
        return _finish(norm2, normCorrection, compensated);
    }

    function _residualWithinTolerance(
        bytes16[] memory a,
        bytes16[] memory b,
        bytes16[] memory x,
        uint256 n,
        bytes16 tol,
        bool compensated
    ) private pure returns (bool) {
        bytes16 residual2 = _residualNormSquared(a, b, x, n, n, compensated);
        bytes16 b2 = _dot(b, b, compensated);
        bytes16 one = MathLib.fromUInt(1);
        bytes16 scale2 = MathLib.cmp(b2, one) > 0 ? b2 : one;
        return MathLib.cmp(residual2, MathLib.mul(tol, scale2)) <= 0;
    }

    function _residualNormSquared(
        bytes16[] memory a,
        bytes16[] memory b,
        bytes16[] memory x,
        uint256 m,
        uint256 n,
        bool compensated
    ) private pure returns (bytes16) {
        bytes16[] memory residual = new bytes16[](m);
        _matvecResidual(a, b, x, residual, m, n, compensated);
        return _dot(residual, residual, compensated);
    }

    function _matvec(
        bytes16[] memory a,
        bytes16[] memory x,
        uint256 m,
        uint256 n,
        bool compensated
    ) private pure returns (bytes16[] memory result) {
        result = new bytes16[](m);
        for (uint256 i = 0; i < m; ++i) {
            bytes16 sum;
            bytes16 correction;
            for (uint256 j = 0; j < n; ++j) {
                (sum, correction) = _add(sum, correction, MathLib.mul(a[i * n + j], x[j]), compensated);
            }
            result[i] = _finish(sum, correction, compensated);
        }
    }

    function _dot(bytes16[] memory a, bytes16[] memory b, bool compensated) private pure returns (bytes16 sum) {
        bytes16 correction;
        for (uint256 i = 0; i < a.length; ++i) {
            (sum, correction) = _add(sum, correction, MathLib.mul(a[i], b[i]), compensated);
        }
        return _finish(sum, correction, compensated);
    }

    function _normalize(bytes16[] memory vector, bool compensated) private pure returns (bytes16[] memory) {
        bytes16 norm = MathLib.sqrt(_dot(vector, vector, compensated));
        require(MathLib.cmp(norm, bytes16(0)) != 0, "SolverAccumulationBenchmark: zero vector");
        for (uint256 i = 0; i < vector.length; ++i) vector[i] = MathLib.div(vector[i], norm);
        return vector;
    }

    function _powerConverged(
        bytes16[] memory next,
        bytes16[] memory old,
        bytes16 tol,
        bool compensated
    ) private pure returns (bool) {
        bytes16 positive;
        bytes16 positiveCorrection;
        bytes16 negative;
        bytes16 negativeCorrection;
        for (uint256 i = 0; i < next.length; ++i) {
            bytes16 delta = MathLib.sub(next[i], old[i]);
            bytes16 sum = MathLib.add(next[i], old[i]);
            (positive, positiveCorrection) = _add(positive, positiveCorrection, MathLib.mul(delta, delta), compensated);
            (negative, negativeCorrection) = _add(negative, negativeCorrection, MathLib.mul(sum, sum), compensated);
        }
        if (MathLib.cmp(MathLib.sqrt(_finish(positive, positiveCorrection, compensated)), tol) < 0) return true;
        return MathLib.cmp(MathLib.sqrt(_finish(negative, negativeCorrection, compensated)), tol) < 0;
    }

    function _eigenResidualSquared(
        bytes16[] memory ax,
        bytes16[] memory x,
        bytes16 lambda,
        bool compensated
    ) private pure returns (bytes16 sum) {
        bytes16 correction;
        for (uint256 i = 0; i < x.length; ++i) {
            bytes16 residual = MathLib.sub(ax[i], MathLib.mul(lambda, x[i]));
            (sum, correction) = _add(sum, correction, MathLib.mul(residual, residual), compensated);
        }
        return _finish(sum, correction, compensated);
    }

    function _add(bytes16 sum, bytes16 correction, bytes16 term, bool compensated)
        private
        pure
        returns (bytes16 nextSum, bytes16 nextCorrection)
    {
        nextSum = MathLib.add(sum, term);
        if (!compensated) return (nextSum, correction);
        if (MathLib.cmp(MathLib.abs(sum), MathLib.abs(term)) >= 0) {
            nextCorrection = MathLib.add(correction, MathLib.add(MathLib.sub(sum, nextSum), term));
        } else {
            nextCorrection = MathLib.add(correction, MathLib.add(MathLib.sub(term, nextSum), sum));
        }
    }

    function _finish(bytes16 sum, bytes16 correction, bool compensated) private pure returns (bytes16) {
        return compensated ? MathLib.add(sum, correction) : sum;
    }

    function _copy(bytes16[] calldata source) private pure returns (bytes16[] memory result) {
        result = new bytes16[](source.length);
        for (uint256 i = 0; i < source.length; ++i) result[i] = source[i];
    }

    function _validateSquare(
        uint256 n,
        uint256 aLength,
        uint256 bLength,
        uint256 xLength,
        uint256 maxIter
    ) private pure {
        require(n > 0 && n <= 8, "SolverAccumulationBenchmark: dimensions");
        require(aLength == n * n && xLength == n, "SolverAccumulationBenchmark: lengths");
        require(bLength == 0 || bLength == n, "SolverAccumulationBenchmark: rhs length");
        require(maxIter > 0 && maxIter <= 64, "SolverAccumulationBenchmark: iterations");
    }
}
