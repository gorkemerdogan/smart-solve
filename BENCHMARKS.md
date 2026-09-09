# SmartSolve tests and benchmarks

SmartSolve reports measurements by execution model rather than presenting one
undifferentiated performance score. A library microbenchmark and a call through
the deployable Diamond answer different questions.

## Test commands

| Command | Intended use |
| --- | --- |
| `npm test` | Default bounded suite for local development and CI. |
| `npm run test:bytecode` | Fast EIP-170 and project-bytecode-budget test for production facets. |
| `npm run test:full` | Full Hardhat suite; can be slow. |
| `npm run test:heavy` | **Expensive, opt-in** linear-solver feasibility corpus. |
| `npm run test:heavy:matrix` | **Expensive, opt-in** MatrixMaster feasibility corpus. |

Heavy suites preserve known gas-boundary cases. Expected OOG outcomes appear in
their summaries; unclassified reverts and other failures still fail the suite.

## Execution models

| Route | Call target | Includes | Excludes | Interpretation |
| --- | --- | --- | --- | --- |
| `harness_direct` | Standalone test harness and numerical library | Harness ABI/memory behaviour and library computation | Diamond fallback, selector lookup, delegatecall, facet composition | Algorithm or arithmetic microbenchmark |
| `facet_direct` | Production facet at its own address | Facet ABI/runtime and linked-library behaviour | Diamond fallback, selector lookup, delegatecall | Facet implementation measurement |
| `diamond_routed` | Production facet ABI encoded to the SmartSolve Diamond | Fallback, selector lookup, delegatecall, proxy-held configuration, facet calldata/memory, linked code | Deployment and selector-installation costs from a one-operation estimate | Deployable SmartSolve operation estimate |

Gas/result metadata is explicit: `estimateGas`, `transaction_receipt_gas`,
`eth_call_result`, and `transaction_plus_call_result`. Integration,
differentiation, ODE, root finding, and optimisation can invoke a target via
`STATICCALL`; their records report callback model and function-evaluation count
where available. Reported gas is end-to-end algorithm-plus-callback gas;
callback gas is not subtracted.

## Benchmark commands

| Command | Purpose |
| --- | --- |
| `npm run benchmark:export:representative` | Export representative Diamond-routed numerical/root-finding results and the MathLib-linkage microbenchmark. |
| `npm run benchmark:export:harness:full` | **Slower.** Export selected generated harness suites. |
| `npm run benchmark:equal-accuracy` | Bounded direct-harness integration/root-finding comparison. |
| `npm run benchmark:equal-accuracy:diamond` | Bounded Diamond-routed integration/root-finding comparison. |
| `npm run benchmark:baseline:polynomial` | Test-only 1e18 fixed-point versus binary128 production-facet Horner baseline. |

Other focused experiments can be run directly:

```sh
npx hardhat test test/mathlib-link-overhead.bench.test.ts
npx hardhat test test/polynomial-horner-linkage-ab.bench.test.ts
npx hardhat test test/summation-strategies.bench.test.ts
npx hardhat test test/integration-compensated-summation.bench.test.ts
npx hardhat test test/matrix-accumulation-production-ab.bench.test.ts
npx hardhat test test/solver-accumulation-ab.bench.test.ts
```

These focused experiments measure specific code paths and input families; they
are not a single uniform SmartSolve performance score. Stable/compensated
accumulation variants may be benchmarked without being production defaults.

## Reproducible local environment

The default benchmark network is Hardhat chain ID `31337`, configured with a
30,000,000 gas block limit and the `merge` hardfork (Paris EVM). Contracts use
Solidity `0.8.28`, optimizer enabled with 200 runs, `viaIR: true`, and
`evmVersion: "paris"`; see [`hardhat.config.ts`](hardhat.config.ts). The
opt-in JSON export records these settings. Figures below are local-Hardhat gas
estimates or receipt values from their named tests, not public-network cost
claims.

## Representative Diamond-routed production-path estimates

Each operation below calls the deployed full SmartSolve Diamond. The facet ABI
only encodes the call to the Diamond address. Values are `estimateGas` paired
with `eth_call` results unless stated otherwise.

| Category | Representative operation | Gas | Callback evaluations | Source |
| --- | --- | ---: | ---: | --- |
| Integration | Simpson 1/3, `n=2` | 80,327 | 3 | `test/numeric-facets.diamond.test.ts` |
| Differentiation | Centered difference, `h=0.01` | 62,393 | 2 | `test/numeric-facets.diamond.test.ts` |
| ODE solving | Euler, 3 steps | 53,837 | 3 | `test/numeric-facets.diamond.test.ts` |
| Polynomial | Horner evaluation, degree 2 | 32,319 | — | `test/numeric-facets.diamond.test.ts` |
| Matrix operations | 2×2 matrix multiplication | 68,996 | — | `test/numeric-facets.diamond.test.ts` |
| Linear solvers | 2×2 Gaussian elimination | 105,374 | — | `test/numeric-facets.diamond.test.ts` |
| Optimisation | Spherical objective, dimension 2, cap 3 | 210,385 | objective-specific | `test/numeric-facets.diamond.test.ts` |
| Root finding | Bisection, `x² − 4` on `[1,4]`, tol `1e-12` | 1,125,674 | 44 | `test/rootfinding.diamond.test.ts` |

The root-finding sample converged in 42 iterations. The full Diamond
deployment plus selector installation used 25,704,257 gas in that same local
test run; deployment/setup gas is not included in the operation estimate.

## Equal-accuracy Diamond sample

`test/equal-accuracy.diamond.bench.test.ts` sweeps a deliberately small set of
configurations and chooses the lowest `diamond_routed` `estimateGas` result
whose exact-rational decoded error meets the target. No-match, revert, OOG, and
other failures are labelled explicitly.

| Problem and target | Selected configuration | Gas | Observed absolute error |
| --- | --- | ---: | ---: |
| Trapezoidal `x²` on `[0,1]`, ε = `1e-2` | `n=8` | 115,274 | 0.00260416666666666666 |
| Trapezoidal `x²` on `[0,1]`, ε = `1e-3` | `n=16` | 186,987 | 0.00065104166666666666 |
| Bisection `x² − 4`, ε = `1e-11` | tol `1e-12`, cap 64 | 1,125,674 | 0.00000000000022737367 |
| Newton `x² − 4`, ε = `1e-11` | tol `1e-12`, cap 64 | 185,883 | 0 |
| Secant `x² − 4`, ε = `1e-11` | tol `1e-12`, cap 64 | 252,443 | 0.00000000000000023984 |

These are bounded sweeps, not global optima. Run
`npm run benchmark:equal-accuracy:diamond` to reproduce the full selection.

## Horner representation baseline

`test/polynomial-fixedpoint-baseline.bench.test.ts` compares
`PolynomialFacet.polyEvaluate` called directly (`facet_direct`) with an
isolated, test-only signed fixed-point 1e18 Horner harness (`harness_direct`).
It is intentionally not a whole-system Diamond comparison. Both use
`estimateGas` and an exact rational oracle from unrounded source inputs.

| Input family | Binary128 facet gas | Fixed-point harness gas | Binary128 absolute error | Fixed-point absolute error |
| --- | ---: | ---: | ---: | ---: |
| Small integer coefficients, degree 2 | 27,345 | 23,345 | 0 | 0 |
| Terminating fractional coefficients, degree 3 | 30,244 | 24,491 | 0 | 0 |
| `1/3` input rounding, degree 2 | 28,968 | 23,729 | ~2.4e-35 | ~1.481481481481481481e-19 |

The non-terminating `1/3` case demonstrates fixed-point input-scaling error;
it does not claim either representation dominates every workload.

| Deployment footprint in the same baseline | Runtime bytes | Deployment gas |
| --- | ---: | ---: |
| Shared linked MathLib | 4,573 | 1,040,850 |
| Production PolynomialFacet | 7,923 | 1,764,249 |
| Production PolynomialFacet plus required MathLib | — | 2,805,099 |
| Test-only fixed-point Horner harness | 474 | 155,555 |

## Horner MathLib linkage experiment

`test/polynomial-horner-linkage-ab.bench.test.ts` is a narrow A/B experiment:
a linked-Horner control versus the current Horner-only inlined add/multiply
path. It checks bit-identical binary128 results for direct-facet and
minimal-Diamond calls. That minimal Diamond installs one selector, so it is not
the full SmartSolve deployment.

| Aggregate across degrees 2, 4, 8, 16, 32 | Linked control gas | Current production gas | Difference | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Direct facet | 363,693 | 242,215 | 121,478 | 1.50× |
| Minimal Diamond route | 393,889 | 269,801 | 124,088 | 1.45× |

At degree 32, direct calls measured 147,427 gas (linked control) versus
88,406 gas (current production); minimal-Diamond calls measured 154,956 versus
94,648 gas. This supports only the Horner-specific change, not global MathLib
inlining.

## Deployment and bytecode feasibility

| Measure | Result | Scope/source |
| --- | ---: | --- |
| Full SmartSolve Diamond deployment plus facet installation | 25,704,257 gas | `test/rootfinding.diamond.test.ts` |
| MatrixMasterFacet runtime | 22,425 bytes | `test/bytecode-size.test.ts` |
| MatrixMasterFacet EIP-170 headroom | 2,151 bytes | 24,576-byte EIP-170 limit |
| MatrixMasterFacet project budget | 23,000 bytes | `test/bytecode-size.test.ts` |
| LinearSolversFacet runtime | 16,015 bytes | `test/bytecode-size.test.ts` |

`npm run test:bytecode` checks all production facets, fails at EIP-170, and
enforces the tighter MatrixMaster and LinearSolvers budgets. MatrixMaster's
headroom is an active deployment constraint.

## Known feasibility boundaries

The following workloads are retained behind
`RUN_HEAVY_MATRIX_SCALABILITY=1` because they approach or cross the configured
30M local block-gas limit. OOG results are reported in feasibility summaries,
not hidden or treated as ordinary numerical-accuracy failures:

- dense–sparse matrix subtraction at `n=92`;
- determinant cases at `n=29`;
- inversion cases at `n=21`; and
- high-bandwidth sparse matrix–vector cases, including `n=128`, bandwidth 32.

`npm run test:heavy` similarly retains its large linear-solver corpus as an
opt-in scalability experiment. Passing bounded tests does not erase these
observed resource boundaries.

## JSON benchmark exports

Exports are opt-in. Set `EXPORT_BENCHMARK_RESULTS=1`, or use the representative
export command which sets it for you:

```sh
npm run benchmark:export:representative
EXPORT_BENCHMARK_RESULTS=1 npm run benchmark:equal-accuracy:diamond
EXPORT_BENCHMARK_RESULTS=1 npm run benchmark:baseline:polynomial
```

One JSON document per suite is written under `benchmark-results/` using the
additive v1 schema:

```json
{
  "format": "smart-solve-benchmark-results-v1",
  "suite": "numeric-facets.diamond",
  "run": { "network": { "name": "hardhat", "chainId": 31337 } },
  "records": [
    {
      "benchmark": "Simpson 1/3, n=2",
      "execution": { "route": "diamond_routed", "gasModel": "estimateGas" },
      "gas": "80327",
      "status": "success"
    }
  ]
}
```

Run metadata includes package/lockfile identity, Git state, tool versions,
platform, invocation and heavy/export flags, compiler settings, network, and
gas-limit context. Records retain available input, callback, gas, error, and
convergence/failure metadata. `benchmark-results/` is gitignored; preserve
exports in a controlled artifact store when they support a comparison.

## Accuracy and scope

The suites use exact-rational, host-side, and method-reference oracles where
appropriate. They do not establish empirical binary128-level accuracy for every
algorithm/input family. Read route, gas model, callback model, and oracle or
tolerance fields before comparing records.

For local/Sepolia deployment, artifact contents, owner responsibility, and
security cautions, see [DEPLOYMENT.md](DEPLOYMENT.md).
