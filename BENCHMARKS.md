# SmartSolve tests and benchmarks

SmartSolve is an experimental numerical-computation and benchmarking project.
The test suite deliberately separates bounded correctness checks from expensive
feasibility experiments, and separates direct harness measurements from calls
routed through the production Diamond.

## Test commands

| Command | Intended use |
| --- | --- |
| `npm test` | Default bounded suite for local development and CI. It includes core Diamond, configuration, validation, bytecode-budget, result-writer, and representative production-path checks. |
| `npm run test:bytecode` | Fast EIP-170 and project bytecode-budget check for all production facets. |
| `npm run test:full` | Entire Hardhat suite. It can be slow and may include feasibility cases near the local 30M block-gas limit. |
| `npm run test:heavy` | Opt-in large linear-solver feasibility/scalability corpus (`RUN_HEAVY_SCALABILITY=1`). |
| `npm run test:heavy:matrix` | Opt-in large MatrixMaster feasibility/scalability corpus (`RUN_HEAVY_MATRIX_SCALABILITY=1`). |

Heavy suites retain out-of-gas boundary cases as reported feasibility outcomes.
They still fail for unclassified reverts or other unexpected failures. They are
not part of the default readiness check.

## Execution models

Benchmark output and JSON records include explicit execution metadata:

- `harness_direct`: a standalone test harness directly invokes library logic.
- `facet_direct`: a direct call to a deployed production facet; it excludes
  Diamond fallback and delegatecall overhead.
- `diamond_routed`: a call through the deployed SmartSolve Diamond proxy,
  including selector lookup, fallback, and delegatecall.
- `estimateGas`: gas from `eth_estimateGas`.
- `transaction_receipt_gas`: gas from a mined transaction receipt.
- `eth_call_result`: a result returned by call simulation.
- `transaction_plus_call_result`: receipt gas paired with a repeated
  `eth_call` to retrieve a return value.

Integration, differentiation, ODE, root-finding, and optimization operations
may call a user-supplied target with `STATICCALL`. Their records report a
callback model and function-evaluation count where available. Reported gas is
end-to-end algorithm-plus-callback gas; callback gas is not subtracted.

## Benchmark commands

| Command | Purpose |
| --- | --- |
| `npm run benchmark:export:representative` | Exports representative Diamond-routed numerical and root-finding measurements, plus the MathLib linking microbenchmark. |
| `npm run benchmark:export:harness:full` | Exports the selected generated harness suites: integration, ODE, root finding, and matrix add/subtract. This is materially slower than the representative export. |
| `npm run benchmark:equal-accuracy` | Runs the controlled equal-accuracy integration/root-finding comparison. |
| `npm run benchmark:baseline:polynomial` | Runs the 1e18 fixed-point versus binary128 production-facet Horner baseline. Add `EXPORT_BENCHMARK_RESULTS=1` to export it. |

The following focused experiments currently have no `npm` alias and can be run
directly when needed:

```sh
npx hardhat test test/mathlib-link-overhead.bench.test.ts
npx hardhat test test/polynomial-horner-linkage-ab.bench.test.ts
npx hardhat test test/summation-strategies.bench.test.ts
npx hardhat test test/integration-compensated-summation.bench.test.ts
npx hardhat test test/matrix-accumulation-production-ab.bench.test.ts
npx hardhat test test/solver-accumulation-ab.bench.test.ts
```

These are targeted evidence, not a single uniform performance score. Some
stable or compensated accumulation variants are benchmarked as alternatives
and are not necessarily the production default.

## JSON benchmark exports

Set `EXPORT_BENCHMARK_RESULTS=1` to enable structured export for suites wired
to the shared result writer:

```sh
EXPORT_BENCHMARK_RESULTS=1 npm run benchmark:baseline:polynomial
```

The writer creates one JSON document per suite under `benchmark-results/`.
Each file retains the additive v1 schema identifier:

```json
{
  "format": "smart-solve-benchmark-results-v1",
  "suite": "example-suite",
  "run": { "timestamp": "...", "network": { "name": "hardhat" } },
  "records": [{ "benchmark": "...", "execution": { "route": "..." } }]
}
```

Run metadata includes the package and lockfile hash, Git commit/branch/dirty
state when available, Node/Hardhat/ethers/TypeScript versions, OS details,
invocation and relevant environment flags, Solidity compiler settings, network,
chain ID, and block-gas limit. Records include available operation inputs, gas,
execution model, status/failure classification, callback metadata, error
metrics, and convergence data.

`benchmark-results/` is gitignored. Treat exports as reproducibility artifacts:
preserve the files alongside experiment records or in controlled artifact
storage when they support a report, rather than relying only on console output.

## Accuracy and feasibility scope

The suites use exact-rational, host-side, and method-reference oracles where
appropriate, but not every accuracy suite establishes empirical binary128-level
accuracy for every workload. Read the execution and precision labels emitted by
each suite before comparing results. Large scalability cases are feasibility
measurements under the configured local environment, not guarantees of public
network deployability or transaction inclusion.

## Deployment

Use [DEPLOYMENT.md](DEPLOYMENT.md) for the local deploy-and-verify workflow,
Sepolia prerequisites, artifact preservation, and ownership/security cautions.
In particular, run `npm run test:bytecode` before deployment: `MatrixMasterFacet`
has limited EIP-170 headroom and is protected by a tighter project budget.
