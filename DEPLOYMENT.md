# SmartSolve deployment and verification

This guide covers the full production SmartSolve Diamond deployment. It does
not deploy to any public network automatically.

## Local readiness

Run these checks before deployment:

```sh
npm test
npm run test:bytecode
npx tsc --noEmit
npm run deploy:local
git diff --check
```

`npm run deploy:local` deploys the full Diamond to an ephemeral local Hardhat
network, writes a deployment artifact, and immediately runs the post-deploy
verifier.

## Local persistent-node workflow

For separate deployment and verification commands, start a local node in one
terminal:

```sh
npx hardhat node
```

Then deploy in another terminal:

```sh
npm run deploy -- --network localhost
```

The deploy command prints an artifact path. Verify that deployment with:

```sh
DEPLOYMENT_ARTIFACT=deployments/<artifact>.json \
  npm run verify:deployment -- --network localhost
```

## Sepolia workflow

Configure the deployment environment with a funded deployer account and an
HTTPS RPC endpoint:

```sh
export SEPOLIA_RPC_URL=https://...
export DEPLOYER_PRIVATE_KEY=0x...
```

Both variables are required before the `sepolia` Hardhat network is enabled.
Deploy explicitly:

```sh
npm run deploy -- --network sepolia
```

Record the artifact path printed by the deploy command, then verify the
existing Diamond using the same network:

```sh
DEPLOYMENT_ARTIFACT=deployments/<artifact>.json \
  npm run verify:deployment -- --network sepolia
```

Optionally set `DIAMOND_ADDRESS` as an additional guard; it must match the
Diamond address in the artifact.

## Deployment artifact

Each deployment writes a JSON artifact containing:

- network name and chain ID;
- deployer address and current Diamond owner;
- Diamond, MathLib, and every facet address;
- installed selectors by facet and their total count;
- per-contract deployment gas, Diamond-cut installation gas, and total gas;
- Solidity compiler and optimization settings.

The verifier checks the recorded chain ID, owner, facets, selector routing,
effective numeric configuration, and a small polynomial call routed through
the Diamond fallback.

`deployments/` is gitignored because artifacts are environment-specific
operational records, not source code. Preserve the testnet artifact in the
release record, a controlled deployment archive, or other access-controlled
backup. Do not rely on a local untracked file as the only record of a public
deployment.

## Deployment cautions

- `MatrixMasterFacet` is currently 22,425 runtime bytes: 2,151 bytes below
  EIP-170's 24,576-byte limit. The bytecode test enforces its tighter 23,000
  byte project budget.
- The Diamond owner controls upgrades and numeric configuration. Use the
  intended owner address and protect its key or multisig process before a
  public deployment.
- Treat `DEPLOYER_PRIVATE_KEY` and RPC credentials as secrets. Keep them out
  of shell history, source control, CI logs, and deployment artifacts.
- Heavy scalability benchmarks (`npm run test:heavy` and
  `npm run test:heavy:matrix`) are opt-in feasibility experiments. They are
  not part of deployment readiness checks.
