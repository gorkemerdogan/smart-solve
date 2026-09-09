import type { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";

const sepoliaRpcUrl = process.env.SEPOLIA_RPC_URL;
const deployerPrivateKey = process.env.DEPLOYER_PRIVATE_KEY;

const config: HardhatUserConfig = {
  networks: {
    hardhat: {
      chainId: 31337,
      // Hardhat names Ethereum's Paris EVM hardfork "merge".
      hardfork: "merge",
      blockGasLimit: 30_000_000,
    },
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
    // Testnet access is opt-in: no remote network is configured unless both
    // secrets are supplied by the deployment environment.
    ...(sepoliaRpcUrl && deployerPrivateKey
      ? {
          sepolia: {
            url: sepoliaRpcUrl,
            accounts: [deployerPrivateKey],
          },
        }
      : {}),
  },
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      viaIR: true,
      evmVersion: "paris",
    }
  }
};

export default config;
