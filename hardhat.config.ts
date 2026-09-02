import type { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";

const config: HardhatUserConfig = {
  networks: {
    hardhat: {
      chainId: 31337,
      // Hardhat names Ethereum's Paris EVM hardfork "merge".
      hardfork: "merge",
      blockGasLimit: 30_000_000,
    },
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
