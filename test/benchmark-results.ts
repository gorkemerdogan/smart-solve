import { artifacts, ethers, network } from "hardhat";
import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { BenchmarkExecutionMetadata, ExecutionFailureKind } from "./test-utils";

type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type BenchmarkExecutionRecord = {
  route: BenchmarkExecutionMetadata["route"];
  gasModel: BenchmarkExecutionMetadata["gas"];
  resultModel: "eth_call_result" | "transaction_plus_call_result";
  transactionPreflight?: boolean;
};

export type BenchmarkResultRecord = {
  suite: string;
  benchmark: string;
  category: string;
  operation: string;
  execution: BenchmarkExecutionRecord;
  input?: Record<string, JsonValue>;
  callback?: {
    model: "target_staticcall";
    functionEvaluations?: string;
    detail?: string;
    gasIncludesCallback: boolean;
  };
  gas?: string;
  status: "success" | "failure" | "skipped";
  failureKind?: ExecutionFailureKind;
  errorMetrics?: Record<string, JsonValue>;
  convergence?: Record<string, JsonValue>;
  variant?: string;
};

export type BenchmarkRunMetadata = {
  timestamp: string;
  git: { commit?: string; branch?: string; dirty?: boolean };
  nodeVersion: string;
  hardhatVersion?: string;
  solidity?: {
    version: string;
    longVersion: string;
    optimizer: { enabled?: boolean; runs?: number };
    viaIR?: boolean;
    evmVersion?: string;
  };
  network?: { name: string; chainId: string };
  blockGasLimit?: string;
};

export type BenchmarkResultDocument = {
  format: "smart-solve-benchmark-results-v1";
  suite: string;
  run: BenchmarkRunMetadata;
  records: BenchmarkResultRecord[];
};

export type BenchmarkResultWriterOptions = {
  suite: string;
  enabled?: boolean;
  outputDirectory?: string;
  runMetadata?: BenchmarkRunMetadata;
};

function gitValue(args: string[]): string | undefined {
  try {
    return execFileSync("git", args, {
      cwd: process.cwd(),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim() || undefined;
  } catch {
    return undefined;
  }
}

function hardhatVersion(): string | undefined {
  try {
    return require("hardhat/package.json").version as string;
  } catch {
    return undefined;
  }
}

function fileSafe(value: string): string {
  return value.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase();
}

/** Converts the existing execution labels into structured fields for JSON. */
export function benchmarkExecutionRecord(metadata: BenchmarkExecutionMetadata): BenchmarkExecutionRecord {
  return {
    route: metadata.route,
    gasModel: metadata.gas,
    resultModel: metadata.gas === "transaction_receipt_gas"
      ? "transaction_plus_call_result"
      : metadata.result,
    ...(metadata.transactionPreflight ? { transactionPreflight: true } : {}),
  };
}

export async function collectBenchmarkRunMetadata(): Promise<BenchmarkRunMetadata> {
  const [chain, latestBlock] = await Promise.all([
    ethers.provider.getNetwork(),
    ethers.provider.getBlock("latest"),
  ]);
  const buildInfo = await artifacts.getBuildInfo("contracts/SmartSolve.sol:SmartSolve");

  return {
    timestamp: new Date().toISOString(),
    git: {
      commit: gitValue(["rev-parse", "HEAD"]),
      branch: gitValue(["branch", "--show-current"]),
      dirty: gitValue(["status", "--porcelain"]) !== undefined,
    },
    nodeVersion: process.version,
    hardhatVersion: hardhatVersion(),
    ...(buildInfo ? {
      solidity: {
        version: buildInfo.solcVersion,
        longVersion: buildInfo.solcLongVersion,
        optimizer: {
          enabled: buildInfo.input.settings.optimizer.enabled,
          runs: buildInfo.input.settings.optimizer.runs,
        },
        viaIR: buildInfo.input.settings.viaIR,
        evmVersion: buildInfo.input.settings.evmVersion,
      },
    } : {}),
    network: { name: network.name, chainId: chain.chainId.toString() },
    ...(latestBlock ? { blockGasLimit: latestBlock.gasLimit.toString() } : {}),
  };
}

/**
 * Buffers a suite's records and writes one JSON document only when explicitly
 * enabled. Disabled writers are no-ops so ordinary tests never require disk
 * output permissions or create benchmark artifacts.
 */
export class BenchmarkResultWriter {
  private readonly records: BenchmarkResultRecord[] = [];

  private constructor(
    private readonly options: Required<Pick<BenchmarkResultWriterOptions, "suite" | "enabled" | "outputDirectory">> & {
      runMetadata: BenchmarkRunMetadata;
    },
  ) {}

  static async create(options: BenchmarkResultWriterOptions): Promise<BenchmarkResultWriter> {
    const enabled = options.enabled ?? process.env.EXPORT_BENCHMARK_RESULTS === "1";
    const fallbackMetadata: BenchmarkRunMetadata = {
      timestamp: new Date().toISOString(),
      git: {},
      nodeVersion: process.version,
    };
    const runMetadata = options.runMetadata ?? (enabled
      ? await collectBenchmarkRunMetadata()
      : fallbackMetadata);

    return new BenchmarkResultWriter({
      suite: options.suite,
      enabled,
      outputDirectory: options.outputDirectory ?? process.env.BENCHMARK_RESULTS_DIR ?? "benchmark-results",
      runMetadata,
    });
  }

  record(record: Omit<BenchmarkResultRecord, "suite">): void {
    if (!this.options.enabled) return;
    this.records.push({ suite: this.options.suite, ...record });
  }

  async flush(): Promise<string | undefined> {
    if (!this.options.enabled) return undefined;

    const timestamp = this.options.runMetadata.timestamp.replace(/[:.]/g, "-");
    const path = join(
      this.options.outputDirectory,
      `${fileSafe(this.options.suite)}-${timestamp}-${process.pid}.json`,
    );
    const document: BenchmarkResultDocument = {
      format: "smart-solve-benchmark-results-v1",
      suite: this.options.suite,
      run: this.options.runMetadata,
      records: this.records,
    };
    await mkdir(this.options.outputDirectory, { recursive: true });
    await writeFile(path, `${JSON.stringify(document, null, 2)}\n`, "utf8");
    console.log(`Benchmark results exported: ${path}`);
    return path;
  }
}
