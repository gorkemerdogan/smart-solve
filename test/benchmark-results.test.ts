import { expect } from "chai";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  BenchmarkResultWriter,
  benchmarkExecutionRecord,
  collectBenchmarkRunMetadata,
  type BenchmarkRunMetadata,
} from "./benchmark-results";
import { HARNESS_ESTIMATE_CALL } from "./test-utils";

const RUN_METADATA: BenchmarkRunMetadata = {
  timestamp: "2026-09-09T12:00:00.000Z",
  git: { commit: "test", branch: "test", dirty: false },
  nodeVersion: "vtest",
  hardhatVersion: "test",
  solidity: {
    version: "0.8.28",
    longVersion: "0.8.28+commit.test",
    optimizer: { enabled: true, runs: 200 },
    viaIR: true,
    evmVersion: "paris",
  },
  network: { name: "hardhat", chainId: "31337" },
  blockGasLimit: "30000000",
};

describe("benchmark result writer", function () {
  let directory: string;

  beforeEach(async function () {
    directory = await mkdtemp(join(tmpdir(), "smart-solve-benchmark-results-"));
  });

  afterEach(async function () {
    await rm(directory, { recursive: true, force: true });
  });

  it("is a no-op when export is disabled", async function () {
    const outputDirectory = join(directory, "disabled");
    const writer = await BenchmarkResultWriter.create({
      suite: "disabled-suite",
      enabled: false,
      outputDirectory,
    });
    writer.record({
      benchmark: "ignored",
      category: "test",
      operation: "ignored",
      execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
      status: "success",
    });

    expect(await writer.flush()).to.equal(undefined);
    try {
      await readdir(outputDirectory);
      expect.fail("disabled writer created an output directory");
    } catch (error) {
      expect((error as NodeJS.ErrnoException).code).to.equal("ENOENT");
    }
  });

  it("writes a structured document with run metadata and records", async function () {
    const outputDirectory = join(directory, "enabled");
    const writer = await BenchmarkResultWriter.create({
      suite: "writer-suite",
      enabled: true,
      outputDirectory,
      runMetadata: RUN_METADATA,
    });
    writer.record({
      benchmark: "sample",
      category: "root-finding",
      operation: "bisection",
      execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
      callback: {
        model: "target_staticcall",
        functionEvaluations: "44",
        gasIncludesCallback: true,
      },
      input: { iterationCap: 100 },
      gas: "1125674",
      status: "success",
      convergence: { converged: true, iterations: "42" },
    });
    writer.record({
      benchmark: "feasibility boundary",
      category: "matrix operations",
      operation: "sparse matrix-vector multiplication",
      execution: benchmarkExecutionRecord(HARNESS_ESTIMATE_CALL),
      input: { n: 128, bandwidth: 32 },
      status: "failure",
      failureKind: "out-of-gas",
    });

    const path = await writer.flush();
    expect(path).to.be.a("string");
    const document = JSON.parse(await readFile(path!, "utf8"));
    expect(document.format).to.equal("smart-solve-benchmark-results-v1");
    expect(document.suite).to.equal("writer-suite");
    expect(document.run).to.deep.equal(RUN_METADATA);
    expect(document.records).to.deep.equal([
      {
        suite: "writer-suite",
        benchmark: "sample",
        category: "root-finding",
        operation: "bisection",
        execution: {
          route: "harness_direct",
          gasModel: "estimateGas",
          resultModel: "eth_call_result",
        },
        callback: {
          model: "target_staticcall",
          functionEvaluations: "44",
          gasIncludesCallback: true,
        },
        input: { iterationCap: 100 },
        gas: "1125674",
        status: "success",
        convergence: { converged: true, iterations: "42" },
      },
      {
        suite: "writer-suite",
        benchmark: "feasibility boundary",
        category: "matrix operations",
        operation: "sparse matrix-vector multiplication",
        execution: {
          route: "harness_direct",
          gasModel: "estimateGas",
          resultModel: "eth_call_result",
        },
        input: { n: 128, bandwidth: 32 },
        status: "failure",
        failureKind: "out-of-gas",
      },
    ]);
  });

  it("collects self-describing, serializable reproducibility metadata", async function () {
    const metadata = await collectBenchmarkRunMetadata("metadata-test-suite");

    expect(metadata.project).to.include({
      packageName: "smart-solve",
      packageVersion: "1.0.0",
    });
    expect(metadata.project?.lockfile).to.deep.include({ file: "package-lock.json" });
    expect(metadata.project?.lockfile?.sha256).to.match(/^[0-9a-f]{64}$/);
    expect(metadata.toolchain?.hardhatVersion).to.be.a("string");
    expect(metadata.toolchain?.ethersVersion).to.be.a("string");
    expect(metadata.toolchain?.typescriptVersion).to.be.a("string");
    expect(metadata.toolchain?.platform).to.be.a("string");
    expect(metadata.toolchain?.architecture).to.be.a("string");
    expect(metadata.invocation?.suiteLabel).to.equal("metadata-test-suite");
    expect(metadata.invocation?.environment).to.have.all.keys(
      "EXPORT_BENCHMARK_RESULTS",
      "RUN_HEAVY_SCALABILITY",
      "RUN_HEAVY_MATRIX_SCALABILITY",
    );
    expect(metadata.solidity).to.include({ version: "0.8.28", evmVersion: "paris" });
    expect(metadata.network).to.deep.include({ name: "hardhat", chainId: "31337" });
    expect(metadata.blockGasLimit).to.equal("30000000");
    expect(() => JSON.stringify(metadata)).not.to.throw();
  });
});
