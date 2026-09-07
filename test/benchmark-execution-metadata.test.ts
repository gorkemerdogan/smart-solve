import { expect } from "chai";
import {
  DIAMOND_ESTIMATE_CALL,
  formatCallbackBenchmark,
  formatBenchmarkExecution,
  HARNESS_ESTIMATE_CALL,
  HARNESS_TRANSACTION_PLUS_CALL,
} from "./test-utils";

describe("benchmark execution metadata", function () {
  it("labels harness estimates and call-simulated results separately", function () {
    expect(formatBenchmarkExecution(HARNESS_ESTIMATE_CALL)).to.equal(
      "harness_direct | estimateGas | eth_call_result"
    );
  });

  it("labels receipt gas plus a call result as a combined execution model", function () {
    expect(formatBenchmarkExecution(HARNESS_TRANSACTION_PLUS_CALL)).to.equal(
      "harness_direct | transaction_receipt_gas | transaction_plus_call_result"
    );
  });

  it("retains the Diamond route in production-path labels", function () {
    expect(formatBenchmarkExecution(DIAMOND_ESTIMATE_CALL)).to.equal(
      "diamond_routed | estimateGas | eth_call_result"
    );
  });

  it("labels callback-intertwined gas without implying subtraction", function () {
    expect(formatCallbackBenchmark({ staticCalls: 4, detail: "per successful run" })).to.equal(
      "target_staticcall | function_evaluations=4 | per successful run | " +
      "gas=algorithm_plus_callback_not_separated"
    );
  });
});
