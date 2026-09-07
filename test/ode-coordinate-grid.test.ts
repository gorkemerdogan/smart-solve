import { expect } from "chai";
import { ethers } from "hardhat";
import {
  binary128ToRational,
  type ExactRational,
} from "./precision-utils";

function abs(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function add(left: ExactRational, right: ExactRational): ExactRational {
  return {
    num: left.num * right.den + right.num * left.den,
    den: left.den * right.den,
  };
}

function multiply(left: ExactRational, right: ExactRational): ExactRational {
  return { num: left.num * right.num, den: left.den * right.den };
}

function multiplyInt(value: ExactRational, multiplier: bigint): ExactRational {
  return { num: value.num * multiplier, den: value.den };
}

/** Exact Euler sum for the decoded binary128 grid x_i = x0 + i*h. */
function eulerGridReference(x0: ExactRational, h: ExactRational, steps: bigint): ExactRational {
  const sumIndices = (steps * (steps - 1n)) / 2n;
  const sumX = add(multiplyInt(x0, steps), multiplyInt(h, sumIndices));
  return multiply(h, sumX);
}

function relativeErrorNumerator(actual: ExactRational, expected: ExactRational): {
  numerator: bigint;
  denominator: bigint;
} {
  return {
    numerator: abs(actual.num * expected.den - expected.num * actual.den),
    denominator: abs(expected.num * actual.den),
  };
}

describe("ODE fixed-grid coordinate regression", function () {
  this.timeout(120_000);

  it("samples a long Euler run on the x0 + i*h grid against a host-side reference", async function () {
    const MathLib = await ethers.getContractFactory("contracts/libraries/MathLib.sol:MathLib");
    const mathLib = await MathLib.deploy();
    await mathLib.waitForDeployment();

    const Harness = await ethers.getContractFactory("ODESolverHarness", {
      libraries: { MathLib: await mathLib.getAddress() },
    });
    const harness = await Harness.deploy();
    await harness.waitForDeployment();

    const x0 = await harness.qFromInt(1_000_000n);
    const y0 = await harness.qFromInt(0n);
    const h = await harness.qFromFrac(1n, 10n);
    const steps = 1_024n;
    const selector = harness.interface.getFunction("f_x")!.selector;

    const args = [
      await harness.getAddress(),
      selector,
      x0,
      y0,
      h,
      steps,
    ] as const;
    const [output, cumulativeOutput, gridGas, cumulativeGas] = await Promise.all([
      harness.eulerIter(...args, { gasLimit: 30_000_000n }),
      harness.eulerIterCumulative(...args, { gasLimit: 30_000_000n }),
      harness.getFunction("eulerIter").estimateGas(...args),
      harness.getFunction("eulerIterCumulative").estimateGas(...args),
    ]);

    const expected = eulerGridReference(
      binary128ToRational(x0),
      binary128ToRational(h),
      steps
    );
    const gridError = relativeErrorNumerator(binary128ToRational(output), expected);
    const cumulativeError = relativeErrorNumerator(binary128ToRational(cumulativeOutput), expected);

    // The host reference intentionally contains no per-operation binary128
    // rounding. This relative envelope permits the solver's arithmetic
    // rounding while detecting a materially drifting coordinate sequence.
    expect(gridError.numerator * (10n ** 27n)).to.be.lessThanOrEqual(gridError.denominator);
    expect(gridError.numerator * cumulativeError.denominator).to.be.lessThan(
      cumulativeError.numerator * gridError.denominator
    );
    console.log(
      `ODE_COORDINATE_GRID | steps=${steps} | gridGas=${gridGas} | ` +
      `cumulativeBaselineGas=${cumulativeGas} | delta=${gridGas - cumulativeGas}`
    );
    expect(gridGas).to.be.greaterThan(cumulativeGas);
  });
});
