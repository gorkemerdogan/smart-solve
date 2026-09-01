import {Contract, Signer} from "ethers";
import {ethers} from "hardhat";

/**
 * A generalized interface for the contract harness, ensuring it has required Ethers methods.
 * It extends Contract to get Ethers' standard properties (like interface and getAddress).
 */
export interface Harness extends Contract {
    // Allows dynamic access to method names for estimateGas fallback check
    [key: string]: any;
}

export type ExecutionFailureKind = "revert" | "out-of-gas" | "failure";

/**
 * Classify an unexpected execution failure without hiding the original error.
 * The classification is intentionally conservative: only recognizable EVM
 * revert and gas-limit messages receive a specific label.
 */
export function classifyExecutionFailure(error: unknown): ExecutionFailureKind {
    const message = error instanceof Error
        ? `${error.name}: ${error.message}`
        : String(error);
    const normalized = message.toLowerCase();

    if (
        normalized.includes("out of gas") ||
        normalized.includes("out-of-gas") ||
        normalized.includes("exceeds block gas limit") ||
        normalized.includes("gas required exceeds allowance") ||
        normalized.includes("intrinsic gas too low")
    ) {
        return "out-of-gas";
    }

    if (
        normalized.includes("revert") ||
        normalized.includes("call exception") ||
        normalized.includes("execution reverted")
    ) {
        return "revert";
    }

    return "failure";
}

// ------------------------------------------------------------
//  Gas Estimation Utilities
// ------------------------------------------------------------

/**
 * @notify        Executes a transaction to touch the gas and confirms it. Failures are classified and propagated.
 * @param harness The contract harness instance.
 * @param method  The name of the contract method to call.
 * @param args    The arguments for the contract method.
 */
export async function touchGas(harness: Harness, method: string, args: any[]): Promise<void> {
    try {
        const data: string = harness.interface.encodeFunctionData(method, args);
        // Get the default signer, assuming it's correctly configured in the Ethers environment
        const [signer]: Signer[] = await ethers.getSigners();
        const to: string = await harness.getAddress();

        // Send a transaction with minimal configuration
        const tx = await signer.sendTransaction({to, data});

        // Wait for the transaction to be mined for a reliable touch
        await tx.wait();
    } catch (error) {
        const kind = classifyExecutionFailure(error);
        const reason = error instanceof Error ? error.message : String(error);
        throw new Error(`touchGas ${kind} for ${method}: ${reason}`);
    }
}

/**
 * @notify        Estimates the gas cost for a given contract method call.
 *                Prioritizes the contract's built-in estimateGas function, then falls back to a raw transaction estimation.
 * @param harness Contract harness instance.
 * @param method  Name of the contract method to call.
 * @param args    Arguments for the contract method.
 * @returns       Estimated gas as a string, or the string "revert" if estimation fails.
 */
export async function estimateGas(harness: Harness, method: string, args: any[]): Promise<string> {
    try {
        // Try built-in estimateGas method on the contract
        const contractMethod = harness[method];
        if (contractMethod && contractMethod.estimateGas) {
            const gasBigInt: bigint = await contractMethod.estimateGas(...args);
            return gasBigInt.toString();
        }

        // Fallback to raw signer.estimateGas
        const data: string = harness.interface.encodeFunctionData(method, args);
        const [signer]: Signer[] = await ethers.getSigners();
        const to: string = await harness.getAddress();

        const gasBigInt: bigint = await signer.estimateGas({to, data});
        return gasBigInt.toString();

    } catch (error) {
        // Return a specific string if estimation fails
        return "revert";
    }
}

// ------------------------------------------------------------
//  Block Printing Interface
// ------------------------------------------------------------

/// Interface for the data block to be printed.
interface PrintBlockMatrixData {
    t: number | string;
    method: string;
    explanation: string;
    gas: string | bigint | number;
    shapeIn?: string;
    shapeOut?: string;
    inHex?: string;
    outHex?: string;
    expectedDec?: string;
    outDec?: string;
    maxAbsError?: string;
    avgAbsError?: string;
    maxRelError?: string;
    avgRelError?: string;
    residual?: string;
    normalizedResidual?: string;
    tolerance?: string;
    withinTol?: string;
    exceededTol?: string;
    worstCase?: string;
}

/**
 * @notify     Prints a standardized, formatted block of information for a matrix test case.
 *             Uses object destructuring and template literals for clean output generation.
 * @param data The structured data to print.
 */
export function printBlockMatrix({
                                     t,
                                     method,
                                     explanation,
                                     gas,
                                     shapeIn = "-",
                                     shapeOut = "-",
                                     inHex = "-",
                                     outHex = "-",
                                     expectedDec = "-",
                                     outDec = "-",
                                     maxAbsError = "-",
                                     avgAbsError = "-",
                                     maxRelError = "-",
                                     avgRelError = "-",
                                     residual = "-",
                                     normalizedResidual = "-",
                                     tolerance = "-",
                                     withinTol = "-",
                                     exceededTol = "-",
                                     worstCase = "-",
                                 }: PrintBlockMatrixData): void {

    const sep: string = "-".repeat(60);

    const logLines: string[] = [
        sep,
        `Test: ${t}`,
        `Method: ${method}`,
        `Explanation: ${explanation}`,
        `Gas Usage: ${gas}`,
        `Shape In: ${shapeIn}`,
        `Shape Out: ${shapeOut}`,
        `Input (Hex): ${inHex}`,
        `Output (Hex): ${outHex}`,
        `Expected Output (Dec): ${expectedDec}`,
        `Output (Dec): ${outDec}`,
        `Max Abs. Error: ${maxAbsError}`,
        `Avg Abs. Error: ${avgAbsError}`,
        `Max Rel. Error: ${maxRelError}`,
        `Avg Rel. Error: ${avgRelError}`,
        `Residual: ${residual}`,
        `Normalized Residual: ${normalizedResidual}`,
        `Tolerance: ${tolerance}`,
        `Within Tol: ${withinTol}`,
        `Exceeded Tol: ${exceededTol}`,
        `Worst Case: ${worstCase}`,
        sep,
    ];

    console.log(logLines.join("\n"));
}

interface PrintBlockRegularData {
    t: number | string;
    method: string;
    explanation: string;
    gas: string | bigint | number;
    inHex?: string;
    expectedHex?: string;
    outHex?: string;
    expectedDec?: string;
    outDec?: string;
    maxAbsError?: string;
    avgAbsError?: string;
    residual?: string;
    normalizedResidual?: string;
    withinTol?: string;
    exceededTol?: string;
    tolerance?: string;
    worstCase?: string;
}

/**
 * @notify     Prints a standardized, formatted block of information for a test case.
 *             Uses object destructuring and template literals for clean output generation.
 * @param data The structured data to print.
 */
export function printBlockRegular({
                                      t,
                                      method,
                                      explanation,
                                      gas,
                                      inHex = "-",
                                      expectedHex = "-",
                                      outHex = "-",
                                      expectedDec = "-",
                                      outDec = "-",
                                      maxAbsError = "-",
                                      avgAbsError = "-",
                                      residual = "-",
                                      normalizedResidual = "-",
                                      tolerance = "-",
                                      withinTol = "-",
                                      exceededTol = "-",
                                      worstCase = "-",
                                  }: PrintBlockRegularData): void {

    const sep: string = "-".repeat(60);

    const logLines: string[] = [
        sep,
        `Test: ${t}`,
        `Method: ${method}`,
        `Explanation: ${explanation}`,
        `Gas Usage: ${gas}`,
        `Input: ${inHex}`,
        `Expected Output (hex): ${expectedHex}`,
        `Output (hex): ${outHex}`,
        `Expected Output (dec): ${expectedDec}`,
        `Output (dec): ${outDec}`,
        `Max Abs. Error: ${maxAbsError}`,
        `Avg Abs. Error: ${avgAbsError}`,
        `Residual: ${residual}`,
        `Normalized Residual: ${normalizedResidual}`,
        `Tolerance: ${tolerance}`,
        `Within Tol: ${withinTol}`,
        `Exceeded Tol: ${exceededTol}`,
        `Worst Case: ${worstCase}`,
        sep,
    ];

    console.log(logLines.join("\n"));
}

interface PrintBlockOptimizationData {
    t: string;
    method: string;
    explanation: string;
    gas: bigint | number | string;
    x0?: string;
    xFinal?: string;
    gx?: string;
    status?: string;
    iters?: string;
    gradNorm?: string;
    xError?: string;
    fError?: string;
    alpha?: string;
    extra?: string;
    withinTol?: string;
    exceededTol?: string;
    tolerance?: string;
    worstCase?: string;
}

export function printBlockOptimization({
                                           t,
                                           method,
                                           explanation,
                                           gas,
                                           x0 = "-",
                                           xFinal = "-",
                                           gx = "-",
                                           status = "-",
                                           iters = "-",
                                           gradNorm = "-",
                                           xError = "-",
                                           fError = "-",
                                           alpha = "-",
                                           extra = "-",
                                           withinTol = "-",
                                           exceededTol = "-",
                                           tolerance = "-",
                                           worstCase = "-",
                                       }: PrintBlockOptimizationData): void {
    const sep = "-".repeat(60);
    const logLines = [
        sep,
        `Test: ${t}`,
        `Method: ${method}`,
        `Explanation: ${explanation}`,
        `Gas Usage: ${gas}`,
        `Initial x: ${x0}`,
        `Final x: ${xFinal}`,
        `g(x_final): ${gx}`,
        `Status: ${status}`,
        `Iterations: ${iters}`,
        `Gradient Norm: ${gradNorm}`,
        `Solution Error: ${xError}`,
        `Objective Error: ${fError}`,
        `Final Step Size: ${alpha}`,
        `Extra: ${extra}`,
        `Tolerance: ${tolerance}`,
        `Within Tol: ${withinTol}`,
        `Exceeded Tol: ${exceededTol}`,
        `Worst Case: ${worstCase}`,
        sep,
    ];
    console.log(logLines.join("\n"));
}

/**
 * @notice Formats a fixed-point BigInt into a human-readable decimal string.
 * @dev    Splits the value into integer and fractional parts. If the scale input is 0,
 *         it defaults to 10^12. Uses padStart to preserve leading zeros in decimals.
 *         Example:
 *          Input: 1050000000000n (with SCALE = 10**12)
 *          Output: "1.050000000000"
 * @param  v The BigInt value to be formatted (supports positive and negative).
 * @param s The SCALE/denominator to use. If 0, defaults to 10**12.
 * @return Formatted decimal string with 12 decimal places.
 */
export function fmt(v: bigint, s?: bigint): string {
    const SCALE = !s || s === 0n ? 10n ** 12n : s; // default -> 10^12
    // Determine the number of decimal places for padding based on the SCALE
    const decimals = SCALE.toString().length - 1;

    const neg = v < 0n;
    const a = neg ? -v : v;

    const integerPart = a / SCALE;
    const fractionalPart = a % SCALE;

    return `${neg ? "-" : ""}${integerPart}.${fractionalPart.toString().padStart(decimals, "0")}`;
}
